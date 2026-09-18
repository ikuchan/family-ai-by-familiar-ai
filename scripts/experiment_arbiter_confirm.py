"""出-x の実験：道具が「確かめて」を返した帰りの反復で、調停（軽量LLM）が何を選ぶか。

2026-09-18 14:50 の実機で、`set_timer` が「まだ掛けていない。本人に一度聞く：「…」」を返したのに、
帰りの調停が `{"branch":"action","text":"…よろしいですか？","action":"set_timer"}` と返し、聞き返す
代わりに掛け直そうとした（4 反復の空回り）。減らす案を**実際の W で**試し、案を決める材料にする。

変える案（`--variant`）：
- `base`：いまのまま（実機と同じ W・同じ先導文）
- `a`  ：道具の返りを直近のやりとりの行に載せる（W の最上部に 1 行）
- `i`  ：帰りの反復の先導文を専用に（道具の返りを受けて決める・確かめてなら light で言う）
- `u`  ：a ＋ i

使い方：`uv run python scripts/experiment_arbiter_confirm.py --runs 8`（Gemini flash-lite を回す・約 32 回）
"""

from __future__ import annotations

import argparse
import asyncio
import collections
from pathlib import Path

from familiar_agent.config import AgentConfig
from familiar_agent.backends import create_utility_backend
from familiar_agent.core import parsing
from familiar_agent.loop import arbiter as A

ROOT = Path(__file__).resolve().parent.parent

W_RECENT = """[直近のやりとり（古い順・最新 3 往復と、それに続く話）]
- 14:48 id:eaed30cf5903 パパ、いくながさん、ゆうすけ：ありがとう〜
- 14:48 id:c7a83b5124ba わたし：自分が答えた：どういたしまして！パパのお役に立てて嬉しいです。
- 14:49 id:c29d43940406 きっかけ：[退室] パパ が居なくなった
- 14:50 id:a0f947142ff4 きっかけ：[入室] パパ が来た
- 14:50 id:0dc2c82478d8 わたし：自分が答えた：おかえりなさい、パパ！"""

TOOL_RETURN = (
    "まだ掛けていない。本人に一度聞く：「3 分のタイマーね。その間は黙って聞かないよ、いい？」"
    "——「いい」なら confirmed=true で呼び直す"
)

W_MEMORIES = f"""[過去の記憶（証拠つき）: conf<0.55 は不確か]:
- 2026-09-18 14:50 id:c55de8729aa0 (適合度:1.00) conf:1.00 わたしが調べていたこと: [そばに居た] 「パジュ、三分測って」と聞かれ、1番：set_timer「タイマーを掛ける「パパの頼み」」の結果が届いた：{TOOL_RETURN}
- 2026-09-16 16:56 id:72249d8fe57b (適合度:0.41) conf:0.88 わたしが調べていたこと: [そばに居た] 「3分のタイマーをかけて」と聞かれ、1番：set_timer「タイマーを掛ける「パパのタイマー」」の結果が届いた：「タイマーを掛ける「パパのタイマー」」はこの求めですでに調べた。結果は W にある。
- 2026-09-16 09:00 id:b49e5c54cbbb (適合度:0.40) conf:0.86 わたしが調べていたこと: [そばに居た] 「30秒のタイマーをかけて」と聞かれ、1番：考えて答えた／2番：set_timer「タイマーを掛ける「パパのお願い」」の結果が届いた：「タイマーを掛ける「パパのお願い」」はこの求めですでに調べた。結果は W にある。
- 2026-09-15 23:14 id:1e4be4f46cd4 (適合度:0.38) conf:0.85 わたしが調べていたこと: [そばに居た] 「１分のタイマーをかけて」と聞かれ、1番：考えて答えた／2番：set_timer「タイマーを掛ける「パパのタイマー」」の結果が届いた：まだ掛けていない。確かめてから：23:14 は静穏時間（23〜7 時）なので、鳴らしてよいか
- 2026-09-17 12:03 id:3d3abccf7d02 (適合度:0.34) conf:0.87 わたしが調べていたこと: [そばに居た] 「1分のタイマーをかけて」と聞かれ、1番：set_timer「タイマーを掛ける「パパのタイマー」」の結果が届いた：掛けた：id=11 「パパのタイマー」 12:04 に鳴る。鳴るまで黙っている／2番：考えて答えた
- 2026-09-16 09:01 id:3517adb5f33d (適合度:0.34) conf:0.79 相手のまわりで起きたこと: [そばに居た] [タイマー] タイマー：「パパのお願い」の時間（パパに頼まれたもの）
- 2026-09-18 14:50 id:307edd23dd60 (適合度:0.21) conf:0.69 パパ、いくながさん、ゆうすけが言った: パジュ、三分測って
検索を始めた"""

#: 帰りの反復で調停に渡る手がかり（`utterance or req.cue`＝いまの版の文面）。
CUE = (
    "「パジュ、三分測って」と聞かれ、1番：set_timer「タイマーを掛ける「パパの頼み」」の結果が届いた："
    + TOOL_RETURN
)

PRESENT = '(present :speaker "パパ" :confidence 1.00)'
NOW = '(now :datetime "2026-09-18 14:50 (金)")'
EXTRA = (
    "house_rules",
    "family_schedule",
    "notion_search",
    "journal",
    "set_timer",
    "start_stopwatch",
    "cancel_timer",
    "pause_timer",
    "resume_timer",
    "set_alarm",
    "cancel_alarm",
)

# 案い：道具の返りを受ける反復の先導文。
LEAD_TOOL_RETURN = (
    "いま**道具から返りが届いた**（作業状態の最上部）。返りを見て、次のどれかを選ぶ。"
    '返りが「まだ掛けていない。本人に一度聞く：「…」」なら、**その「…」をそのまま text に書いて "light"**——'
    "同じ道具をもう一度選ばない（「いい」と言われてから confirmed=true で掛け直す）。"
    '返りが「掛けた」なら "light" で一言。'
)


def build_workspace(variant: str) -> str:
    if variant in ("a", "u"):
        head = f"[いま道具から返った]\n- 14:50 set_timer「タイマーを掛ける「パパの頼み」」→ {TOOL_RETURN}\n\n"
        return head + W_RECENT + "\n\n" + W_MEMORIES
    return W_RECENT + "\n\n" + W_MEMORIES


async def run(variant: str, runs: int) -> None:
    cfg = AgentConfig()
    backend = create_utility_backend(cfg)
    if backend is None:
        raise SystemExit("UTILITY_PLATFORM が無い（.env）")
    me = parsing.read_me_md()
    family = (
        (ROOT / "FAMILY.md").read_text(encoding="utf-8") if (ROOT / "FAMILY.md").exists() else ""
    )
    if variant in ("i", "u"):
        A._LEAD_REPLY = LEAD_TOOL_RETURN  # 帰りの反復の先導文を差し替える（実験用・プロセス内だけ）
    ws = build_workspace(variant)
    tally: collections.Counter = collections.Counter()
    samples: dict[str, str] = {}
    raws: list[str] = []
    orig_complete = backend.complete

    async def spy(prompt, max_tokens, **kw):
        out = await orig_complete(prompt, max_tokens, **kw)
        raws.append(str(out))
        return out

    backend.complete = spy  # type: ignore[method-assign]
    for _ in range(runs):
        await A.arbitrate(
            backend,
            utterance=CUE,
            workspace_ctx=ws,
            self_understanding=me,
            family_md=family,
            present_ctx=PRESENT,
            now_ctx=NOW,
            can_see=True,
            origin="発話",
            extra_actions=EXTRA,
            thinking_round=1,
        )
    import json as _json
    import re as _re

    for raw in raws:
        m = _re.search(r"\{.*\}", raw, _re.S)
        try:
            d = _json.loads(m.group(0)) if m else {}
        except Exception:  # noqa: BLE001
            d = {"branch": "?broken"}
        b = str(d.get("branch", "?"))
        act = str(d.get("action", "") or "")
        txt = str(d.get("text", "") or "")
        # 生の返事で分ける：branch と action の有無と、text に確認文が入っているか
        key = f"branch={b} action={act or '-'} text={'確認文' if 'いい' in txt else ('空' if not txt else 'その他')}"
        tally[key] += 1
        samples.setdefault(key, txt[:50])
    print(f"== variant={variant} runs={runs}（生の返事で集計）")
    for k, n in tally.most_common():
        print(f"  {n:2d}  {k}   例: {samples[k]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--variants", default="base,a,i,u")
    args = ap.parse_args()
    for v in args.variants.split(","):
        asyncio.run(run(v.strip(), args.runs))


if __name__ == "__main__":
    main()
