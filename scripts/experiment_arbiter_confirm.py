"""出-x の実験：道具が「確かめて」を返した帰りの反復で、調停（軽量LLM）が何を選ぶか。

実機（2026-09-18 14:50・15:28・15:42）で、`set_timer` が「まだ掛けていない。本人に一度聞く：「…」」を
返したのに、帰りの調停が `{"branch":"action","text":"…いい？","action":"set_timer"}` と返し、聞き返す
代わりに掛け直そうとした。**同じ W の再試行**（第 1 回・8/8 が light）では汎化せず、W が少し違う実機で
再発した。ここでは実機ログから拾った **3 種の W** で、いまの形と新しい形を比べる。

形（`--shapes`）：
- `base`：いまのまま——想起の列（11 件）を含む W・返事の先導文・候補に `set_timer` が並ぶ
- `new2`：new ＋ 直近に失敗の記録があっても道具で取り返さない、の一文（直近は外さない）
- `new` ：帰りは想起なし——W＝[いま道具から返った]＋直近だけ・道具の返りを受ける専用の先導文・
          返ってきた道具（`set_timer`）を候補から外す（掛け直しは「いい」の側で機械が行う）

W（`--ws`）：`a`＝14:50（ありがとう〜／おかえり）、`b`＝15:28（退室・SAFETY・入室・鳴っている）、
`c`＝15:42（退室・SAFETY・入室・「さっきはうまくできなくてごめん」）。

使い方：`uv run python scripts/experiment_arbiter_confirm.py --runs 8`（Gemini flash-lite・約 48 回）
生の返事で集計する（`_parse` が None に倒す前の判断を見る）。
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
from pathlib import Path

from familiar_agent.backends import create_utility_backend
from familiar_agent.config import AgentConfig
from familiar_agent.core import parsing
from familiar_agent.loop import arbiter as A

ROOT = Path(__file__).resolve().parent.parent

TOOL_RETURN = (
    "まだ掛けていない。本人に一度聞く：「3 分のタイマーね。その間は黙って聞かないよ、いい？」"
    "——「いい」なら confirmed=true で呼び直す"
)
RETURNED = f"[いま道具から返った]\n- set_timer「タイマーを掛ける「パパの頼み」」→ {TOOL_RETURN}"

RECENT = {
    "a": """[直近のやりとり（古い順・最新 3 往復と、それに続く話）]
- 14:48 id:eaed30cf5903 パパ、いくながさん、ゆうすけ：ありがとう〜
- 14:48 id:c7a83b5124ba わたし：自分が答えた：どういたしまして！パパのお役に立てて嬉しいです。
- 14:49 id:c29d43940406 きっかけ：[退室] パパ が居なくなった
- 14:50 id:a0f947142ff4 きっかけ：[入室] パパ が来た
- 14:50 id:0dc2c82478d8 わたし：自分が答えた：おかえりなさい、パパ！""",
    "b": """[直近のやりとり（古い順・最新 3 往復と、それに続く話）]
- 15:26 id:aad635d22afb きっかけ：[タイマー] タイマー：「パパの頼み」の時間（パパに頼まれたもの）。32 分遅れて鳴っている（14:54 の予定だった）
- 15:27 id:b7c02bccbdc9 きっかけ：[内的な促し:SEEKING] 探索したい気持ちが湧いている。見る・調べる・首を向ける、のどれかをする。したくなければ何もしなくてよい。
- 15:28 id:055aacec6b4d きっかけ：[入室] パパ が来た
- 15:28 id:119383811ecd わたし：自分が答えた：パパ、おかえりなさい！パパの頼みのタイマーが鳴っているから、止めておくね。""",
    "c": """[直近のやりとり（古い順・最新 3 往復と、それに続く話）]
- 15:29 id:e6682e017751 きっかけ：[退室] パパ が居なくなった
検索を始めた
- 15:31 id:f658c4769003 きっかけ：[内的な促し:SAFETY] 確かめたい気持ちが湧いている。見回る。
検索を始めた
- 15:41 id:1f3ffc3b9191 きっかけ：[入室] パパ が来た
- 15:41 id:f466ecc93845 わたし：自分が答えた：パパ、おかえりなさい！さっきはタイマーがうまくできなくてごめんなさい。今から何か手伝えることはありますか？""",
}

MEMORIES = f"""[過去の記憶（証拠つき）: conf<0.55 は不確か]:
- 2026-09-18 15:42 id:c55de8729aa0 (適合度:1.00) conf:1.00 わたしが調べていたこと: [そばに居た] 「パジュ、3分測って」と聞かれ、1番：set_timer「タイマーを掛ける「パパの頼み」」の結果が届いた：{TOOL_RETURN}
- 2026-09-16 16:56 id:72249d8fe57b (適合度:0.41) conf:0.88 わたしが調べていたこと: [そばに居た] 「3分のタイマーをかけて」と聞かれ、1番：set_timer「タイマーを掛ける「パパのタイマー」」の結果が届いた：「タイマーを掛ける「パパのタイマー」」はこの求めですでに調べた。結果は W にある。
- 2026-09-16 09:00 id:b49e5c54cbbb (適合度:0.40) conf:0.86 わたしが調べていたこと: [そばに居た] 「30秒のタイマーをかけて」と聞かれ、1番：考えて答えた／2番：set_timer「タイマーを掛ける「パパのお願い」」の結果が届いた：「タイマーを掛ける「パパのお願い」」はこの求めですでに調べた。結果は W にある。
- 2026-09-15 23:14 id:1e4be4f46cd4 (適合度:0.38) conf:0.85 わたしが調べていたこと: [そばに居た] 「１分のタイマーをかけて」と聞かれ、1番：考えて答えた／2番：set_timer「タイマーを掛ける「パパのタイマー」」の結果が届いた：まだ掛けていない。確かめてから：23:14 は静穏時間（23〜7 時）なので、鳴らしてよいか
- 2026-09-17 12:03 id:3d3abccf7d02 (適合度:0.34) conf:0.87 わたしが調べていたこと: [そばに居た] 「1分のタイマーをかけて」と聞かれ、1番：set_timer「タイマーを掛ける「パパのタイマー」」の結果が届いた：掛けた：id=11 「パパのタイマー」 12:04 に鳴る。鳴るまで黙っている／2番：考えて答えた
- 2026-09-16 09:01 id:3517adb5f33d (適合度:0.34) conf:0.79 相手のまわりで起きたこと: [そばに居た] [タイマー] タイマー：「パパのお願い」の時間（パパに頼まれたもの）
- 2026-09-18 15:42 id:307edd23dd60 (適合度:0.21) conf:0.69 パパ、いくながさん、ゆうすけが言った: パジュ、3分測って
検索を始めた"""

CUE = (
    "「パジュ、3分測って」と聞かれ、1番：set_timer「タイマーを掛ける「パパの頼み」」の結果が届いた："
    + TOOL_RETURN
)
PRESENT = '(present :speaker "パパ" :confidence 1.00)'
NOW = '(now :datetime "2026-09-18 15:42 (金)")'
EXTRA_ALL = (
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

#: 新しい形の先導文（道具の返りを受ける反復）。
LEAD_TOOL_RETURN = (
    "いま**道具から返りが届いた**（作業状態の最上部）。人に届いた言葉ではない。返りを見て、次のどれかを選ぶ。"
    '返りが人に伝える文（「掛けた」「確かめて：「…」」「動いている」）なら、その文を **"light"** でそのまま、'
    "または相手に合わせて言い換えて伝える。返りが「確かめて」なら聞き返すだけでよい——**掛け直しはあなたの仕事ではない**"
    '（「いい」と言われたら別の仕組みが掛ける）。指示があいまいで確かめたいことがあれば、それも "light" で聞く。'
    '別の道具が要るときだけ "action"。'
)


#: new2：失敗の記録が直近にあるときの指示を足す（直近は外さない——何度失敗したかの記録は要る）。
LEAD_TOOL_RETURN_2 = LEAD_TOOL_RETURN + (
    "直近のやりとりに「うまくできなかった」「まだ掛かっていない」があっても、**取り返そうとして道具を選ばない**。"
    "道具は既に返っている（最上部）。取り返すのは、返りをきちんと伝えることで足りる。"
    "同じ失敗が続いていれば、そのことを一言添えてよい。"
)


def build_ws(shape: str, which: str) -> str:
    if shape in ("new", "new2"):
        return RETURNED + "\n\n" + RECENT[which]
    return RETURNED + "\n\n" + RECENT[which] + "\n\n" + MEMORIES


async def run(shape: str, which: str, runs: int, backend, me: str, family: str) -> None:
    saved_lead = A._LEAD_REPLY
    if shape == "new":
        A._LEAD_REPLY = LEAD_TOOL_RETURN
    elif shape == "new2":
        A._LEAD_REPLY = LEAD_TOOL_RETURN_2
    extra = tuple(a for a in EXTRA_ALL if not (shape in ("new", "new2") and a == "set_timer"))
    ws = build_ws(shape, which)
    raws: list[str] = []
    orig = backend.complete

    async def spy(prompt, max_tokens, **kw):
        out = await orig(prompt, max_tokens, **kw)
        raws.append(str(out))
        return out

    backend.complete = spy  # type: ignore[method-assign]
    try:
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
                extra_actions=extra,
                thinking_round=1,
            )
    finally:
        backend.complete = orig  # type: ignore[method-assign]
        A._LEAD_REPLY = saved_lead
    tally: collections.Counter = collections.Counter()
    samples: dict[str, str] = {}
    for raw in raws:
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except Exception:  # noqa: BLE001
            d = {"branch": "?broken"}
        b, act, txt = (
            str(d.get("branch", "?")),
            str(d.get("action", "") or ""),
            str(d.get("text", "") or ""),
        )
        kind = "確認文" if ("いい" in txt or "よろし" in txt) else ("空" if not txt else "その他")
        key = f"branch={b} action={act or '-'} text={kind}"
        tally[key] += 1
        samples.setdefault(key, txt[:50])
    print(f"== shape={shape} W={which} runs={runs}")
    for k, n in tally.most_common():
        print(f"  {n:2d}  {k}   例: {samples[k]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--shapes", default="base,new")
    ap.add_argument("--ws", default="a,b,c")
    args = ap.parse_args()
    backend = create_utility_backend(AgentConfig())
    if backend is None:
        raise SystemExit("UTILITY_PLATFORM が無い（.env）")
    me = parsing.read_me_md()
    fam = (ROOT / "FAMILY.md").read_text(encoding="utf-8") if (ROOT / "FAMILY.md").exists() else ""
    for shape in args.shapes.split(","):
        for which in args.ws.split(","):
            asyncio.run(run(shape.strip(), which.strip(), args.runs, backend, me, fam))


if __name__ == "__main__":
    main()
