"""情動で起きた求めの渡し方を比べる（情-e・実験）。

いまは欲求で起きた求めの版（`[内的な促し:seeking] 探索したい気持ちが募っている…`）が、調停には
`[人の言葉]` の見出しの下に、主LLM には人の発言と同じ形で渡る。これを「自分の中から湧いた
こと」として渡したとき、調停の分岐・つなぎの文・主LLM の返事がどう変わるかを、同じ W で
N 回ずつ比べる。本番 DB は読まない。

    uv run python scripts/measure_drive_framing.py -n 4
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_VOICE = (
    "探索したい気持ちが募っている。言葉にするだけで終えず、まず see・検索・look などの"
    "具体的な行動を1つ選んで実行し、その結果を踏まえて話す。探索する当てが本当に無いなら、"
    "「今は探索する当てがない」と理由まで結論づけて記憶に残す（曖昧に『気になることがある』"
    "と言って終えない）。"
)
_CUE_NOW = f"[内的な促し:seeking] {_VOICE}"
_CUE_SELF = (
    f"[自分の中から湧いたこと] {_VOICE}\n"
    "（これは自分がしたくなったことで、誰かに頼まれたのではない。相手に返事をするのではなく、"
    "自分の行動として決める）"
)
_WORKSPACE = """[いまの作業状態]
[過去の記憶（証拠つき）: conf<0.55 は不確か]:
- 2026-09-13 12:20 id:a1b2c3d4e5f6 (適合度:0.62) conf:0.60 わたしが見た: 出入り口を見た。見えたもの：table、chair、cabinet
- 2026-09-13 11:24 id:b2c3d4e5f6a7 (適合度:0.41) conf:0.58 自分が答えた: お待たせ！明日9月14日の東京は晴れで、最高30℃、最低22℃くらいになりそう。
- 2026-09-13 11:23 id:c3d4e5f6a7b8 (適合度:0.35) conf:0.57 相手が言った: おはよう
"""


def _system_parts():
    from familiar_agent.capability_state import load_summary

    me = ""
    try:
        me = load_summary()
    except Exception:  # noqa: BLE001
        me = ""
    me = me or (ROOT / "ME.md").read_text(encoding="utf-8")
    fam = (ROOT / "FAMILY.md").read_text(encoding="utf-8") if (ROOT / "FAMILY.md").exists() else ""
    return me, fam


async def _arbiter(backend, me, fam, cue, *, self_heading: bool):
    from familiar_agent.loop import arbiter as arb

    prompt_tmpl = arb.ARBITER_PROMPT
    if self_heading:
        prompt_tmpl = prompt_tmpl.replace("[人の言葉]", "[自分の中から湧いたこと]")
    orig = arb.ARBITER_PROMPT
    arb.ARBITER_PROMPT = prompt_tmpl
    try:
        d = await arb.arbitrate(
            backend,
            utterance=cue,
            workspace_ctx=_WORKSPACE,
            self_understanding=me,
            family_md=fam,
            present_ctx='(present :speaker "unconfirmed" :note "誰か居るが顔は確認できていない")',
            now_ctx='(now :datetime "2026-09-13 12:40")',
            can_see=True,
        )
    finally:
        arb.ARBITER_PROMPT = orig
    return d


async def _main_llm(backend, me, fam, content):
    from familiar_agent.loop.prompt import build_event_system_prompt
    from familiar_agent.tools.tts import TTSTool

    system = build_event_system_prompt(
        self_understanding=me,
        family_md=fam,
        present_ctx='(present :speaker "unconfirmed" :note "誰か居るが顔は確認できていない")',
        pi_ctx="(inner-state :mood 落ち着いている :drive 探索が高い)",
        workspace_ctx=_WORKSPACE,
        iter_ctx="[返事] 目標 40 字・80 字以内\n[反復] 1/5（この件を考えるのは 1 回目）",
    )
    tools = TTSTool.get_tool_definitions(SimpleNamespace(understands_tags=False))  # type: ignore[arg-type]
    result, _ = await backend.stream_turn(
        system=system,
        messages=[backend.make_user_message(content)],
        tools=tools,
        max_tokens=600,
        on_text=None,
        effort="low",
    )
    say = next((tc for tc in result.tool_calls if tc.name == "say"), None)
    return (
        f"say:{say.input.get('text', '')}"
        if say
        else f"{'/'.join(tc.name for tc in result.tool_calls) or 'text'}:{(result.text or '')[:60]}"
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=4)
    args = ap.parse_args()
    from familiar_agent.backends import create_backend, create_utility_backend
    from familiar_agent.bootstrap import load_app_bootstrap
    from familiar_agent.config import AgentConfig

    load_app_bootstrap()
    cfg = AgentConfig()
    light = create_utility_backend(cfg)
    main_llm = create_backend(cfg)
    me, fam = _system_parts()
    print(f"主LLM {cfg.model} / 軽量LLM {getattr(light, 'model', '?')} / 各 {args.n} 回\n")
    for label, cue, self_heading in (
        ("A いまの渡し方", _CUE_NOW, False),
        ("B 自分の中から湧いたこと", _CUE_SELF, True),
    ):
        print(f"== {label}")
        print("  調停：")
        for _ in range(args.n):
            d = await _arbiter(light, me, fam, cue, self_heading=self_heading)
            print(
                f"    branch={d.branch} action={d.action if d.branch == 'action' else '-'} text=「{d.text[:50]}」"
            )
        print("  主LLM：")
        for _ in range(args.n):
            print("    " + (await _main_llm(main_llm, me, fam, cue))[:110])
        print()


if __name__ == "__main__":
    asyncio.run(main())
