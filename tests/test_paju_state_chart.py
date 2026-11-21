"""環-bb 一段目：パジュの状態図の図は、コードから作る（2026-11-17 本人の決定 (2)「図はコードから自動で作る」）。

`loop/paju_state_chart.mermaid()` が、状態図の型（`PajuState`）から領域・状態・移り・事象を読み出し、Mermaid の文字にする。
設計の文書（`設計方針_パジュの状態` §2.1）の図は手で書かず、これを貼る。文書の図が作った図と違えば落ちる（図がコードから
ずれない）。図にする範囲は状態図だけ（仮・本人の確認待ち）。
"""

from __future__ import annotations

import re
from pathlib import Path

from familiar_agent.loop import paju_state_chart as chart

DOCS = Path(__file__).resolve().parents[1] / "docs" / "NewModesDocs"


def _doc_block() -> str:
    docs = list(DOCS.glob("familiar-ai_設計方針_パジュの状態_v*.md"))
    assert len(docs) == 1, [d.name for d in docs]  # 版は 1 つだけ置く
    [doc] = docs
    text = doc.read_text(encoding="utf-8")
    sec = text[text.index("### 2.1 実装した頭") :]
    m = re.search(r"```mermaid\n(.*?)```", sec, re.S)
    assert m, "§2.1 に Mermaid の図が無い"
    return m.group(1)


def test_the_design_document_shows_the_chart_made_from_the_code():
    assert _doc_block() == chart.mermaid(), (
        "§2.1 の図がコードとずれている。`uv run python -m familiar_agent.loop.paju_state_chart` の出力を貼る"
    )


def test_every_transition_of_the_chart_is_drawn():
    """移りの元・先・事象の組が、図の行とちょうど一致する（描き漏れも余分も無い）。図からは行を読み、型からは
    ライブラリの移りを直に数える（作り手と同じ絞り込みを使わない）。"""
    from familiar_agent.loop.paju_state import PajuState

    def walk(st):
        yield st
        for k in st.states or []:
            yield from walk(k)

    want = set()
    for top in PajuState.states:
        for st in walk(top):
            for t in st.transitions:
                if t.event and not t.internal and t.source is not t.target:
                    want.add((chart._name(t.source), chart._name(t.target), t.event))
    got = set()
    for line in chart.mermaid().splitlines():
        m = re.match(r"\s*(\S+) --> (\S+) : (.*)", line)
        if m:
            for label in m.group(3).split("、"):
                got.add((m.group(1), m.group(2), label.split("（")[0]))
    assert got == want


def test_drawn_names_do_not_collide():
    names = chart.names()
    assert len(names) == len(set(names)), sorted(n for n in names if names.count(n) > 1)


def test_regions_and_states_carry_japanese_names():
    text = chart.mermaid()
    assert text.startswith("stateDiagram-v2\n    direction TB\n")
    for ident in ("head", "listening", "no_voice", "alarm_quiet", "hearing"):
        assert ident not in text  # 英字の id は出さない
    assert "聞いている --> 聞かない : request_opened（頭が寝ていなければ）、speech_began" in text


def test_each_transition_is_logged_at_info_for_replay(caplog):
    """移りの 1 行は情報の段に残す（環-bb 二段目・本人の決定 (4)「移りの 1 行を info に上げてログに残し」）。入るのは事象と
    状態の名前だけで、家族の言葉（事象と一緒に渡すデータ）は入らない。"""
    import asyncio
    import logging

    from familiar_agent.loop.paju_state import PajuState

    async def go():
        sm = PajuState()
        await sm.start()
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await sm.fire("ring_began", text="ひみつのことば")  # データは 1 行に出さない

    asyncio.run(go())
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and "で移った" in r.getMessage()
    ]
    assert lines and re.fullmatch(
        r"状態図 ring_began で移った（頭=\S* 耳=\S*/窓=\S*/答え=\S*/入口=\S* 口=\S*/声=\S*/知らせの音=\S*）",
        lines[-1],
    )
    assert all("ひみつ" not in r.getMessage() for r in caplog.records)
