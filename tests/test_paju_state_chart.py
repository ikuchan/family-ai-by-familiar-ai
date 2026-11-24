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
    docs = [next(DOCS.rglob("familiar-ai_設計方針_パジュの状態.md"))]
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
                    # 入れ子の中へ直に向かう移りは、入れ子の箱へ向けて描く（行き先は事象のあとに添える）
                    target = t.target
                    while target.parent is not None and target.parent is not t.source.parent:
                        target = target.parent
                    want.add((chart._name(t.source), chart._name(target), t.event))
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
    assert (
        "答えを言っている --> 考えている : speech_ended（書き留めているへ）" in text
    )  # 箱の中へ直に向けない


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
        r"状態図 ring_began で移った（頭=\S* 耳=\S*/窓=\S*/答え=\S*/入口=\S* 口=\S*/声=\S*/知らせの音=\S*/歌=\S*）",
        lines[-1],
    )
    assert all("ひみつ" not in r.getMessage() for r in caplog.records)


# ── 三段目・四段目：いまの状態を光らせる、ログから読む ─────────────────────────


def test_the_live_states_are_named_and_highlighted():
    import asyncio

    from familiar_agent.loop.paju_state import PajuState

    async def go():
        sm = PajuState()
        await sm.start()
        await sm.fire("ring_began")
        return sm

    sm = asyncio.run(go())
    now = chart.active_names(sm)
    assert {"何もしていない", "聞いている", "閉じている", "答えなし", "音鳴っている"} <= now
    assert not now & {"パジュ", "頭", "耳"}  # 領域は光らせない
    text = chart.highlighted(now)
    assert text.startswith(chart.mermaid())
    assert "    音鳴っている:::now\n" in text and "classDef now " in text


def test_transitions_are_read_from_the_log():
    lines = [
        "2026-11-21 20:01:02,345 [v0.x] [INFO] familiar_agent.loop.paju_state: 状態図 request_opened で移った"
        "（頭=考えている/思い出している 耳=聞かない/窓=開いている/答え=なし/入口=開いている 口=話してよい/声=なし/知らせの音=止まっている）",
        "2026-11-21 20:01:03,000 [v0.x] [INFO] familiar_agent.loop.event_loop: ほかの行",
    ]
    [(at, event, desc)] = chart.transitions_in(lines)
    assert at == "2026-11-21 20:01:02" and event == "request_opened"
    assert desc.startswith("頭=考えている/思い出している 耳=聞かない")


def test_a_logged_state_line_maps_back_to_drawn_names():
    """再生（四段目）：ログの状態の行を、図の上の名前に戻す。名前が重なる状態（なし・開いている）も領域で見分ける。"""
    desc = (
        "頭=考えている/思い出している 耳=聞かない/窓=開いている/答え=なし/入口=開いている 口=話してよい/声=なし"
        "/知らせの音=止まっている/歌=鳴っている"
    )
    assert chart.names_from_describe(desc) == {
        "考えている",
        "思い出している",
        "聞かない",
        "開いている",
        "答えなし",
        "入口が開いている",
        "話してよい",
        "声なし",
        "音止まっている",
        "歌鳴っている",
    }
    assert chart.names_from_describe("頭=寝ている 耳=聞いている") == {"寝ている", "聞いている"}


def test_the_live_and_logged_names_agree():
    """動いている状態図から作った名前と、その状態の行から戻した名前が同じ（2 つの読み方がずれない）。"""
    import asyncio

    from familiar_agent.loop.paju_state import PajuState

    async def go():
        sm = PajuState()
        await sm.start()
        for ev in ("request_opened", "decide_began", "window_opened", "ring_began", "cue_began"):
            await sm.fire(ev)
        return sm

    sm = asyncio.run(go())
    assert chart.names_from_describe(sm.describe()) == chart.active_names(sm)


def test_state_names_have_no_separators():
    """状態の行は「 」と「/」で区切るので、状態の名前に含めない（含めると再生で光らない）。"""
    from familiar_agent.loop.paju_state import _NAMES

    assert not [n for n in _NAMES.values() if " " in n or "/" in n or "=" in n]
