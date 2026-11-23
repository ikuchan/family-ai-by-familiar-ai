"""知-bo 段 2：「印刷して」と頼まれたら、主LLM が直前の会話の中身を題・日付・本文にまとめて刷る（2026-11-22）。

本人の決定（2026-11-21）：刷る前に題を言って 1 回確かめる（「『〇〇』を 1 枚刷るね。いい？」・タイマーの確かめと同じ形）。
確かめは機械が持つ（`confirm_state.PendingConfirm`）。「いい」なら機械が預かった入力で刷る（`confirmed=True`）。本物の紙は刷らない。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core import completion_kind, confirm_state
from familiar_agent.tools.print_tool import PrintTool

INPUT = {"title": "カレーの作り方", "date": "2026年11月22日", "body": "- 玉ねぎを炒める\n- 煮込む"}


def _tool(tmp_path, *, send=None, pages=1):
    asked: list = []
    rendered: list = []
    written: list = []

    async def ask(pc):
        asked.append(pc)

    def render(title, date_text, body, path: Path):
        rendered.append((title, date_text, body, path.name))
        path.write_bytes(b"%PDF-1.4")
        return pages

    oif = MagicMock()

    async def write(mi, **kw):
        written.append(mi.content)
        return "obs-1"

    oif.write = write
    tool = PrintTool(
        oif=oif,
        speaker=lambda: "ママ",
        printer="Brother_DCP_J552N",
        folder=tmp_path,
        ask=ask,
        render=render,
        send=send or (lambda path, *, printer: "Brother_DCP_J552N-7"),
    )
    return tool, asked, rendered, written


def test_the_tool_is_offered_with_title_date_and_body(tmp_path):
    tool, *_ = _tool(tmp_path)
    (d,) = tool.get_tool_definitions()
    assert d["name"] == "print_document"
    assert set(d["input_schema"]["required"]) == {"title", "body"}


def test_it_asks_once_with_the_title_before_printing(tmp_path):
    tool, asked, rendered, _ = _tool(tmp_path)
    text, ok = asyncio.run(tool.call("print_document", INPUT))
    assert ok and text.startswith("まだ刷っていない")
    assert [r[3] for r in rendered] == ["count.pdf"]  # 枚数を数えるために組んだだけ
    assert list(tmp_path.iterdir()) == []  # 数えた PDF は残さない
    (pc,) = asked
    assert pc.action == "print_document" and pc.tool_input == INPUT
    assert pc.text == "『カレーの作り方』を 1 枚刷るね。いい？"


def test_confirmed_it_prints_and_writes_a_record(tmp_path):
    tool, asked, rendered, written = _tool(tmp_path)
    text, ok = asyncio.run(tool.call("print_document", INPUT, confirmed=True))
    assert ok and asked == [] and text.startswith("刷った")
    assert rendered == [
        ("カレーの作り方", "2026年11月22日", "- 玉ねぎを炒める\n- 煮込む", "print.pdf")
    ]
    assert written == ["ママに頼まれて『カレーの作り方』を 1 枚刷った（受付 Brother_DCP_J552N-7）"]
    assert list(tmp_path.iterdir()) == []  # 会話の中身をファイルに残さない


def test_a_long_body_is_asked_with_the_real_number_of_pages(tmp_path):
    tool, asked, *_ = _tool(tmp_path, pages=2)
    asyncio.run(tool.call("print_document", INPUT))
    assert asked[0].text == "『カレーの作り方』を 2 枚刷るね。いい？"


def test_a_folder_that_cannot_be_made_is_a_failure_not_a_maybe(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    tool, *_ = _tool(blocker)  # ファイルの下にフォルダは作れない
    text, ok = asyncio.run(tool.call("print_document", INPUT, confirmed=True))
    assert not ok and text.startswith("刷れなかった")


def test_a_missing_title_or_body_is_refused(tmp_path):
    tool, asked, *_ = _tool(tmp_path)
    text, ok = asyncio.run(tool.call("print_document", {"title": "", "body": "x"}))
    assert not ok and asked == []


def test_a_printer_that_refuses_is_a_failure(tmp_path):
    from familiar_agent.io.printer import PrintError

    def send(path, *, printer):
        raise PrintError("プリンターが見つかりません")

    tool, *_ = _tool(tmp_path, send=send)
    text, ok = asyncio.run(tool.call("print_document", INPUT, confirmed=True))
    assert not ok and "見つかりません" in text


def test_the_prompt_and_the_completion_know_printing():
    pc = confirm_state.PendingConfirm(action="print_document")
    assert (
        confirm_state.prompt_text(pc)
        == "いま印刷するか確かめているところだよ。はいか、いいえで答えてね"
    )
    assert "print_document" in confirm_state.BLOCKED_WHILE_CONFIRMING
    asked = completion_kind.kind_of(
        "print_document", failed=False, result="まだ刷っていない。確かめてから", origin="発話"
    )
    assert asked == ("確認の問いを出した", ("silent",))
    done = completion_kind.kind_of("print_document", failed=False, result="刷った", origin="発話")
    assert done is not None and done[1] == ("tell_light",)


def test_a_print_pending_goes_to_the_print_tool_and_a_no_says_it_is_not_printed(monkeypatch):
    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.loop.paju_state import PajuState

    async def go():
        a = EmbodiedAgent.__new__(EmbodiedAgent)
        a.paju_state = PajuState(confirm_ttl_sec=300.0)
        a._print_tool = MagicMock()
        a._print_tool.call = AsyncMock(return_value=("刷った", True))
        monkeypatch.setattr("familiar_agent.agent.time.time", lambda: 10_000.0)
        pc = confirm_state.PendingConfirm(
            action="print_document",
            tool_input=INPUT,
            asked_at=10_000.0,
            text="q",
            what="『カレーの作り方』",
        )
        await a.ask_confirm(pc)
        await a.resolve_confirm(True, now="t")
        a._print_tool.call.assert_awaited_once_with("print_document", INPUT, confirmed=True)
        await a.ask_confirm(pc)
        text, ok = await a.resolve_confirm(False, now="t")
        assert ok and text == "やめた：『カレーの作り方』は刷らない"

    asyncio.run(go())


def test_the_main_llm_gets_the_tool_and_the_loop_runs_it():
    from familiar_agent.loop import actions, workspace

    assert "print_document" in actions._FULL_ACTIONS
    assert "print_document" in actions._LOOKUP_ACTIONS
    assert "print_document" in workspace.RETURN_WITHOUT_RECALL
    assert actions._query_label("print_document", {"title": "カレー"}) == "印刷する「カレー」"


def test_without_a_qt_app_the_tool_is_not_offered(tmp_path, monkeypatch):
    from PySide6.QtGui import QGuiApplication

    from familiar_agent.loop import actions

    tool, *_ = _tool(tmp_path)
    agent = MagicMock(_print_tool=tool)
    monkeypatch.setattr(QGuiApplication, "instance", staticmethod(lambda: None))
    assert actions._print_def(agent) == []
    monkeypatch.setattr(QGuiApplication, "instance", staticmethod(lambda: object()))
    assert [d["name"] for d in actions._print_def(agent)] == ["print_document"]
