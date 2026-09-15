"""自発の発話は、人が話しかける前でも画面に出る（2026-09-15 実機 21:50）。

パジュへのメモの求めが起動直後に走り、主LLM は 66 字を「話した」（`DIF speak`）のに、GUI の
吹き出しには載らなかった。ループの出口（`_on_text`／`_on_action`）は人の発話（`push_utterance`）
で初めて結ばれていたため。アプリは起動時に出口を渡す（`agent.set_output`）。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.loop.event_loop import InformationProcessing


def test_emit_reaches_an_action_sink_registered_at_startup():
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    seen: list = []
    ip.set_output(None, on_action=lambda name, inp: seen.append((name, inp)))
    ip._emit("こんばんは")
    assert seen == [("say", {"text": "こんばんは"})]


def test_emit_with_no_sink_warns_instead_of_vanishing(caplog):
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    with caplog.at_level("WARNING", logger="familiar_agent.loop.event_loop"):
        ip._emit("誰にも見えない")
    assert any("表示先が無い" in r.message for r in caplog.records)


def test_the_agent_exposes_set_output_for_the_app():
    from familiar_agent.agent import EmbodiedAgent as Agent

    a = MagicMock(spec=Agent)
    a._info_processing = MagicMock()
    a._ensure_event_loop = lambda on_text=None, on_action=None: a._info_processing.set_output(
        on_text, on_action=on_action
    )
    Agent.set_output(a, None, on_action="sink")
    a._info_processing.set_output.assert_called_once_with(None, on_action="sink")


def test_the_gui_registers_its_bubble_sink_after_starting_autonomy():
    import inspect

    from familiar_agent import gui

    src = inspect.getsource(gui.FamiliarWindow._initialize_agent)
    assert "start_autonomy()" in src and "set_output(" in src
    assert src.index("start_autonomy()") < src.index("set_output(")
