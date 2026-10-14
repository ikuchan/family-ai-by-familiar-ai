"""反復ごとの続き先の判定を撤去した（記-t 段 3・2026-10-15・本人の決定）。

継起は、最終版を書いた直後に最終版どうしで結ぶ（段 2・`_link_succession`）。反復ごとに W の行 → 起点へ張っていた
判定（`judge_follows`）は、起点どうしの意味のない網を作り、書かれたのも 3 本だけだった。計測ログの「続き先」「直近」と、
それを読む層 3 の窓の調整（続きが 20 件に届かず一度も動かなかった）も外す。直近の往復数はいまの値で固定する。
"""

from __future__ import annotations

import inspect
from pathlib import Path

from familiar_agent.backends import ToolCall
from familiar_agent.core import jev_judges, measure
from familiar_agent.loop import rest_settings, workspace
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.loop.iteration import Iteration
from tests.test_event_loop import _agent, _run, _turn


def test_the_per_iteration_judgement_is_gone():
    for name in ("_judge_follows", "_note_follows", "_follows_view"):
        assert not hasattr(InformationProcessing, name), name
    assert "follows" not in inspect.getsource(Iteration)
    assert not hasattr(jev_judges, "judge_follows")
    assert not hasattr(jev_judges, "JudgeFailed")
    assert not hasattr(workspace, "link_follows")
    assert not hasattr(workspace.Workspace, "for_jev_follows")


def test_settings_no_longer_move_the_recent_window():
    assert not hasattr(rest_settings, "adjust_window")
    assert not hasattr(measure, "summarize_window")


def test_a_turn_writes_neither_follows_nor_recent_to_the_measure_log(tmp_path):
    measure.setup(base_dir=tmp_path)
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    _run(a, utterance="さっきの話だけど")
    path: Path = tmp_path / "rest_logs" / "measure.log"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    assert not [x for x in lines if " 続き先 " in x or " 直近 " in x]


def test_the_only_succession_writer_is_between_finals():
    """**反証の側**：継起を書くのは最終版どうし（段 2）だけで、それは残っている。"""
    src = inspect.getsource(InformationProcessing._link_succession)
    assert "oif.link(KIND_SUCCESSION" in src
