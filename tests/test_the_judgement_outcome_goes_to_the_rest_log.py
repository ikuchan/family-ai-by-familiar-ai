"""続き先の判定の結末を、REST が読む専用の計測ログへ書く（記-i・2026-09-13）。

結末は 5 通り——続き（辺を書いた）／途切れ（none）／未判定（言葉が無い・W が空）／
落ちた（例外・時間切れ）／不一致（返った id が W に無い・自分の起点）。以前は「続き」しか
形に残らず、継起が 0 本のとき「判定が壊れている」のか「続きの場面が無かった」のかを
切り分けられなかった。

計測ログは `~/.cache/familiar-ai/rest_logs/measure.log`（`logs/` の兄弟）。起動時に回転せず、
REST 内省が読み終えたら改名する（記-a-に）。書き手は `WatchedFileHandler` なので、改名の
あとは次の行で新しい file を作り直す。`app.log` には流さない。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock

from familiar_agent.backends import ToolCall
from familiar_agent.core import measure
from tests.test_event_loop import _agent, _run, _turn


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def test_the_measure_log_lives_next_to_logs_and_is_not_rotated_at_startup(tmp_path):
    path = measure.setup(base_dir=tmp_path)
    assert path == tmp_path / "rest_logs" / "measure.log"
    measure.record("続き先", 結末="途切れ", 起点="c08eb01eccd8")
    measure.record("続き先", 結末="続き", 相手="2449f3006662", 起点="4b5becd539d0")
    got = _lines(path)
    assert len(got) == 2
    assert got[0].endswith(" 続き先 結末=途切れ 起点=c08eb01eccd8")
    assert "結末=続き 相手=2449f3006662" in got[1]
    # もう一度 setup しても（再起動）、行は消えない。
    measure.setup(base_dir=tmp_path)
    measure.record("続き先", 結末="未判定")
    assert len(_lines(path)) == 3


def test_a_renamed_file_is_recreated_on_the_next_line(tmp_path):
    """REST が読み終えて改名したあと、書き手は新しい file を作る（logrotate と同じ約束）。"""
    path = measure.setup(base_dir=tmp_path)
    measure.record("続き先", 結末="途切れ")
    rotated = path.with_name("measure.log.20260913T2130")
    path.rename(rotated)
    measure.record("続き先", 結末="続き", 相手="aaaaaaaaaaaa")
    assert _lines(rotated) == [line for line in _lines(rotated) if "途切れ" in line]
    assert len(_lines(path)) == 1 and "結末=続き" in _lines(path)[0]


def test_the_measure_logger_does_not_leak_into_the_app_log(tmp_path):
    measure.setup(base_dir=tmp_path)
    assert logging.getLogger(measure.LOGGER_NAME).propagate is False


# ── 5 通りの結末 ──────────────────────────────────────────────────────────────


def _loop(tmp_path, *, judge):
    measure.setup(base_dir=tmp_path)
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    a._evaluator.judge_follows = judge
    return a, tmp_path / "rest_logs" / "measure.log"


def _outcomes(path: Path) -> list[str]:
    return [line.split("結末=")[1].split(" ")[0] for line in _lines(path) if " 続き先 " in line]


def test_a_continuation_is_recorded_with_the_target(tmp_path):
    a, path = _loop(tmp_path, judge=AsyncMock(return_value="m1"))
    _run(a, utterance="さっきの話だけど")
    assert _outcomes(path) == ["続き"]
    assert "相手=m1" in _lines(path)[-1] or "相手=" in _lines(path)[-1]


def test_none_is_a_break(tmp_path):
    a, path = _loop(tmp_path, judge=AsyncMock(return_value=None))
    _run(a, utterance="はじめまして")
    assert _outcomes(path) == ["途切れ"]


def test_an_id_outside_the_workspace_is_a_mismatch(tmp_path):
    a, path = _loop(tmp_path, judge=AsyncMock(return_value="deadbeefdead"))
    _run(a, utterance="こんにちは")
    assert _outcomes(path) == ["不一致"]


def test_a_failing_judge_is_recorded_as_such(tmp_path):
    a, path = _loop(tmp_path, judge=AsyncMock(side_effect=RuntimeError("軽量LLM が落ちた")))
    _run(a, utterance="ねえ")
    assert _outcomes(path) == ["落ちた"]


def test_a_turn_without_words_is_not_judged(tmp_path):
    """入室・情動が起点の反復には人の言葉が無く、判定は呼ばれない。"""
    from familiar_agent.loop.event_loop import InformationProcessing

    a, path = _loop(tmp_path, judge=AsyncMock(return_value="m1"))
    ip = InformationProcessing(a)

    async def scenario():
        ip.push_device("入室", "[入室] こうき が来た")
        ip.start()
        for _ in range(200):
            if a.backend.stream_turn.await_count >= 1:
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())
    # 言葉が無いので、実物の `judge_follows` は軽量LLM を呼ばずに None を返す（ここでは
    # 偽物なので呼ばれた形跡だけ残る）。結末は「未判定」で、「途切れ」と混ざらない。
    assert _outcomes(path) == ["未判定"]


def test_the_app_wires_the_measure_log_at_startup():
    import inspect

    from familiar_agent import main as main_mod

    src = inspect.getsource(main_mod.setup_logging)
    assert "measure.setup(log_dir)" in src
