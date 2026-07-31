"""一次絞り件数 N（軸あたり）と W 載せ上限 K の分離。

`課題5_パラメータ仮案` は一次絞り件数 N=50（軸あたり・確定）と W 載せ上限 K=7 を別の
つまみとして定めている。N は「フルLLM には渡らず再スコア用」なのでトークン量に関係なく、
K だけが W に載る量を決める。実装は両者を `recall()` の `n` 一つで兼ねており、各軸が
5 件しか集めていなかった。

ここでは (1) min_score の有無に依らず各軸が N 件取ること、(2) N が Config で差し替え
られること、(3) 返る件数は呼び出し側の n（K）で絞られることを見る。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from familiar_agent.config import MemoryConfig
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


@pytest.fixture()
def memory():
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        yield ObservationMemory()


def test_primary_n_default_is_50() -> None:
    """一次絞り件数の既定は正本の N=50。"""
    assert MemoryConfig().recall_primary_n == 50


def test_every_axis_takes_primary_n(memory) -> None:
    """関連軸と時間軸は、min_score の有無に依らず N 件取る。

    以前は min_score>0 のときだけ n×3（上限20）へ増やしていた。床を課すかどうかは
    採点後の話で、候補をいくつ集めるかとは別の決定である。
    """
    with (
        patch.object(memory._observations, "by_vector", return_value=[]) as bv,
        patch.object(memory._observations, "by_time", return_value=[]) as bt,
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        memory.recall("q", n=7, min_score=0.05)
        assert bv.call_args.args[1] == 50, "関連軸が N=50 を取っていない"
        assert bt.call_args.args[1] == 50, "時間軸が N=50 を取っていない"

        bv.reset_mock()
        bt.reset_mock()
        memory.recall("q", n=7, min_score=0.0)
        assert bv.call_args.args[1] == 50, "床が無いと N が縮んでいる"
        assert bt.call_args.args[1] == 50, "床が無いと N が縮んでいる"


def test_primary_n_is_configurable(memory, monkeypatch) -> None:
    """N は Config（env）で差し替えられる。"""
    monkeypatch.setenv("RECALL_PRIMARY_N", "80")
    with (
        patch.object(memory._observations, "by_vector", return_value=[]) as bv,
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        memory.recall("q", n=7, min_score=0.05)
        assert bv.call_args.args[1] == 80, "RECALL_PRIMARY_N が効いていない"


def test_overfetch_knobs_are_gone() -> None:
    """過剰取得の係数と上限は N に役目を譲って撤去された。"""
    cfg = MemoryConfig()
    assert not hasattr(cfg, "recall_overfetch_factor"), "recall_overfetch_factor が残っている"
    assert not hasattr(cfg, "recall_overfetch_cap"), "recall_overfetch_cap が残っている"


def test_event_loop_passes_min_score() -> None:
    """イベントループの想起は床（min_score）を渡す。

    渡さないと `recall()` の既定 0.0 で床が効かず、無関係な記録まで W の枠を埋める。
    床は正本 [D-想起合成] が「無関係排除の主たる足切り」と定めるもので、連想想起
    （`agent.py`）は既に渡していた。イベントループだけが渡していなかった。
    """
    from familiar_agent.backend import ToolCall

    from tests.test_event_loop import _agent, _run_chain, _turn

    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "はい"})])])
    _run_chain(a, utterance="ねえ")

    mem = a._active_memory()
    assert mem.recall_async.await_count >= 1, "想起が呼ばれていない（前提が崩れている）"
    kwargs = mem.recall_async.await_args_list[0].kwargs
    assert kwargs.get("min_score") == pytest.approx(MemoryConfig().recall_min_score), (
        "イベントループが床を渡していない"
    )


def test_recall_k_default_is_7() -> None:
    """W 載せ上限の既定は正本の K=7（人間の作業記憶に倣う）。"""
    assert MemoryConfig().recall_k == 7


def test_old_name_recall_n_is_gone() -> None:
    """旧名 `recall_n` は残っていない。

    N（一次絞り）と K（W 載せ）は別のつまみである。`recall_n` という名前は
    どちらとも読めるので、正本の語 K へ揃える。
    """
    assert not hasattr(MemoryConfig(), "recall_n"), "旧名 recall_n が残っている"
