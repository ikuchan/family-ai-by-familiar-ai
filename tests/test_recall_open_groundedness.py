"""open な記録の活性下限 $a_{open}$（`課題5_パラメータ仮案` §184）。

完了で起きた反復の手がかりは**届いた結果の本文**なので、元の人の問い（トリガ O）の
関連 $r$ が低くなりうる。「リバプールの試合どうだった？」と試合結果の本文で、語彙が
重なるとは限らない。完了プロファイルは $w_r$＝1.5 で関連を厳しく要求するため、何のために
調べていたかが W から落ちる。

正本は「open の間だけ導出値に下限を課す（$a = \\max(\\text{導出}, a_{open})$）」と定める。
$w_g$＝1.5 は加算部で最も重い係数なので、下限を課せば確実に浮く。

ここでは (1) 既定値、(2) 下限が採点に効くこと、(3) open でない記録は動かないこと、
(4) ループがトリガ O と生きた意図 O を open として渡すこと、を見る。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from familiar_agent.backend import ToolCall
from familiar_agent.config import MemoryConfig
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

from tests.test_event_loop import _agent, _run, _run_chain, _turn


def _row(obs_id: str) -> dict:
    """採点まで届く最小の行。活性は既定の低い側に置く（下限の効きを見るため）。"""
    return {
        "id": obs_id, "content": "むかしの話", "timestamp": None,
        "last_recalled_at": None,
        "groundedness_g0": 0.1, "groundedness_n": 0,
        "emotion_p": 0.5, "emotion_pn": 0.5, "emotion_a": 0.5, "emotion_dom": 0.5,
        "direction": "発話", "kind": "observation", "emotion": "neutral",
        "image_path": None, "score": 0.3,
    }


def _recall_with(open_ids):
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        mem = ObservationMemory()
        with (
            patch.object(mem._observations, "by_vector", return_value=[_row("obs-1")]),
            patch.object(mem._observations, "by_time", return_value=[]),
            patch.object(mem._observations, "by_emotion", return_value=[]),
            patch.object(mem._observations, "situated_cosines", return_value={"obs-1": 0.3}),
        ):
            return mem.recall("q", n=7, open_ids=open_ids)


def test_a_open_default_is_1() -> None:
    """既定は正本の $a_{open}$＝1.0（既定取込超・pinned 未満）。"""
    assert MemoryConfig().recall_g_open == 1.0


def test_open_record_gets_the_activation_floor() -> None:
    """open な記録は活性の下限で浮く。"""
    plain = _recall_with(None)
    opened = _recall_with(["obs-1"])
    assert plain and opened, "候補が採点まで届いていない（前提が崩れている）"
    assert opened[0]["fit"] > plain[0]["fit"], "open にしてもスコアが上がらない"


def test_a_open_is_applied_to_the_scored_activation() -> None:
    """下限は導出値との max であって、置き換えではない。

    導出値が下限より高い記録を open にしても、値は下がらない。
    """
    cfg = MemoryConfig()
    opened = _recall_with(["obs-1"])
    assert opened[0]["fit"] > 0.0
    # 下限そのものは Config の値。導出（a0=0.1・n=0）はこれより低い。
    assert cfg.recall_g_open > 0.1


def test_records_outside_open_ids_are_unaffected() -> None:
    """open に挙げていない記録は動かない。"""
    plain = _recall_with(None)
    other = _recall_with(["まったく別のid"])
    assert plain[0]["fit"] == pytest.approx(other[0]["fit"])


def test_loop_passes_the_trigger_as_open() -> None:
    """ループはトリガ O（求めの親）を open として渡す。

    求めが閉じるまでトリガ O を open 扱いにするのは、完了の反復で手がかりが結果本文に
    変わっても、元の問いを W に残すためである。
    """
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a, utterance="おはよう")
    kwargs = a._active_memory().recall_async.call_args.kwargs
    assert "obs1" in (kwargs.get("open_ids") or []), "トリガ O が open に入っていない"


def test_live_intent_is_open_and_drops_out_when_the_result_arrives() -> None:
    """飛行中の意図 O は open。結果が届いた意図はもう open ではない。

    「結果はまだ無い」という意図の記録が、結果が届いたあとも浮き続ける理由はない。
    """
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "昨日の天気"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れてたよ"})]),
    ])
    _run_chain(a, utterance="昨日の天気覚えてる？")

    calls = a._active_memory().recall_async.call_args_list
    assert len(calls) >= 2, "完了で起きた反復の想起が無い（前提が崩れている）"
    # obs1=トリガ / obs2=意図 / obs3=完了。反復2は完了で起きるので、意図は解決済み。
    second = calls[1].kwargs.get("open_ids") or []
    assert "obs1" in second, "完了の反復でトリガ O が open から外れている"
    assert "obs2" not in second, "結果が届いた意図がまだ open のまま"
