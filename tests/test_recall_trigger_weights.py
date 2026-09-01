"""trigger 別の5軸重みプロファイル。

`課題5_パラメータ仮案` §280 は「W 構築の5軸重みを trigger 種別で決める」を確定として
いる。実装は Config 固定の1組を全 trigger で使っており、手がかりの性質が違っても同じ
選び方をしていた。

重みは2種類が混ざっている。$w_r$ は関連ゲートの**指数**（`適合度 = r^{w_r} × 地力`）で、
$r \\in [0,1]$ なので上げると関連の薄い候補がより強く罰される。残る4つは加算部 $M$ の
加重平均係数で、分母で正規化されるため比だけが意味を持つ。

ここでは (1) 4プロファイルの既定値、(2) 反復を駆動したものに応じて選ばれること、
(3) env で差し替えられること、(4) 渡した重みが実際に採点へ効くこと、を見る。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from familiar_agent.backend import ToolCall
from familiar_agent.config import MemoryConfig
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

from tests.test_event_loop import _agent, _run, _run_chain, _turn

# 承認済みの仮値（`改造方針_W構築の統一` の重み表）。順は (w_r, w_t, w_e, w_g, w_p)。
_EXPECTED = {
    "発話": (1.5, 1.0, 0.8, 1.5, 1.5),
    "機器": (1.2, 1.2, 0.8, 1.5, 2.0),
    "情動": (0.5, 1.5, 1.5, 1.5, 1.0),
    "完了": (1.5, 1.5, 0.5, 1.5, 1.0),
}


@pytest.mark.parametrize("trigger,expected", sorted(_EXPECTED.items()))
def test_profile_defaults(trigger, expected) -> None:
    """4つの trigger それぞれに既定のプロファイルがある。"""
    w = MemoryConfig().recall_weights(trigger)
    assert (w.w_r, w.w_t, w.w_e, w.w_g, w.w_p) == expected


def test_unknown_trigger_falls_back_to_base() -> None:
    """知らない trigger は基底（正本の (1,1,1,1.5,1.0)）へ落ちる。

    trigger を増やすたびに落ちるより、基底で動いて挙動が読めるほうがよい。
    """
    cfg = MemoryConfig()
    w = cfg.recall_weights("知らない種別")
    assert (w.w_r, w.w_t, w.w_e, w.w_g, w.w_p) == (
        cfg.recall_w_r, cfg.recall_w_t, cfg.recall_w_e, cfg.recall_w_g, cfg.recall_w_p
    )


def test_profile_is_configurable(monkeypatch) -> None:
    """プロファイルの各軸は env で差し替えられる。"""
    monkeypatch.setenv("RECALL_W_R_AFFECT", "0.2")
    assert MemoryConfig().recall_weights("情動").w_r == pytest.approx(0.2)


@pytest.fixture()
def no_jitter(monkeypatch):
    """揺らぎを止める。どのプロファイルが選ばれたかを等値で確かめるために使う。"""
    for axis in ("R", "T", "E", "G", "P"):
        monkeypatch.setenv(f"RECALL_W_{axis}_JITTER", "0")
    return None


def test_jitter_widths_defaults() -> None:
    """軸ごとの揺らぎ幅（絶対値・trigger 共通）の既定値。"""
    cfg = MemoryConfig()
    assert (cfg.recall_w_r_jitter, cfg.recall_w_t_jitter, cfg.recall_w_e_jitter,
            cfg.recall_w_g_jitter, cfg.recall_w_p_jitter) == (0.3, 0.1, 0.2, 0.3, 0.1)


def test_jitter_stays_within_the_width() -> None:
    """採用値は基底 ± 幅の内側に収まる。"""
    import random

    cfg = MemoryConfig()
    base = cfg.recall_weights("完了")
    widths = (cfg.recall_w_r_jitter, cfg.recall_w_t_jitter, cfg.recall_w_e_jitter,
              cfg.recall_w_g_jitter, cfg.recall_w_p_jitter)
    rng = random.Random(0)
    for _ in range(200):
        w = cfg.jitter_weights(base, rng)
        for got, b, width in zip(
            (w.w_r, w.w_t, w.w_e, w.w_g, w.w_p),
            (base.w_r, base.w_t, base.w_e, base.w_g, base.w_p),
            widths,
        ):
            assert b - width - 1e-9 <= got <= b + width + 1e-9, "幅の外へ出ている"


def test_jitter_actually_varies() -> None:
    """揺らぎは呼び出しごとに変わる（固定値を返していない）。"""
    import random

    cfg = MemoryConfig()
    base = cfg.recall_weights("発話")
    rng = random.Random(1)
    seen = {cfg.jitter_weights(base, rng) for _ in range(20)}
    assert len(seen) > 1, "毎回同じ値を返している"


def test_zero_width_axis_does_not_move(monkeypatch) -> None:
    """幅 0 の軸は動かない（テストと再現確認をここで止める）。"""
    monkeypatch.setenv("RECALL_W_T_JITTER", "0")
    cfg = MemoryConfig()
    base = cfg.recall_weights("情動")
    for _ in range(50):
        assert cfg.jitter_weights(base).w_t == pytest.approx(base.w_t)


def test_jitter_never_goes_negative(monkeypatch) -> None:
    """幅より小さい基底を入れても、重みは負にならない（意味が反転する）。"""
    monkeypatch.setenv("RECALL_W_R_AFFECT", "0.05")
    monkeypatch.setenv("RECALL_W_R_JITTER", "1.0")
    cfg = MemoryConfig()
    base = cfg.recall_weights("情動")
    for _ in range(200):
        assert cfg.jitter_weights(base).w_r >= 0.0


def test_utterance_iteration_uses_the_utterance_profile(no_jitter) -> None:
    """人の発話で起きた反復は、発話プロファイルで想起する。"""
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a, utterance="おはよう")
    kwargs = a._active_memory().recall_async.call_args.kwargs
    assert kwargs.get("weights") == MemoryConfig().recall_weights("発話")


def test_completion_driven_iteration_uses_the_completion_profile(no_jitter) -> None:
    """完了が届いて起きた反復は、完了プロファイルで想起する。

    重みを選ぶ基準は「この求めを何が始めたか」ではなく「この反復が何を手がかりに動くか」
    である。反復1の手がかりは人の言葉だが、反復2の手がかりは届いた結果の本文であり、
    性質が違う。
    """
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "昨日の天気"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れてたよ"})]),
    ])
    _run_chain(a, utterance="昨日の天気覚えてる？")

    calls = a._active_memory().recall_async.call_args_list
    assert len(calls) >= 2, "完了で起きた反復の想起が無い（前提が崩れている）"
    assert calls[0].kwargs.get("weights") == MemoryConfig().recall_weights("発話")
    assert calls[1].kwargs.get("weights") == MemoryConfig().recall_weights("完了")


def test_adopted_weights_are_logged_at_info(caplog) -> None:
    """採用値と上位のスコアを INFO に残す（記憶の内容は出さない）。"""
    import logging

    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    a._active_memory().recall_async.return_value = [
        {"memory_id": "m1", "summary": "秘密のはなし", "fit": 0.42},
    ]
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.event_loop"):
        _run(a, utterance="おはよう")

    lines = [r.getMessage() for r in caplog.records if "想起 trigger=" in r.getMessage()]
    assert lines, "採用値の INFO ログが無い"
    assert "trigger=発話" in lines[0]
    assert "0.420" in lines[0], "上位のスコアが出ていない"
    assert "秘密のはなし" not in lines[0], "記憶の内容が INFO へ漏れている"


def test_completion_driven_iteration_keeps_the_origin_kind() -> None:
    """反復の駆動源で重みを選んでも、求めの起点（静穏時間のゲート）は動かさない。

    起点を「完了」へ書き換えると、人に話しかけられて始まった求めが夜間に静穏時間の
    ゲートへ掛かり、返事が保留されて翌朝に届く（`_should_hold` の分岐）。
    """
    import asyncio

    from familiar_agent.loop.event_loop import InformationProcessing

    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "昨日の天気"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "晴れてたよ"})]),
    ])

    async def scenario():
        ip = InformationProcessing(a)
        await ip.run_iteration("昨日の天気覚えてる？")
        for _ in range(400):
            if a.backend.stream_turn.await_count >= a._expected_turns and not ip._tasks:
                break
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)
        await ip.close()
        return ip._origin_kind

    assert asyncio.run(scenario()) == "発話"


def test_weights_reach_the_scorer() -> None:
    """渡した重みが採点に効く（$w_r$ を上げるとスコアが下がる）。

    $r \\in [0,1]$ の累乗なので、指数を上げれば同じ候補のスコアは下がる。逆向きに
    実装すると、関連を厳しくしたつもりが緩める側へ働く。
    """
    from familiar_agent.config import RecallWeights

    row = {
        "id": "obs-1", "content": "むかしの話", "timestamp": None,
        "last_recalled_at": None,
        "groundedness_g0": 1.0, "groundedness_n": 0,
        "emotion_p": 0.5, "emotion_pn": 0.5, "emotion_a": 0.5, "emotion_dom": 0.5,
        "direction": "発話", "kind": "observation", "emotion": "neutral",
        "image_path": None, "score": 0.5,
    }
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        mem = ObservationMemory()
        with (
            patch.object(mem._observations, "by_vector", return_value=[dict(row)]),
            patch.object(mem._observations, "by_time", return_value=[]),
            patch.object(mem._observations, "by_emotion", return_value=[]),
            patch.object(mem._observations, "situated_cosines", return_value={"obs-1": 0.5}),
        ):
            soft = mem.recall("q", n=7, weights=RecallWeights(0.5, 1.0, 1.0, 1.5, 1.0))
            hard = mem.recall("q", n=7, weights=RecallWeights(2.0, 1.0, 1.0, 1.5, 1.0))

    assert soft and hard, "候補が採点まで届いていない（前提が崩れている）"
    assert soft[0]["fit"] > hard[0]["fit"], "w_r を上げてもスコアが下がらない"
