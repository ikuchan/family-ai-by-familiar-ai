"""求めの寿命の状態に、持ち主を与える（環-e-に・に-5-に-1）。

`InformationProcessing` は 30 個の可変状態を1つの名前空間に平らに置いていた。そのうち
**求めが始まってから閉じるまでだけ生きるもの**は、`_begin_request`・`_finish`・
`_abort_lookups` の3つ——**求めの寿命の3つの遷移**——が書いている。

振る舞い（生成器・動作器・想起）で割ると、30 個中 14 個が境界をまたぐ。**所有を分けるのは
振る舞いではなく寿命である**（に-2 の結論）。`loop/request.py` の `Request` がその持ち主に
なる。

**丸ごと作り直すことはしない。** 18 個のうち 5 個は求めごとにリセットされていない
（`_request_generation` は打ち切りの検出に使う単調増加で、戻せば壊れる）。束ごとに、
**いまと同じリセット点をそのまま保つ**。

**挙動は変えない。**
"""

from __future__ import annotations

import inspect

from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.loop.request import Request


# ── 束 A：数え ─────────────────────────────────────────────────────────────


def test_the_request_owns_the_counters():
    r = Request()
    assert r.iterations == 0
    assert r.iterations_capped is False


def test_a_fresh_request_starts_from_zero():
    """作り直しがリセットになる（散らばった `= 0` の意味は「新しい求め」である）。"""
    r = Request()
    r.iterations = 3
    r.iterations_capped = True
    assert Request().iterations == 0
    assert Request().iterations_capped is False


# ── 束 B：発話の持ち越し ───────────────────────────────────────────────────


def test_the_request_owns_what_was_already_said():
    """つなぎと、配る保留。どちらも打ち切りと `_finish` で空になる＝求めの寿命である。"""
    r = Request()
    assert r.said_fillers == []
    assert r.speech_to_deliver == []


def test_each_request_gets_its_own_lists():
    """**入れ物を共有しない。** 既定値を1つにすると、次の求めに前の一言が残る。"""
    a, b = Request(), Request()
    a.said_fillers.append("調べますね")
    a.speech_to_deliver.append("さっき言いたかったこと")
    assert b.said_fillers == []
    assert b.speech_to_deliver == []


# ── 束 C：やりとり ─────────────────────────────────────────────────────────


def test_the_request_owns_the_turn_records():
    """このターンが作った記録と、やりとりの切り出し位置。`_finish` で空へ戻る。

    **`recent_cursor` はここに入れない。** あれは会話の履歴をたどる位置で、求めごとに
    戻らない（`_note_origin` と `_recent_ctx` が書くだけ）。装置の寿命である。
    """
    r = Request()
    assert r.turn_records == []
    assert r.exchange_start == 0


def test_each_request_gets_its_own_turn_records():
    a, b = Request(), Request()
    a.turn_records.append(("obs1", "起点"))
    assert b.turn_records == []


# ── 束 D：版 ───────────────────────────────────────────────────────────────


def test_the_request_owns_the_version_chain():
    """求めそのもの（文面・起点の id・生きている版）と、この求めの手がかり。

    どれも `_begin_request` で置き直され、`_finish` と打ち切りで戻る。
    """
    r = Request()
    assert r.request_id is None
    assert r.live_version_id is None
    assert r.request_text == ""
    assert r.cue == ""
    assert r.utterance == ""
    assert r.trigger_kind == "発話"


def test_no_live_request_is_still_the_absent_id():
    """**求めが無いことは `request_id is None` が表す。** 器を `None` にはしない。"""
    r = Request()
    assert r.request_id is None
    r.request_id = "obs1"
    assert Request().request_id is None


# ── 持ち主 ─────────────────────────────────────────────────────────────────


def test_the_loop_holds_exactly_one_request():
    """**`None` にしない。** 求めが無いことは、いまも `_request_id is None` が表している。

    18 個ぶんの `if self._req else` を 40 箇所へ入れれば、寿命を表すどころか読みにくくなる。
    """
    src = inspect.getsource(InformationProcessing.__init__)
    assert "self._req = Request()" in src


def test_the_old_flat_names_are_gone():
    """旧名の grep が 0 件であることを完了条件にする（数え上げでは代えない）。"""
    src = inspect.getsource(InformationProcessing)
    assert "self._iterations" not in src
    assert "self._iterations_capped" not in src
    assert "self._said_fillers" not in src
    assert "self._speech_to_deliver" not in src
    assert "self._turn_records" not in src
    assert "self._exchange_start" not in src
    for name in ("_request_id", "_request_text", "_live_version_id", "_cue", "_utterance"):
        assert f"self.{name}" not in src, name
    assert "self._trigger_kind" not in src
