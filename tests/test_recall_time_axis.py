"""時間軸（基準時刻からの隔たり）。

軸が表すのは「新しいこと」ではなく**基準時刻からの隔たり**である。既定では基準が
「いま」なので結果として新しさになるが、それは特別な場合にすぎない。調停（軽量LLM）が
人の言葉から**基準時刻と幅**を読み取り、想起の出発点を毎回動かせるようにする。

- 「去年の夏の話なんだけど」→ 基準＝去年の8月あたり、幅＝数十日
- 何も手がかりが無ければ基準＝いま、幅＝既定（3日）

**幅はそのまま半減期**にする（案A）。幅の内側は $t \\ge 0.5$、幅の2倍で $t \\approx 0.25$。
換算の係数を新たに置かずに済む。

一次絞りは、幅の指定が無ければ `COALESCE(last_recalled_at, timestamp)` の1本、
**幅の指定があれば `timestamp` と `last_recalled_at` の両方**で基準の前後から取る。
書かれた時刻（その頃の出来事）と使った時刻（その頃に思い出していたこと）は別の手がかりで、
時期を指定されたときはどちらも要る。
"""

from __future__ import annotations

import inspect

from familiar_agent.store.observations import ObservationStore
from familiar_agent.time_decay import DecayState


def _state(origin: float, half_life_seconds: float = 3 * 86400.0) -> DecayState:
    return DecayState(origin_epoch=origin, half_life_seconds=half_life_seconds, floor=0.001)


def test_distance_is_symmetric_around_the_reference():
    # 現行は max(0, now - origin) と負を切り捨てており、基準が過去へ動くと
    # それ以降の記録が全部 t=1 になる。基準からの隔たりは絶対値で測る。
    ref = 1_000_000.0
    before = _state(ref - 86400.0).score(ref)
    after = _state(ref + 86400.0).score(ref)
    assert abs(before - after) < 1e-9
    assert before < 1.0                      # 1日離れていれば 1 より小さい


def test_the_reference_itself_scores_one():
    ref = 1_000_000.0
    assert _state(ref).score(ref) == 1.0


def test_span_is_the_half_life():
    # 案A：幅をそのまま半減期にする。幅ちょうど離れたら 0.5。
    span = 30 * 86400.0
    ref = 1_000_000.0
    assert abs(_state(ref - span, half_life_seconds=span).score(ref) - 0.5) < 1e-6


def test_time_axis_uses_the_same_origin_as_scoring():
    # 採点は last_recalled_at（無ければ timestamp）を起点にする。一次絞りが
    # timestamp だけで並べていると、「古いが最近よく使う記憶」が候補に入らない。
    src = inspect.getsource(ObservationStore.by_time)
    assert "COALESCE(o.last_recalled_at, o.timestamp)" in src


def test_time_axis_scans_both_sides_of_the_reference():
    # 基準が過去へ動くと、その前後どちらも近い。片側だけでは取りこぼす。
    src = inspect.getsource(ObservationStore.by_time)
    assert "<=" in src and ">" in src
    assert "DESC" in src and "ASC" in src


def test_time_axis_uses_both_columns_when_a_span_is_given():
    # 幅の指定があるときは、書かれた時刻と使った時刻の両方で探す。
    src = inspect.getsource(ObservationStore.by_time)
    assert "o.timestamp" in src and "o.last_recalled_at" in src
    assert "span" in inspect.signature(ObservationStore.by_time).parameters


def test_time_axis_skips_dead_records_and_keeps_the_perspective_scope():
    src = inspect.getsource(ObservationStore.by_time)
    assert "o.superseded_by IS NULL" in src
    assert "s.person_id = %s" in src


def test_recall_count_does_not_extend_the_half_life():
    """強化A（想起回数で実効半減期を伸ばす）は想起の t 軸で使わない。

    `課題5` F節が廃止と確定させている（重要さは activation の n が担い、t は純粋な
    時間減衰）。実装に残っていたため `recall_count` が 20 なら半減期が 3×2^20 日＝
    8600年になり、**何度も想起された古い記録が永久に t=1** になっていた。実機で 47日前の
    挨拶が t=1.000 で上位を占め、5秒前の自分の発話を押し出した（「おかえりなさい」を2回）。
    """
    import datetime as dt

    from familiar_agent.tools.memory import _score_breakdown

    ref = dt.datetime(2026, 7, 27, tzinfo=dt.timezone.utc)
    old_ts = ref - dt.timedelta(days=47)

    def t_of(recall_count: int) -> float:
        parts = _score_breakdown(
            0.5, old_ts, None, recall_count, 1.0, 0,
            half_life_days=3.0, floor=0.001, reference_epoch=ref.timestamp(),
        )
        return parts.t

    assert t_of(0) < 0.01                      # 47日前は半減期3日でほぼ 0
    assert t_of(20) == t_of(0)                 # 想起回数で伸びない
