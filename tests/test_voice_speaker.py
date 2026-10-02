"""声で話者を決める規則（知-ae 段 2・2026-10-02・`設計方針_声で話者を見分ける` v0.1）。純関数。

- **続ける**：いまの話者に緩い閾値（0.25）以上で似ていれば、その人のまま（声の揺れで外れない）。
- **付け替える**：別の人に厳しい閾値（0.35）以上で、いまの話者より近ければ、その人へ（似た声への取り違えを防ぐ）。
- **分からない**：どちらでもなければ、既定の人に戻す（本人の決定イ）。
- **いまの話者に基準が無い**（`/speaker` で決めたばかりで名乗ったことのない人）：その人ではないとは言えないので
  続ける。厳しい閾値で別の人に当たったときだけ付け替える（改造方針で承認した解釈）。
- **誰の基準も無い**：照らせない（何もしない）。
- 厳しい閾値以上で当たった発話は、今日の声に足す印（`sure`）を持つ。
"""

from __future__ import annotations

import numpy as np
import pytest

from familiar_agent.core import voice_speaker as vs

LOOSE, STRICT = 0.25, 0.35


def _d(current, scores):
    return vs.decide(current, scores, loose=LOOSE, strict=STRICT)


def test_the_current_speaker_continues_on_the_loose_threshold():
    v = _d("papa", {"papa": 0.28, "taiki": 0.10})
    assert (v.action, v.person_id, v.sure) == ("keep", "papa", False)


def test_someone_else_on_the_strict_threshold_takes_over():
    v = _d("papa", {"papa": 0.20, "taiki": 0.40})
    assert (v.action, v.person_id, v.sure) == ("switch", "taiki", True)


def test_someone_else_only_on_the_loose_threshold_does_not_take_over():
    v = _d("papa", {"papa": 0.26, "taiki": 0.30})
    assert (v.action, v.person_id) == ("keep", "papa")


def test_the_current_speaker_closer_than_a_strict_other_stays():
    v = _d("papa", {"papa": 0.50, "taiki": 0.40})
    assert (v.action, v.person_id, v.sure) == ("keep", "papa", True)


def test_neither_threshold_means_unknown():
    v = _d("papa", {"papa": 0.20, "taiki": 0.30})
    assert v.action == "unknown" and v.person_id is None


def test_a_current_speaker_without_a_voice_continues():
    assert _d("koki", {"papa": 0.30}).action == "keep"
    v = _d("koki", {"papa": 0.36})
    assert (v.action, v.person_id) == ("switch", "papa")


def test_no_voices_at_all_means_nothing_to_compare():
    assert _d("papa", {}).action == "skip"
    assert _d(None, {}).action == "skip"


def test_from_the_default_person_only_the_strict_threshold_names_someone():
    v = _d(None, {"papa": 0.36})
    assert (v.action, v.person_id) == ("switch", "papa")
    assert _d(None, {"papa": 0.30}).action == "unknown"


def test_scores_take_the_closer_of_registered_and_today():
    voice = np.asarray([1.0, 0.0], dtype=np.float32)
    registered = {"papa": np.asarray([0.0, 1.0], dtype=np.float32)}
    today = {
        "papa": np.asarray([1.0, 0.1], dtype=np.float32),
        "taiki": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    got = vs.scores(voice, registered, today)
    assert got["papa"] == pytest.approx(1.0 / np.sqrt(1.01))
    assert got["taiki"] == pytest.approx(0.0)


def test_a_zero_vector_scores_nothing():
    zero = np.zeros(2, dtype=np.float32)
    assert vs.scores(zero, {"papa": np.asarray([1.0, 0.0], dtype=np.float32)}, {}) == {}
