"""声の照合と名乗りは在席を待たない（知-ai 段 4・2026-10-05・本人の決定・`設計方針_在席と顔ぶれ` v0.1）。

声で誰か分かるなら、それは顔ぶれ（特定の誰がいるか）の証拠になる。以前は在席（カメラ）があるときだけ声を照らし、
名乗りを付けていた（在席を顔ぶれの門に使う、上下が逆の形）。

- **付け替える閾値は状況で変える**（仮の値）：在席あり 0.35（`VOICE_SWITCH_THRESHOLD`）、何もなし 0.45
  （`VOICE_ALONE_THRESHOLD`）。いまの話者を続けるのは状況によらず 0.25。続けたときも顔ぶれの持ち時間を数え直す。
- **名乗り**は言葉だけでは顔ぶれに入れない。その発話の声を名乗った本人とだけ照らし、0.30（`VOICE_CLAIM_THRESHOLD`）
  以上なら話者に付け、登録の声と今日の声に足す。本人の声の基準が無ければ何もしない（最初の登録は `/voice`）。
- 名乗りの預かり（知-w-ろ）はやめた。
"""

from __future__ import annotations

import asyncio
import math

import numpy as np

from familiar_agent.loop import arbiter
from tests.test_voice_picks_the_speaker import _ip, _Store

PAPA = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
TAIKI = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)


def _toward(axis: int, cos: float) -> np.ndarray:
    """`axis` の人にちょうど `cos` で当たり、もう一人には 0 で当たる声。"""
    v = [0.0, 0.0, math.sqrt(1 - cos * cos)]
    v[axis] = cos
    return np.asarray(v, dtype=np.float32)


def _store():
    return _Store(registered={"papa": PAPA, "taiki": TAIKI})


def test_the_thresholds(monkeypatch):
    from familiar_agent.config import RecognitionConfig

    for k in ("VOICE_CLAIM_THRESHOLD", "VOICE_ALONE_THRESHOLD", "VOICE_SWITCH_THRESHOLD"):
        monkeypatch.delenv(k, raising=False)
    cfg = RecognitionConfig()
    assert (cfg.voice_claim_threshold, cfg.voice_switch_threshold, cfg.voice_alone_threshold) == (
        0.30,
        0.35,
        0.45,
    )


def test_a_voice_is_matched_without_occupancy():
    ip, a, s = _ip(present=0.0, store=_store())
    asyncio.run(ip._match_voice(TAIKI))
    a._persons.set_active.assert_called_once_with("たいき")


def test_without_occupancy_the_switch_needs_more():
    voice = _toward(1, 0.40)  # たいきに 0.40・パパに 0
    ip, a, _ = _ip(present=0.0, store=_store())
    asyncio.run(ip._match_voice(voice))
    a._persons.set_active.assert_not_called()  # 0.45 に届かない → 分からない
    a._persons.reset_to_default.assert_called_once()
    ip, a, _ = _ip(present=1.0, store=_store())
    asyncio.run(ip._match_voice(voice))
    a._persons.set_active.assert_called_once_with("たいき")  # 在席ありは 0.35


def test_keeping_the_speaker_refreshes_their_presence():
    ip, a, _ = _ip(present=0.0, store=_store())
    asyncio.run(ip._match_voice(_toward(0, 0.27)))  # パパに緩く
    a._persons.set_active.assert_not_called()
    a._pmm.refresh_signal.assert_called_with("papa")


# ── 名乗り ───────────────────────────────────────────────────────────────────


def _claim(ip, name: str):
    asyncio.run(
        ip._apply_speaker_claim(arbiter.Decision(branch="light", text="x", speaker_claim=name))
    )


def test_a_claim_with_a_matching_voice_names_the_speaker_without_occupancy():
    ip, a, s = _ip(present=0.0, speaker=None, store=_store())
    ip._req.voice = _toward(0, 0.31)
    _claim(ip, "パパ")
    a._persons.set_active.assert_called_once_with("パパ")
    assert [x[:3] for x in s.added] == [
        ("papa", "voice", "registered"),
        ("papa", "voice", "today"),
    ]


def test_a_claim_whose_voice_falls_short_does_nothing():
    ip, a, s = _ip(present=1.0, speaker=None, store=_store())
    ip._req.voice = _toward(0, 0.29)
    _claim(ip, "パパ")
    a._persons.set_active.assert_not_called()
    assert s.added == []


def test_a_claim_by_someone_without_a_voice_does_nothing():
    ip, a, s = _ip(present=1.0, speaker=None, store=_Store(registered={"taiki": TAIKI}))
    ip._req.voice = PAPA
    _claim(ip, "パパ")
    a._persons.set_active.assert_not_called()
    assert s.added == []


def test_a_claim_without_a_voice_does_nothing():
    ip, a, s = _ip(present=1.0, speaker=None, store=_store())
    ip._req.voice = None  # キーボード・短い断片
    _claim(ip, "パパ")
    a._persons.set_active.assert_not_called()


def test_the_claim_is_no_longer_kept():
    from familiar_agent.loop import event_loop

    ip, _, _ = _ip(present=0.0, speaker=None, store=_store())
    assert not hasattr(event_loop, "PENDING_CLAIM_SEC")
    assert not hasattr(ip, "_pending_claim")
    assert not hasattr(ip, "_apply_pending_claim")
