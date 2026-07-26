"""イベント駆動ループが「誰と話しているか」を渡す。

CUI にはカメラが無く、PMM の在席は常に空になる。話者は `/speaker` や `[名前]` で
設定される `PersonRegistry.active_name` にしか現れないが、イベントループはこれを
読んでいなかった。結果、相手が誰でも「分からない」に倒れ、ME.md の「分からないときは
大人として扱い丁寧に話す」だけが効き続けた（実機で観測）。

案B：既定の話者は置かない。明示的に指定されたときだけ相手が定まる。家族の誰かを
決め打ちして、その人向けの口調で話し始めることを避ける。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.loop.event_loop import _present_ctx
from familiar_agent.relationship import PersonRegistry


def _agent(*, rows=None, explicit: str | None = None, spoken_to=True):
    a = MagicMock()
    a._pmm = MagicMock()
    a._pmm.presence_status = MagicMock(return_value=rows or [])
    a._social_presence_permission = MagicMock(return_value=1.0 if spoken_to else 0.0)
    persons = PersonRegistry("あなた")
    if explicit is not None:
        persons.set_active(explicit)
    a._persons = persons
    return a


def test_explicit_speaker_is_passed_with_its_origin():
    ctx = _present_ctx(_agent(explicit="パパ"))
    assert '"パパ"' in ctx
    assert "自己申告" in ctx          # 顔で確かめた話者と混同しない


def test_no_speaker_without_an_explicit_one():
    # 既定名（PersonRegistry の初期値）を話者として渡さない（案B）。
    ctx = _present_ctx(_agent(explicit=None))
    assert "あなた" not in ctx
    assert "unconfirmed" in ctx


def test_face_recognised_speaker_wins_over_self_declared():
    rows = [{"name": "たいき", "is_speaker": True, "confidence": 0.92}]
    ctx = _present_ctx(_agent(rows=rows, explicit="パパ"))
    assert '"たいき"' in ctx and "パパ" not in ctx
