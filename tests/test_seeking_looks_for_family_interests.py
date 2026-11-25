"""記-ag 段 1：SEEKING の発火で、家族が気にかけていることを調べる（`設計方針_先回りして調べておく` §3・2026-11-25 本人の決定）。

何を調べるかは決め打ちしない。SEEKING の内なる声に抽象的な指示（家族が気にかけていて、新しい結果や情報が出るものを
思い出して調べる）を足し、何を調べるかの問いは、いまどおり W（思い出した記憶を含む）から書く。SEEKING の求めは
いまどおり話しかけない（出-as §2.1）。
"""

from __future__ import annotations


def test_the_seeking_voice_asks_to_look_into_what_the_family_cares_about(monkeypatch):
    from familiar_agent.config import DriveConfig

    monkeypatch.delenv("VOICE_SEEKING", raising=False)
    voice = DriveConfig().voice_seeking
    assert "家族が気にかけて" in voice and "新しい結果や情報" in voice
    assert "思い出" in voice  # 何を調べるかは記憶から選ぶ


def test_the_seeking_voice_does_not_tell_to_talk(monkeypatch):
    """**反証の側**：SEEKING は話しかけない軸のまま（知らせるのは別の求め・段 3）。"""
    from familiar_agent.config import DriveConfig

    monkeypatch.delenv("VOICE_SEEKING", raising=False)
    voice = DriveConfig().voice_seeking
    assert "話しかけ" not in voice and "伝える" not in voice
