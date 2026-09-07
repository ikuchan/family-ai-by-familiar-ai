"""能力表を実物に合わせる（出-j）。

`capabilities.yaml` は自己理解（`capability_state`）の材料で、ここに書いたことがパジュの
「できること」になる。**ずれたまま置くと、パジュは無い能力を語り、ある能力を語らない。**

`filter_enabled()` は `enabled_env: X` の X が未設定なら、その能力の項ごと落とす。出-a
（音声出力のローカル化・2026-07-30）で TTS と STT の既定はローカルへ移ったのに、条件が
`ELEVENLABS_API_KEY` のまま残っていた。鍵の無い機体では、パジュは**自分に声と耳が無い**と
自己認識することになる。

外部 API は「戻せる逃げ道」であって前提ではない。
"""

from __future__ import annotations

import re

from familiar_agent.capability_state import filter_enabled, load_manifest


def _ids(manifest: str) -> set[str]:
    return set(re.findall(r"(?m)^  - id: (\S+)", manifest))


def _no_env() -> str:
    """鍵も host も何も無い機体。ローカルだけで動く構成である。"""
    return filter_enabled(load_manifest(), env={})


# ── 声と耳は鍵に依らない ────────────────────────────────────────────────────


def test_the_voice_survives_without_an_api_key():
    """既定は `TTS_ENGINE=sbv2`（Style-Bert-VITS2・ローカル）。鍵は要らない。"""
    assert "tts" in _ids(_no_env())


def test_the_ears_survive_without_an_api_key():
    """既定は `STT_ENGINE=whisper`（faster-whisper・ローカル）。鍵は要らない。"""
    ids = _ids(_no_env())
    assert "stt" in ids
    assert "realtime_stt" in ids


def test_the_local_engine_is_named_in_what_it_says_about_itself():
    """自己理解の材料なので、既定の担い手が書かれていること。"""
    m = load_manifest()
    tts = m.split("  - id: tts", 1)[1].split("  - id: ", 1)[0]
    assert "Style-Bert-VITS2" in tts or "TTS_ENGINE" in tts


# ── 反証側：本当に外部条件が要るものは、まだ落ちる ──────────────────────────


def test_a_capability_that_really_needs_hardware_still_drops():
    """カメラは `CAMERA_HOST` が無ければ本当に無い。落とす仕組み自体は生きている。"""
    ids = _ids(_no_env())
    assert "camera_vision" not in ids
    assert "mobility" not in ids


def test_the_env_gate_still_keeps_what_is_configured():
    ids = _ids(filter_enabled(load_manifest(), env={"CAMERA_HOST": "192.168.0.2"}))
    assert "camera_vision" in ids


# ── 実在しない module を指さない（出-j-ろ）────────────────────────────────


def test_every_module_it_points_at_exists():
    """能力表が指す file は実在すること。

    自己理解は毎回この file から組み直されるので、無い module を指す項を残せば、
    パジュは実装の無い能力を語り続ける。数え上げでは網羅を証明できないので、
    **書かれている参照を全部引いて実在を確かめる**。
    """
    from pathlib import Path

    src = Path(__file__).parent.parent / "src/familiar_agent"
    refs = set(re.findall(r"`([a-z_][a-z_0-9/]*\.py)`", load_manifest()))
    assert refs, "参照が1つも取れていない（正規表現が壊れている）"
    missing = sorted(r for r in refs if not (src / r).exists())
    assert missing == [], f"実在しない module を指している: {missing}"


def test_the_removed_capabilities_are_gone():
    """実装が無い能力は項ごと落とす。残せば語られる。"""
    ids = _ids(load_manifest())
    for gone in ("theory_of_mind", "social_policy", "interoception_bridge", "meta_monitor"):
        assert gone not in ids


def test_the_capabilities_that_moved_point_at_their_new_home():
    """実物のあるものは参照先を直す（項は残す）。"""
    m = load_manifest()
    assert "appraisal_engine" in _ids(m) and "default_mode_network" in _ids(m)


def test_the_key_modules_list_only_names_files_that_exist():
    """自己理解の材料。無い file は黙って飛ばされるので、痩せていても気づけない。"""
    from pathlib import Path

    from familiar_agent.capability_state import _KEY_MODULES

    src = Path(__file__).parent.parent / "src/familiar_agent"
    missing = sorted(n for n in _KEY_MODULES if not (src / n).exists())
    assert missing == [], f"実在しない module を材料に挙げている: {missing}"
