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
    for gone in (
        "theory_of_mind",
        "social_policy",
        "interoception_bridge",
        "meta_monitor",
        # 環-d（2026-09-14）：書き手ごと撤去したストア
        "self_narrative",
        "tape_planning",
        "concern_engine",
    ):
        assert gone not in ids


def test_the_capabilities_that_moved_point_at_their_new_home():
    """実物のあるものは参照先を直す（項は残す）。"""
    m = load_manifest()
    assert "appraisal_engine" in _ids(m) and "default_mode_network" in _ids(m)


# ── 担い手を足したら台帳も直す（環-y・2026-09-24） ────────────────────────


def _engines(module: str) -> set[str]:
    """`engine == "…"` で分岐している担い手の名前。**コードが正**である。"""
    from pathlib import Path

    src = (Path(__file__).parent.parent / "src/familiar_agent" / module).read_text(encoding="utf-8")
    return (
        set(re.findall(r'engine\s*==\s*"([a-z0-9_]+)"', src))
        | set(re.findall(r'engine\s*!=\s*"([a-z0-9_]+)"', src))
        | set(re.findall(r'\bengine\s*:\s*str\s*=\s*"([a-z0-9_]+)"', src))
    )


def test_every_voice_engine_is_named_in_the_manifest():
    """**声の担い手を足したら台帳も直す。**

    2026-09-24 に Gemini TTS と控えの仕組みを入れたが、台帳の `tts` 項は SBV2 と
    ElevenLabs のままだった。**誰も気づかなかった**——`capabilities.yaml` を書くのは
    機能を作る人で、人の記憶に頼っていたからである。ここで落ちれば忘れられない。

    見るのは**担い手の名前だけ**にする。設定名まで全部求めると、いまの台帳で 43 件中
    38 件が赤くなり、鳴り止まない番人は無視されるようになる。
    """
    m = load_manifest()
    missing = sorted(e for e in _engines("tools/tts.py") if e not in m)
    assert missing == [], f"台帳に書いていない声の担い手: {missing}"


def test_every_ear_engine_is_named_in_the_manifest():
    m = load_manifest()
    missing = sorted(e for e in _engines("tools/stt.py") if e not in m)
    assert missing == [], f"台帳に書いていない耳の担い手: {missing}"


def test_the_spare_engine_is_named_in_the_manifest():
    """控え（`TTS_FALLBACK`）は**仕組みそのもの**なので、名前で確かめる（環-y）。"""
    m = load_manifest()
    assert "TTS_FALLBACK" in m, "控えの担い手の仕組みが台帳に無い"
