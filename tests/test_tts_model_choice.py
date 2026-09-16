"""ElevenLabs のモデルを `.env` で選ぶ（`ELEVENLABS_MODEL`・2026-09-16 実機）。

`eleven_v3` は 73 字の合成に 5.4 秒かかり、返事が出るまで毎回それだけ待たせていた
（ログの `DIF 声 19.07 秒（86 字）` は合成 5〜6 秒＋再生 12〜13 秒）。同じ文で
`eleven_flash_v2_5` は 0.7 秒、`eleven_multilingual_v2` は 1.7 秒。聞き比べて flash を
既定にし、必要なら v3 へ戻せるようにする。

角括弧タグ（[cheerful] など）を指示として解するのは v3 だけなので、タグを残すかどうかは
担い手だけでなく**モデル**で決める。flash に渡せばそのまま音になる。
"""

from __future__ import annotations

import os
from unittest.mock import patch

from familiar_agent.config import TTSConfig
from familiar_agent.tools.tts import TTSTool


def test_the_model_defaults_to_flash():
    with patch.dict(os.environ, {}, clear=True):
        assert TTSConfig().elevenlabs_model == "eleven_flash_v2_5"


def test_the_model_can_be_switched_back_to_v3():
    with patch.dict(os.environ, {"ELEVENLABS_MODEL": "eleven_v3"}, clear=True):
        assert TTSConfig().elevenlabs_model == "eleven_v3"


def test_only_v3_understands_bracket_tags():
    v3 = TTSTool("k", "v", engine="elevenlabs", elevenlabs_model="eleven_v3")
    flash = TTSTool("k", "v", engine="elevenlabs", elevenlabs_model="eleven_flash_v2_5")
    assert v3.understands_tags is True
    assert flash.understands_tags is False


def test_the_tool_defaults_to_flash_too():
    assert TTSTool("k", "v", engine="elevenlabs").elevenlabs_model == "eleven_flash_v2_5"
