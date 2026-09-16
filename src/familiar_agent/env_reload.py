"""`.env` を動かしたまま読み直す（GUI の `/reload`・2026-09-16）。

`.env` は起動時に一度だけ読まれる（`bootstrap.py`）。マイクの倍率や `STT_*` の値を
試すたびに再起動していたので、読み直しの口を置く。**読み直して効くのは、集音の開始時に
読まれるものだけ**である（マイク機器・`AUDIO_INPUT_GAIN`・`STT_*`・担い手）。LLM・鍵・
カメラ・声の担い手は起動時に組んだままなので、応答でその旨を伝える。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .bootstrap import resolve_env_path

logger = logging.getLogger(__name__)

# 読み直しで効く値（集音の立て直しで読まれるもの）。応答に並べて、何が効いたかを見せる。
LISTENER_KEYS = (
    "AUDIO_INPUT_DEVICE",
    "AUDIO_INPUT_GAIN",
    "STT_ENGINE",
    "STT_VAD_SILENCE_SEC",
    "STT_MIN_SEGMENT_SEC",
    "STT_HOLD_GIVE_UP_SEC",
    "STT_NO_SPEECH_MAX",
)


@dataclass(frozen=True)
class EnvReload:
    path: Path
    found: bool

    def summary(self) -> str:
        """吹き出しに出す一行。効いた値と、効かないものがあることを伝える。"""
        if not self.found:
            return f"[error] .env が見つからない：{self.path}"
        values = "・".join(
            f"{k}={os.environ.get(k, '').strip() or '（既定）'}" for k in LISTENER_KEYS
        )
        return f"🔄 .env を読み直した（{values}）——LLM・鍵・カメラ・声の担い手は再起動で"


def reload_env(path: Path | None = None) -> EnvReload:
    """`.env` を読み直して環境変数を上書きする（無い file なら何もしない）。"""
    path = path or resolve_env_path()
    if not path.exists():
        logger.warning(".env を読み直せない（無い）：%s", path)
        return EnvReload(path=path, found=False)
    load_dotenv(path, override=True)
    logger.info(".env を読み直した：%s", path)
    return EnvReload(path=path, found=True)
