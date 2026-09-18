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
    # タイマーの振る舞い（知-o・2026-09-18）。集音ではないが、保存した瞬間に効くものとして要約に出す。
    "TIMER_SILENCE",
    "TIMER_MIC_CLOSE",
    "TIMER_CONFIRM",
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
    # 読む側（`bootstrap`）と同じ順：素の `.env` を読んでから、上書き file（`FAMILIAR_ENV_FILE`）を重ねる
    # （環-n・2026-09-18）。設定画面は素の `.env` に書くので、保存直後に呼んでも上書き file の値が勝つ。
    # 素の `.env` は **呼び手が `path` で渡したときだけ**読む（GUI は `settings_env_path()` を渡す）。
    # 渡されなければ上書き file（`FAMILIAR_ENV_FILE`・無ければ素の `.env` の位置）だけ。**本物の `.env` を
    # 暗黙に読まない**——読むとテストのプロセスの `DATABASE_URL` が本番に変わり、後のテストが本番へ書いた
    # （環-r・2026-09-19・168 行・片付け済み）。`conftest` の番人が `DATABASE_URL` の変化を落とす。
    base = path
    overlay = resolve_env_path() if os.environ.get("FAMILIAR_ENV_FILE") else None
    read: list[Path] = []
    if base is not None and base.exists():
        load_dotenv(base, override=True)
        read.append(base)
    if overlay is not None and overlay != base and overlay.exists():
        load_dotenv(overlay, override=True)
        read.append(overlay)
    if not read:
        logger.warning(".env を読み直せない（無い）：%s", base or overlay)
        return EnvReload(path=base or overlay or Path(".env"), found=False)
    logger.info(".env を読み直した：%s", "・".join(str(p) for p in read))
    return EnvReload(path=read[-1], found=True)
