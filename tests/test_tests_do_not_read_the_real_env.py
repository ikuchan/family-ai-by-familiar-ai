"""テストは本物の `.env` を読まない（環-aj・2026-10-11・本人の決定 案 1）。

`config.py` の import で `load_dotenv()` が本物の `.env` を読み、`STT_SAVE_AUDIO` がテストへ漏れていた。書き起こしの
テストが本物の録音の置き場（`~/.cache/familiar-ai/stt_audio`）へ偽の音を書き、置き場は最新 20 本までなので、
テストを回すたびに貯めていた本物の録音（知-z-は）を押し出していた。`conftest.py` が印（`FAMILIAR_SKIP_DOTENV`）を
立て、`config.py` はその印があれば `.env` を読まない。置き場の中身は読まない（件数と最新の時刻だけ）。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from familiar_agent.config import STTConfig
from tests.test_local_stt import _engine

_REAL_FOLDER = Path(os.path.expanduser("~/.cache/familiar-ai/stt_audio"))


def _folder_state() -> "tuple[int, float]":
    if not _REAL_FOLDER.exists():
        return (0, 0.0)
    files = list(_REAL_FOLDER.iterdir())
    return (len(files), max((f.stat().st_mtime for f in files), default=0.0))


def test_the_save_flag_from_the_real_env_does_not_reach_tests():
    # 設定を丸ごと比べない。落ちたときの表示に鍵（`elevenlabs_api_key`）が出る（2026-10-11 の RED で出た）。
    on = STTConfig().save_audio
    assert on is False


def test_the_test_database_stays():
    """**反証側**：`.env` を読まなくても、テスト DB の向きは `conftest` が決めたまま。"""
    assert ":5433/" in os.environ["DATABASE_URL"]


def test_a_written_segment_does_not_touch_the_real_folder():
    before = _folder_state()
    engine = _engine(vad_says=[])
    asyncio.run(engine._write(b"\x00\x40" * 16000))
    assert _folder_state() == before
