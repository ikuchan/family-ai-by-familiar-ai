"""不変条件テストの共通部品。

ここに置くのは、システムが「守ると言っていること」を端から端まで確かめる少数の
テストである。個々の機能の正しさではなく、**壊れていたら気づけること**を目的に
する。2026-06-29 から 2026-07-20 まで、会話しても記憶が一切書かれない状態に3週間
気づかなかった。そのとき落ちるものを常設する。

実 DB を使うので、通常の一式とはマーカー `invariant` で分けて走らせる。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


@pytest.fixture
def memory():
    """埋め込みモデルを読み込まない ObservationMemory。"""
    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory(person_id=DEFAULT_PERSON_ID)
    with patch.object(_EmbeddingModel, "encode_document", return_value=[[0.1] * 1024]):
        yield mem
