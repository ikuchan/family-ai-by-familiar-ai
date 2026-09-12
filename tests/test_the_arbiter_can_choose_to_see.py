"""調停（軽量LLM）も see を選べる（`イベント駆動ループ` v0.44）。

カメラを向ける判断が主LLM だけにあると、「何が見えますか？」でも想起→調停→主LLM を経て
やっと see に着く（実機 2.6 秒）。調停は「見るべき」と分かっても手段が無く、検索へ逃げた。
カメラが無い構成では候補に載せない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.arbiter import _parse, arbitrate


def test_see_is_accepted_when_the_agent_can_see() -> None:
    d = _parse('{"branch": "action", "action": "see", "text": "見てみますね"}', can_see=True)
    assert d is not None and d.branch == "action" and d.action == "see"
    assert d.query == "目の前を見る", "見出しは固定なので query は埋める"


def test_see_is_rounded_to_recall_without_a_camera() -> None:
    d = _parse('{"branch": "action", "action": "see", "query": "部屋"}', can_see=False)
    assert d is not None and d.action == "recall"


def _prompt_of(can_see: bool) -> str:
    backend = MagicMock()
    backend.complete = AsyncMock(return_value='{"branch": "full"}')
    asyncio.run(arbitrate(backend, utterance="何が見える？", workspace_ctx="", can_see=can_see))
    return backend.complete.call_args.args[0]


def test_the_prompt_offers_see_only_with_a_camera() -> None:
    assert '"see"' in _prompt_of(True)
    assert '"see"' not in _prompt_of(False)


def test_the_prompt_says_the_photo_goes_to_the_main_llm() -> None:
    assert "写真そのものは主LLM" in _prompt_of(True)


def test_the_prompt_lets_light_answer_from_the_labels() -> None:
    """ラベルで足りる問いは light（v0.45）。写真を見て語るなら full。"""
    prompt = _prompt_of(True)
    assert 'そのラベルで "light"' in prompt
    assert "80 種" in prompt
