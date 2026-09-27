"""調停（軽量LLM）も see を選べる（`イベント駆動ループ` v0.44）。

カメラを向ける判断が主LLM だけにあると、「何が見えますか？」でも想起→調停→主LLM を経て
やっと see に着く（実機 2.6 秒）。調停は「見るべき」と分かっても手段が無く、検索へ逃げた。
カメラが無い構成では候補に載せない。
"""

from __future__ import annotations

from tests._arbiter_compat import _parse


def test_see_is_accepted_when_the_agent_can_see() -> None:
    d = _parse('{"branch": "action", "action": "see", "filler": "見てみますね"}', can_see=True)
    assert d is not None and d.branch == "action" and d.action == "see"
    assert d.query == "目の前を見る", "見出しは固定なので query は埋める"


def test_see_is_rounded_to_recall_without_a_camera() -> None:
    d = _parse('{"branch": "action", "action": "see", "query": "部屋"}', can_see=False)
    assert d is not None and d.action == "recall"


def _questions(can_see: bool) -> dict:
    from familiar_agent.loop.arbiter import Arbiter, ArbiterInput

    return Arbiter(jev=None, writer=None)._questions(
        ArbiterInput(utterance="何が見える？", workspace_ctx="", can_see=can_see)
    )


def _state(can_see: bool) -> str:
    from familiar_agent.loop.arbiter import Arbiter, ArbiterInput

    return Arbiter(jev=None, writer=None)._state(
        ArbiterInput(utterance="何が見える？", workspace_ctx="", can_see=can_see)
    )


def test_see_is_offered_only_with_a_camera() -> None:
    # 動作は Jev の選択肢（出-au 段 5-7d）。
    assert "see" in _questions(True)["action"]["criteria"]
    assert "see" not in _questions(False)["action"]["criteria"]


def test_the_guide_says_the_photo_goes_to_the_main_llm() -> None:
    assert "写真そのものは主LLM に渡る" in _state(True)
    assert "写真そのものは主LLM に渡る" not in _state(False)


def test_the_guide_lets_light_answer_from_the_reading() -> None:
    """写真の読み取り（『見えたもの』）で足りる問いは light。写真を見て語るなら full（出-au 段 5-7a）。"""
    state = _state(True)
    assert "『見えたもの』" in state and "light でよい" in state
