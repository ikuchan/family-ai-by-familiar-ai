"""調停（軽量LLM）も see を選べる（`イベント駆動ループ` v0.44）。

カメラを向ける判断が主LLM だけにあると、「何が見えますか？」でも想起→調停→主LLM を経て
やっと see に着く（実機 2.6 秒）。調停は「見るべき」と分かっても手段が無く、検索へ逃げた。
カメラが無い構成では候補に載せない。
"""

from __future__ import annotations

from familiar_agent.loop.arbiter import assemble


def test_see_is_accepted_when_the_agent_can_see() -> None:
    d = assemble({"branch": "action", "action": "see", "filler": "見てみますね"}, can_see=True)
    assert d is not None and d.branch == "action" and d.action == "see"
    assert d.query == "目の前を見る", "見出しは固定なので query は埋める"


def test_see_is_rounded_to_recall_without_a_camera() -> None:
    d = assemble({"branch": "action", "action": "see", "query": "部屋"}, can_see=False)
    assert d is not None and d.action == "recall"


def _offered(can_see: bool) -> set:
    """発話の問いに並ぶ動作（段 4-4b から、見る・首を向けるは意味「見る依頼」の動作）。"""
    from familiar_agent.core import utterance_meaning as um

    qs = um.fanout_questions(confirming=False, music=False, camera=can_see, family=[])
    return {a for k, q in qs.items() if k.startswith("action_") for a in q["criteria"]}


def test_see_is_offered_only_with_a_camera() -> None:
    assert "see" in _offered(True)
    assert "see" not in _offered(False)


# 写真の読み取りで答えるか主LLM が写真を見て語るかの古い目安（`SEE_GUIDE`）は、出-ay 段 5c で外した。見た結果が届いた
# あとは完了の表（頼まれて見た結果 → 軽く返す／もう一度見る・2026-10-09 本人）で決める（`test_completion_kind`）。
