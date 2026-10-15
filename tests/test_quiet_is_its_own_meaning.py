"""黙る依頼は、時間から独立した意味「話しかけ方の依頼」で聞く（出-cn・2026-10-15・本人の決定カ2）。

「黙って」系の言い方は、意味の段で「その他の会話」か「時間に関する依頼」に読まれ、動作の段で「黙る（何も言わず、何も
しない）」（`silent`）を選んで、黙る依頼（`quiet`）にならなかった（実験 1・2：18 回中 quiet 0 回）。意味「話しかけ方の
依頼」（`speaking`）を足し、動作を黙る依頼・解く・聞き返すの 3 つにした（`silent` を並べない）ところ、18 回中 18 回 quiet
になった（実験 3）。「待って」「ちょっと待ってね」「おやすみ」は quiet にならなかった。
"""

from __future__ import annotations

import asyncio

from familiar_agent.core import utterance_meaning as um
from familiar_agent.store import arbiter_records
from tests._arbiter_fakes import decide, writer_says
from tests.test_arbiter_utterance_by_meaning import _a, _jev


def test_speaking_is_a_meaning_of_its_own():
    assert "speaking" in um.MEANINGS
    name, desc = um.MEANINGS["speaking"]
    assert name == "話しかけ方の依頼"
    assert "静かにして" in desc and "待って" in desc  # 待っては当たらないと書く
    assert um.ACTIONS_BY_MEANING["speaking"] == ("quiet", "lift_quiet", "ask_back")


def test_time_no_longer_holds_silence():
    assert not {"quiet", "lift_quiet"} & set(um.ACTIONS_BY_MEANING["time"])
    assert "黙っていて" not in um.MEANINGS["time"][1]
    assert "話していいよ" not in um.MEANINGS["time"][1]


def test_the_length_is_asked_with_speaking():
    qs = um.fanout_questions(music=False, camera=False, family=[], silenced=False)
    assert set(qs["action_speaking"]["criteria"]) == {"quiet", "ask_back"}
    assert qs["quiet_minutes"]["type"] == "choice"
    lifted = um.fanout_questions(music=False, camera=False, family=[], silenced=True)
    assert "lift_quiet" in lifted["action_speaking"]["criteria"]


def test_a_speaking_quiet_answer_sets_the_minutes(monkeypatch):
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    jev = _jev(
        {
            "meaning": _a("speaking", 0.99),
            "action_speaking": _a("quiet", 0.96),
            "quiet_minutes": _a("10", 0.9),
        }
    )
    d = asyncio.run(decide(jev=jev, writer=writer_says({"text": "わかった"}), utterance="黙って"))
    assert (d.branch, d.silence_minutes) == ("light", 10)


def test_what_stays_the_same():
    """**反証の側**：silent の説明は元のまま・時間はタイマーのまま・しきい値はまだ見る。"""
    assert um.ACTIONS["silent"] == "黙る（何も言わず、何もしない）"
    assert "set_timer" in um.ACTIONS_BY_MEANING["time"]
    assert um.decide(_a("time", 0.99), _a("set_timer", 0.9)).final == "set_timer"
    assert um.decide(_a("speaking", 0.5), _a("quiet", 0.9)).final == "unsure"
    assert um.decide(_a("speaking", 0.9), _a("quiet", 0.5)).final == "unsure"
