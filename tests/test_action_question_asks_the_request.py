"""出-cq：意味ごとの動作の問いは、続きとしてパジュがどうするかでなく、この人が何を頼んでいるかを聞く。

「直前のやりとりの続きとして」と聞くと、直前に音楽を頼んだやりとりが残っているとき「ストップ」でも play_music を
選んだ（id 455・play 0.64）。頼みを聞く形にすると stop 0.99（実験 3）。名乗り・否定と意味の問いはそのまま。
"""

from familiar_agent.core import utterance_meaning as um

ASKS = "この人の言葉は、パジュに何をしてほしいと頼んでいるか"
CONTINUATION = "直前のやりとりの続きとして"


def _qs():
    return um.fanout_questions(
        music=True, camera=True, family=["パパ", "ママ"], suggesting=True, speaker="パパ"
    )


def test_action_questions_ask_what_the_person_requests():
    qs = _qs()
    actions = [
        k for k in qs if k.startswith("action_") and k not in ("action_claim", "action_deny")
    ]
    assert "action_music" in actions and "action_time" in actions
    for k in actions:
        text = qs[k]["instructions"]
        assert ASKS in text, k
        assert CONTINUATION not in text, k


def test_naming_and_denial_questions_keep_their_wording():
    qs = _qs()
    for k in ("action_claim", "action_deny"):
        assert ASKS not in qs[k]["instructions"], k
        assert CONTINUATION in qs[k]["instructions"], k


def test_the_meaning_question_still_reads_the_flow():
    # 意味の問いは直前のやりとりを踏まえて読む。文は出-db 段 B で「話題を変えた言葉も、その中身で選ぶ」形に改めた。
    assert "直前のやりとりを踏まえて読む" in _qs()["meaning"]["instructions"]


def test_the_choices_do_not_change():
    qs = _qs()
    assert qs["action_music"]["criteria"] == {
        k: um.ACTIONS[k] for k in um.ACTIONS_BY_MEANING["music"]
    }
