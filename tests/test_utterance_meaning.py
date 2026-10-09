"""発話の意味と動作の問い（出-ay 段 4-4a・2026-10-09・`設計方針_判定の段` §2.2.5）。調停からはまだ呼ばない部品。

実験（`scripts/experiment_meaning.py`）で決めた形を本体へ移す。1 回目に意味（12 個・明確なものからあやふやなものの順）、
2 回目に意味ごとの動作。先読みで 1 回に並べる（意味の問いと、意味ごとの「もしこの言葉が○○なら」の問い）。決まり：
成立しないものは黙る。文脈に合わない言葉は、聞き返すを 0.6 以上で選んだときだけ聞き返し、それ以外は黙る。ほかはしきい値
0.6。聞き返すは確信度に関係なく。越えなければ「よく考えるか、軽く聞き返すか」を聞き、それも 0.6 未満なら軽く聞き返す。
"""

from __future__ import annotations

from familiar_agent.core import utterance_meaning as um


def _a(choice: str, confidence: float) -> dict:
    return {"choice": choice, "confidence": confidence}


def test_the_meanings_go_from_clear_to_vague():
    assert list(um.MEANINGS) == [
        "confirm",
        "claim",
        "deny",
        "accepted_check",
        "music",
        "time",
        "look",
        "research",
        "answerable",
        "other",
        "off_context",
        "unformed",
    ]


def test_unusable_meanings_are_not_offered():
    got = um.offered(confirming=False, music=False, camera=False)
    assert "confirm" not in got and "music" not in got and "look" not in got
    assert um.offered(confirming=True, music=True, camera=True) == list(um.MEANINGS)


def test_every_meaning_offers_ask_back_and_silence_except_the_fixed_ones():
    for meaning, keys in um.ACTIONS_BY_MEANING.items():
        if meaning == "unformed":
            assert keys == ("silent",)
        elif meaning in ("claim", "deny"):
            continue  # 2 回目は誰か・どの呼び方か（家族の呼び方から作る）
        else:
            assert {"ask_back", "silent"} <= set(keys), meaning
    assert "reply_full" not in um.ACTIONS_BY_MEANING["research"]
    assert {"quiet", "lift_quiet"} <= set(um.ACTIONS_BY_MEANING["time"])


def test_the_fan_out_asks_the_meaning_and_each_meanings_actions_at_once():
    qs = um.fanout_questions(confirming=False, music=True, camera=True, family=["パパ", "ママ"])
    assert "meaning" in qs
    assert "action_music" in qs and "action_research" in qs
    assert "action_unformed" not in qs  # 1 つしか無いので聞かない
    assert set(qs["action_claim"]["criteria"]) == {"パパ", "ママ", "other"}
    assert "直前のやりとり" in qs["meaning"]["instructions"]


def test_the_rules():
    d = um.decide
    assert d(_a("unformed", 0.1), None).final == "silent"
    assert d(_a("off_context", 0.2), _a("ask_back", 0.7)).final == "ask_back"
    assert d(_a("off_context", 0.2), _a("ask_back", 0.4)).final == "silent"
    assert d(_a("music", 0.5), _a("play_music", 0.9)).final == "unsure"
    assert d(_a("music", 0.9), _a("ask_back", 0.2)).final == "ask_back"
    assert d(_a("music", 0.9), _a("play_music", 0.5)).final == "unsure"
    assert d(_a("music", 0.9), _a("play_music", 0.9)).final == "play_music"
    claimed = d(_a("claim", 0.9), _a("パパ", 0.9))
    assert (claimed.final, claimed.name) == ("claim", "パパ")


def test_when_unsure_think_or_ask_back():
    assert um.resolve_unsure(_a("reply_full", 0.7)) == "reply_full"
    assert um.resolve_unsure(_a("ask_back", 0.7)) == "ask_back"
    assert um.resolve_unsure(_a("reply_full", 0.4)) == "ask_back"
    assert set(um.unsure_question()["action"]["criteria"]) == {"reply_full", "ask_back"}


def test_ask_back_is_twenty_characters():
    assert um.ASK_BACK_MAX_CHARS == 20


def test_the_fan_out_also_asks_how_deeply_to_think():
    """考える深さ（low・medium・high）も同じ 1 回で聞く（2026-10-09 本人の決定ウ）。考えて返すとき主LLM に渡す。"""
    qs = um.fanout_questions(confirming=False, music=True, camera=True, family=[])
    assert set(qs["effort"]["criteria"]) == {"low", "medium", "high"}
