"""発話の意味と動作の問い（出-ay 段 4-4a・2026-10-09・`設計方針_判定の段` §2.2.5）。調停からはまだ呼ばない部品。

実験（`scripts/experiment_meaning.py`）で決めた形を本体へ移す。1 回目に意味（12 個・明確なものからあやふやなものの順）、
2 回目に意味ごとの動作。先読みで 1 回に並べる（意味の問いと、意味ごとの「もしこの言葉が○○なら」の問い）。決まり：
成立しないものは黙る。文脈に合わない言葉は、聞き返すを 0.6 以上で選んだときだけ聞き返し、それ以外は黙る。ほかは確率で
決める（出-db 段 A・`tests/test_meaning_by_probability.py`）。
"""

from __future__ import annotations

from familiar_agent.core import utterance_meaning as um


def _a(choice: str, confidence: float) -> dict:
    return {"choice": choice, "confidence": confidence}


def test_the_meanings_go_from_clear_to_vague():
    assert list(um.MEANINGS) == [
        "suggestion",
        "claim",
        "deny",
        "accepted_check",
        "music",
        "time",
        "speaking",
        "look",
        "research",
        "家族の情報源",
        "answerable",
        "other",
        "off_context",
        "unformed",
    ]


def test_unusable_meanings_are_not_offered():
    got = um.offered(music=False, camera=False)
    assert "music" not in got and "look" not in got
    assert "suggestion" not in got
    assert um.offered(music=True, camera=True, suggesting=True) == list(um.MEANINGS)


def test_every_meaning_offers_ask_back_and_silence_except_the_fixed_ones():
    for meaning, keys in um.ACTIONS_BY_MEANING.items():
        if meaning == "unformed":
            assert keys == ("silent",)
        elif meaning in ("claim", "deny"):
            continue  # 2 回目は誰か・どの呼び方か（家族の呼び方から作る）
        elif meaning == "speaking":
            assert (
                "ask_back" in keys and "silent" not in keys
            )  # 黙るは「黙って」とぶつかる（出-cn）
        else:
            assert {"ask_back", "silent"} <= set(keys), meaning
    assert "reply_full" not in um.ACTIONS_BY_MEANING["research"]
    assert {"quiet", "lift_quiet"} <= set(um.ACTIONS_BY_MEANING["speaking"])


def test_the_fan_out_asks_the_meaning_and_each_meanings_actions_at_once():
    qs = um.fanout_questions(music=True, camera=True, family=["パパ", "ママ"])
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
    assert d(_a("music", 0.5), _a("play_music", 0.9)).final == "play_music"
    assert d(_a("music", 0.9), _a("ask_back", 0.2)).final == "ask_back"
    assert (
        d(_a("time", 0.9), _a("set_timer", 0.5)).final == "set_timer"
    )  # 確率で決める（出-db 段 A）
    assert d(_a("music", 0.9), _a("play_music", 0.9)).final == "play_music"
    claimed = d(_a("claim", 0.9), _a("パパ", 0.9))
    assert (claimed.final, claimed.name) == ("claim", "パパ")


def test_ask_back_is_twenty_characters():
    assert um.ASK_BACK_MAX_CHARS == 20


def test_the_fan_out_also_asks_how_deeply_to_think():
    """考える深さ（low・medium・high）も同じ 1 回で聞く（2026-10-09 本人の決定ウ）。考えて返すとき主LLM に渡す。"""
    qs = um.fanout_questions(music=True, camera=True, family=[])
    assert set(qs["effort"]["criteria"]) == {"low", "medium", "high"}


# ── 段 5（2026-10-10 本人の決定ア）：古い問いにあった決まりを新しい問いへ移す ──────────────────────


def _qs(**kw):
    base = dict(music=False, camera=False, family=["パパ", "たいき"])
    return um.fanout_questions(**{**base, **kw})


def test_a_nameless_denial_can_point_to_the_current_speaker():
    """「ちがうよ」（名前なし）は、いま話者としている人を否定したとして選べる。"""
    deny = _qs(speaker="パパ")["action_deny"]["criteria"]
    assert "名前を言わず" in deny["パパ"]
    stranger = _qs(speaker="おきゃくさん")["action_deny"]["criteria"]
    assert "名前を言わず" in stranger["おきゃくさん"]
    assert "名前を言わず" not in str(_qs()["action_deny"]["criteria"])


def test_waiting_is_not_a_request_for_silence():
    assert "待って" in um.ACTIONS["quiet"] and "これには当たらない" in um.ACTIONS["quiet"]


def test_lifting_the_silence_is_offered_only_while_silenced():
    assert "lift_quiet" in _qs(silenced=True)["action_speaking"]["criteria"]
    assert "lift_quiet" not in _qs(silenced=False)["action_speaking"]["criteria"]


def test_the_house_tools_are_offered_only_when_connected():
    # 家の道具は意味「家族に関わる情報源を見に行く問い」の動作（知-ax）。調べる問いには並ばない。
    family = _qs(tools={"family_schedule"})["action_家族の情報源"]["criteria"]
    assert {"recall", "family_schedule"} <= set(family)
    assert not {"house_rules", "notion_search", "journal", "vault"} & set(family)
    assert "family_schedule" not in _qs(tools=set())["action_家族の情報源"]["criteria"]
    research = _qs(tools={"family_schedule"})["action_research"]["criteria"]
    assert set(research) == {"search_deferred", "recall", "ask_back", "silent"}


def test_the_confirm_meaning_is_retired():
    """確認待ちのあいだは調停が 3 択だけで決める（知-o・2026-10-13）ので、意味の「確認待ちへの答え」は撤去した。"""
    import inspect

    assert "confirm" not in um.MEANINGS
    assert "confirm" not in um.ACTIONS and "decline" not in um.ACTIONS
    assert "confirm" not in um.ACTIONS_BY_MEANING
    assert "confirming" not in inspect.signature(um.offered).parameters
    assert "confirming" not in inspect.signature(um.fanout_questions).parameters


def test_time_is_read_from_lengths_and_clock_times():
    """書き起こしが崩れても、何分・何秒・何時があれば時間と読む（出-cb・10/13 19:58「3分のダイマーをかけて」を音楽と読んだ）。"""
    time_desc = um.MEANINGS["time"][1]
    assert "何分・何秒・何時" in time_desc and "書き起こしが崩れていても" in time_desc
    assert "音楽ではない" in um.MEANINGS["music"][1]


def test_music_takes_the_likeliest_action_whatever_its_confidence():
    """音楽の 2 回目は見込みを見ない（出-cf・10/14 00:46 に「ストップ」を確信度不足で聞き返した・本人の決定ウ）。"""
    got = um.decide(_a("music", 0.9), _a("stop_music", 0.45))
    assert got.final == "stop_music"
