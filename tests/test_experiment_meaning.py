"""出-ay の実験の道具（発話の意味の分類）の読み取りと問いの組み立て（2026-10-08）。Jev は呼ばない。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "experiment_meaning.py"
_spec = importlib.util.spec_from_file_location("experiment_meaning", _PATH)
em = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["experiment_meaning"] = em  # dataclass が自分のモジュールを引けるように
_spec.loader.exec_module(em)  # type: ignore[union-attr]

_MD = """# Jev の正解

## 2026-10-08 18:16:32（2）
- 起点：発話
- 言葉：「バージュ音楽をかけて」
- 正解の動作：play_music
- 正解の意味：音楽に関する依頼

## 2026-10-08 18:16:02（1）
- 起点：情動
- 情動の発火：seeking
- 正解の動作：look か search_deferred

## 2026-10-08 21:23:34（29）
- 起点：発話
- 言葉：「パージュ、明日の天気は?」
- 正解の動作：手順で決める（W 次第）
"""


def test_only_utterances_with_a_gold_meaning_are_read():
    (case,) = em.parse_gold(_MD)
    assert (case.n, case.at, case.words, case.gold) == (
        "2",
        "2026-10-08 18:16:32",
        "バージュ音楽をかけて",
        "music",
    )


def test_the_meanings_go_from_clear_to_vague():
    assert list(em.MEANINGS) == [
        "confirm",
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
    q = em.meaning_question(confirming=False, music=False, camera=False)
    assert set(q["meaning"]["criteria"]) == {
        "accepted_check",
        "time",
        "research",
        "answerable",
        "other",
        "off_context",
        "unformed",
    }
    q = em.meaning_question(confirming=True, music=True, camera=True)
    assert set(q["meaning"]["criteria"]) == set(em.MEANINGS)


def test_every_meaning_offers_ask_back_and_silence_and_unformed_only_silence():
    for meaning, keys in em.ACTIONS_BY_MEANING.items():
        if meaning == "unformed":
            assert keys == ("silent",)
            assert em.action_question(meaning) is None  # 1 つしか無ければ聞かない
        else:
            assert {"ask_back", "silent"} <= set(keys), meaning
    assert em.ACTIONS_BY_MEANING["accepted_check"] == ("state_light", "ask_back", "silent")


def test_the_gold_action_is_read():
    md = _MD.replace("- 正解の動作：play_music", "- 正解の動作：play_music")
    (case,) = em.parse_gold(md)
    assert case.action == "play_music"


def test_the_rules_decide_the_final_action():
    d = em.decide
    assert d({"choice": "unformed", "confidence": 0.1}, None)[0] == "silent"
    assert em.THRESHOLD == 0.6  # 本人の決定（2026-10-09）
    assert d({"choice": "music", "confidence": 0.5}, None)[0] == "fallback"
    assert set(em.UNSURE_ACTIONS) == {"reply_full", "ask_back"}
    ask = {"choice": "ask_back", "confidence": 0.2}
    assert d({"choice": "music", "confidence": 0.9}, ask)[0] == "ask_back"
    assert (
        d({"choice": "music", "confidence": 0.9}, {"choice": "play_music", "confidence": 0.5})[0]
        == "fallback"
    )
    assert (
        d({"choice": "music", "confidence": 0.9}, {"choice": "play_music", "confidence": 0.9})[0]
        == "play_music"
    )


def test_research_offers_only_tools_and_answering_is_its_own_meaning():
    assert "reply_full" not in em.ACTIONS_BY_MEANING["research"]
    assert em.ACTIONS_BY_MEANING["answerable"][:2] == ("reply_full", "reply_light")


def test_off_context_asks_back_only_when_sure():
    d = em.decide
    ctx = {"choice": "off_context", "confidence": 0.2}
    assert d(ctx, {"choice": "ask_back", "confidence": 0.7})[0] == "ask_back"
    assert d(ctx, {"choice": "ask_back", "confidence": 0.4})[0] == "silent"
    assert d(ctx, {"choice": "silent", "confidence": 0.9})[0] == "silent"
