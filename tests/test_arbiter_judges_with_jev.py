"""調停の判定を Jev が決める（出-au 段 5-7b・2026-09-27・`設計方針_判定の段` v0.4 §2.2.2）。

`Arbiter`（調停を 1 つのクラスにまとめた・本人の決定 ア）の判定の段。Jev に 1 回で次をまとめて聞く。
分岐・深さ・動作（そのとき使える動作だけ）・時期を指しているか・黙る依頼とその長さ・解く依頼・名乗りと誰か・打ち消しと
どれか。答えは `_parse` の守りへ渡す辞書の形で返す。分岐か動作の確信度が 0.6 未満・Jev が使えないときは None
（呼び手が full へ倒す）。道具の帰りの反復では、黙る依頼・名乗りは聞かない（発話は古い・情-n）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.loop.arbiter import Arbiter, ArbiterInput

_FAMILY = """## パパ
- 名前: ゆうすけ
- 呼び方: パパ、おとうさん
## たいき
- 名前: たいき
- 呼び方: たいき、たいきくん
"""


def _inp(**kw):
    base = dict(
        utterance="パジュ、右向いて",
        workspace_ctx="[直近のやりとり]\n- 相手：おはよう",
        present_ctx="パパ",
        now_ctx="2026-09-27 10:00",
        origin="発話",
        can_see=True,
        extra_actions=("set_timer",),
        family_md=_FAMILY,
        current_speaker="パパ",
        silenced=False,
    )
    base.update(kw)
    return ArbiterInput(**base)


def _jev(answers, ok=True):
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(return_value=JevAnswer(ok=ok, answers=answers))
    return c


def _judge(jev, inp):
    return asyncio.run(Arbiter(jev=jev, writer=MagicMock(), min_conf=0.6)._judge(inp))


def _c(pick, conf=0.9):
    return {"choice": pick, "confidence": conf}


def test_one_call_asks_everything_with_only_the_usable_actions():
    j = _jev({"branch": _c("action"), "action": _c("look"), "effort": _c("low")})
    got = _judge(j, _inp())
    assert got["branch"] == "action" and got["action"] == "look"
    j.ask.assert_awaited_once()
    state, qs = j.ask.await_args.args
    assert {
        "branch",
        "effort",
        "action",
        "refers_time",
        "asks_quiet",
        "quiet_minutes",
        "claims",
        "claimed",
        "denies",
        "denied",
    } <= set(qs)
    actions = set(qs["action"]["criteria"])
    assert {"look", "see", "set_timer", "recall", "search_deferred"} <= actions
    assert "family_schedule" not in actions  # 繋がっていない道具は並べない
    assert "パジュ、右向いて" in state and "おはよう" in state
    assert set(qs["claimed"]["criteria"]) == {"パパ", "たいき", "other"}


def test_without_a_camera_see_and_look_are_not_offered():
    j = _jev({"branch": _c("full")})
    _judge(j, _inp(can_see=False))
    assert not {"see", "look"} & set(j.ask.await_args.args[1]["action"]["criteria"])


def test_capped_offers_only_answering():
    j = _jev({"branch": _c("full")})
    _judge(j, _inp(capped=True))
    assert set(j.ask.await_args.args[1]["branch"]["criteria"]) == {"light", "full"}


def test_low_confidence_or_failure_means_fall_back():
    assert _judge(_jev({"branch": _c("light", 0.5)}), _inp()) is None
    assert _judge(_jev({"branch": _c("action"), "action": _c("look", 0.4)}), _inp()) is None
    assert _judge(_jev({}, ok=False), _inp()) is None
    assert _judge(None, _inp()) is None


def test_quiet_claims_denials_and_time_are_read():
    j = _jev(
        {
            "branch": _c("light"),
            "refers_time": {"noul": 0.8},
            "asks_quiet": {"noul": 0.9},
            "quiet_minutes": _c("default"),
            "lifts_quiet": {"noul": 0.1},
            "claims": {"noul": 0.7},
            "claimed": _c("たいき"),
            "denies": {"noul": 0.2},
        }
    )
    got = _judge(j, _inp())
    assert got["silence_minutes"] == -1  # 長さの指定なし＝既定
    assert got["speaker_claim"] == "たいき"
    assert got["not_person"] == "" and got["refers_time"] is True
    assert got["lift_silence"] is False


def test_a_tool_return_does_not_read_quiet_or_claims():
    j = _jev({"branch": _c("light")})
    _judge(j, _inp(tool_return=True))
    qs = j.ask.await_args.args[1]
    assert not {"asks_quiet", "claims", "denies"} & set(qs)


def test_self_driven_turns_describe_the_branches_as_own_actions():
    j = _jev({"branch": _c("light")})
    _judge(j, _inp(origin="情動", utterance="[内的な促し:SEEKING] 探索したい"))
    assert "黙る" in j.ask.await_args.args[1]["branch"]["criteria"]["light"]


# ── 倒れた理由を残す（出-ay 段 1・2026-10-07）────────────────────────────────
#
# 10/6〜10/7 の実機で、調停 119 回のうち 79 回が「調停を決められなかったのでフルへ倒す（判定=なし）」だった。どの条件で
# 倒れたかがログに無いので、倒す前に理由を置き、警告に足す。倒す・倒さないの結果は変えない。


def _why(jev, inp=None):
    arb = Arbiter(jev=jev, writer=MagicMock(), min_conf=0.6)
    got = asyncio.run(arb._judge(inp or _inp()))
    return got, arb._why


def _p(pick, conf, probs):
    return {"choice": pick, "confidence": conf, "probabilities": probs}


def test_why_when_jev_does_not_answer():
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(return_value=JevAnswer(ok=False, error="時間切れ"))
    got, why = _why(c)
    assert got is None and "Jev が答えなかった" in why and "時間切れ" in why


def test_why_when_the_branch_is_unsure():
    got, why = _why(
        _jev({"branch": _p("full", 0.55, {"full": 0.55, "light": 0.40, "action": 0.05})})
    )
    assert got is None
    assert why == "分岐の確信度 0.55＜0.6（1 番 full 0.55・2 番 light 0.40）"


def test_why_when_the_action_is_unsure():
    got, why = _why(
        _jev(
            {
                "branch": _c("action"),
                "action": _p("search_deferred", 0.56, {"search_deferred": 0.56, "set_timer": 0.30}),
            }
        )
    )
    assert got is None
    assert why == "動作の確信度 0.56＜0.6（1 番 search_deferred 0.56・2 番 set_timer 0.30）"


def test_why_when_the_action_is_not_offered():
    got, why = _why(_jev({"branch": _c("action"), "action": _c("play_music", 0.9)}))
    assert got is None and why == "動作 play_music は候補に無い"


def test_no_why_when_it_decides():
    got, why = _why(_jev({"branch": _c("light"), "effort": _c("low")}))
    assert got is not None and why == ""


def test_the_warning_carries_the_why(caplog):
    arb = Arbiter(
        jev=_jev({"branch": _p("full", 0.55, {"full": 0.55, "light": 0.40})}),
        writer=MagicMock(),
        min_conf=0.6,
    )
    with caplog.at_level("WARNING"):
        asyncio.run(arb.decide(_inp()))
    assert "判定=なし・分岐の確信度 0.55＜0.6" in caplog.text
