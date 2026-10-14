"""記憶の申告は Jev だけが出す（記-r・2026-10-14・本人の決定）。

主LLM は `say` の必須の引数 `memory_verdicts` で、自分が W の過去の記憶をどう使ったかを申告していた。軽量LLM の道は
Jev が背景で申告していた。申告の源を Jev 1 つにする：

- 主LLM の自己申告（`say` の引数・規則 `declare-memory-use`）は外す。
- 主LLM の道では、発話前の検査と**同じ Jev の 1 回の呼び出し**で申告も聞く（待ちを増やさない）。検査には、使ったと
  申告された記憶の代わりに、W の過去の記憶の中身（1 件 300 字まで）を事実として渡す。
- 検査で差し戻された返事は声にならないので、申告を当てない。言い直した返事（検査しない）と検査を切っているときは、
  軽量LLM の道と同じく背景で聞く。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges
from familiar_agent.io.oif import MI, Recalled, Verdict
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.loop.prompt import EVENT_SYSTEM_PROMPT

from tests.test_event_loop import _agent
from tests.test_send_back_is_dispatched import _ip, _say


def _recalled(obs_id, content):
    mi = MI(id=obs_id, content=content, timestamp=datetime.now(), direction="発話", obs_id=obs_id)
    return Recalled(mi=mi, fit=0.5, groundedness=0.5, confidence=0.6)


# ── 主LLM には申告させない ───────────────────────────────────────────────────────


def test_the_say_tool_has_no_verdicts():
    from familiar_agent.tools.tts import TTSTool

    schema = TTSTool.get_tool_definitions(MagicMock())[0]["input_schema"]
    assert "memory_verdicts" not in schema["properties"]
    assert schema["required"] == ["text"]


def test_the_prompt_does_not_ask_for_verdicts():
    assert "declare-memory-use" not in EVENT_SYSTEM_PROMPT
    assert "memory_verdicts" not in EVENT_SYSTEM_PROMPT


# ── 検査と申告を 1 回で聞く ─────────────────────────────────────────────────────


def _client(answers):
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(return_value=JevAnswer(ok=True, answers=answers))
    return c


def _c(choice, conf=0.9):
    return {"choice": choice, "confidence": conf}


def _judge(client):
    return asyncio.run(
        jev_judges.judge_speech_and_verdicts(
            client,
            response="明日は晴れだよ",
            recent="",
            facts="見たか：いいえ",
            rules={"no-fake-perception": "見たと言えるのは写真に写っていることだけ"},
            rows={"aaaaaaaaaaaa": "明日は晴れ・最高 30℃", "bbbbbbbbbbbb": "おはよう"},
            min_conf=0.6,
        )
    )


def test_the_check_and_the_verdicts_are_one_call():
    c = _client({"broken": _c("ok"), "aaaaaaaaaaaa": _c("referred"), "bbbbbbbbbbbb": _c("unused")})
    violation, verdicts = _judge(c)
    assert c.ask.await_count == 1
    _state, questions = c.ask.await_args.args
    assert set(questions) == {"broken", "aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert violation is None
    assert verdicts == [
        {"id": "aaaaaaaaaaaa", "verdict": "referred"},
        {"id": "bbbbbbbbbbbb", "verdict": "unused"},
    ]


def test_a_low_confidence_memory_is_not_declared_and_the_violation_still_comes():
    c = _client({"broken": _c("no-fake-perception"), "aaaaaaaaaaaa": _c("important", 0.4)})
    violation, verdicts = _judge(c)
    assert violation and violation.startswith("決まり no-fake-perception")
    assert verdicts == []


def test_the_reply_is_sent_whole_for_the_verdicts():
    """申告は返事のどこで使ったかを見るので、返事を 600 字で切らない（検査だけのころの切り方）。"""
    c = _client({"broken": _c("ok")})
    long = "あ" * 700 + "最高 30℃"
    asyncio.run(
        jev_judges.judge_speech_and_verdicts(
            c,
            response=long,
            recent="",
            facts="",
            rules={"x": "y"},
            rows={"aaaaaaaaaaaa": "最高 30℃"},
            min_conf=0.6,
        )
    )
    state, _ = c.ask.await_args.args
    assert "最高 30℃" in state.split("[家のロボット（パジュ）がいま言おうとしている返事]")[1]


# ── 主LLM の道：検査を通った返事に、Jev の申告を当てる ───────────────────────────────


def test_the_jev_verdicts_land_and_the_main_llm_verdicts_are_ignored():
    ip, a = _ip()
    _oif(a)
    ip._speech_check_violation = AsyncMock(
        return_value=(None, [{"id": "abcdef123456", "verdict": "important"}])
    )
    # **反証の側**：主LLM がまだ申告を書いてきても、使わない（Jev は important、主LLM は useless）。
    _run_with_map(ip, _say("はい", memory_verdicts=[{"id": "abcdef123456", "verdict": "useless"}]))
    a._oif.feedback.assert_called_once_with({"m-full": Verdict.IMPORTANT}, viewpoint="__self__")


def test_the_jev_verdicts_fold_what_was_wanted_to_be_said():
    ip, a = _ip()
    _oif(a)
    ip._speech_check_violation = AsyncMock(
        return_value=(None, [{"id": "abcdef123456", "verdict": "referred"}])
    )
    _run_with_map(
        ip,
        _say("そういえば明日は雨だよ"),
        memories=[_recalled("m-full", "パパに言いたかったこと：明日は雨")],
    )
    assert ip._req.told_unsaid == [("m-full", "パパに言いたかったこと：明日は雨")]


def test_a_sent_back_reply_lands_no_verdicts():
    ip, a = _ip()
    _oif(a)
    ip._speech_check_violation = AsyncMock(
        return_value=("決まり x：y", [{"id": "abcdef123456", "verdict": "important"}])
    )
    _run_with_map(ip, _say("そこに本があるね"))
    ip._dispatch_main_llm.assert_called_once()
    a._oif.feedback.assert_not_called()


def _oif(a):
    """記憶の口。版は裏で書かれる（`oif.write` を待つ）ので、待てる偽物にする。"""
    a._oif = MagicMock()
    a._oif.write = AsyncMock(return_value="ver-1")


def _run_with_map(ip, result, *, memories=None, retried=False):
    from familiar_agent.loop.event_loop import Decision

    return asyncio.run(
        ip._act_on_decision(
            Decision(
                result=result,
                memories=memories or [],
                w_id_map={"abcdef123456": "m-full"},
                verdict_view="__self__",
                recent_frame="",
                system=("安定", "可変"),
                effort="high",
                capped=False,
                retried=retried,
                original_text="",
            ),
            utterance="こんばんは",
            gen=0,
        )
    )


# ── 検査で聞かなかった返事は、背景で聞く ───────────────────────────────────────────


def test_a_retried_reply_is_declared_in_the_background():
    ip, _a = _ip()
    ip._declare_light_memory_use = MagicMock()
    _run_with_map(ip, _say("直した"), retried=True)
    ip._speech_check_violation.assert_not_called()
    kw = ip._declare_light_memory_use.call_args.kwargs
    assert kw["w_id_map"] == {"abcdef123456": "m-full"}
    assert kw["verdict_view"] == "__self__"
    assert kw["reply"] == "直した"


def test_an_unchecked_reply_is_declared_in_the_background():
    """検査を切っている（`speech_check` が偽）と、検査は申告を聞かない（`None`）。"""
    ip, _a = _ip()
    ip._declare_light_memory_use = MagicMock()
    ip._speech_check_violation = AsyncMock(return_value=(None, None))
    _run_with_map(ip, _say("はい"))
    ip._declare_light_memory_use.assert_called_once()


def test_a_checked_reply_is_not_declared_again():
    """**反証の側**：検査で申告を聞いた返事は、背景でもう一度聞かない。"""
    ip, a = _ip()
    _oif(a)
    ip._declare_light_memory_use = MagicMock()
    ip._speech_check_violation = AsyncMock(return_value=(None, []))
    _run_with_map(ip, _say("はい"))
    ip._declare_light_memory_use.assert_not_called()


# ── 検査に渡す事実：W の過去の記憶の中身 ───────────────────────────────────────────


def test_the_checker_gets_the_past_memories_and_asks_about_them():
    a = _agent(stream_returns=[])
    a.config.speech_check = True
    a.config.jev_confidence_min = 0.6
    a._jev = _client({"broken": _c("ok"), "aaaaaaaaaaaa": _c("referred")})
    ip = InformationProcessing(a)
    memories = [
        _recalled("aaaaaaaaaaaa-1", "自分が答えた：明日の東京は晴れで、最高30℃。" + "い" * 400),
        _recalled("cccccccccccc-1", "直近の枠に載った行"),  # 対応表に無い＝過去の列ではない
    ]

    async def scenario():
        got = await ip._speech_check_violation(
            "明日は晴れだよ", "", memories, w_id_map={"aaaaaaaaaaaa": "aaaaaaaaaaaa-1"}
        )
        await ip.close()
        return got

    violation, verdicts = asyncio.run(scenario())
    state, questions = a._jev.ask.await_args.args
    assert "最高30℃" in state
    assert "い" * 300 not in state  # 1 件 300 字まで
    assert "直近の枠に載った行" not in state
    assert set(questions) == {"broken", "aaaaaaaaaaaa"}
    assert violation is None
    assert verdicts == [{"id": "aaaaaaaaaaaa", "verdict": "referred"}]


def test_a_long_state_shrinks_only_the_memories():
    """上限（出-bw の `jev_state_max_chars`）を超えたら、記憶 1 件の字数だけを縮める。返事は削らない。"""
    a = _agent(stream_returns=[])
    a.config.speech_check = True
    a.config.jev_confidence_min = 0.6
    a.config.jev_state_max_chars = 3000
    a._jev = _client({"broken": _c("ok")})
    ip = InformationProcessing(a)
    ids = [f"{i:012d}" for i in range(10)]
    memories = [_recalled(f"{i}-full", f"記憶{i}：" + "う" * 400) for i in ids]
    reply = "明日は晴れだよ" + "え" * 500

    async def scenario():
        got = await ip._speech_check_violation(
            reply, "", memories, w_id_map={i: f"{i}-full" for i in ids}
        )
        await ip.close()
        return got

    asyncio.run(scenario())
    state, questions = a._jev.ask.await_args.args
    assert reply in state  # 返事は削らない
    assert len(state) + sum(len(q["instructions"]) for q in questions.values()) < 3600
    assert set(questions) == {"broken", *ids}  # 問いの数は減らさない


def test_a_short_state_keeps_300_chars():
    """**反証の側**：収まっていれば縮めない。"""
    from familiar_agent.loop.speech_check import PAST_CHARS, fitting_cap

    assert fitting_cap(100, {"a": "x" * 300}, 28_000) == PAST_CHARS
    assert fitting_cap(100, {"a": "x" * 300, "b": "x" * 300}, 900) == 200
