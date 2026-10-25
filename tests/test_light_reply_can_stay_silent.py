"""軽量LLM が人の言葉に軽く返す・聞き返すときは、返すか黙るかを先に決めさせる（出-db 段 E・2026-10-25・本人の決定ウ・形 2）。

返しようのない言葉（テレビや動画の声、家族どうしの話、崩れた書き起こし）に、軽量LLM は何か一言を返してしまう。書く呼び出しの
中で確かめさせると 12 件中 5 件しか黙らず、短い指示で「返す／黙る」だけを先に聞くと 9 件黙った（gemini-3.1-flash-lite）。
待ちを足さないよう、判定と書く呼び出しは同時に走らせ、黙るなら書いた文を捨てる。黙るときは主LLM に回さない（文が空だと
「書けなかった」として主LLM に回っていた）。判定が失敗したら返す。入れるのは人の言葉への軽く返す・聞き返すだけ。
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from tests._arbiter_fakes import decide


def _jev(answers: dict) -> MagicMock:
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(return_value=JevAnswer(ok=True, answers=answers))
    return jev


def _a(choice: str, confidence: float = 0.9) -> dict:
    return {"choice": choice, "confidence": confidence}


def _writer(
    *, judge: str = "判定：返す", text: str = "うん、いいよ。", judge_raises=None, delay=0.0
):
    """判定の呼び出し（system に「判定：」の書き方がある）と、書く呼び出しを分けて答える偽の軽量LLM。"""
    calls: list = []

    async def complete(prompt, max_tokens=300, *, system=None):
        is_judge = "「判定：返す」か「判定：黙る」" in (system or "")
        calls.append(("judge" if is_judge else "write", asyncio.get_running_loop().time()))
        if delay:
            await asyncio.sleep(delay)
        if is_judge:
            if judge_raises is not None:
                raise judge_raises
            return judge
        return json.dumps({"text": text}, ensure_ascii=False)

    w = MagicMock()
    w.complete = AsyncMock(side_effect=complete)
    w.calls = calls
    return w


_LIGHT = {"meaning": _a("other"), "action_other": _a("reply_light")}
_ASK = {"meaning": _a("other"), "action_other": _a("ask_back")}


def _run(answers, writer, **inp):
    inp.setdefault("origin", "発話")
    return asyncio.run(
        decide(jev=_jev(answers), writer=writer, utterance="きょうも退屈してたんだ", **inp)
    )


def test_a_light_reply_judged_silent_is_silent_without_the_main_llm():
    w = _writer(judge="判定：黙る")
    d = _run(_LIGHT, w)
    assert (d.branch, d.text) == ("light", "")


def test_an_ask_back_judged_silent_is_silent_and_keeps_no_heard_check():
    w = _writer(judge="判定：黙る", text="しりとりのこと？")
    d = _run(_ASK, w)
    assert (d.branch, d.text) == ("light", "")
    assert d.heard_check is None


def test_a_light_reply_judged_to_reply_is_spoken():
    """**反証の側**：返すと決めたら、書いた文をそのまま話す。"""
    d = _run(_LIGHT, _writer(judge="判定：返す"))
    assert (d.branch, d.text) == ("light", "うん、いいよ。")


def test_a_failed_judgement_replies():
    d = _run(_LIGHT, _writer(judge_raises=RuntimeError("時間切れ")))
    assert (d.branch, d.text) == ("light", "うん、いいよ。")


def test_the_judgement_and_the_writing_run_at_once():
    """待ちを足さない：判定と書く呼び出しは同時に始まる（どちらも 0.2 秒かかっても、合わせて 0.2 秒ほど）。"""
    w = _writer(delay=0.2)
    _run(_LIGHT, w)
    kinds = sorted(k for k, _t in w.calls)
    assert kinds == ["judge", "write"]
    starts = [t for _k, t in w.calls]
    assert abs(starts[0] - starts[1]) < 0.1


def test_the_judge_reads_the_names_the_recent_talk_and_the_words():
    w = _writer()
    _run(_LIGHT, w, agent_names=("パジュ", "パチュ"), recent_ctx="- 18:42 パパ：しりとりしよう")
    judge_call = next(
        c for c in w.complete.await_args_list if "判定：" in (c.kwargs.get("system") or "")
    )
    prompt, system = judge_call.args[0], judge_call.kwargs["system"]
    assert "パジュ、パチュ" in system
    assert "しりとりしよう" in prompt and "きょうも退屈してたんだ" in prompt


def test_other_light_words_are_not_judged():
    """**反証の側**：状態を伝える・道具の結果が返ったとき・内からの求めには入れない（本人の決定イ・ア）。"""
    w = _writer(judge="判定：黙る")
    d = _run({"meaning": _a("accepted_check"), "action_accepted_check": _a("state_light")}, w)
    assert d.text == "うん、いいよ。"
    assert all(k == "write" for k, _t in w.calls)
    w2 = _writer(judge="判定：黙る")
    d2 = _run({"action": _a("reply_light")}, w2, returned=(("search_deferred", False, "晴れ"),))
    assert all(k == "write" for k, _t in w2.calls)
    assert d2.text == "うん、いいよ。"


def test_typed_words_are_not_judged():
    """キーボードで打った言葉には、テレビも崩れた書き起こしも来ないので判定しない。"""
    w = _writer(judge="判定：黙る")
    d = _run(_LIGHT, w, typed=True)
    assert d.text == "うん、いいよ。" and all(k == "write" for k, _t in w.calls)


def test_a_denial_reply_is_not_judged():
    """名乗り・否定・黙る依頼の受け答えは、頼まれたことへの返事なので判定しない（欄だけ入れて黙ることがない）。"""
    w = _writer(judge="判定：黙る")
    answers = {"meaning": _a("deny"), "action_deny": _a("other")}
    d = _run(answers, w)
    assert d.text == "うん、いいよ。" and all(k == "write" for k, _t in w.calls)


def test_a_reason_with_asking_back_still_reads_the_verdict():
    """答えに「聞き返す」が混じっても、「判定：」の行だけを読む。"""
    d = _run(_LIGHT, _writer(judge="判定：黙る（迷って聞き返すより黙る）"))
    assert d.text == ""
    d2 = _run(_LIGHT, _writer(judge="よく分からない"))
    assert d2.text == "うん、いいよ。"  # **反証の側**：読めない答えは返す
