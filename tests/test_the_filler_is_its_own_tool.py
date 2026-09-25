"""つなぎは `say` と別の道具にし、主LLM の会話に置く（出-aq 段 2・2026-09-25）。

**主LLM は、つなぎを知らなかった。** W は調停の前に組まれ（`workspace_ctx` を文字列で
取る）、つなぎはその後に出る。主LLM へ渡るのは、つなぎの前に組んだ W のままだった
（実機 9/21 15:49 でも組み直しの跡は無い）。

つなぎをどう渡すかを、実機 15:49 の条件で測った（`根拠台帳` §48）。

| 渡し方 | 本題を `say` で言った | 挨拶を繰り返した |
|---|---|---|
| 渡さない（いまの実機） | 8/8 | 8/8 |
| **`say` を使った発言**として会話に置く | **1/8**（7 回は「もう答えた」で黙った） | 1/8 |
| **別の道具**を使った発言として置く | **10/10** | 8〜10 割 |

`say` は「返事」なので、つなぎを `say` で表すと**返事をもう出した**ことになり、主LLM は
「`(応答は既に送信済み)`」と書いて黙った。**つなぎと返事を同じ道具で表したので、
見分けがつかなかった。** 別の道具にすれば、主LLM は「`say` はまだ使っていない」と
道具の区別で知る。繰り返しは言い回しでは減らなかったので、機械（`drop_echo`）が落とす。

道具の定義は、測った中で落とした後の漏れが 0 だった形（返事の前半・`say` はその続き・
重複は機械が消す）を使う。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.types import ToolCall
from familiar_agent.loop.event_loop import _FULL_ACTIONS, InformationProcessing
from familiar_agent.tools.tts import TTSTool

from tests.test_event_loop import _agent


def _tts() -> TTSTool:
    t = TTSTool.__new__(TTSTool)
    t.engine, t.elevenlabs_model = "sbv2", "eleven_flash_v2_5"
    return t


# ── 道具の定義 ────────────────────────────────────────────────────────────


def test_the_voice_offers_say_first_and_filler_second():
    """`say` を先頭のまま保つ（既存の読み手が `[0]` を `say` と見ている）。"""
    names = [d["name"] for d in _tts().get_tool_definitions()]
    assert names == ["say", "filler"]


def test_the_filler_says_what_it_is():
    """**規則ではなく、つなぎが何であるか**を書く。言い回しの規則は効かなかった。"""
    d = {x["name"]: x for x in _tts().get_tool_definitions()}["filler"]
    desc = d["description"]
    assert "opening of your reply" in desc
    assert "`say` continues" in desc
    assert "removes them" in desc  # 重複は機械が消す
    assert set(d["input_schema"]["properties"]) == {"text"}


def test_the_voice_can_speak_a_filler_directly():
    """どこから呼ばれても鳴らせる（ループは横取りするが、道具としても成り立たせる）。"""
    t = _tts()
    t.say = AsyncMock(return_value="Said: ちょっと待ってね")
    out, _ = asyncio.run(t.call("filler", {"text": "ちょっと待ってね"}))
    t.say.assert_awaited_once()
    assert out.startswith("Said:")


def test_the_dif_hands_them_out_separately():
    """`say` と `filler` は別々に渡す。**上限の反復（`say` だけ）でつなぎを渡さない**ため。"""
    from familiar_agent.io.dif import DIF

    dif = DIF.__new__(DIF)
    dif._tts = _tts()
    assert [d["name"] for d in dif.speak_defs()] == ["say"]
    assert [d["name"] for d in dif.filler_defs()] == ["filler"]


def test_the_main_llm_is_offered_the_filler():
    assert "filler" in _FULL_ACTIONS
    assert "filler" in InformationProcessing._ACTIONS


# ── つなぎを主LLM の会話に置く ────────────────────────────────────────────


def _ip():
    ip = InformationProcessing(_agent(stream_returns=[]))
    return ip


def test_no_filler_adds_nothing():
    ip = _ip()
    assert ip._filler_turns(MagicMock()) == []


def test_the_fillers_become_a_call_and_its_result():
    ip = _ip()
    ip._req.said_fillers.extend(["こんにちは！ちょっと待ってくださいね。", "もう少しです。"])
    backend = MagicMock()
    backend.make_tool_call_message = lambda calls: {"call": [c.id for c in calls]}
    backend.make_tool_results = lambda calls, results: [
        {"result": [c.id for c in calls], "text": [t for t, _ in results]}
    ]
    turns = ip._filler_turns(backend)
    assert len(turns) == 2
    assert turns[0]["call"] == turns[1]["result"], "同じ id で対になる"
    assert turns[1]["text"] == [
        "Said: こんにちは！ちょっと待ってくださいね。",
        "Said: もう少しです。",
    ]


def test_the_result_is_a_report_not_an_instruction():
    """**道具の返りに指示を紛れ込ませない**（本人の指摘）。返りは「言った」の報告だけ。"""
    ip = _ip()
    ip._req.said_fillers.append("こんにちは。")
    backend = MagicMock()
    backend.make_tool_call_message = lambda calls: {}
    got: list = []
    backend.make_tool_results = lambda calls, results: got.extend(results) or []
    ip._filler_turns(backend)
    assert got == [("Said: こんにちは。", None)]


def test_a_backend_without_the_door_adds_nothing():
    """口の無い担い手でも壊れない（重複は機械の落としが受ける）。"""
    ip = _ip()
    ip._req.said_fillers.append("こんにちは。")
    backend = SimpleNamespace(make_tool_results=lambda *a: [])
    assert ip._filler_turns(backend) == []


def test_the_main_llm_call_carries_the_fillers():
    """主LLM を呼ぶ 1 箇所（`_run_main_llm`）で足す。通常と差し戻しの両方がそこを通る。"""
    import inspect

    src = inspect.getsource(InformationProcessing._run_main_llm)
    assert "_filler_turns" in src
    assert src.index("_filler_turns") < src.index("stream_turn")


# ── 主LLM が自分で `filler` を使ったとき ──────────────────────────────────


def _decision(*calls):
    return SimpleNamespace(
        result=SimpleNamespace(tool_calls=list(calls), text=""),
        capped=False,
        retried=False,
        mem=None,
        memories=[],
        w_id_map={},
        recent_frame="",
    )


def _run_act(calls):
    ip = _ip()
    ip._say_filler = AsyncMock()
    ip._start_lookup = MagicMock()
    spoken: list[str] = []

    async def _speak(text, **kw):
        spoken.append(text)
        return text, "発話"

    ip._speak = _speak
    ip._finish = AsyncMock()
    ip._coherence_violation = AsyncMock(return_value=None)
    ip._apply_seen_people = AsyncMock()
    gen = ip._request_generation
    asyncio.run(ip._act_on_decision(_decision(*calls), utterance="こんにちは", gen=gen))
    return ip, spoken


def test_filler_with_a_lookup_is_spoken_as_a_filler():
    """口 3。**格上げの細工が要らなくなる**——主LLM は最初から「つなぎ＋調べもの」を返せる。"""
    ip, spoken = _run_act(
        [
            ToolCall(id="a", name="filler", input={"text": "ちょっと調べますね。"}),
            ToolCall(id="b", name="search_deferred", input={"query": "明日の天気"}),
        ]
    )
    ip._say_filler.assert_awaited_once_with("ちょっと調べますね。")
    assert ip._start_lookup.called, "調べものは投げた"
    assert spoken == [], "つなぎを返事として出していない"


def test_filler_then_say_speaks_both_in_order():
    ip, spoken = _run_act(
        [
            ToolCall(id="a", name="filler", input={"text": "こんにちは！ちょっと待ってね。"}),
            ToolCall(id="b", name="say", input={"text": "タイマー止めておいたよ。"}),
        ]
    )
    ip._say_filler.assert_awaited_once_with("こんにちは！ちょっと待ってね。")
    assert spoken == ["タイマー止めておいたよ。"]


def test_a_lone_filler_is_not_swallowed():
    """**つなぎだけ返したら、それを返事として出す。** 黙らせない。"""
    ip, spoken = _run_act([ToolCall(id="a", name="filler", input={"text": "えっと…"})])
    assert spoken == ["えっと…"]
