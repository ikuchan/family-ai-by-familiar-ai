"""主LLM に窓の会話を渡し、キャッシュを効かせる（環-ar 段 D・2026-10-25・本人の決定）。

窓の会話の中なら、主LLM の system は「変わらない部分 → 窓の会話（見出しと 1 行ずつを別のまとまり）→ 毎回変わる部分」
の順に並ぶ。キャッシュの印は、変わらない部分と窓の会話の最後の行に付く。行が足されても前のまとまりは 1 字も変わらない
ので、前の呼び出しの分がそのまま当たる。W（`for_main`）からは [直近のやりとり] を外す。

窓の中では、言い直し・差し戻し・上限のときも道具の一覧を変えず（変えると道具から後ろのキャッシュが全部外れる）、
「この回は say だけ」を毎回変わる部分で伝え、say 以外の呼び出しは投げずに捨てる。窓の外では、いまどおり say だけを渡す。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from familiar_agent.backends.anthropic import AnthropicBackend
from familiar_agent.backends.types import ToolCall, TurnResult
from familiar_agent.core.context_parts import Stance, build_context
from familiar_agent.backends.types import system_text
from familiar_agent.loop.prompt import build_event_system_prompt
from familiar_agent.loop.event_loop import _caches_window as _real_caches_window
from tests.test_jev_gets_a_measured_state import _ws
from tests.test_window_talk_goes_to_jev import TALK, _past, _ws_across_the_window

HEAD = "[窓の会話（19:00:00 から・古い順・全部）]"
L1 = "- 19:00:00 パパ：しりとりしよう"
L2 = "- 19:00:02 わたし（パパへ）：りんご！"
L3 = "- 19:00:05 パパ：ごりら"


def _built(window: "tuple[str, ...]"):
    return build_event_system_prompt(
        self_understanding="[ME] ぼく",
        family_md="[FAMILY] パパ・ママ",
        present_ctx='(present :speaker "パパ")',
        pi_ctx="",
        workspace_ctx="[想起]昔の話",
        window=window,
    )


# ── system の並び ─────────────────────────────────────────────────


def test_the_window_lines_sit_between_the_stable_and_the_changing_parts():
    parts = _built((HEAD, L1, L2))
    assert parts[1:-1] == (HEAD, L1, L2)
    assert "[ME] ぼく" in parts[0] and "[想起]昔の話" in parts[-1]


def test_with_no_window_the_system_is_still_two_parts():
    """**反証の側**：窓の会話が無ければ、いまどおり (安定部, 可変部) の 2 つ。"""
    assert len(_built(())) == 2


def test_a_new_line_leaves_every_earlier_part_as_it_was():
    before, after = _built((HEAD, L1, L2)), _built((HEAD, L1, L2, L3))
    assert after[: len(before) - 1] == before[:-1]
    assert after[-2] == L3


def test_the_context_keeps_the_window_apart_from_the_stable_text():
    ctx = build_context(stance=Stance.PAJU, self_understanding="私", family="家", window=(HEAD, L1))
    assert ctx.window == (HEAD, L1)
    assert L1 not in ctx.stable and L1 not in ctx.variable


# ── Anthropic の印 ─────────────────────────────────────────────────


def test_the_cache_marks_go_on_the_stable_part_and_the_last_line():
    blocks = AnthropicBackend._build_system_param(("安定", HEAD, L1, L2, "可変"))
    assert [b["text"] for b in blocks] == ["安定", HEAD, L1, L2, "可変"]
    assert [i for i, b in enumerate(blocks) if "cache_control" in b] == [0, 3]


def test_without_a_window_the_marks_are_as_before():
    """**反証の側**：2 つなら、印は安定部にだけ。"""
    blocks = AnthropicBackend._build_system_param(("安定", "可変"))
    assert [i for i, b in enumerate(blocks) if "cache_control" in b] == [0]


# ── ほかの backend は 1 本の文 ─────────────────────────────────────────


def test_other_backends_join_the_lines_as_one_frame():
    assert system_text(("安定", HEAD, L1, "可変")) == f"安定\n\n---\n\n{HEAD}\n{L1}\n\n---\n\n可変"
    assert system_text(("安定", "可変")) == "安定\n\n---\n\n可変"  # いまどおり
    assert system_text("文") == "文"


def test_an_openai_style_backend_gets_the_window_as_one_frame():
    from familiar_agent.backends.openai_compat import OpenAICompatibleBackend

    flat = OpenAICompatibleBackend._flatten_messages(
        object.__new__(OpenAICompatibleBackend), ("安定", HEAD, L1, "可変"), []
    )
    assert f"{HEAD}\n{L1}" in flat[0]["content"]


# ── 主LLM の W ─────────────────────────────────────────────────


def test_the_main_view_drops_the_recent_frame_in_the_window():
    ws = _ws()
    ws.window_talk = TALK
    text = ws.for_main
    assert "直近のひとこと" not in text and "窓の会話" not in text


def test_with_no_window_the_main_view_keeps_the_recent_frame():
    """**反証の側**：窓の外なら、いまどおり。"""
    assert "直近のひとこと" in _ws().for_main


def test_the_main_view_leaves_out_only_what_the_window_talk_shows():
    text = _ws_across_the_window().for_main
    assert "窓の前のはなし" in _past(text)
    assert "窓の中のはなし" not in text
    assert text.count("[タイマー] 3 分たった") == 1


# ── ループ：system を組む・言い直しの道具 ─────────────────────────────────


@pytest.fixture(autouse=True)
def _anthropic_like(monkeypatch):
    """偽物の backend を、行ごとにキャッシュの印を付ける backend とみなす（窓の中の say だけの回を見るため）。"""
    from familiar_agent.loop import event_loop

    monkeypatch.setattr(event_loop, "_caches_window", lambda backend: True)


def _ip_in_window():
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._current_speaker_name = lambda: "パパ"
    ip._wake_window().open(time.monotonic())
    ip._req.window_session = ip._wake_window().session
    ip._talk_line("パパ", "しりとりしよう")
    return ip


def test_the_loop_puts_the_window_lines_into_the_main_system():
    ip = _ip_in_window()
    blocks = ip._window_talk().blocks()
    system = ip._build_system(present_ctx="", workspace_ctx="", iter_ctx="", window=blocks)
    assert tuple(system[1:-1]) == blocks


def _run(ip, monkeypatch, *, calls, capped=False, retried=False, **kw):
    from familiar_agent.loop import event_loop

    monkeypatch.setattr(event_loop, "main_llm_cache_on", lambda: True)
    backend = ip._agent.backend
    backend.stream_turn = AsyncMock(
        return_value=(TurnResult(stop_reason="tool_use", text="", tool_calls=calls), None)
    )
    system = ("安定", HEAD, L1, "可変")
    asyncio.run(
        ip._run_main_llm(
            index=1,
            messages=[],
            system=system,
            effort=None,
            capped=capped,
            memories=[],
            w_id_map={},
            verdict_view="",
            recent_frame="",
            max_tokens=100,
            retried=retried,
            **kw,
        )
    )
    sent = backend.stream_turn.await_args.kwargs
    decided = ip._triggers.get_nowait().decision
    return sent, decided


def _names(tools):
    return [t["name"] for t in tools]


@pytest.mark.parametrize("kw", [{"capped": True}, {"retried": True}, {"say_asked": True}])
def test_a_say_only_round_in_the_window_keeps_the_tool_list(monkeypatch, kw):
    ip = _ip_in_window()
    from familiar_agent.loop.event_loop import _FULL_ACTIONS

    full = _names(ip._tools(actions=_FULL_ACTIONS))
    say = ToolCall(id="s", name="say", input={"text": "ごりら"})
    other = ToolCall(id="r", name="recall", input={"query": "x"})
    sent, decided = _run(ip, monkeypatch, calls=[other, say], **kw)
    assert _names(sent["tools"]) == full  # 道具の一覧は変えない
    assert sent["system"][:-1] == ("安定", HEAD, L1)  # 前のまとまりは変えない
    assert "say だけ" in sent["system"][-1]  # 毎回変わる部分で伝える
    assert [tc.name for tc in decided.result.tool_calls] == ["say"]  # say 以外は捨てる


def test_a_say_only_round_outside_the_window_still_narrows_the_tools(monkeypatch):
    """**反証の側**：窓の外の言い直しは、いまどおり say だけを渡す。"""
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    say = ToolCall(id="s", name="say", input={"text": "ごりら"})
    sent, _decided = _run(ip, monkeypatch, calls=[say], retried=True)
    assert _names(sent["tools"]) == ["say"]
    assert "say だけ" not in system_text(sent["system"])


def test_a_full_round_in_the_window_has_no_say_only_note(monkeypatch):
    """**反証の側**：ふつうの回は、窓の中でも全部の道具を使える。"""
    ip = _ip_in_window()
    other = ToolCall(id="r", name="recall", input={"query": "x"})
    sent, decided = _run(ip, monkeypatch, calls=[other])
    assert "say だけ" not in sent["system"][-1]
    assert [tc.name for tc in decided.result.tool_calls] == ["recall"]


# ── 反復を通して ─────────────────────────────────────────────────


def _main_system(monkeypatch, *, window: bool):
    from familiar_agent.loop.arbiter import Arbiter, Decision
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests._arbiter_fakes import jev_says, writer_says
    from tests.test_event_loop import _agent, _turn
    from tests.test_iteration_characterization import _run_until

    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    a._jev = jev_says("full")
    a._utility_backend = writer_says({"text": ""})

    async def full(self, inp, on_decided=None):
        return Decision(branch="full")

    monkeypatch.setattr(Arbiter, "decide", full)
    from familiar_agent.loop import event_loop

    monkeypatch.setattr(event_loop, "main_llm_cache_on", lambda: True)
    ip = InformationProcessing(a)
    ip._current_speaker_name = lambda: "パパ"
    if window:
        ip._wake_window().open(time.monotonic())
        ip._wake_window().until += 3600  # 反復が回るあいだ開けておく
        ip._talk_line("パパ", "しりとりしよう")
    _run_until(
        ip,
        lambda: a.backend.stream_turn.await_count,
        push=lambda: ip.push_affect("CALM", "落ち着いている"),
    )
    return a.backend.stream_turn.await_args.kwargs["system"]


def test_the_iteration_gives_the_window_lines_to_the_main_llm(monkeypatch):
    system = _main_system(monkeypatch, window=True)
    assert system[1].startswith("[窓の会話（") and system[2].endswith("パパ：しりとりしよう")


def test_outside_the_window_the_main_llm_gets_two_parts(monkeypatch):
    """**反証の側**：窓が開いていなければ、いまどおり 2 つ（キャッシュを切っていれば 1 本の文）。"""
    system = _main_system(monkeypatch, window=False)
    assert "窓の会話" not in system_text(system)


# --- 見直しの指摘（1 回目）---


def test_a_stride_mark_keeps_hits_when_many_lines_were_added():
    """印から前へ探すのは 20 まとまりまで。16 まとまりごとの区切りの最後にも印を付ける（印は全部で 4 つまで）。"""
    lines = tuple(f"- 行{i}" for i in range(17))
    blocks = AnthropicBackend._build_system_param(("安定", HEAD, *lines, "可変"))
    assert [i for i, b in enumerate(blocks) if "cache_control" in b] == [0, 16, 18]


def test_a_short_window_has_no_stride_mark():
    """**反証の側**：16 まとまり以下なら、区切りの印は付けない。"""
    lines = tuple(f"- 行{i}" for i in range(15))
    blocks = AnthropicBackend._build_system_param(("安定", HEAD, *lines, "可変"))
    assert [i for i, b in enumerate(blocks) if "cache_control" in b] == [0, 16]


def test_the_say_only_note_is_added_once():
    from familiar_agent.loop.event_loop import SAY_ONLY_NOTE, _with_say_only_note

    once = _with_say_only_note(("安定", HEAD, "可変"))
    assert _with_say_only_note(once) == once
    assert once[-1].count(SAY_ONLY_NOTE) == 1


def test_when_only_other_tools_were_called_it_asks_again_with_say_only(monkeypatch):
    """捨てたら何も言えないので、say だけを渡してもう一度呼ぶ（黙って終わらない）。"""
    from familiar_agent.loop import event_loop

    ip = _ip_in_window()
    monkeypatch.setattr(event_loop, "main_llm_cache_on", lambda: True)
    other = ToolCall(id="r", name="recall", input={"query": "x"})
    say = ToolCall(id="s", name="say", input={"text": "ごりら"})
    ip._agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (TurnResult(stop_reason="tool_use", text="", tool_calls=[other]), None),
            (TurnResult(stop_reason="tool_use", text="", tool_calls=[say]), None),
        ]
    )
    asyncio.run(
        ip._run_main_llm(
            index=1, messages=[], system=("安定", HEAD, L1, "可変"), effort=None, capped=False,
            memories=[], w_id_map={}, verdict_view="", recent_frame="", max_tokens=100, retried=True,
        )
    )  # fmt: skip
    calls = ip._agent.backend.stream_turn.await_args_list
    assert len(calls) == 2 and _names(calls[1].kwargs["tools"]) == ["say"]
    assert [tc.name for tc in ip._triggers.get_nowait().decision.result.tool_calls] == ["say"]


def test_with_the_cache_off_a_say_only_round_still_narrows_the_tools(monkeypatch):
    """**反証の側**：キャッシュを使わない日は守るものが無いので、窓の中でもいまどおり say だけ。"""
    from familiar_agent.loop import event_loop

    ip = _ip_in_window()
    say = ToolCall(id="s", name="say", input={"text": "ごりら"})
    ip._agent.backend.stream_turn = AsyncMock(
        return_value=(TurnResult(stop_reason="tool_use", text="", tool_calls=[say]), None)
    )
    monkeypatch.setattr(event_loop, "main_llm_cache_on", lambda: False)
    asyncio.run(
        ip._run_main_llm(
            index=1, messages=[], system=("安定", HEAD, L1, "可変"), effort=None, capped=False,
            memories=[], w_id_map={}, verdict_view="", recent_frame="", max_tokens=100, retried=True,
        )
    )  # fmt: skip
    sent = ip._agent.backend.stream_turn.await_args.kwargs
    assert _names(sent["tools"]) == ["say"] and "say だけ" not in sent["system"]
    assert f"{HEAD}\n{L1}" in sent["system"]  # 1 本の文でも、窓の会話は 1 つの枠


def test_a_pre_window_exchange_shown_as_a_memory_can_be_declared():
    """窓の中では、窓が開く前のやりとりが記憶の列に出る。その id でも記憶の申告を受ける（対応表に入る）。"""
    from datetime import datetime

    from familiar_agent.core.window_talk import WindowTalk

    ws = _ws_across_the_window()
    assert "q1" not in ws.verdict_map  # 窓の外では直近の枠に載るので申告の母数ではない
    talk = WindowTalk(session=1, opened_at=datetime(2026, 10, 25, 19, 0))
    talk.add(datetime(2026, 10, 25, 19, 0), "パパ", "窓の中のはなし", "q3")
    ws.set_window(talk)
    assert "q1" in ws.verdict_map and ws.id_map["q1"] == "q1"
    assert "q3" not in ws.verdict_map and "q2" not in ws.verdict_map
    assert ws.window_blocks == talk.blocks()


def test_a_recall_moved_to_another_time_keeps_the_window_talk(monkeypatch):
    from types import SimpleNamespace

    from familiar_agent.loop import workspace

    ip = _ip_in_window()
    moved = _ws()

    async def recall(*a, **k):
        return moved

    monkeypatch.setattr(workspace, "recall", recall)
    decision = SimpleNamespace(time_ref="2025-08-15T00:00:00", time_span_days=30)
    got = asyncio.run(ip._recall_at(decision, _ws(), cue="", viewpoint=None, weights=None))
    assert got is moved and "パパ：しりとりしよう" in (got.window_talk or "")
    assert got.window_blocks == ip._window_talk().blocks()


# --- 見直しの指摘（2 回目）---


def test_a_backend_that_joins_the_system_still_narrows_the_tools(monkeypatch):
    """**反証の側**：system を 1 本の文にする backend には守るキャッシュが無いので、窓の中でもいまどおり say だけ。"""
    from familiar_agent.loop import event_loop

    monkeypatch.setattr(event_loop, "_caches_window", lambda backend: False)
    say = ToolCall(id="s", name="say", input={"text": "ごりら"})
    sent, _decided = _run(_ip_in_window(), monkeypatch, calls=[say], retried=True)
    assert _names(sent["tools"]) == ["say"]


def test_only_the_anthropic_backend_marks_the_window_lines():
    # 差し替える前の本物（`_anthropic_like` はどのテストにも効く）。
    assert _real_caches_window(object.__new__(AnthropicBackend))
    assert not _real_caches_window(object())


def test_the_speech_check_sees_the_window_talk():
    """主LLM は窓の古い行も見て返すので、発話前の検査にも窓の会話を渡す（作り話と読まない）。"""
    ws = _ws()
    ws.window_talk = TALK
    assert TALK in ws.recent_for_checker() and "直近のひとこと" not in ws.recent_for_checker()
    assert "直近のひとこと" in _ws().recent_for_checker()  # 窓の外はいまどおり
