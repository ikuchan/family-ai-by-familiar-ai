"""掛かるまでを壊した 3 つの穴（実機 2026-09-18 20:46〜20:47・quiet）。

- **情-n**：道具の帰りの反復で調停が `silence_minutes` を書き、確認文「その間は黙って待機します」を
  人の「黙って」の依頼として 60 分の沈黙が掛かった（「いいよ」が飲まれた）。帰りの反復では
  `silence_minutes`／`lift_silence` を機械で落とす。
- **情-l-ろ**：明示の沈黙が退室（誰も居ないを 60 秒）で解けても記録が `until` まで残り、タイマーの沈黙
  （`hush_for_timer`）が負けた／人が映れば復活する。解けたら記録も消す。タイマー由来は消さない。
- **出-aa**：「再開」を調停が light で受け流し「再開しますね」と言うだけだった。`[タイマー]` に動いて
  いる／一時停止中があるとき、操作の言葉（`is_control_word`）の light は full へ倒す（主LLM が道具で決める）。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

from familiar_agent.loop import arbiter
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.silence_state import SilenceRequest

from tests.test_event_loop import _agent


def _backend(reply: str):
    async def complete(prompt, max_tokens, **kw):
        return reply

    b = MagicMock()
    b.complete = complete
    return b


# ── 情-n ─────────────────────────────────────────────────────────────────


def test_a_tool_return_never_carries_a_silence_request():
    b = _backend(
        '{"branch":"light","text":"3 分ね、いい？","silence_minutes":-1,"lift_silence":true}'
    )
    d = asyncio.run(
        arbiter.arbitrate(b, utterance="パジュ、3分測って", workspace_ctx="", tool_return=True)
    )
    assert d.branch == "light" and d.silence_minutes == 0 and d.lift_silence is False


def test_a_first_iteration_still_carries_it():
    b = _backend('{"branch":"light","text":"うん、黙るね","silence_minutes":-1}')
    d = asyncio.run(arbiter.arbitrate(b, utterance="パジュ、静かにして", workspace_ctx=""))
    assert d.silence_minutes == -1


# ── 情-l-ろ ───────────────────────────────────────────────────────────────


def _ip(req, *, nobody_since):
    a = _agent(stream_returns=[])
    a.nobody_since = MagicMock(return_value=nobody_since)
    ip = InformationProcessing(a)
    ip._load_silence = lambda: req
    ip.push_device = MagicMock()
    return ip


def test_an_explicit_silence_lifted_by_absence_is_cleared(monkeypatch):
    cleared = []
    monkeypatch.setattr("familiar_agent.silence_state.clear_silence", lambda: cleared.append(1))
    ip = _ip(SilenceRequest(person="パパ", until=time.time() + 3000), nobody_since=time.time() - 61)
    ip.check_silence_lifted()  # 器が空でも消す
    assert cleared == [1]


def test_a_timer_silence_is_not_cleared_by_absence(monkeypatch):
    cleared = []
    monkeypatch.setattr("familiar_agent.silence_state.clear_silence", lambda: cleared.append(1))
    ip = _ip(
        SilenceRequest(person="パパ", until=time.time() + 120, reason="timer:1"),
        nobody_since=time.time() - 61,
    )
    ip.check_silence_lifted()
    assert cleared == []
    ip2 = _ip(SilenceRequest(person="パパ", until=time.time() + 3000), nobody_since=None)
    ip2.check_silence_lifted()  # 人が見えているなら解けない → 消さない
    assert cleared == []


# ── 出-aa ─────────────────────────────────────────────────────────────────


def test_a_control_word_answered_lightly_falls_to_full_while_a_timer_is_running():
    b = _backend('{"branch":"light","text":"はい、再開しますね。"}')
    d = asyncio.run(arbiter.arbitrate(b, utterance="再開", workspace_ctx="", timer_active=True))
    assert d.branch == "full"
    d = asyncio.run(arbiter.arbitrate(b, utterance="再開", workspace_ctx=""))
    assert d.branch == "light"  # タイマーが無ければ会話（「再開発の話？」）
    d = asyncio.run(
        arbiter.arbitrate(b, utterance="こんにちは", workspace_ctx="", timer_active=True)
    )
    assert d.branch == "light"


def test_the_control_guard_is_not_applied_on_a_tool_return():
    b = _backend('{"branch":"light","text":"止めておくね。"}')
    d = asyncio.run(
        arbiter.arbitrate(
            b, utterance="一時停止", workspace_ctx="", timer_active=True, tool_return=True
        )
    )
    assert d.branch == "light"


def test_decide_passes_whether_a_timer_is_running():
    seen = {}

    async def fake_arbitrate(backend, **kw):
        seen.update(kw)
        return arbiter.Decision(branch="light", text="x")

    a = _agent(stream_returns=[])
    a._timer_tool = MagicMock()
    a._timer_tool.frame = MagicMock(return_value="[タイマー]\n- id=1 パスタ 一時停止中")
    ip = InformationProcessing(a)
    ip._req.trigger_kind = "発話"
    import familiar_agent.loop.event_loop as el

    orig = el.arbitrate
    el.arbitrate = fake_arbitrate
    try:
        asyncio.run(
            ip._decide(utterance="再開", workspace_ctx="", present_ctx="", capped=False, round_=1)
        )
    finally:
        el.arbitrate = orig
    assert seen.get("timer_active") is True
