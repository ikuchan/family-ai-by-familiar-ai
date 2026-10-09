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

import pytest

import asyncio
import time
from unittest.mock import MagicMock

from familiar_agent.loop import arbiter
from tests._arbiter_fakes import decide, jev_says, writer_says
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.silence_state import SilenceRequest

from tests.test_event_loop import _agent


def _light(text: str, **jev):
    """Jev は light と決め（ほかの答えは `jev`）、軽量LLM が `text` を書く。"""
    return dict(jev=jev_says("light", **jev), writer=writer_says({"text": text}))


# ── 情-n ─────────────────────────────────────────────────────────────────


def test_a_tool_return_never_carries_a_silence_request():
    # Jev の答えに黙る依頼と解く依頼が混じっていても、道具の帰りでは読まない。
    said = _light("3 分ね、いい？", quiet="default", lifts_quiet=True)
    d = asyncio.run(decide(**said, utterance="パジュ、3分測って", tool_return=True))
    assert d.branch == "light" and d.silence_minutes == 0 and d.lift_silence is False


@pytest.mark.xfail(
    strict=True,
    reason="段 4-4c で戻す：発話の新しい問い（出-ay 段 4-4b）は、黙る依頼・解く・名乗り・否定・時期をまだ聞かない（本人：一時的に効かないのはかまわない）",
)
def test_a_first_iteration_still_carries_it():
    said = _light("うん、黙るね", quiet="default")
    d = asyncio.run(decide(**said, utterance="パジュ、静かにして"))
    assert d.silence_minutes == -1


# ── 情-l-ろ ───────────────────────────────────────────────────────────────


def _ip(req):
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._load_silence = lambda: req
    ip.push_device = MagicMock()
    return ip


def test_a_live_silence_is_not_cleared(monkeypatch):
    """期限までは消さない。誰も居なくなっても解けない（出-as §2.5・2026-09-26。以前は 60 秒で解けた）。"""
    cleared = []
    monkeypatch.setattr("familiar_agent.silence_state.clear_silence", lambda: cleared.append(1))
    _ip(
        SilenceRequest(person="パパ", until=time.time() + 120, reason="timer:1")
    ).check_silence_lifted()
    _ip(SilenceRequest(person="パパ", until=time.time() + 3000)).check_silence_lifted()
    assert cleared == []


# ── 出-aa ─────────────────────────────────────────────────────────────────


def test_a_control_word_answered_lightly_falls_to_full_while_a_timer_is_running():
    def branch(utterance: str, **kw) -> str:
        return asyncio.run(
            decide(**_light("はい、再開しますね。"), utterance=utterance, **kw)
        ).branch

    assert branch("再開", timer_active=True) == "full"
    assert branch("再開") == "light"  # タイマーが無ければ会話（「再開発の話？」）
    assert branch("こんにちは", timer_active=True) == "light"


def test_the_control_guard_is_not_applied_on_a_tool_return():
    d = asyncio.run(
        decide(
            **_light("止めておくね。"), utterance="一時停止", timer_active=True, tool_return=True
        )
    )
    assert d.branch == "light"


def test_decide_passes_whether_a_timer_is_running():
    seen = {}

    async def fake_decide(self, inp):
        seen["inp"] = inp
        return arbiter.Decision(branch="light", text="x")

    a = _agent(stream_returns=[])
    a._timer_tool = MagicMock()
    a._timer_tool.frame = MagicMock(return_value="[タイマー]\n- id=1 パスタ 一時停止中")
    ip = InformationProcessing(a)
    ip._req.trigger_kind = "発話"
    from unittest.mock import patch

    with patch("familiar_agent.loop.arbiter.Arbiter.decide", new=fake_decide):
        asyncio.run(
            ip._decide(utterance="再開", workspace_ctx="", present_ctx="", capped=False, round_=1)
        )
    assert seen["inp"].timer_active is True
