"""「黙っていて」と頼まれたら、その人が居るあいだ黙る。

- **気づくのは軽量LLM**（調停）。言い方は「黙って」「うるさい」「あとにして」「いま集中
  したい」と無数にあり、文字列の一覧は必ず漏れる。ツールを渡してフルLLM に呼ばせる形も
  取らない（調停は毎反復の頭で必ず通るので、そこで判断すれば発話の出口すべてに効く）。
- **止めるのは発話すべて**。自発だけでなく、話しかけられても話さない。
- **解けるのは退室と時間**。時間は Config（既定60分）。
- **黙っているあいだの言葉は捨てず `pending_speech` へ溜める**。「聞きたくない」ではなく
  「いまは」なので、解けたときに配る。
"""

from __future__ import annotations

import time

from familiar_agent.silence_state import (
    SilenceRequest,
    is_silenced,
)


def test_silenced_within_the_deadline():
    req = SilenceRequest(person="パパ", until=time.time() + 600)
    assert is_silenced(req, now=time.time()) is True


def test_an_explicit_request_lifts_when_the_room_has_been_empty_for_a_minute():
    """退室で解ける——ただし在席表（誰か・失効で消える）でなく、**居るかの層**で見る（情-l・2026-09-18）。"""
    req = SilenceRequest(person="パパ", until=time.time() + 600)
    now = time.time()
    assert is_silenced(req, now=now, nobody_since=now - 30) is True  # 30 秒ではまだ
    assert is_silenced(req, now=now, nobody_since=now - 70) is False  # 60 秒誰も見ていない → 解ける
    assert is_silenced(req, now=now, nobody_since=None) is True  # センサが人を見ている・または無い


def test_a_timer_silence_does_not_lift_on_absence():
    """タイマー由来（`reason=timer:`）は鳴る・止めるまで（在席表の失効で解けた実機 2026-09-18 14:51）。"""
    req = SilenceRequest(person="パパ", until=time.time() + 180, reason="timer:12")
    now = time.time()
    assert is_silenced(req, now=now, nobody_since=now - 600) is True
    assert is_silenced(req, now=now + 200, nobody_since=now - 600) is False  # 期限では解ける


def test_not_silenced_after_the_time_runs_out():
    req = SilenceRequest(person="パパ", until=time.time() - 1)
    assert is_silenced(req, now=time.time()) is False


def test_no_request_means_no_silence():
    assert is_silenced(None, now=time.time()) is False


def test_arbiter_can_flag_a_silence_request():
    # 気づくのは軽量LLM。言い方は無数にあるので、文字列の一覧を持たない。
    from familiar_agent.loop.arbiter import ARBITER_PROMPT, Decision, arbitrate
    import asyncio
    from unittest.mock import AsyncMock

    assert "silence_minutes" in ARBITER_PROMPT
    b = AsyncMock()
    b.complete = AsyncMock(return_value='{"branch":"light","text":"わかった","silence_minutes":-1}')
    d: Decision = asyncio.run(arbitrate(b, utterance="ちょっと静かにして", workspace_ctx=""))
    assert d.silence_minutes == -1  # 頼まれたが長さの指定なし


# ── 沈黙は出口でなく入口で見る（情-h・2026-09-16）───────────────────────────
#
# 以前は配信ゲート（出口）が「黙っているよう頼まれている」を返し、発話ごとに求めが立って
# 調停・主LLM が回ってから止まっていた。他人への返事を通す例外（案イ・2026-09-13）もここに
# あった。いまは入口（`_swallow_if_unheard`・`test_silence_hold`）で、誰の声でも・機器でも・
# 情動でも求めを立てない。案イは撤回した。出口は沈黙を知らない。


def _ip_with(*, speaker: str, others: tuple[str, ...] = ()):
    from unittest.mock import MagicMock

    from familiar_agent.loop.event_loop import InformationProcessing

    a = MagicMock()
    rows = [{"name": speaker, "is_speaker": True, "confidence": 1.0}]
    rows += [{"name": n, "is_speaker": False, "confidence": 1.0} for n in others]
    a._pmm.presence_status = MagicMock(return_value=rows)
    a._social_presence_permission = MagicMock(return_value=1.0)
    a._in_quiet_hours = MagicMock(return_value=False)
    return InformationProcessing(a)


def test_the_exit_gate_no_longer_looks_at_the_silence_request(monkeypatch):
    """黙っている依頼が生きていても出口は止めない（止めるのは入口）。"""
    import familiar_agent.silence_state as ss

    ip = _ip_with(speaker="パパ", others=("たいきくん",))
    monkeypatch.setattr(
        ss, "load_silence", lambda: SilenceRequest(person="パパ", until=time.time() + 600)
    )
    for kind in ("発話", "情動", "機器"):
        ip._req.trigger_kind = kind
        assert ip._delivery_block_reason() == "", kind


def test_default_duration_is_an_hour():
    """長さを言われなかったときの既定。

    一度 15 分へ縮めたが（「言わずに頼んだだけで 1 時間黙るのは長い」）、2026-09-13 に
    **60 分**へ戻した（課題5 G 章〔確定〕・情-d）。「黙って」と頼まれたら 1 時間は黙る。
    上限（`silence_max_minutes`）も 60 分。
    """
    from familiar_agent.config import AgentConfig

    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {}, clear=True):
        assert AgentConfig().silence_minutes == 60
        assert AgentConfig().silence_max_minutes == 60


# ── 「もう話していいよ」と解かれたら消す（2026-09-16 実機）────────────────────
#
# 11:47 の「待てぃ」を調停が「黙っていて」と読み、60 分の依頼になった。解ける条件は退室か
# 期限だけで `clear_silence()` はどこからも呼ばれておらず、解く口が無かった。話しかけられた
# だけでは解かない（黙っていてほしい人が用事だけ言うことはある）。頼んだ本人が話していいと
# 言ったとき（調停が `lift_silence` を立てる）に消す。


def _ip_with_speaker(speaker: str):
    from unittest.mock import MagicMock

    from familiar_agent.loop.event_loop import InformationProcessing

    a = MagicMock()
    a._pmm.presence_status = MagicMock(
        return_value=[{"name": speaker, "is_speaker": True, "confidence": 1.0}]
    )
    return InformationProcessing(a)


def _patched(monkeypatch, req):
    import familiar_agent.silence_state as ss

    cleared: list[bool] = []
    monkeypatch.setattr(ss, "load_silence", lambda: req)
    monkeypatch.setattr(ss, "clear_silence", lambda: cleared.append(True))
    return cleared


def test_the_asker_saying_she_may_talk_lifts_the_request(monkeypatch):
    ip = _ip_with_speaker("パパ")
    cleared = _patched(monkeypatch, SilenceRequest(person="パパ", until=time.time() + 3600))
    ip._release_silence()
    assert cleared == [True]


def test_someone_else_cannot_lift_it(monkeypatch):
    ip = _ip_with_speaker("たいきくん")
    cleared = _patched(monkeypatch, SilenceRequest(person="パパ", until=time.time() + 3600))
    ip._release_silence()
    assert cleared == []


def test_nothing_to_lift_when_no_request(monkeypatch):
    ip = _ip_with_speaker("パパ")
    cleared = _patched(monkeypatch, None)
    ip._release_silence()
    assert cleared == []


def test_merely_speaking_to_her_does_not_lift_it(monkeypatch):
    """話しかけただけでは解かない（`push_utterance` の入口で消さない）。"""
    import asyncio
    from unittest.mock import AsyncMock

    ip = _ip_with_speaker("パパ")
    cleared = _patched(monkeypatch, SilenceRequest(person="パパ", until=time.time() + 3600))
    ip._ensure_driver = lambda: None
    ip._abort_lookups = AsyncMock()

    async def run():
        task = asyncio.create_task(ip.push_utterance("ねえ、明日の予定は？"))
        trig = await ip._triggers.get()
        trig.future.set_result("")
        await task

    asyncio.run(run())
    assert cleared == []


def test_the_arbiter_can_flag_a_release():
    from familiar_agent.loop.arbiter import ARBITER_PROMPT, arbitrate
    import asyncio
    from unittest.mock import AsyncMock

    assert "lift_silence" in ARBITER_PROMPT
    b = AsyncMock()
    b.complete = AsyncMock(return_value='{"branch":"light","text":"はい","lift_silence":true}')
    d = asyncio.run(arbitrate(b, utterance="もう話していいよ", workspace_ctx=""))
    assert d.lift_silence is True and d.silence_minutes == 0
    b.complete = AsyncMock(return_value='{"branch":"light","text":"はい"}')
    assert asyncio.run(arbitrate(b, utterance="おはよう", workspace_ctx="")).lift_silence is False


def test_waiting_is_not_a_request_for_silence():
    from familiar_agent.loop.arbiter import ARBITER_PROMPT

    assert "待って" in ARBITER_PROMPT and "沈黙の依頼ではない" in ARBITER_PROMPT


def test_the_decision_is_applied_through_one_door(monkeypatch):
    """掛けるのも解くのも `_apply_silence` の 1 口（反復本体を厚くしない）。"""
    from familiar_agent.loop.arbiter import Decision

    ip = _ip_with_speaker("パパ")
    calls: list[str] = []
    ip._accept_silence = lambda m: calls.append(f"掛ける{m}")
    ip._release_silence = lambda: calls.append("解く")
    ip._apply_silence(Decision(branch="light", silence_minutes=-1))
    ip._apply_silence(Decision(branch="light", lift_silence=True))
    ip._apply_silence(Decision(branch="light"))
    assert calls == ["掛ける-1", "解く"]


def test_the_loop_reads_nobody_since_from_the_agent_and_a_mock_is_not_a_clock():
    from unittest.mock import MagicMock

    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing.__new__(InformationProcessing)
    ip._agent = MagicMock()
    ip._agent.nobody_since = MagicMock(return_value=1234.5)
    assert ip._nobody_since() == 1234.5
    ip._agent.nobody_since = MagicMock(return_value=MagicMock())  # 読めない → 解けない側
    assert ip._nobody_since() is None
