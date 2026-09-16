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


def test_silenced_while_the_asker_is_present():
    req = SilenceRequest(person="パパ", until=time.time() + 600)
    assert is_silenced(req, present={"パパ"}, now=time.time()) is True


def test_not_silenced_when_the_asker_has_left():
    # 退室で解ける。
    req = SilenceRequest(person="パパ", until=time.time() + 600)
    assert is_silenced(req, present={"たいきくん"}, now=time.time()) is False


def test_not_silenced_after_the_time_runs_out():
    req = SilenceRequest(person="パパ", until=time.time() - 1)
    assert is_silenced(req, present={"パパ"}, now=time.time()) is False


def test_no_request_means_no_silence():
    assert is_silenced(None, present={"パパ"}, now=time.time()) is False


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


def test_silence_blocks_speech_even_when_spoken_to():
    # 止めるのは発話すべて。自発だけでなく、話しかけられても話さない。
    import time as _time
    from unittest.mock import MagicMock

    from familiar_agent.loop.event_loop import InformationProcessing

    a = MagicMock()
    a._pmm.presence_status = MagicMock(
        return_value=[{"name": "パパ", "is_speaker": True, "confidence": 1.0}]
    )
    a._social_presence_permission = MagicMock(return_value=1.0)  # 相手は居る
    a._in_quiet_hours = MagicMock(return_value=False)  # 静穏時間でもない
    ip = InformationProcessing(a)
    req = SilenceRequest(person="パパ", until=_time.time() + 600)
    import familiar_agent.silence_state as ss

    original, ss.load_silence = ss.load_silence, lambda: req
    try:
        assert ip._delivery_block_reason() == "黙っているよう頼まれている"
    finally:
        ss.load_silence = original


def _ip_silenced_by(asker: str, *, speaker: str, others: tuple[str, ...] = ()):
    """`asker` に黙れと頼まれ、いま `speaker` が話している装置。他の在席者は `others`。"""
    import time as _time
    from unittest.mock import MagicMock

    from familiar_agent.loop.event_loop import InformationProcessing

    a = MagicMock()
    rows = [{"name": speaker, "is_speaker": True, "confidence": 1.0}]
    rows += [{"name": n, "is_speaker": False, "confidence": 1.0} for n in others]
    a._pmm.presence_status = MagicMock(return_value=rows)
    a._social_presence_permission = MagicMock(return_value=1.0)
    a._in_quiet_hours = MagicMock(return_value=False)
    ip = InformationProcessing(a)
    req = SilenceRequest(person=asker, until=_time.time() + 600)
    return ip, req


def _gate_with(ip, req) -> str:
    import familiar_agent.silence_state as ss

    original, ss.load_silence = ss.load_silence, lambda: req
    try:
        return ip._delivery_block_reason()
    finally:
        ss.load_silence = original


def test_a_reply_to_someone_else_passes_while_the_asker_is_still_there():
    """頼んだ本人以外が話しかけたら、その返事は通す（案イ・2026-09-13）。

    実機で、たいきの「だまってて」のあとパパが「今度の火曜日の天気は？」と聞いても、
    たいきが居る限り答えが `pending_speech` に溜まった。たいきの依頼は「ぼくの邪魔を
    しないで」であって、パパの質問まで止めるものではない。依頼は消さない——たいきへの
    返事と自発の発話は止めたまま、退室か時間で解ける。
    """
    ip, req = _ip_silenced_by("たいきくん", speaker="パパ", others=("たいきくん",))
    ip._req.trigger_kind = "発話"
    assert _gate_with(ip, req) == ""


def test_the_asker_is_still_answered_with_silence():
    ip, req = _ip_silenced_by("たいきくん", speaker="たいきくん", others=("パパ",))
    ip._req.trigger_kind = "発話"
    assert _gate_with(ip, req) == "黙っているよう頼まれている"


def test_spontaneous_speech_stays_blocked_even_if_someone_else_spoke_last():
    # 自発（情動が起点）は、最後に話した人が誰であっても止めたまま。
    ip, req = _ip_silenced_by("たいきくん", speaker="パパ", others=("たいきくん",))
    ip._req.trigger_kind = "情動"
    assert _gate_with(ip, req) == "黙っているよう頼まれている"


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


# ── 頼んだ本人が話しかけてきたら解く（2026-09-16 実機）────────────────────────
#
# 11:47 の「待てぃ」を調停が「黙っていて」と読み、60 分の依頼になった。解ける条件は退室か
# 期限だけで `clear_silence()` はどこからも呼ばれておらず、本人が話しかけ直しても 12:47 まで
# 返事が全部保留になった。黙っていてほしい人は話しかけない——話しかけてきたなら、その時点で
# 依頼は終わっている。


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


def test_the_asker_speaking_again_lifts_the_request(monkeypatch):
    ip = _ip_with_speaker("パパ")
    cleared = _patched(monkeypatch, SilenceRequest(person="パパ", until=time.time() + 3600))
    ip._lift_silence_if_asker_speaks()
    assert cleared == [True]


def test_someone_else_speaking_leaves_the_request(monkeypatch):
    ip = _ip_with_speaker("たいきくん")
    cleared = _patched(monkeypatch, SilenceRequest(person="パパ", until=time.time() + 3600))
    ip._lift_silence_if_asker_speaks()
    assert cleared == []


def test_nothing_to_lift_when_no_request(monkeypatch):
    ip = _ip_with_speaker("パパ")
    cleared = _patched(monkeypatch, None)
    ip._lift_silence_if_asker_speaks()
    assert cleared == []


def test_speaking_to_her_lifts_it_before_the_turn_runs(monkeypatch):
    """`push_utterance` の入口で解く。同じ発話で改めて頼まれれば、その反復が掛け直す。"""
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
    assert cleared == [True]


def test_waiting_is_not_a_request_for_silence():
    from familiar_agent.loop.arbiter import ARBITER_PROMPT

    assert "待って" in ARBITER_PROMPT and "沈黙の依頼ではない" in ARBITER_PROMPT
