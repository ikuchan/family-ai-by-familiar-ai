"""名乗りで話者を付ける（知-w・2026-09-19）。在席があるときだけ。

「パジュ、こんにちはパパだよ」がカメラに映らない位置からの声で入口に飲まれた（実機 11:32）。在席（居るか）は
カメラだけで決める（マイクは証拠にしない・変えない）。**カメラが人を見ているとき**は、声の名乗り（「パパだよ」
「僕はたいき」）から話者を付ける（`/speaker` と同じ効き：`set_active`・`_speaker_set_at`・PMM 同期）。
名乗りの読みは調停（`Decision.speaker_claim`）、検めは機械（在席あり・家族の名前か呼び名に一致）。
道具の帰りの反復では読まない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core.speaker_claim import resolve_claim
from familiar_agent.loop import arbiter
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent

FAMILY = "## ゆうすけ\n- **名前**：ゆうすけ\n- **呼び方**：パパ\n- **英字**：Yusuke Ikunaga\n\n## たいき\n- **名前**：たいき\n- **呼び方**：たいき\n- **英字**：Taiki Ikunaga\n"


def test_a_claim_resolves_to_a_family_name_or_nothing():
    assert resolve_claim("パパ", FAMILY) == "パパ"
    assert (
        resolve_claim("ゆうすけ", FAMILY) == "パパ"
    )  # 名前で名乗っても呼び方に寄せる（/speaker パパ と同じ表記）
    assert resolve_claim("たいき", FAMILY) == "たいき"
    assert resolve_claim("太郎", FAMILY) is None
    assert resolve_claim("", FAMILY) is None


def _backend(reply: str):
    async def complete(prompt, max_tokens, **kw):
        return reply

    b = MagicMock()
    b.complete = complete
    return b


def test_the_arbiter_carries_the_claim_and_drops_it_on_a_tool_return():
    b = _backend('{"branch":"light","text":"パパ、おかえり","speaker_claim":"パパ"}')
    d = asyncio.run(arbiter.arbitrate(b, utterance="パパだよ", workspace_ctx=""))
    assert d.speaker_claim == "パパ"
    d = asyncio.run(arbiter.arbitrate(b, utterance="パパだよ", workspace_ctx="", tool_return=True))
    assert d.speaker_claim == ""
    assert "speaker_claim" in arbiter.ARBITER_PROMPT and "名乗" in arbiter.ARBITER_PROMPT


def _ip(*, present: float):
    a = _agent(stream_returns=[])
    a._family_md = FAMILY
    a._social_presence_permission = MagicMock(return_value=present)
    a._persons.active_name = "推定話者"
    a._sync_pmm_speaker = AsyncMock()
    ip = InformationProcessing(a)
    return ip, a


def test_a_claim_with_presence_sets_the_speaker_like_the_command():
    ip, a = _ip(present=1.0)
    d = arbiter.Decision(branch="light", text="x", speaker_claim="ゆうすけ")
    asyncio.run(ip._apply_speaker_claim(d))
    a._persons.set_active.assert_called_once_with("パパ")
    assert isinstance(a._speaker_set_at, float)
    a._sync_pmm_speaker.assert_awaited_once_with("パパ")


def test_no_presence_or_unknown_name_sets_nothing():
    ip, a = _ip(present=0.0)
    asyncio.run(
        ip._apply_speaker_claim(arbiter.Decision(branch="light", text="x", speaker_claim="パパ"))
    )
    a._persons.set_active.assert_not_called()
    ip, a = _ip(present=1.0)
    asyncio.run(
        ip._apply_speaker_claim(arbiter.Decision(branch="light", text="x", speaker_claim="太郎"))
    )
    a._persons.set_active.assert_not_called()
    a._persons.active_name = "パパ"
    asyncio.run(
        ip._apply_speaker_claim(arbiter.Decision(branch="light", text="x", speaker_claim="パパ"))
    )
    a._persons.set_active.assert_not_called()  # 既に同じ話者なら何もしない


# ── 名乗りの預かり（知-w-ろ・2026-09-19）────────────────────────────────────
#
# 映らない位置の「パパだよ出入口を見て」→ 見回りが首を向けた帰り（13:27:22）に名乗りを読んだが、センサが人を
# 見たのは 6 秒後（13:27:28）→「在席が無いので話者にしない」→ 入室で「おかえりなさい、パパ」（話者は付かず）。
# 在席が無くて使えなかった名乗りを **30 秒**預かり、センサが人を見た最初の求めで生きていれば話者に付ける。


def test_a_claim_without_presence_is_kept_for_thirty_seconds(monkeypatch):
    ip, a = _ip(present=0.0)
    clock = {"t": 1000.0}
    monkeypatch.setattr("familiar_agent.loop.event_loop.time.time", lambda: clock["t"])
    asyncio.run(
        ip._apply_speaker_claim(arbiter.Decision(branch="light", text="x", speaker_claim="パパ"))
    )
    a._persons.set_active.assert_not_called()
    assert ip._pending_claim == ("パパ", 1000.0)
    # 人が映った最初の求めで付く
    a._social_presence_permission = MagicMock(return_value=1.0)
    clock["t"] = 1006.0
    asyncio.run(ip._begin_request(kind="機器", text="[入室] 誰か が来た"))
    a._persons.set_active.assert_called_once_with("パパ")
    assert ip._pending_claim is None


def test_a_kept_claim_expires_and_an_unknown_name_is_not_kept(monkeypatch):
    ip, a = _ip(present=0.0)
    clock = {"t": 1000.0}
    monkeypatch.setattr("familiar_agent.loop.event_loop.time.time", lambda: clock["t"])
    asyncio.run(
        ip._apply_speaker_claim(arbiter.Decision(branch="light", text="x", speaker_claim="パパ"))
    )
    a._social_presence_permission = MagicMock(return_value=1.0)
    clock["t"] = 1031.0  # 30 秒を過ぎた
    asyncio.run(ip._begin_request(kind="機器", text="[入室] 誰か が来た"))
    a._persons.set_active.assert_not_called()
    assert ip._pending_claim is None
    ip, a = _ip(present=0.0)
    asyncio.run(
        ip._apply_speaker_claim(arbiter.Decision(branch="light", text="x", speaker_claim="太郎"))
    )
    assert ip._pending_claim is None  # 家族に無い名前は預からない
