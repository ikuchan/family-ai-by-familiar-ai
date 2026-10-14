"""求めを閉じるときに、最終版（聞かれて答えた塊）を書く（記-t 段 1・2026-10-14・本人の決定）。

いまは「何を聞かれ、何を答えたか」が 1 件の記録になっていなかった（起点＝人の言葉だけ、版＝聞かれたことと調べた結果で
答えは入らない、答え＝言ったことだけ）。調停が軽く返した求めには版が 1 つも無かった。

- 最終版は版の鎖の最後（親は起点・前の版は改訂で畳む・やりとりの「版」の項）。どの求めでも、どの閉じ方でも書く。
- 文の頭に印と時刻（「〔最終版 2026-10-14 10:13:40〜10:14:05〕」）。向きは `求め` のまま（`kind` は記-u で別に考える）。
- 上限は `completion_content_max`。超えたら背景で主LLM が要約し、閉じた時刻を渡して書く。
- 直近の枠に起点が載ったやりとりの最終版は、思い出したことの列に出さない。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

from familiar_agent.core import final_version as fv

JST = timezone(timedelta(hours=9))
_START = datetime(2026, 10, 14, 1, 13, 40, tzinfo=timezone.utc)  # 10:13:40 JST
_END = datetime(2026, 10, 14, 1, 14, 5, tzinfo=timezone.utc)  # 10:14:05 JST


def _compose(**kw):
    base = dict(
        trigger_kind="発話",
        asked="明日の天気は？",
        added=[],
        outcome="発話",
        answer="明日は晴れで 26℃ だよ",
        block_reason="",
        steps=[],
        start=_START,
        end=_END,
        tz=JST,
    )
    base.update(kw)
    return fv.compose(**base)


# ── 文の形（純関数）──────────────────────────────────────────────────────────────


def test_a_spoken_answer():
    assert _compose() == (
        "〔最終版 2026-10-14 10:13:40〜10:14:05〕「明日の天気は？」と聞かれ、「明日は晴れで 26℃ だよ」と答えた"
    )


def test_the_course_and_the_added_words():
    got = _compose(
        added=["東京の"], steps=["1番：search_deferred「東京 天気」の結果が届いた：晴れ"]
    )
    assert got.endswith(
        "「明日の天気は？」と聞かれ、「東京の」と言い足され、「明日は晴れで 26℃ だよ」と答えた。"
        "経緯：1番：search_deferred「東京 天気」の結果が届いた：晴れ"
    )


def test_device_and_emotion_heads():
    assert "〔[タイマー] パスタが鳴った〕が起きて、" in _compose(
        trigger_kind="機器", asked="[タイマー] パスタが鳴った"
    )
    assert "〔[内的な促し:bond] 話したい〕が湧いて、" in _compose(
        trigger_kind="情動", asked="[内的な促し:bond] 話したい"
    )


def test_each_way_of_closing():
    assert _compose(outcome="沈黙", answer="雨かも").endswith("声には出さず「雨かも」と思った")
    assert _compose(outcome="沈黙", answer="").endswith("何も言わなかった")
    assert _compose(outcome="独白", answer="雨だよ", block_reason="誰も居ない").endswith(
        "「雨だよ」と言おうとしたが、言えなかった（誰も居ない）"
    )
    assert _compose(outcome="打ち切り", answer="").endswith("打ち切った")


def test_the_mark_tells_a_final_version():
    assert fv.is_final(_compose())
    assert not fv.is_final("「明日の天気は？」と聞かれ、1番：…")  # ふつうの版
    assert fv.is_final("[そばに居た] " + _compose())  # 想起の頭が付いても見分ける


# ── 閉じるときに書く ──────────────────────────────────────────────────────────


def _ip(trigger_kind="発話", request_text="明日の天気は？", live_version="ver-1"):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a.config.completion_content_max = 8192
    ip = InformationProcessing(a)
    ip._req.request_id = "origin-1"
    ip._req.request_text = request_text
    ip._req.trigger_kind = trigger_kind
    ip._req.began_at = _START
    ip._req.live_version_id = live_version
    return a, ip


def _finals(a):
    return [c for c in a._memory.save_async_with_id.call_args_list if fv.is_final(c.args[0])]


def _close(ip, text, outcome):
    async def go():
        await ip._finish(text, [], outcome)
        await ip.settle_finals()
        await ip.close()

    asyncio.run(go())


def test_closing_writes_one_final_version_on_the_chain():
    a, ip = _ip()
    _close(ip, "明日は晴れだよ", "発話")
    finals = _finals(a)
    assert len(finals) == 1
    call = finals[0]
    assert call.kwargs["direction"] == "求め"
    assert call.kwargs["parent_id"] == "origin-1"
    assert "「明日は晴れだよ」と答えた" in call.args[0]
    final_id = a._memory.save_async_with_id.call_args_list.index(call) + 1
    a._memory.mark_superseded.assert_any_call("ver-1", f"obs{final_id}", kind="改訂")


def test_the_final_version_is_a_member_of_the_exchange():
    a, ip = _ip()
    _close(ip, "明日は晴れだよ", "発話")
    from familiar_agent.store.relations import KIND_EXCHANGE

    final_id = a._memory.save_async_with_id.call_args_list.index(_finals(a)[0]) + 1
    links = [c.args for c in a._memory.link.call_args_list if c.args[0] == KIND_EXCHANGE]
    assert any((f"obs{final_id}", "版") in [(i, r) for i, r, _ in m] for _, m in links)


def test_a_request_without_versions_still_gets_one():
    a, ip = _ip(live_version=None)
    _close(ip, "おはよう", "発話")
    assert len(_finals(a)) == 1
    a._memory.mark_superseded.assert_not_called()  # 畳む前の版が無い


def test_an_unsaid_close_still_gets_one():
    a, ip = _ip()
    ip._req.block_reason = "誰も居ない"
    _close(ip, "雨だよ", "独白")
    assert _finals(a)[0].args[0].endswith("「雨だよ」と言おうとしたが、言えなかった（誰も居ない）")


def test_aborting_writes_the_final_version():
    a, ip = _ip()

    async def go():
        await ip._abort_lookups()
        await ip.settle_finals()
        await ip.close()

    asyncio.run(go())
    finals = _finals(a)
    assert len(finals) == 1 and finals[0].args[0].endswith("打ち切った")


# ── 長すぎれば、背景で主LLM が要約し、閉じた時刻で書く ───────────────────────────────


def _long(ip):
    from familiar_agent.loop.event_loop import Lookup

    lk = Lookup(
        index=1, action="search_deferred", query="天気", generation=0, result="長い結果" * 200
    )
    ip._req.lookups = [lk]


def test_a_long_final_version_is_summarized_by_the_main_llm_and_dated_at_the_close():
    a, ip = _ip()
    a.config.completion_content_max = 300
    a.backend.complete = AsyncMock(return_value="天気を調べて晴れと答えた")
    _long(ip)
    before = datetime.now(timezone.utc)
    _close(ip, "明日は晴れだよ", "発話")
    finals = _finals(a)
    assert len(finals) == 1
    text = finals[0].args[0]
    assert text.startswith(fv.MARK) and "天気を調べて晴れと答えた" in text
    assert len(text) <= 300
    at = finals[0].kwargs["at"]
    assert before <= at <= datetime.now(timezone.utc)  # 閉じた時刻（要約を書いた時刻ではない）
    a.backend.complete.assert_awaited_once()


def test_a_failed_summary_still_writes_a_cut_final_version():
    a, ip = _ip()
    a.config.completion_content_max = 300
    a.backend.complete = AsyncMock(side_effect=RuntimeError("落ちた"))
    _long(ip)
    _close(ip, "明日は晴れだよ", "発話")
    text = _finals(a)[0].args[0]
    assert len(text) <= 300
    assert "「明日は晴れだよ」と答えた" in text  # 聞かれたことと答えは残し、経緯の末尾を切る


def test_a_short_final_version_is_not_summarized():
    """**反証の側**：収まれば主LLM は呼ばない。"""
    a, ip = _ip()
    a.backend.complete = AsyncMock(return_value="要約")
    _close(ip, "明日は晴れだよ", "発話")
    a.backend.complete.assert_not_awaited()


# ── 書き込みに時刻を渡す口 ────────────────────────────────────────────────────


def test_the_mouth_passes_the_time():
    from familiar_agent.io.oif import MI
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    asyncio.run(
        a._oif.write(
            MI(id="", content="x", timestamp=None, direction="求め"), writer_id="__self__", at=_END
        )
    )
    assert a._memory.save_async_with_id.call_args.kwargs["at"] == _END


def test_without_a_time_the_mouth_passes_none():
    """**反証の側**：渡さなければ、いまの時刻のまま（店が決める）。"""
    from familiar_agent.io.oif import MI
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    asyncio.run(
        a._oif.write(MI(id="", content="x", timestamp=None, direction="求め"), writer_id="__self__")
    )
    assert a._memory.save_async_with_id.call_args.kwargs.get("at") is None


# ── W：直近の枠に起点が載ったやりとりの最終版は出さない ─────────────────────────────


def _rec(obs_id, content, parent_id):
    from familiar_agent.io.oif import MI, Recalled

    mi = MI(
        id=obs_id,
        obs_id=obs_id,
        content=content,
        timestamp=_END,
        direction="求め",
        parent_id=parent_id,
    )
    return Recalled(mi=mi, fit=0.5, groundedness=0.5, confidence=0.8)


def test_the_final_version_of_a_recent_exchange_is_not_shown_again():
    from familiar_agent.loop import workspace
    from familiar_agent.loop.request import Request

    recent = _rec("fin-1", _compose(), "origin-recent")
    older = _rec("fin-2", _compose(), "origin-older")
    shown = workspace._shown([recent, older], Request(), {"origin-recent"})
    assert [r.mi.obs_id for r in shown] == ["fin-2"]
