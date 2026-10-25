"""窓の最終版を書き直す（環-ar 段 E・2026-10-25・本人の決定）。

窓の中の 2 つ目からの求めを閉じると、最終版は窓が開いた時刻から始まり、窓の会話の全部を持つ。前の窓の最終版は
改訂で畳み、窓の中では続きの辺を Jev に聞かない（前の最終版は畳まれ、窓の会話がひと続きを持つ）。窓の 1 つ目の求めと
窓の外の求めは、いまどおり（その求めの「聞かれ、答えた」・続きの辺を聞く）。
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core import final_version
from familiar_agent.store.relations import KIND_REVISION


def _ip(*, window: bool = True, prev_final: str = ""):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a.config.completion_content_max = 8192
    a.config.jev_confidence_min = 0.6
    a.config.jev_state_max_chars = 28_000
    a.config.memory.succession_window_sec = 300
    ip = InformationProcessing(a)
    ip._current_speaker_name = lambda: "パパ"
    ip._req.request_id = "origin-2"
    ip._req.request_text = "ごりら"
    ip._req.trigger_kind = "発話"
    ip._req.began_at = datetime.now(timezone.utc)
    if window:
        ip._wake_window().open(time.monotonic())
        ip._req.window_session = ip._wake_window().session
        ip._talk_line("パパ", "しりとりしよう")
        ip._talk_line("わたし（パパへ）", "りんご！")
        ip._talk_line("パパ", "ごりら")
        ip._window_talk().final_id = prev_final
    a._oif.write = AsyncMock(return_value="final-new")
    a._oif.supersede = MagicMock()
    a._oif.recent_finals = MagicMock(return_value=[])
    return a, ip


def _close(ip, text="らっぱ！"):
    async def go():
        await ip._finish(text, [], "発話")
        await ip.settle_finals()
        await ip.close()

    asyncio.run(go())


def _final_text(a) -> str:
    texts = [c.args[0].content for c in a._oif.write.await_args_list]
    finals = [t for t in texts if final_version.is_final(t)]
    assert len(finals) == 1, texts
    return finals[0]


def _folded(a) -> "list[str]":
    return [
        c.args[0]
        for c in a._oif.supersede.call_args_list
        if c.args[1] == "final-new" and c.kwargs.get("kind") == KIND_REVISION
    ]


# ── 窓の 2 つ目からの求め ─────────────────────────────────────────────────


def test_a_later_request_writes_the_whole_window_talk_from_its_opening():
    a, ip = _ip(prev_final="final-1")
    opened = ip._window_talk().opened_at
    _close(ip)
    text = _final_text(a)
    assert text.startswith(final_version.head(opened, opened).split("〜")[0])
    for line in ("パパ：しりとりしよう", "わたし（パパへ）：りんご！", "パパ：ごりら", "らっぱ！"):
        assert line in text


def test_the_previous_window_final_is_folded_as_a_revision():
    a, ip = _ip(prev_final="final-1")
    _close(ip)
    assert "final-1" in _folded(a)
    assert ip._window_talk().final_id == "final-new"  # 次の求めはこれを畳む


def test_inside_the_window_jev_is_not_asked_for_the_link():
    a, ip = _ip(prev_final="final-1")
    _close(ip)
    a._oif.recent_finals.assert_not_called()


# ── 窓の 1 つ目の求めと、窓の外（反証の側）────────────────────────────────


def test_the_first_request_of_the_window_is_written_as_before():
    """**反証の側**：窓の 1 つ目の求めは、いまどおりその求めの「聞かれ、答えた」。続きの辺も聞く。"""
    a, ip = _ip(prev_final="")
    _close(ip)
    text = _final_text(a)
    assert "「ごりら」と聞かれ" in text and "しりとりしよう" not in text
    assert _folded(a) == []
    a._oif.recent_finals.assert_called_once()
    assert ip._window_talk().final_id == "final-new"  # 2 つ目の求めが畳む


def test_a_request_outside_the_window_is_written_as_before():
    a, ip = _ip(window=False)
    _close(ip)
    assert "「ごりら」と聞かれ" in _final_text(a)
    assert _folded(a) == []
    a._oif.recent_finals.assert_called_once()


# ── 長すぎるとき ─────────────────────────────────────────────────


def test_a_long_window_final_is_shortened_in_the_background_and_still_folds():
    a, ip = _ip(prev_final="final-1")
    a.config.completion_content_max = 60
    a.backend.complete = AsyncMock(return_value="しりとりをした")
    _close(ip)
    assert _final_text(a).endswith("しりとりをした")
    assert "final-1" in _folded(a)
    assert ip._window_talk().final_id == "final-new"
    a._oif.recent_finals.assert_not_called()


def test_a_notice_in_the_window_does_not_fold_the_window_final():
    """**反証の側**：窓の会話の行にならない求め（タイマーの知らせ）は、窓の最終版を畳まず、控えも替えない。"""
    a, ip = _ip(prev_final="final-1")
    ip._req.trigger_kind, ip._req.request_text = "機器", "3 分たった"
    ip._talking = lambda: False
    _close(ip, text="3 分たったよ")
    assert "〔3 分たった〕が起きて" in _final_text(a)
    assert _folded(a) == []
    assert ip._window_talk().final_id == "final-1"


# --- 見直しの指摘（1 回目）---


def test_two_requests_closed_back_to_back_fold_in_order():
    """窓の最終版は窓ごとの列で順に書く。前が背景で書き終わる前に次が閉じても、同じものを二度畳まない。"""
    a, ip = _ip(prev_final="final-1")
    finals = iter(["final-2", "final-3"])

    async def write(mi, **k):  # 返事の記録も同じ口を通るので、最終版だけに id を振る
        if not final_version.is_final(mi.content):
            return "answer"
        got = next(finals)
        if got == "final-2":
            await asyncio.sleep(0.05)  # 1 つ目がまだ書いている最中に、2 つ目が閉じる
        return got

    a._oif.write = AsyncMock(side_effect=write)

    async def go():
        await ip._finish("らっぱ！", [], "発話")
        ip._req.request_id, ip._req.request_text = "origin-3", "ぱんだ"
        ip._req.began_at = datetime.now(timezone.utc)
        ip._talk_line("パパ", "ぱんだ")
        await ip._finish("だちょう！", [], "発話")
        await ip.settle_finals()
        await ip.close()

    asyncio.run(go())
    folds = [(c.args[0], c.args[1]) for c in a._oif.supersede.call_args_list]
    assert ("final-1", "final-2") in folds and ("final-2", "final-3") in folds
    assert ("final-1", "final-3") not in folds
    assert ip._window_talk().final_id == "final-3"


def test_a_cut_window_final_keeps_the_newest_lines():
    head = final_version.head(datetime(2026, 1, 1), datetime(2026, 1, 1))
    lines = [f"- 19:00:{i:02d} パパ：ことば{i}" for i in range(50)]
    body = final_version.window_body(lines)
    text = final_version.cut_window(head, body, "", 300)
    assert len(text) <= 300 and "ことば49" in text and "ことば0\n" not in text


def test_an_aborted_close_is_kept_in_the_window_final():
    a, ip = _ip(prev_final="final-1")

    async def go():
        pending = await ip._close_final("打ち切り", "", aborted=True)
        ip._spawn_final(pending, None)
        await ip.settle_finals()

    asyncio.run(go())
    assert "「ごりら」と聞かれ、打ち切った" in _final_text(a)
    assert "「ごりら」と聞かれ、打ち切った" in ip._window_talk().steps  # 次の書き直しでも消えない


def test_the_course_of_earlier_requests_stays_in_the_window_final():
    a, ip = _ip(prev_final="final-1")
    ip._window_talk().steps.append("調べた：明日は晴れ")
    _close(ip)
    assert "調べた：明日は晴れ" in _final_text(a)


def _successions(a):
    from familiar_agent.store.relations import KIND_SUCCESSION

    return [c.args[1] for c in a._oif.link.call_args_list if c.args[0] == KIND_SUCCESSION]


def test_the_rewritten_final_keeps_the_link_to_before_the_window():
    """窓の 1 つ目の最終版が結んだ「前」を、書き直した最終版にも付ける（畳むと、窓の前とのつながりが切れる）。"""
    a, ip = _ip(prev_final="final-1")
    a._oif.link = MagicMock()
    ip._window_talk().before_id = "pre-window"
    _close(ip)
    assert [("pre-window", "前", 0), ("final-new", "後", 1)] in _successions(a)


def test_the_first_final_of_the_window_remembers_what_it_follows():
    from tests.test_succession_between_finals import _CANDS, _client

    a, ip = _ip(prev_final="")
    a._oif.link = MagicMock()
    a._oif.recent_finals = MagicMock(
        return_value=[{"id": "pre-window", "content": _CANDS["c1"], "timestamp": None}]
    )
    a._jev = _client("pre-window")
    _close(ip)
    assert ip._window_talk().before_id == "pre-window"


def test_a_cut_window_final_keeps_lines_even_with_a_long_course():
    head = final_version.head(datetime(2026, 1, 1), datetime(2026, 1, 1))
    body = final_version.window_body([f"- 19:00:{i:02d} パパ：ことば{i}" for i in range(50)])
    text = final_version.cut_window(
        head, body, final_version.course(["調べた：" + "晴れ" * 500]), 600
    )
    assert len(text) <= 600 and "ことば49" in text


def test_a_cut_keeps_the_newest_line_and_the_newest_course():
    head = final_version.head(datetime(2026, 1, 1), datetime(2026, 1, 1))
    body = final_version.window_body(["- 19:00:00 パパ：やあ", "- 19:00:01 わたし：" + "あ" * 400])
    course = final_version.course(["古い：" + "a" * 300, "新しい：打ち切った"])
    text = final_version.cut_window(head, body, course, 300)
    assert len(text) <= 300 and "わたし：あ" in text and "新しい：打ち切った" in text


def test_nothing_dropped_means_no_ellipsis():
    """**反証の側**：行を落としていなければ「…」を置かない。"""
    head = final_version.head(datetime(2026, 1, 1), datetime(2026, 1, 1))
    body = final_version.window_body(["- 19:00:00 パパ：やあ", "- 19:00:01 わたし：うん"])
    text = final_version.cut_window(head, body, final_version.course(["x" * 400]), 300)
    assert "\n…\n" not in text and "パパ：やあ" in text
