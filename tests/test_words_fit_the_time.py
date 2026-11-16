"""話す言葉を時刻に合わせる（出-ds・2026-11-16・本人の決定）。

夜の発火で「おはよう」と言い出した（11/15 19:40・19:50）。W には朝の記憶が日時つきで並んでいたが、いまから
どれだけ前かは書いていなかった。LLM に渡す文にだけ、曜日と「N 分前」を載せる。DB に書く文には距離を入れない
（あとで思い出したとき、いまからの距離に読めてしまう）。
"""

from __future__ import annotations

import pytest

from familiar_agent.store import clock


@pytest.mark.parametrize(
    ("sec", "want"),
    [
        (0, "0 秒前"),
        (59, "59 秒前"),
        (60, "1 分前"),
        (3599, "59 分前"),
        (3600, "1 時間前"),
        (86399, "23 時間前"),
        (86400, "1 日前"),
        (3 * 86400 + 5, "3 日前"),
    ],
)
def test_ago_truncates_at_each_unit(sec, want):
    assert clock.ago(sec) == want


def test_ago_treats_a_negative_gap_as_now():
    """時計のずれで先の時刻になっても、負の距離は書かない。"""
    assert clock.ago(-5) == "0 秒前"


# ── 項目 2：いまの日時に曜日を添える ──────────────


def test_the_prompt_time_carries_the_weekday(monkeypatch):
    """`strftime("%a")` は言語の設定で変わるので、曜日は日本語の表から引く。"""
    from datetime import datetime, timedelta, timezone

    jst = timezone(timedelta(hours=9), "JST")

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 15, 19, 50, tzinfo=jst)

    monkeypatch.setattr(clock, "datetime", _Fixed)
    assert clock.now_local_str() == "2026-11-15（日）19:50 JST(+0900)"


# ── 項目 3：W の思い出したことに、いまからの距離を添える ──────────────

from datetime import datetime, timedelta, timezone  # noqa: E402

from familiar_agent.io.oif import MI, Recalled  # noqa: E402
from familiar_agent.loop import workspace  # noqa: E402
from familiar_agent.loop.request import Request  # noqa: E402

JST = timezone(timedelta(hours=9))
NOW = datetime(2026, 11, 15, 19, 50, tzinfo=JST)


def _r(obs_id: str, when: datetime) -> Recalled:
    return Recalled(
        mi=MI(id=obs_id, obs_id=obs_id, content="おはよう", timestamp=when, direction="発話"),
        fit=0.5,
        groundedness=1.0,
        confidence=0.8,
    )


def test_a_recalled_line_says_how_long_ago():
    got = workspace._lines([_r("a", datetime(2026, 11, 15, 9, 25, tzinfo=JST))], {}, now=NOW)
    assert "- 2026-11-15 09:25（10 時間前） id:a" in got


def test_a_naive_time_is_read_as_local_time():
    """tz の無い時刻（読み手の偽物など）は、ローカル時刻として距離を出す。"""
    local_now = datetime(2026, 11, 15, 19, 50).astimezone()
    got = workspace._lines([_r("a", datetime(2026, 11, 15, 19, 45))], {}, now=local_now)
    assert "（5 分前）" in got


def test_without_now_the_line_is_unchanged():
    """**反証の側**：`now` を渡さない呼び手（引き直しの道具の返り）は、いまの書式のまま。"""
    got = workspace._lines([_r("a", datetime(2026, 11, 15, 9, 25, tzinfo=JST))], {})
    assert "- 2026-11-15 09:25 id:a" in got
    assert "前）" not in got


def test_the_w_carries_the_distance(monkeypatch):
    """W を組む `compose` は、いまの時刻を渡す。"""
    from unittest.mock import MagicMock

    monkeypatch.setattr(workspace.clock, "now_utc", lambda: NOW)
    text, _ = workspace.compose(
        MagicMock(), [_r("a", datetime(2026, 11, 15, 9, 25, tzinfo=JST))], Request()
    )
    assert "09:25（10 時間前）" in text


# ── 項目 4：直近のやりとりと、そのあいだに起きたことに、いまからの距離を添える ──────────────


def _said(obs_id: str, content: str, when: datetime, *, direction: str = "発話"):
    from types import SimpleNamespace

    return SimpleNamespace(
        obs_id=obs_id, role="起点", content=content, when=when, direction=direction
    )


class _NoVoices:
    def voices(self, ids):
        return {}


def test_the_recent_lines_say_how_long_ago():
    rows = [
        _said("o1", "ただいま", datetime(2026, 11, 15, 19, 40, tzinfo=JST)),
        _said(
            "o2",
            "[タイマー] 鳴った",
            datetime(2026, 11, 15, 19, 49, 30, tzinfo=JST),
            direction="機器",
        ),
    ]
    _, text, _ = workspace.render_recent(_NoVoices(), [("o1", rows)], 3, now=NOW)
    assert "- 19:40（10 分前） id:o1 相手：ただいま" in text
    assert "- 19:49（30 秒前） id:o2 [タイマー] 鳴った" in text


def test_the_recent_frame_reads_the_clock_without_now(monkeypatch):
    """呼び手（W と、記録から引く直近）は `now` を渡さないので、いまの時刻を自分で読む。"""
    monkeypatch.setattr(workspace.clock, "now_utc", lambda: NOW)
    rows = [_said("o1", "ただいま", datetime(2026, 11, 15, 19, 40, tzinfo=JST))]
    _, text, _ = workspace.render_recent(_NoVoices(), [("o1", rows)], 3)
    assert "19:40（10 分前）" in text


def test_the_window_talk_has_no_distance():
    """**反証の側**：窓の会話の行には距離を付けない（行ごとに変わると、文の写しが使い回せない・環-ar 段 D）。"""
    from familiar_agent.core.window_talk import WindowTalk

    talk = WindowTalk(session=1, opened_at=datetime(2026, 11, 15, 19, 39, tzinfo=JST))
    talk.add(datetime(2026, 11, 15, 19, 40, tzinfo=JST), "相手", "ただいま", "o1")
    assert "前）" not in talk.render()


# ── 項目 5：調べものの行に、投げてからの距離を添える（LLM に渡す文だけ） ──────────────

from familiar_agent.loop.request import Lookup  # noqa: E402

SENT = 1_000_000.0


def _searching_req(*, returned_now: bool = False) -> Request:
    req = Request()
    req.request_text = "明日の天気は？"
    req.lookups.append(
        Lookup(
            index=1,
            action="search_deferred",
            query="明日 天気",
            generation=0,
            result="晴れ",
            sent_at=SENT,
        )
    )
    if returned_now:
        req.just_returned.append(1)
    req.live_version_id = "v-live"
    return req


def test_a_lookup_keeps_when_it_was_sent():
    """器を作った時刻を持つ（器を作る口は `_open_lookup` の 1 か所で、そこは変えない）。"""
    import time

    before = time.time()
    lk = Lookup(index=1, action="search_deferred", query="q", generation=0)
    assert before <= lk.sent_at <= time.time()


def test_the_lookup_line_says_how_long_ago_it_was_sent():
    got = workspace.lookup_lines(_searching_req(), now=SENT + 180)
    assert got == ["1番：search_deferred「明日 天気」（3 分前に投げた）の結果が届いた：晴れ"]


def test_a_body_elsewhere_line_also_says_when():
    """本文を最上部へ回した行（同じ反復で返った結果）も、届いた結果の行なので距離を添える。"""
    got = workspace.lookup_lines(_searching_req(), now=SENT + 180, omit=frozenset({1}))
    assert got == [
        "1番：search_deferred「明日 天気」（3 分前に投げた）の結果が届いた（中身は [いま道具から返った]）"
    ]


def test_the_written_version_has_no_distance():
    """**反証の側**：O に書く版の文（`now` を渡さない）は、いまの書式のまま。"""
    assert (
        workspace.version_text(_searching_req())
        == "「明日の天気は？」と聞かれ、1番：search_deferred「明日 天気」の結果が届いた：晴れ"
    )


def _ws_of(req: Request):
    from unittest.mock import MagicMock

    from tests.test_recent_window_is_a_workspace_frame import _oif, _said
    from tests.test_workspace_is_the_core import _rec

    oif = _oif(["q1"], {"q1": [_said("q1", "直近のひとこと", "起点", 0)]})
    oif.roles = MagicMock(return_value={})
    oif.actors = MagicMock(return_value={})
    live = _rec("v-live", workspace.version_text(req), fit=0.95)
    return workspace.Workspace.build(oif, [live], req, n_arbiter=1, n_main=2)


def test_the_w_and_the_jev_cue_say_when(monkeypatch):
    """LLM に渡す W（いまの求めの版）と、Jev の [いま湧いたこと] には距離を載せる。"""
    monkeypatch.setattr(workspace.time, "time", lambda: SENT + 180)
    ws = _ws_of(_searching_req())
    assert "（3 分前に投げた）の結果が届いた：晴れ" in ws.for_main
    assert "（3 分前に投げた）の結果が届いた：晴れ" in ws.jev_cue()


def _ip_with(req: Request):
    from familiar_agent.loop.event_loop import InformationProcessing

    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._req = req
    return ip


def test_the_speech_check_gets_the_distance(monkeypatch):
    """発話前の検査へ渡す「届いた結果」にも距離を載せる。"""
    from familiar_agent.loop import event_loop
    from tests.test_workspace_is_the_core import _rec

    monkeypatch.setattr(event_loop.time, "time", lambda: SENT + 180)
    ip = _ip_with(_searching_req())
    got = ip._arrived([_rec("v-live", "版の記録の中身", fit=0.9)])
    assert got == ["1番：search_deferred「明日 天気」（3 分前に投げた）の結果が届いた：晴れ"]


def test_the_version_written_to_o_has_no_distance():
    """**反証の側**：O に書く版（`_version_content`）には距離を入れない。"""
    ip = _ip_with(_searching_req())
    assert "投げた" not in ip._version_content()


# ── 項目 6：日時と曜日に合う言葉で話す、と書く ──────────────


def test_the_main_rules_say_fit_the_time():
    """主LLM の規則に、now の日時と曜日に合う言葉で話すことを書く。"""
    from familiar_agent.loop.prompt import EVENT_SYSTEM_PROMPT, rules_section

    want = ":id fit-the-time"
    assert want in rules_section()
    assert "now の日時と曜日に合う言葉で話す。" in EVENT_SYSTEM_PROMPT


def test_the_speech_check_does_not_judge_the_time():
    """**反証の側**：発話前の検査へは渡さない（照らす規則は `CHECKER_RULE_IDS` に限る）。"""
    from familiar_agent.loop.prompt import CHECKER_RULE_IDS

    assert "fit-the-time" not in CHECKER_RULE_IDS


@pytest.mark.parametrize("field", ["text_talk", "text_self"])
def test_the_writer_is_told_to_fit_the_time(field):
    """軽量LLM の書き手にも、話しかける一言とひとりごとで、[いま] の日時と曜日に合わせると書く。"""
    from familiar_agent.loop.arbiter import _FIELD_TEXT

    assert "[いま] の日時と曜日に合う言葉で" in _FIELD_TEXT[field]
