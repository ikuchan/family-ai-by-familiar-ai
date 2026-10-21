"""音楽のおすすめを毎晩選び直し、軽く話しかけるときにも渡す（知-aq・2026-10-22・本人の決定）。

10/12 に用意した候補が 9 日間、一度も勧められないまま残っていた。晩の REST は候補が残っていれば何もせず、勧める材料
`[音楽のおすすめ]` は主LLM にしか載らなかった——bond・esteem の発火は 4 日で 1 回で、そのときも調停は軽く話しかける
（`talk_light`）を選び、軽量LLM の文には材料が無かった。

- 軽く話しかけるときも、書き手に材料を渡す（本人の決定ア）。
- 晩ごとに選び直す（ア）。新しい 5 件のうち Spotify で通ったものと前日の候補を並べ、Jev が家族のいまの様子・
  プレイリストのアーティスト・前日のおすすめを見て選ぶ。1 番目の確率が 0.5〔仮〕以上なら一つに固まったとみなし、
  2 番目以降から条件付き確率で引く。固まっていなければ 1 番目（イ）。Jev が答えなければ前日の候補を残す（ア）。
"""

from __future__ import annotations

import asyncio

from tests._arbiter_fakes import decide, jev_says, prompt_of, writer_says

FRAME = "[音楽のおすすめ]\n家族にすすめたい曲がある：「忘れられないの」（サカナクション）。"


# ── 段 1：軽く話しかける文にも材料を渡す ─────────────────────────────────────


def _write(origin: str, suggestion: str, **kw) -> str:
    writer = writer_says({"text": ""})
    asyncio.run(
        decide(
            jev=jev_says("light", action="talk_light"),
            writer=writer,
            utterance="[内的な促し:BOND] 誰かと居たい気持ちが湧いている。"
            if origin == "情動"
            else "こんにちは",
            origin=origin,
            suggestion_ctx=suggestion,
            **kw,
        )
    )
    return prompt_of(writer)


def test_a_light_talk_on_bond_carries_the_suggestion():
    assert FRAME in _write("情動", FRAME, fired_axis="bond", talking=True)


def test_without_a_suggestion_nothing_is_added():
    """**反証の側**：材料が空なら見出しも出さない。"""
    assert "[音楽のおすすめ]" not in _write("情動", "", fired_axis="bond", talking=True)


def test_a_reply_to_someone_does_not_carry_it():
    """**反証の側**：会話の返事（発話）には、材料があっても添えない（勧めるのは自分から話しかけるときだけ）。"""
    assert "[音楽のおすすめ]" not in _write("発話", FRAME)


def test_the_loop_hands_the_frame_only_on_a_talking_drive(monkeypatch):
    """`_arbiter_input` は、bond・esteem の情動のときだけ `[音楽のおすすめ]` を渡す。"""
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    monkeypatch.setattr(ip, "_suggestion_frame", lambda: FRAME)
    ip._req.trigger_kind = "情動"
    ip._req.fired_axis = "bond"
    assert (
        ip._arbiter_input(utterance="x", workspace_ctx="", present_ctx="").suggestion_ctx == FRAME
    )
    ip._req.fired_axis = "safety"
    assert ip._arbiter_input(utterance="x", workspace_ctx="", present_ctx="").suggestion_ctx == ""


# ── 段 2：選び方 ───────────────────────────────────────────────────────────


class _Rng:
    """`random()` が決まった値を返す。"""

    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value


def test_below_the_line_the_first_is_taken():
    from familiar_agent.core import music_suggestion as ms

    probs = {"a": 0.49, "b": 0.3, "c": 0.21}
    assert ms.pick(probs, fixed_at=0.5, rng=_Rng(0.0)) == ("a", 1, False)
    assert ms.pick(probs, fixed_at=0.5, rng=_Rng(0.99)) == ("a", 1, False)  # **反証の側**：引かない


def test_at_the_line_the_rest_are_drawn_by_conditional_probability():
    """p₁ = 0.5 なら 1 番目は出ない。残りは pᵢ ÷ (1 − p₁)：b は 0.3/0.5 = 0.6、c は 0.2/0.5 = 0.4。"""
    from familiar_agent.core import music_suggestion as ms

    probs = {"c": 0.2, "a": 0.5, "b": 0.3}
    assert ms.pick(probs, fixed_at=0.5, rng=_Rng(0.0)) == ("b", 2, True)
    assert ms.pick(probs, fixed_at=0.5, rng=_Rng(0.59)) == ("b", 2, True)
    assert ms.pick(probs, fixed_at=0.5, rng=_Rng(0.61)) == ("c", 3, True)
    assert ms.pick(probs, fixed_at=0.5, rng=_Rng(0.999)) == ("c", 3, True)


def test_a_single_candidate_is_taken_even_when_fixed():
    from familiar_agent.core import music_suggestion as ms

    assert ms.pick({"a": 1.0}, fixed_at=0.5, rng=_Rng(0.5)) == ("a", 1, False)


def test_all_on_the_first_falls_back_to_the_second_evenly():
    """残りの確率がすべて 0 でも、1 番目には戻さない（残りから等しく引く）。"""
    from familiar_agent.core import music_suggestion as ms

    probs = {"a": 1.0, "b": 0.0, "c": 0.0}
    assert ms.pick(probs, fixed_at=0.5, rng=_Rng(0.0)) == ("b", 2, True)
    assert ms.pick(probs, fixed_at=0.5, rng=_Rng(0.7)) == ("c", 3, True)


def test_the_last_pick_round_trips_and_old_json_still_reads():
    from familiar_agent.core import music_suggestion as ms

    s = ms.Suggestions(last_pick={"title": "t", "artist": "a", "reason": "r", "uri": "u"})
    assert ms._from_json(ms._to_json(s)).last_pick == s.last_pick
    assert ms._from_json({"candidate": None}).last_pick == {}  # 欄の無い古い JSON


# ── 段 3：Jev の判定 ─────────────────────────────────────────────────────────

CANDS = [
    {"title": "灯", "artist": "甲", "reason": "夜に合う"},
    {"title": "新宝島", "artist": "サカナクション", "reason": "よく聞く人の別の曲"},
]


def _jev(answers: dict, ok: bool = True):
    from unittest.mock import AsyncMock, MagicMock

    from familiar_agent.backends.jev import JevAnswer

    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(return_value=JevAnswer(ok=ok, answers=answers))
    return jev


def _judge(jev):
    from familiar_agent.core.jev_judges import judge_music_pick

    return asyncio.run(
        judge_music_pick(
            jev,
            candidates=CANDS,
            family_now="パパ：最近は朝にジャズを聞いている",
            artists="- サカナクション（12）",
            previous="「灯」（甲）。まだ勧めていない",
        )
    )


def test_the_judge_shows_the_family_the_playlists_and_yesterday():
    jev = _jev(
        {"pick": {"choice": "c2", "confidence": 0.7, "probabilities": {"c1": 0.3, "c2": 0.7}}}
    )
    assert _judge(jev) == {"c1": 0.3, "c2": 0.7}
    state, questions = jev.ask.await_args.args
    assert "[家族のいまの様子]\nパパ：最近は朝にジャズを聞いている" in state
    assert "[プレイリストのアーティスト]\n- サカナクション（12）" in state
    assert "[前日のおすすめ]\n「灯」（甲）。まだ勧めていない" in state
    assert set(questions["pick"]["criteria"]) == {"c1", "c2"}
    assert "新宝島" in questions["pick"]["criteria"]["c2"]


def test_without_jev_or_probabilities_there_is_no_answer():
    """**反証の側**：使えない・失敗・確率の無い答えは None（呼び手が前日の候補を残す）。"""
    from unittest.mock import MagicMock

    down = MagicMock()
    down.available = False
    assert _judge(down) is None
    assert _judge(_jev({}, ok=False)) is None
    assert _judge(_jev({"pick": {"choice": "c2", "confidence": 0.7}})) is None


# ── 段 4：晩の流れ ───────────────────────────────────────────────────────────

import json  # noqa: E402
from datetime import datetime  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import pytest  # noqa: E402

from familiar_agent.core import family_now as fn  # noqa: E402
from familiar_agent.core import music_catalog as mc  # noqa: E402
from familiar_agent.core import music_suggestion as ms  # noqa: E402
from familiar_agent.loop.rest_music import prepare_suggestion  # noqa: E402

CATALOG = mc.Catalog(
    playlists=[
        mc.Playlist(
            "p1",
            "夜のドライブ",
            "spotify:playlist:p1",
            [{"title": "新宝島", "artist": "サカナクション", "uri": "spotify:track:shin"}],
        )
    ]
)
OLD = ms.Candidate("前の曲", "サカナクション", "spotify:track:old", "前の理由", 1)


@pytest.fixture
def state(monkeypatch):
    data: dict = {"s": ms.Suggestions()}
    monkeypatch.setattr(ms, "stored", lambda: data["s"])
    monkeypatch.setattr(ms, "store", lambda s: data.__setitem__("s", s) or True)
    monkeypatch.setattr(mc, "stored", lambda: CATALOG)
    monkeypatch.setattr(
        fn, "stored", lambda: {"パパ": fn.Now("朝にジャズを聞いている", "", datetime(2026, 10, 21))}
    )
    return data


def _night(titles, found, jev):
    """`found` は Spotify に有る曲名 → URI。"""
    a = MagicMock()
    a.backend.complete = AsyncMock(
        return_value=json.dumps(
            {
                "candidates": [
                    {"title": t, "artist": "サカナクション", "reason": f"{t} が合う"}
                    for t in titles
                ]
            },
            ensure_ascii=False,
        )
    )

    def search(query, kind):
        title = query.rsplit(" ", 1)[0]
        uri = found.get(title)
        return [{"title": title, "artist": "サカナクション", "uri": uri}] if uri else []

    a._music_tool._web.search = MagicMock(side_effect=search)
    a._jev = jev
    a.config.music_pick_fixed_at = 0.5
    return a


def _probs(**p):
    top = max(p, key=lambda k: p[k])
    return _jev({"pick": {"choice": top, "confidence": p[top], "probabilities": p}})


FOUND = {"灯": "spotify:track:akari", "夜": "spotify:track:yoru"}


def test_a_left_candidate_is_compared_again_every_night(state):
    """候補が残っていても選び直す。前日の候補も並べ、選ばれれば勧めた回数を引き継ぐ。"""
    state["s"] = ms.Suggestions(candidate=ms.Candidate(*OLD.__dict__.values()))
    jev = _probs(c1=0.2, c2=0.1, c3=0.4)  # c3 = 前日の候補（新しいものの後ろに並ぶ）
    assert asyncio.run(prepare_suggestion(_night(["灯", "夜"], FOUND, jev), rng=_Rng(0.0)))
    c = state["s"].candidate
    assert (c.uri, c.offered) == ("spotify:track:old", 1)
    criteria = jev.ask.await_args.args[1]["pick"]["criteria"]
    assert "前日から残っている候補" in criteria["c3"]


def test_every_suggestion_is_checked_on_spotify(state):
    a = _night(["灯", "幻", "夜"], FOUND, _probs(c1=0.4, c2=0.3))
    asyncio.run(prepare_suggestion(a, rng=_Rng(0.0)))
    assert a._music_tool._web.search.call_count == 3
    assert [m["title"] for m in state["s"].missing] == ["幻"]


def test_a_fixed_first_gives_way_to_the_others(state, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="familiar_agent.loop.rest_music")
    asyncio.run(
        prepare_suggestion(_night(["灯", "夜"], FOUND, _probs(c1=0.9, c2=0.1)), rng=_Rng(0.0))
    )
    assert state["s"].candidate.uri == "spotify:track:yoru"
    assert state["s"].last_pick["uri"] == "spotify:track:yoru"
    assert any("おすすめを選んだ" in r.message and "固まった" in r.message for r in caplog.records)


def test_an_open_field_takes_the_first(state):
    """**反証の側**：固まっていなければ 1 番目。"""
    asyncio.run(
        prepare_suggestion(_night(["灯", "夜"], FOUND, _probs(c1=0.3, c2=0.45)), rng=_Rng(0.0))
    )
    assert state["s"].candidate.uri == "spotify:track:yoru"


def test_yesterday_is_told_to_jev(state):
    state["s"] = ms.Suggestions(
        liked=[{"title": "前の曲", "artist": "サカナクション", "uri": "spotify:track:old"}],
        last_pick={
            "title": "前の曲",
            "artist": "サカナクション",
            "reason": "r",
            "uri": "spotify:track:old",
        },
    )
    jev = _probs(c1=0.4, c2=0.3)
    asyncio.run(prepare_suggestion(_night(["灯", "夜"], FOUND, jev), rng=_Rng(0.0)))
    s = jev.ask.await_args.args[0]
    assert "[前日のおすすめ]\n「前の曲」（サカナクション）。勧めて、気に入ってもらえた" in s
    assert "パパ：朝にジャズを聞いている" in s
    assert "- サカナクション（1）" in s


def test_without_jev_yesterdays_candidate_stays(state):
    from unittest.mock import MagicMock as _M

    down = _M()
    down.available = False
    state["s"] = ms.Suggestions(candidate=ms.Candidate(*OLD.__dict__.values()))
    assert asyncio.run(prepare_suggestion(_night(["灯"], FOUND, down))) is False
    assert state["s"].candidate.uri == "spotify:track:old"


def test_without_jev_and_nothing_left_the_first_found_is_taken(state):
    """**反証の側**：前日の候補が無ければ、通った最初の 1 件（今までと同じ）。"""
    from unittest.mock import MagicMock as _M

    down = _M()
    down.available = False
    assert asyncio.run(prepare_suggestion(_night(["幻", "灯", "夜"], FOUND, down))) is True
    assert state["s"].candidate.uri == "spotify:track:akari"


def test_the_line_is_a_plain_default():
    from familiar_agent.config import AgentConfig

    got = AgentConfig.__dataclass_fields__["music_pick_fixed_at"].default
    assert got == 0.5
