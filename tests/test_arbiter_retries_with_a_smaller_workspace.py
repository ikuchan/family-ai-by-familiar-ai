"""完了の道で Jev に断られたら、思い出したことを 1 件ずつ外して聞き直す（出-bm・2026-10-12・本人の決定）。

Jev は `state` といちばん長い問いを足して 32k 語までしか受けない（docs.typesafe.ai の models.md）。W をまるごと
送る（本人の決定イ）と、W に積もった検索の結果で超え、HTTP 400 で断られて主LLM に倒れていた（10/10〜 53 件・
すべて完了の道）。完了の道だけ、400 のあいだ、W の過去の記憶を適合度の低い順に 1 件ずつ外して聞き直す。
今回の結果・直近のやりとりは外さない。主LLM に渡す W と、申告の対応表は変えない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.loop import workspace
from familiar_agent.loop.request import Lookup, Request
from tests._arbiter_fakes import decide, writer_says
from tests.test_recent_window_is_a_workspace_frame import _oif, _said
from tests.test_workspace_is_the_core import _rec

# --- 調停に送る W を、過去の記憶だけ k 件外して組む ---


def _ws():
    oif = _oif(["q1"], {"q1": [_said("q1", "直近のひとこと", "起点", 0)]})
    oif.roles = MagicMock(return_value={})
    oif.actors = MagicMock(return_value={})
    req = Request()
    req.lookups.append(
        Lookup(index=1, action="search_deferred", query="天気", generation=0, result="今日の天気")
    )
    req.just_returned.append(1)
    memories = [
        _rec("m1", "高い記憶", fit=0.9),
        _rec("m2", "中くらいの記憶", fit=0.5),
        _rec("m3", "低い記憶", fit=0.1),
    ]
    return workspace.Workspace.build(oif, memories, req, n_arbiter=1, n_main=2)


def test_the_lowest_fit_memory_goes_first():
    ws = _ws()
    text = ws.for_arbiter_without(1)
    assert text is not None
    assert "低い記憶" not in text
    assert "高い記憶" in text and "中くらいの記憶" in text


def test_the_return_and_the_recent_exchange_stay():
    text = _ws().for_arbiter_without(3)
    assert text is not None
    assert "[思い出したこと]" not in text
    assert "[いま道具から返った]" in text and "今日の天気" in text
    assert "直近のひとこと" in text


def test_nothing_left_to_drop_returns_none():
    assert _ws().for_arbiter_without(4) is None


def test_the_main_workspace_and_the_id_map_do_not_change():
    """**反証の側**：調停の写しを縮めても、主LLM の W と申告の対応表はそのまま。"""
    ws = _ws()
    before = (ws.for_arbiter, ws.for_main, dict(ws.id_map))
    ws.for_arbiter_without(2)
    assert (ws.for_arbiter, ws.for_main, dict(ws.id_map)) == before
    assert "低い記憶" in ws.for_arbiter


# --- 完了の道で、400 のあいだ 1 件ずつ外して聞き直す ---

_REJECTED = JevAnswer(ok=False, error="HTTP 400")
_RETURNED = (("search_deferred", False, "今日の天気は晴れ"),)


def _jev(*answers) -> MagicMock:
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(side_effect=list(answers))
    return jev


def _picks(choice: str) -> JevAnswer:
    return JevAnswer(ok=True, answers={"action": {"choice": choice, "confidence": 0.3}})


def _smaller(limit: int = 9):
    """k 件外した W。外せるのは `limit` 件まで。"""
    return MagicMock(side_effect=lambda k: None if k > limit else f"W から {k} 件外した")


def _run(jev, without, *, origin="情動", returned=_RETURNED, monkeypatch=None):
    kept: list = []
    if monkeypatch is not None:
        from familiar_agent.store import arbiter_records

        monkeypatch.setattr(arbiter_records, "record", lambda **kw: kept.append(kw))
    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer_says({"text": "晴れだって"}),
            utterance="今日の天気は晴れ",
            origin=origin,
            fired_axis="seeking",
            returned=returned,
            workspace_ctx="W まるごと",
            workspace_without=without,
        )
    )
    return d, kept


def _sent(jev) -> "list[str]":
    return [c.args[0] for c in jev.ask.await_args_list]


def test_a_rejected_completion_is_asked_again_with_one_less_memory_each_time(monkeypatch):
    jev = _jev(_REJECTED, _REJECTED, _picks("silent"))
    without = _smaller()
    d, kept = _run(jev, without, monkeypatch=monkeypatch)
    sent = _sent(jev)
    assert len(sent) == 3
    assert "W まるごと" in sent[0]
    assert "W から 1 件外した" in sent[1] and "W から 2 件外した" in sent[2]
    assert (d.branch, d.text) == ("light", "")  # silent が通った（主LLM に倒れていない）
    assert "W から 2 件外した" in kept[-1]["state"]  # 記録には最後に送った文


def test_other_failures_fall_to_full_without_asking_again():
    """**反証の側**：時間切れ・429 は長さのせいではないので、聞き直さずにすぐ倒す。"""
    for why in ("時間切れ（2.0 秒）", "HTTP 429"):
        jev = _jev(JevAnswer(ok=False, error=why))
        without = _smaller()
        d, _ = _run(jev, without)
        assert jev.ask.await_count == 1, why
        without.assert_not_called()
        assert d.branch == "full"


def test_when_nothing_is_left_to_drop_it_falls_to_full():
    jev = _jev(*([_REJECTED] * 4))
    d, _ = _run(jev, _smaller(limit=3))
    assert jev.ask.await_count == 4  # まるごと 1 回 ＋ 外せる 3 件ぶん
    assert d.branch == "full"


def test_other_paths_are_not_asked_again():
    """**反証の側**：発話・情動の道は W をまるごと送ったまま（本人の決定イ）。400 でも聞き直さない。"""
    for origin, returned in (("発話", ()), ("情動", ())):
        jev = MagicMock()
        jev.available = True
        jev.ask = AsyncMock(return_value=_REJECTED)
        without = _smaller()
        _run(jev, without, origin=origin, returned=returned)
        without.assert_not_called()


# --- 反復が W の口を調停へ渡す ---


def test_the_iteration_hands_the_workspace_shrinker_to_the_arbiter(monkeypatch):
    from familiar_agent.loop.arbiter import Arbiter, Decision
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests._arbiter_fakes import jev_says
    from tests.test_event_loop import _agent
    from tests.test_iteration_characterization import _run_until

    a = _agent(stream_returns=[])
    a._jev = jev_says("light")
    a._utility_backend = writer_says({"text": ""})
    got: list = []

    async def capture(self, inp, on_decided=None):
        got.append(inp)
        return Decision(branch="light", text="")

    monkeypatch.setattr(Arbiter, "decide", capture)
    ip = InformationProcessing(a)
    _run_until(ip, lambda: got, push=lambda: ip.push_affect("CALM", "落ち着いている"))
    without = got[0].workspace_without
    # この反復で組んだ W の口そのもの（主LLM の W と同じ想起から縮める）
    assert getattr(without, "__func__", None) is workspace.Workspace.for_arbiter_without
    assert isinstance(without.__self__, workspace.Workspace)
