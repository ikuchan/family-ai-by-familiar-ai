"""Jev へ送る文の重なりを除き、送る前に測って一度で削る（出-bw・2026-10-12・本人の決定）。

Jev は state といちばん長い問いを足して 32k 語までしか受けない。調べた結果の全文が、W の最上部
[いま道具から返った]・W の記憶（いまの求めの版）・情動の求めでは [いま湧いたこと]（版の文）の 3 か所に入り、
断られた回が 64 件あった（`根拠台帳` §54）。断られると記憶を 1 件ずつ外して聞き直していた（出-bm・22:15 に 5 回）。

- 重なりは Jev へ送る文だけで除く。全文は [いま道具から返った] に 1 回、版の文からは同じ反復で返った結果の本文だけ外す。
- 送る前に字数を測り、上限（`jev_state_max_chars`・28,000 字・仮）を超えたら、収まるところまで一度に削る。
  調停は記憶を適合度の低い順に外す。続き先の判定は各項目を同じ字数まで縮める（行の数は減らさない）。
- 収まらなければ聞かずに倒し先へ倒す。断られても聞き直さない。主LLM の W は変えない。
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.loop import workspace
from familiar_agent.loop.request import Lookup, Request
from tests._arbiter_fakes import decide, writer_says
from tests.test_recent_window_is_a_workspace_frame import _oif, _said
from tests.test_workspace_is_the_core import _rec

RESULT = "晴れ時々くもり、最高二十五度。" * 40  # 600 字の検索結果
VERSION = "v-live"


def _req(*, returned_now: bool = True) -> Request:
    req = Request()
    req.request_text = "明日の天気は？"
    req.lookups.append(
        Lookup(index=1, action="search_deferred", query="明日 天気", generation=0, result=RESULT)
    )
    if returned_now:
        req.just_returned.append(1)
    req.live_version_id = VERSION
    return req


def _ws(memories=(), *, returned_now: bool = True):
    oif = _oif(["q1"], {"q1": [_said("q1", "直近のひとこと", "起点", 0)]})
    oif.roles = MagicMock(return_value={})
    oif.actors = MagicMock(return_value={})
    req = _req(returned_now=returned_now)
    live = _rec(VERSION, workspace.version_text(req), fit=0.95)
    return workspace.Workspace.build(oif, [live, *memories], req, n_arbiter=1, n_main=2)


# ── 版の文 ───────────────────────────────────────────────────────


def test_the_version_text_drops_only_the_bodies_returned_now():
    req = _req()
    req.lookups.append(
        Lookup(index=2, action="search_deferred", query="週末", generation=0, result="雨の週末")
    )
    text = workspace.version_text(req, omit=frozenset({1}))
    assert RESULT not in text
    assert "1番：search_deferred「明日 天気」の結果が届いた（中身は [いま道具から返った]）" in text
    assert "2番：search_deferred「週末」の結果が届いた：雨の週末" in text  # 前に返った結果は残す


def test_the_written_version_keeps_the_whole_result():
    """**反証の側**：O に書く版の文は全文のまま（組み方を移しても 1 字も変わらない）。"""
    req = _req()
    assert (
        workspace.version_text(req)
        == f"「明日の天気は？」と聞かれ、1番：search_deferred「明日 天気」の結果が届いた：{RESULT}"
    )


# ── 調停へ送る W ─────────────────────────────────────────────────


def test_the_result_appears_once_for_jev_and_twice_for_the_main_llm():
    ws = _ws()
    assert ws.for_jev_arbiter(28_000).count(RESULT) == 1
    assert ws.for_main.count(RESULT) == 2  # 主LLM の W は変えない（本人）


def test_a_result_from_an_earlier_iteration_stays_in_the_version():
    """**反証の側**：この反復で返っていない結果は、版にしか無いので残す。"""
    ws = _ws(returned_now=False)
    assert ws.for_jev_arbiter(28_000).count(RESULT) == 1


def test_nothing_is_dropped_within_the_limit():
    mems = [_rec("m1", "高い記憶", fit=0.9), _rec("m2", "低い記憶", fit=0.1)]
    text = _ws(mems).for_jev_arbiter(28_000)
    assert "高い記憶" in text and "低い記憶" in text


def test_the_lowest_fit_memories_go_at_once():
    big = "長い記憶の中身。" * 250  # 2,000 字
    mems = [
        _rec("m1", "高い" + big, fit=0.9),
        _rec("m2", "中くらい" + big, fit=0.5),
        _rec("m3", "低い" + big, fit=0.1),
    ]
    ws = _ws(mems)
    full = ws.for_jev_arbiter(1_000_000)
    limit = len(full) - 3_000  # 2 件外せば収まる
    text = ws.for_jev_arbiter(limit)
    assert text is not None and len(text) <= limit
    assert "高い" + big in text
    assert "中くらい" + big not in text and "低い" + big not in text
    assert RESULT in text and "直近のひとこと" in text  # 判断の中身と直近は削らない


def test_none_when_nothing_can_make_it_fit():
    assert _ws([_rec("m1", "記憶", fit=0.5)]).for_jev_arbiter(100) is None


def test_the_cue_for_jev_drops_the_returned_body():
    ws = _ws()
    assert RESULT not in ws.jev_cue()
    assert "（中身は [いま道具から返った]）" in ws.jev_cue()


# ── 続き先の判定へ送る W ─────────────────────────────────────────────


def test_follows_shrinks_each_item_and_keeps_every_row():
    from familiar_agent.core.jev_judges import workspace_rows

    big = "長い記憶の中身。" * 250
    mems = [_rec(f"m{i}", f"記憶{i}" + big, fit=0.5) for i in range(3)]
    ws = _ws(mems)
    whole = ws.for_jev_follows("さっきの話", 1_000_000)
    limit = len(whole) // 2
    text = ws.for_jev_follows("さっきの話", limit)
    assert text is not None and len(text) + len("さっきの話") <= limit
    assert set(workspace_rows(text)) == set(workspace_rows(whole))  # 選択肢の行は減らさない


def test_follows_is_untouched_within_the_limit():
    """**反証の側**：内側なら縮めない（重なりだけ除く）。"""
    ws = _ws([_rec("m1", "短い記憶", fit=0.5)])
    text = ws.for_jev_follows("さっきの話", 28_000)
    assert "短い記憶" in text and text.count(RESULT) == 1


# ── 調停：送る前に測る ───────────────────────────────────────────────

_RETURNED = (("search_deferred", False, RESULT),)


def _jev(*answers) -> MagicMock:
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(side_effect=list(answers))
    return jev


def _picks(choice: str) -> JevAnswer:
    return JevAnswer(ok=True, answers={"action": {"choice": choice, "confidence": 0.3}})


def _run(jev, jev_workspace, *, writer=None, jev_utterance=""):
    return asyncio.run(
        decide(
            jev=jev,
            writer=writer or writer_says({"text": "晴れだって"}),
            utterance="全文の版の文",
            jev_utterance=jev_utterance,
            origin="情動",
            fired_axis="seeking",
            returned=_RETURNED,
            workspace_ctx="全文の W",
            jev_workspace=jev_workspace,
            jev_max_chars=28_000,
        )
    )


def test_the_arbiter_sends_the_measured_workspace_once():
    budgets: list[int] = []

    def jev_ws(budget):
        budgets.append(budget)
        return "Jev 用の W"

    jev = _jev(_picks("silent"))
    _run(jev, jev_ws, jev_utterance="本文を外した版の文")
    sent = jev.ask.await_args.args[0]
    assert jev.ask.await_count == 1
    assert "Jev 用の W" in sent and "全文の W" not in sent
    assert "本文を外した版の文" in sent and "全文の版の文" not in sent
    assert 0 < budgets[0] < 28_000 and len(sent) <= 28_000


def test_too_long_is_not_asked(caplog):
    jev = _jev()
    with caplog.at_level("INFO"):
        d = _run(jev, lambda budget: None)
    assert jev.ask.await_count == 0
    assert d.branch == "full"
    assert "長すぎるので Jev に聞かなかった" in caplog.text


def test_a_refusal_is_not_asked_again():
    """断られても記憶を外して聞き直さない（出-bm を撤去）。倒し先へ倒す。"""
    jev = _jev(JevAnswer(ok=False, error="HTTP 400"), _picks("silent"))
    d = _run(jev, lambda budget: "Jev 用の W")
    assert jev.ask.await_count == 1
    assert d.branch == "full"


def test_the_writer_reads_the_whole_workspace():
    """**反証の側**：軽量LLM の書き手は、今までどおり全文の W を見る。"""
    writer = writer_says({"text": "晴れだって"})
    _run(_jev(_picks("talk_light")), lambda budget: "Jev 用の W", writer=writer)
    prompt = writer.complete.await_args.args[0]
    assert "全文の W" in prompt and "Jev 用の W" not in prompt


# ── 反復が口を渡す・設定 ─────────────────────────────────────────────


def test_the_iteration_hands_the_jev_view_to_the_arbiter(monkeypatch):
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
    view = got[0].jev_workspace
    assert getattr(view, "__func__", None) is workspace.Workspace.for_jev_arbiter
    assert got[0].jev_max_chars == 28_000


def test_the_limit_comes_from_config():
    from familiar_agent.config import AgentConfig

    with patch.dict(os.environ, {}, clear=True):
        assert AgentConfig().jev_state_max_chars == 28_000
