"""「しっかり考えて」で、別の高度なモデルに調べさせて答えをそのまま話す（出-dh 段 1・2026-11-02・本人の決定）。

調べもの（`search_deferred`）と並ぶ動作 `deep_research` を足す。投げると裏で別モデル（既定は Claude Fable 5.1・考える深さ low）が
Anthropic の側の Web 検索（5 回まで）で調べて考え、終わったら完了で届く。答えは主LLM に書き直させず、そのまま声にする（目標
400 字・800 字以内を別モデルに書かせる）。2 分で打ち切り、1 つの求めで 1 回まで。値は `config.py` の既定値（`.env` は読まない）。
段 0（2026-11-02・`jev-1.14.0`・3 回）：「しっかり考えて」「じっくり調べて」「時間かけていいから詳しく」は新しい動作 0.99、
ふつうの「〜調べて」は `search_deferred` 0.96〜1.00。決めた指示で記号は 0、出どころの言い方はあり、申し出で終わらなかった。
**本物の API は呼ばない**（回帰テストで毎回お金をかけない・本人）。別モデルの返りは偽物で作る。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from familiar_agent.tools import deep_research as dr

ME = "名前：パジュ、パージュ\n\n## 話し方\n- 大人には：ていねいに話す\n- 子どもには：やさしく話す\n"


# ── 指示 ─────────────────────────────────────────────────────────────


def test_the_system_prompt_reads_me_md_and_keeps_the_rules():
    system = dr.system_prompt(ME, self_text="自分の要約", target=400, limit=800, adult=True)
    assert "「パジュ」" in system and "自分の要約" in system
    assert "ていねいに話す" in system  # 話し方の行は ME.md から
    assert "必ず 800 字以内" in system and "400 字" in system
    assert "記号" in system and "信頼できる" in system
    assert "申し出" in system  # 申し出で終わらない（出-dj の預かりが効かないため）
    assert "最後にまとめて書く" in system  # 検索のあいだに答えを書くと、最後の段しか残らない


def test_the_name_is_not_written_in_the_code():
    """**反証の側**：名前は ME.md から読む。ME.md の名前が違えば、指示の名前も違う。"""
    system = dr.system_prompt("名前：ほかの子\n", self_text="", target=400, limit=800, adult=True)
    assert "「ほかの子」" in system and "パジュ" not in system


# ── 呼び出し（偽物）───────────────────────────────────────────────────────


def _resp(text: str, stop: str = "end_turn", searches: int = 1):
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=200,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        server_tool_use=SimpleNamespace(web_search_requests=searches),
    )
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason=stop, usage=usage
    )


class _Client:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kw):
        self.calls.append(kw)
        r = self._responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def _ask(client, **kw):
    params = dict(
        model="claude-fable-5-1",
        effort="low",
        max_searches=5,
        max_tokens=4096,
        system="指示",
        user="頼み",
    )
    params.update(kw)
    return asyncio.run(dr.research(client, **params))


def test_the_call_carries_the_config_values():
    client = _Client(_resp("答えです。"))
    got = _ask(client)
    kw = client.calls[0]
    assert kw["model"] == "claude-fable-5-1" and kw["max_tokens"] == 4096
    assert kw["output_config"] == {"effort": "low"}
    assert kw["tools"][0]["type"] == "web_search_20260209" and kw["tools"][0]["max_uses"] == 5
    assert (got.text, got.failed) == ("答えです。", False)


def test_pause_turn_is_continued_and_the_text_joined():
    client = _Client(_resp("前半。", stop="pause_turn"), _resp("後半。"))
    got = _ask(client)
    assert got.text == "前半。後半。" and len(client.calls) == 2
    assert got.searches == 2


@pytest.mark.parametrize(
    "responses",
    [(_resp("", stop="refusal"),), (RuntimeError("つながらない"),), (_resp("  "),)],
)
def test_a_refusal_an_error_or_nothing_is_a_failure(responses):
    """**反証の側**：断られた・落ちた・何も書かなかったは失敗。"""
    got = _ask(_Client(*responses))
    assert got.failed is True


# ── 調停の候補 ─────────────────────────────────────────────────────────


def test_research_offers_deep_only_when_available():
    from familiar_agent.core import utterance_meaning as um

    qs = um.fanout_questions(music=False, camera=False, family=["パパ"], tools={"deep_research"})
    assert "deep_research" in qs["action_research"]["criteria"]
    qs = um.fanout_questions(music=False, camera=False, family=["パパ"], tools=set())
    assert "deep_research" not in qs["action_research"]["criteria"]  # **反証の側**


def test_the_arbiter_throws_deep_research(monkeypatch):
    from familiar_agent.store import arbiter_records
    from tests._arbiter_fakes import decide, writer_says
    from tests.test_arbiter_utterance_by_meaning import _a, _jev

    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    jev = _jev({"meaning": _a("research", 0.9), "action_research": _a("deep_research", 0.95)})
    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer_says({"query": "子どもの読書の習慣"}),
            utterance="しっかり考えて、子どもの読書の習慣のつけ方を調べて",
            origin="発話",
            extra_actions=("deep_research",),
        )
    )
    assert (d.branch, d.action) == ("action", "deep_research")


# ── 完了：そのまま話す ───────────────────────────────────────────────────


def test_the_completion_is_spoken_as_it_is(monkeypatch):
    from familiar_agent.store import arbiter_records
    from tests._arbiter_fakes import decide, writer_says

    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    writer = writer_says({"text": "書き手の文"})
    d = asyncio.run(
        decide(
            jev=MagicMock(),
            writer=writer,
            utterance="しっかり考えて",
            origin="発話",
            returned=(("deep_research", False, "調べた答えです。"),),
        )
    )
    assert (d.branch, d.text) == ("light", "調べた答えです。")
    writer.complete.assert_not_awaited()  # 主LLM も書き手も通さない


def test_a_failed_deep_research_is_told_lightly():
    """**反証の側**：失敗は、いまどおり軽く伝える。"""
    from familiar_agent.core.completion_kind import kind_of

    got = kind_of("deep_research", failed=True, result="時間内に調べきれなかった", origin="発話")
    assert got is not None and got[1] == ("tell_light",)


# ── ループ ──────────────────────────────────────────────────────────────


def _ip(platform: str = "anthropic"):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a.config.platform = platform
    a.config.api_key = "鍵"
    a.config.deep_model = "claude-fable-5-1"
    a.config.deep_effort = "low"
    a.config.deep_max_searches = 5
    a.config.deep_max_tokens = 4096
    a.config.deep_timeout_sec = 120.0
    a.config.deep_reply_target = 400
    a.config.deep_reply_limit = 800
    ip = InformationProcessing(a)
    ip._req.trigger_kind = "発話"
    ip._req.request_text = "しっかり考えて、読書の習慣を調べて"
    return ip


def test_deep_is_offered_once_and_only_on_anthropic():
    ip = _ip()
    assert "deep_research" in ip._extra_actions()
    ip._req.spent_actions.add("deep_research")
    assert "deep_research" not in ip._extra_actions()  # 1 つの求めで 1 回まで
    assert "deep_research" not in _ip("gemini")._extra_actions()  # **反証の側**：鍵の担い手が違う


def _run_body(ip, answer):
    from familiar_agent.loop.event_loop import Lookup

    seen: dict = {}

    async def fake_research(client, **kw):
        seen.update(kw)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    async def go():
        ip._req.lookups.append(
            Lookup(index=1, action="deep_research", query="読書の習慣", generation=0)
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(dr, "research", fake_research)
            await ip._run_lookup_body("deep_research", {}, "読書の習慣", None, 1)
        t = ip._triggers.get_nowait()
        await ip.close()
        return t

    return asyncio.run(go()), seen


def test_the_loop_runs_deep_research_with_the_config():
    ok = dr.Answer(text="調べた答えです。", failed=False, why="", seconds=1.0, searches=2)
    t, seen = _run_body(_ip(), ok)
    assert (t.kind, t.result, t.failed) == ("完了", "調べた答えです。", False)
    assert seen["model"] == "claude-fable-5-1" and seen["effort"] == "low"
    assert seen["max_searches"] == 5 and seen["max_tokens"] == 4096
    assert "読書の習慣を調べて" in seen["user"]  # 頼みの言葉を渡す


def test_a_timeout_is_a_failed_completion():
    """**反証の側**：2 分を過ぎたら、失敗の完了にする。"""
    t, _ = _run_body(_ip(), asyncio.TimeoutError())
    assert t.failed is True and "調べきれなかった" in t.result


def test_dispatching_spends_the_one_time():
    ip = _ip()

    async def go():
        ip._dispatch_lookup("deep_research", {}, "読書の習慣", None)
        await ip.close()

    with pytest.MonkeyPatch.context() as mp:

        async def nothing(*a, **k):
            return None

        mp.setattr(ip, "_run_lookup", nothing)
        asyncio.run(go())
    assert "deep_research" in ip._req.spent_actions


def test_waiting_for_it_counts_and_a_name_stops_it():
    """投げたら待っているものに数える（電子音が鳴る・声を聞かない）。名前で打ち切れば、別モデルの呼び出しも止まる。"""
    ip = _ip()
    stopped: list = []
    started = asyncio.Event()

    async def slow_research(client, **kw):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            stopped.append(True)
            raise

    async def go():
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(dr, "research", slow_research)
            ip._dispatch_lookup("deep_research", {}, "読書の習慣", None)
            await asyncio.wait_for(started.wait(), timeout=5)
            waiting = [lk.action for lk in ip._waiting_on()]
            await ip._abort_lookups()
            await asyncio.sleep(0.05)
        await ip.close()
        return waiting

    assert asyncio.run(go()) == ["deep_research"]
    assert stopped == [True]


# ── 見直しで足した確かめ（1 回目）───────────────────────────────────────────


def _blocks(*blocks, stop="end_turn", searches=1):
    r = _resp("", stop=stop, searches=searches)
    r.content = [
        SimpleNamespace(type="text", text=b) if isinstance(b, str) else SimpleNamespace(type=b[0])
        for b in blocks
    ]
    return r


def test_the_preamble_before_the_search_is_not_spoken():
    client = _Client(
        _blocks(
            "最新の情報を調べますね。", ("server_tool_use",), ("web_search_tool_result",), "答え。"
        )
    )
    assert _ask(client).text == "答え。"


@pytest.mark.parametrize(
    "responses",
    [
        (_resp("途中まで", stop="max_tokens"),),
        tuple(_resp("前置き", stop="pause_turn") for _ in range(5)),
    ],
)
def test_a_cut_answer_is_a_failure(responses):
    """出力の上限で切れた・続きを受けきれなかった答えは、尻切れのまま話さない。"""
    got = _ask(_Client(*responses))
    assert got.failed is True and got.text == ""


def test_searches_are_capped_across_continuations():
    client = _Client(_resp("", stop="pause_turn", searches=3), _resp("答え。", searches=2))
    _ask(client)
    assert client.calls[0]["tools"][0]["max_uses"] == 5
    assert client.calls[1]["tools"][0]["max_uses"] == 2  # 合わせて 5 回まで


def _dispatch(ip):
    ran: list = []

    async def run(*a, **k):
        ran.append(a)

    async def go():
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ip, "_run_lookup", run)
            ip._dispatch_lookup("deep_research", {}, "読書の習慣", None)
            await asyncio.sleep(0)
        got = None if ip._triggers.empty() else ip._triggers.get_nowait()
        await ip.close()
        return got

    return asyncio.run(go()), ran


def test_the_throwing_side_also_keeps_once_and_anthropic():
    """主LLM が文で書いた呼び出しも投げる口を通るので、そこでも 1 回まで・Anthropic のときだけ。"""
    ip = _ip()
    ip._req.spent_actions.add("deep_research")
    got, ran = _dispatch(ip)
    assert ran == [] and got is not None and got.failed is True
    got, ran = _dispatch(_ip("gemini"))
    assert ran == [] and got is not None and got.failed is True


def test_an_earlier_search_with_the_same_words_does_not_block_it():
    from familiar_agent.loop.event_loop import Lookup

    ip = _ip()
    ip._req.lookups.append(
        Lookup(
            index=1, action="search_deferred", query="読書の習慣", generation=0, result="前の結果"
        )
    )
    got, ran = _dispatch(ip)
    assert len(ran) == 1 and got is None


def test_the_platform_is_read_like_the_backend_factory():
    """担い手の名前が空でも、ほかの名前でなければ Anthropic（`create_backend` と同じ決め方）。"""
    assert _ip("")._deep_available() is True
    assert _ip("openai")._deep_available() is False


# ── 見直しで足した確かめ（2 回目）───────────────────────────────────────────


def test_no_continuation_after_the_searches_are_used_up():
    """検索を使い切ったのに続きを求められたら、道具を外して頼まず、切れた扱いの失敗にする。"""
    client = _Client(_resp("", stop="pause_turn", searches=5), _resp("答え。"))
    got = _ask(client)
    assert len(client.calls) == 1 and got.failed is True


def test_a_second_call_while_the_first_runs_queues_nothing():
    """1 回目が走っているあいだの 2 回目は、失敗を積まない（それが届いた反復で答える）。"""
    from familiar_agent.loop.event_loop import Lookup

    ip = _ip()
    ip._req.spent_actions.add("deep_research")
    ip._req.lookups.append(
        Lookup(index=1, action="deep_research", query="前の頼み（深く調べる）", generation=0)
    )
    got, ran = _dispatch(ip)
    assert ran == [] and got is None


def test_its_box_is_labelled_apart_from_a_search():
    """器の見出しを分ける（同じ語の調べものと取り違えない）。"""
    ip = _ip()
    _dispatch(ip)
    assert [lk.query for lk in ip._req.lookups] == ["読書の習慣（深く調べる）"]
