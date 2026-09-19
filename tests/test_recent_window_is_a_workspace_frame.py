"""直近のやりとりは W の枠（記-h・`設計方針_MI間の関係` v0.15 段 4 改訂）。

時系列で最新 n 往復（無条件）＋ 各々から継起の辺があるぶんさかのぼった鎖。組むのは
`workspace` の 1 箇所で、調停と主LLM は同じ作り方の W を受け取る（差は窓 n だけ）。
いまの反復の続き先の判定は待たず、辺を書くだけで W の中身を決めない。

実機で、こうきと話した直後に待たされていた入室の反復が起き、直近が空のまま
「おかえり、こうき！」と挨拶した（2026-09-13・F）。継起だけを頼ると話題が切り替わった
瞬間に直近が空になる。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from familiar_agent.io.oif import Said
from familiar_agent.loop import workspace

_T0 = datetime(2026, 9, 13, 17, 30, tzinfo=timezone.utc)


def _said(
    obs_id: str, content: str, role: str, minute: int, depth: int = 0, direction: str = "発話"
) -> Said:
    return Said(
        obs_id=obs_id,
        content=content,
        role=role,
        when=_T0 + timedelta(minutes=minute),
        depth=depth,
        direction=direction,
    )


def _oif(origins: list[str], chains: dict[str, list[Said]], names: dict[str, str] | None = None):
    """`latest_origins(n)` と `exchanges(origin)` を持つ偽の口。"""
    oif = MagicMock()
    oif.latest_origins = MagicMock(side_effect=lambda n: origins[:n])
    oif.exchanges = MagicMock(side_effect=lambda o: chains.get(o, []))
    oif.actors = MagicMock(return_value=names or {})
    return oif


def test_the_latest_exchanges_are_shown_in_time_order_without_edges():
    # 3 往復・辺なし。新しい 2 つが古い順に、逐語で載る。
    chains = {
        "q1": [_said("q1", "今日の天気は？", "起点", 0), _said("a1", "晴れだよ", "答え", 1)],
        "q2": [_said("q2", "予定は？", "起点", 5), _said("a2", "フットサルだよ", "答え", 6)],
        "q3": [_said("q3", "お話できる？", "起点", 9), _said("a3", "もちろん", "答え", 10)],
    }
    rows, text, id_map = workspace.recent_window(_oif(["q3", "q2", "q1"], chains), 2)
    assert [r.obs_id for r in rows] == ["q2", "a2", "q3", "a3"]
    assert "今日の天気は？" not in text and "お話できる？" in text
    assert text.index("予定は？") < text.index("お話できる？")


def test_a_chain_reaches_beyond_the_window_through_succession_edges():
    # 最新 1 往復（q3）が q1 の続き（辺あり）なら、q1 も載る。q2 は窓の外で辺も無い。
    chains = {
        "q3": [
            _said("q1", "サッカーの話", "起点", 0, depth=1),
            _said("a1", "いいね", "答え", 1, depth=1),
            _said("q3", "その続きだけど", "起点", 9, depth=0),
            _said("a3", "うん", "答え", 10, depth=0),
        ],
        "q2": [_said("q2", "予定は？", "起点", 5)],
    }
    rows, text, _ = workspace.recent_window(_oif(["q3", "q2"], chains), 1)
    assert [r.obs_id for r in rows] == ["q1", "a1", "q3", "a3"]
    assert "予定は？" not in text


def test_a_record_reached_twice_is_shown_once():
    # q3 と q2 が両方 q1 に繋がっていても、q1 は 1 度だけ。
    q1 = [_said("q1", "根の話", "起点", 0, depth=1)]
    chains = {
        "q3": q1 + [_said("q3", "三", "起点", 9)],
        "q2": q1 + [_said("q2", "二", "起点", 5)],
    }
    rows, text, _ = workspace.recent_window(_oif(["q3", "q2"], chains), 2)
    assert [r.obs_id for r in rows] == ["q1", "q2", "q3"]
    assert text.count("根の話") == 1


def test_each_line_carries_a_twelve_digit_id_and_who_said_it():
    # 判定（続き先）と申告が直近の記録も名指せるように、12 桁の id を印字し対応表に入れる。
    chains = {
        "3f2b9c1d8e7a6b5c4d3e2f1a0b9c": [
            _said("3f2b9c1d8e7a6b5c4d3e2f1a0b9c", "お話できる？", "起点", 0),
            _said("a3", "もちろん", "答え", 1),
        ]
    }
    oif = _oif(
        ["3f2b9c1d8e7a6b5c4d3e2f1a0b9c"], chains, names={"3f2b9c1d8e7a6b5c4d3e2f1a0b9c": "こうき"}
    )
    _, text, id_map = workspace.recent_window(oif, 1)
    assert "id:3f2b9c1d8e7a" in text and id_map["3f2b9c1d8e7a"] == "3f2b9c1d8e7a6b5c4d3e2f1a0b9c"
    assert "こうき：お話できる？" in text
    assert "わたし：もちろん" in text


def test_an_origin_that_is_not_a_persons_words_is_not_labelled_as_the_other_side():
    """情動・入室が起点のやりとりは「相手：」ではなく「きっかけ：」（実機 2026-09-13 21:21）。"""
    chains = {
        "q1": [
            _said("q1", "[内的な促し:SEEKING] 探索したい", "起点", 0, direction="情動"),
            _said("a1", "見てみよう", "答え", 1),
        ],
        "q2": [
            _said("q2", "[入室] こうき が来た", "起点", 5, direction="機器"),
            _said("a2", "おかえり", "答え", 6),
        ],
        "q3": [_said("q3", "おはなしできる？", "起点", 9), _said("a3", "もちろん", "答え", 10)],
    }
    _, text, _ = workspace.recent_window(_oif(["q3", "q2", "q1"], chains), 3)
    assert "きっかけ：[内的な促し:SEEKING] 探索したい" in text
    assert "きっかけ：[入室] こうき が来た" in text
    assert "相手：おはなしできる？" in text
    assert text.count("相手：") == 1


def test_zero_or_no_exchanges_gives_an_empty_frame():
    rows, text, id_map = workspace.recent_window(_oif([], {}), 3)
    assert rows == [] and text == "" and id_map == {}
    rows, text, _ = workspace.recent_window(_oif(["q1"], {"q1": [_said("q1", "x", "起点", 0)]}), 0)
    assert rows == [] and text == ""


# ── W は 3 枠・組むのは 1 箇所・窓だけが違う ─────────────────────────────────


def _recalled(obs_id: str, content: str):
    from tests.test_workspace_is_the_core import _rec

    return _rec(obs_id, content)


def _ws(oif, memories, *, n_arbiter=1, n_main=2):
    from familiar_agent.loop.request import Request

    return workspace.Workspace.build(oif, memories, Request(), n_arbiter=n_arbiter, n_main=n_main)


def test_the_workspace_has_three_frames_in_order_and_no_duplicate():
    chains = {
        "q1": [_said("q1", "今日の天気は？", "起点", 0), _said("a1", "晴れだよ", "答え", 1)],
        "q2": [_said("q2", "お話できる？", "起点", 5), _said("a2", "もちろん", "答え", 6)],
    }
    oif = _oif(["q2", "q1"], chains)
    oif.roles = MagicMock(return_value={})
    # 想起は a2（直近にも居る）と、古い記憶 m9 を返した。
    ws = _ws(oif, [_recalled("a2", "もちろん"), _recalled("m9", "去年の夏の話")], n_main=2)
    text = ws.render(2)
    assert text.index("[直近のやりとり") < text.index("[過去の記憶")
    assert text.count("もちろん") == 1, "直近に載った記録が過去の列にも出ている"
    assert "去年の夏の話" in text
    # 対応表は両方の枠を含む。
    assert {"q1", "q2", "a1", "a2", "m9"} <= set(ws.id_map.values())


def test_the_two_windows_are_the_same_build_with_a_different_width():
    chains = {
        "q1": [_said("q1", "一", "起点", 0)],
        "q2": [_said("q2", "二", "起点", 5)],
        "q3": [_said("q3", "三", "起点", 9)],
    }
    oif = _oif(["q3", "q2", "q1"], chains)
    oif.roles = MagicMock(return_value={})
    ws = _ws(oif, [_recalled("m9", "昔")], n_arbiter=1, n_main=3)
    narrow, wide = ws.render(1), ws.render(3)
    assert "三" in narrow and "二" not in narrow and "一" not in narrow
    assert "三" in wide and "二" in wide and "一" in wide
    # 直近以外（過去の記憶・言った一言）は同じ。
    assert narrow.split("[過去の記憶")[1] == wide.split("[過去の記憶")[1]
    # 直近は 1 度しか引かない（窓の最大で引いて、狭い方は切り出す）。
    assert oif.latest_origins.call_count == 1


def test_the_verdict_map_holds_only_the_past_column():
    """申告の母数は過去の記憶の列だけ（出-n 4・2026-09-13）。

    記-h で直近の行にも id を印字し対応表に入れたため `記憶の判定 7/19 件` のように母数が
    増えた。直近は無条件に載せたもので、大事／不要を申告させて根づきを動かす意味がない。
    判定（続き先）は両方を名指せるので `id_map` は両方を持つ。
    """
    chains = {"q2": [_said("q2", "お話できる？", "起点", 5), _said("a2", "もちろん", "答え", 6)]}
    oif = _oif(["q2"], chains)
    oif.roles = MagicMock(return_value={})
    ws = _ws(oif, [_recalled("a2", "もちろん"), _recalled("m9", "去年の夏の話")], n_main=2)
    assert set(ws.verdict_map.values()) == {"m9"}  # a2 は直近の枠に載ったので過去の列に無い
    assert set(ws.id_map.values()) == {"q2", "a2", "m9"}


# ── いま道具から返ったもの（出-x・2026-09-18）────────────────────────────────


def test_a_tool_return_that_just_arrived_is_shown_at_the_top():
    """道具の返りは版（過去の列の全文）に載るだけで、調停は「記憶」として読み、いま届いた返りとして
    扱わなかった——「確かめて」の返りに聞き返さず `set_timer` を掛け直した（実機 14:50）。
    実験（`scripts/experiment_arbiter_confirm.py`）：最上部に 1 行載せると 8/8 で light の確認文。"""
    from familiar_agent.loop.request import Lookup, Request

    chains = {"q1": [_said("q1", "三分測って", "起点", 0)]}
    oif = _oif(["q1"], chains)
    oif.roles = MagicMock(return_value={})
    req = Request()
    req.lookups.append(
        Lookup(
            index=1,
            action="set_timer",
            query="タイマーを掛ける「パパの頼み」",
            generation=0,
            result="まだ掛けていない。本人に一度聞く：「3 分のタイマーね、いい？」",
        )
    )
    req.just_returned.append(1)  # 取込がこの反復で受けた返り（次の反復の頭で空になる）
    ws = workspace.Workspace.build(oif, [], req, n_arbiter=1, n_main=2)
    req.just_returned.clear()  # 組んだあとに空にしても（`_iterate` はそうする）枠は残る（遅延評価で消えた・実機 15:28）
    text = ws.render(1)
    assert text.startswith("[いま道具から返った]")
    assert "set_timer" in text and "本人に一度聞く" in text
    assert text.index("[いま道具から返った]") < text.index("[直近のやりとり")


def test_no_block_when_nothing_just_returned():
    from familiar_agent.loop.request import Lookup, Request

    chains = {"q1": [_said("q1", "三分測って", "起点", 0)]}
    oif = _oif(["q1"], chains)
    oif.roles = MagicMock(return_value={})
    req = Request()
    req.lookups.append(
        Lookup(index=1, action="set_timer", query="x", generation=0, result="掛けた")
    )
    # 前の反復で受けた返り（`just_returned` は空）→ 載せない（版の全文にはある）
    assert "[いま道具から返った]" not in workspace.Workspace.build(
        oif, [], req, n_arbiter=1, n_main=2
    ).render(1)


def test_intake_marks_the_returns_and_the_next_iteration_clears_them():
    import asyncio

    from familiar_agent.loop.event_loop import InformationProcessing, Lookup, Trigger

    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.lookups.append(
        Lookup(index=1, action="set_timer", query="タイマーを掛ける「x」", generation=0)
    )
    ip._drained_completions.append(
        Trigger(kind="完了", query="タイマーを掛ける「x」", result="まだ掛けていない", index=1)
    )
    asyncio.run(ip._intake())
    assert ip._req.just_returned == [1]
    ip._req.just_returned.clear()  # `_iterate` が W を組んだあとに空にする（本体は次の test）


# ── 直近の窓には時間の上限もある（出-ae(2)・2026-09-19）───────────────────────


def test_rows_older_than_the_time_cap_are_not_shown():
    """再起動をまたいで 1 時間半前の「パパ、…」が最上部に居続け、調停が名前を埋めた（実機 13:14）。"""
    chains = {
        "q1": [_said("q1", "今日の天気は？", "起点", 0), _said("a1", "パパ、晴れだよ", "答え", 1)],
        "q2": [_said("q2", "予定は？", "起点", 7), _said("a2", "フットサルだよ", "答え", 8)],
        "q3": [_said("q3", "お話できる？", "起点", 9), _said("a3", "もちろん", "答え", 10)],
    }
    now = _T0 + timedelta(minutes=11)
    rows, text, id_map = workspace.recent_window(
        _oif(["q3", "q2", "q1"], chains), 3, max_age_sec=300, now=now
    )
    assert [r.obs_id for r in rows] == ["q2", "a2", "q3", "a3"]
    assert "パパ" not in text
    assert "q1" not in id_map.values()


def test_the_time_cap_also_cuts_the_chain_reached_through_edges():
    # 続きの鎖でさかのぼった古い行も、上限より古ければ載せない（想起の列が担う）。
    chains = {
        "q3": [
            _said("q1", "サッカーの話", "起点", 0, depth=1),
            _said("q3", "その続きだけど", "起点", 9, depth=0),
        ]
    }
    now = _T0 + timedelta(minutes=10)
    rows, text, _ = workspace.recent_window(_oif(["q3"], chains), 1, max_age_sec=300, now=now)
    assert [r.obs_id for r in rows] == ["q3"]
    assert "サッカー" not in text


def test_a_zero_cap_means_no_time_limit():
    chains = {"q1": [_said("q1", "今日の天気は？", "起点", 0)]}
    now = _T0 + timedelta(days=3)
    rows, _, _ = workspace.recent_window(_oif(["q1"], chains), 1, max_age_sec=0, now=now)
    assert [r.obs_id for r in rows] == ["q1"]


def test_the_workspace_uses_the_configured_cap(monkeypatch):
    """`recall` が組む W は `MemoryConfig.recent_exchanges_max_sec`（既定 300）で直近を切る。"""
    import asyncio

    from familiar_agent.config import MemoryConfig
    from familiar_agent.loop.request import Request

    assert MemoryConfig().recent_exchanges_max_sec == 300
    chains = {"q1": [_said("q1", "パパ、今日の天気は？", "起点", 0)]}
    oif = _oif(["q1"], chains)
    oif.roles = MagicMock(return_value={})

    async def _recall(_cue, _view):
        return []

    oif.recall = _recall
    monkeypatch.setattr(workspace.clock, "now_utc", lambda: _T0 + timedelta(hours=1))
    ws = asyncio.run(workspace.recall(oif, "天気", viewpoint="v", weights=None, req=Request()))
    assert ws.max_age_sec == 300
    assert "パパ" not in ws.render(2)
    assert "q1" not in ws.id_map.values()
    # 上限なしなら載る（時計だけの違い）。
    ws0 = _ws(oif, [], n_main=2)
    assert "パパ" in ws0.render(2)
