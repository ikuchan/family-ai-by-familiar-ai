"""W ではパジュ自身を呼び名で書く（記-x・2026-11-02・本人の決定）。

W を読むのは主LLM・軽量LLM・Jev・発話前の検査の 4 つで、指示文の立場がそろっていない（Jev の問いは「パジュは次に
どうするか」、軽量LLM は「あなたは家庭のロボット」）。「わたし」と書くと、誰を指すかが読み手の立場に依存する。
ME.md の呼び名の先頭で書けば、どの読み手にも同じ一人を指す。

- 呼び名が無い（ME.md に無い・テスト）ときは、いまどおり「わたし」（本人の決定ア）。
- 相手の判定は「わたし」の文字ではなく、主体の id（`AGENT_SELF_ID`）で見る。
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from unittest.mock import MagicMock

from familiar_agent.io.oif import MI, OIF, Recalled
from familiar_agent.loop import workspace
from familiar_agent.loop.request import Request
from familiar_agent.person_memory_manager import AGENT_SELF_ID
from familiar_agent.tools.memory import me_name, subject_line

NAMES = ["パジュ", "パジュちゃん"]


def _r(obs_id, content, direction="発話"):
    return Recalled(
        mi=MI(
            id=obs_id,
            obs_id=obs_id,
            content=content,
            timestamp=datetime(2026, 11, 2, 12, 0),
            direction=direction,
        ),
        fit=0.5,
        groundedness=1.0,
        confidence=0.8,
    )


def _oif(voices, roles, names):
    """本物の口（OIF）に、偽の記憶を渡す。呼び名は動いているエージェントの設定から毎回読む。"""
    mem = MagicMock()
    mem.voices_of = MagicMock(side_effect=lambda ids: {i: voices[i] for i in ids if i in voices})
    mem.exchange_roles_of = MagicMock(return_value=roles)
    return OIF(mem, names=lambda: names)


# ── 名前を決める所は 1 か所 ─────────────────────────────────────────────────


def test_the_name_is_the_first_call_name():
    assert me_name(NAMES) == "パジュ"


def test_without_names_it_is_watashi():
    """**反証の側**：呼び名が無ければ、いまどおり「わたし」（決定ア）。"""
    assert me_name([]) == "わたし"
    assert me_name(None) == "わたし"


# ── 想起の見出し ──────────────────────────────────────────────────────────


def test_a_recalled_own_line_is_headed_by_the_name():
    oif = _oif({"a": (AGENT_SELF_ID, [])}, {"a": "答え"}, NAMES)
    text, _ = workspace.compose(oif, [_r("a", "自分が答えた：はい")], Request())
    assert "パジュが言った" in text
    assert "わたしが言った" not in text


def test_an_unresolved_speaker_is_the_other_side_even_with_a_name():
    """**反証の側**：主体が自分でも役割が起点なら相手。文字で比べていれば「パジュが言った」になる。"""
    oif = _oif({"q": (AGENT_SELF_ID, [])}, {"q": "起点"}, NAMES)
    text, _ = workspace.compose(oif, [_r("q", "何が見えますか？")], Request())
    assert "相手が言った: 何が見えますか？" in text
    assert "パジュが言った" not in text


def test_without_names_the_heading_stays_watashi():
    """**反証の側**：呼び名が無ければ「わたしが言った」のまま。"""
    oif = _oif({"a": (AGENT_SELF_ID, [])}, {"a": "答え"}, [])
    text, _ = workspace.compose(oif, [_r("a", "自分が答えた：はい")], Request())
    assert "わたしが言った" in text


def test_a_renamed_agent_is_read_at_the_next_w():
    """ME.md を読み直して呼び名が変わったら、次の W から新しい名前（口は呼び名を毎回読む）。"""
    names = list(NAMES)
    oif = _oif({"a": (AGENT_SELF_ID, [])}, {"a": "答え"}, names)
    names[0] = "ポポ"
    text, _ = workspace.compose(oif, [_r("a", "自分が答えた：はい")], Request())
    assert "ポポが言った" in text


# ── 直近のやりとり ────────────────────────────────────────────────────────


def _row(obs_id, role, content):
    from datetime import timezone
    from types import SimpleNamespace

    return SimpleNamespace(
        obs_id=obs_id,
        role=role,
        content=content,
        when=datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc),
        direction="発話",
    )


def test_my_recent_answer_is_named():
    oif = _oif({"o1": (AGENT_SELF_ID, ["パパ"])}, {}, NAMES)
    rows = [_row("o1", "答え", "自分が答えた：おかえりなさい")]
    _, text, _ = workspace.render_recent(oif, [("o1", rows)], 3)
    assert "パジュ（パパへ）：" in text
    assert "わたし" not in text


def test_an_unresolved_recent_origin_is_the_other_side():
    """**反証の側**：起点の行の主体が自分（話者が分からない）なら「相手」。名前は書かない。"""
    oif = _oif({"q": (AGENT_SELF_ID, [])}, {}, NAMES)
    rows = [_row("q", "起点", "何が見えますか？")]
    _, text, _ = workspace.render_recent(oif, [("q", rows)], 3)
    assert "相手：何が見えますか？" in text
    assert "パジュ" not in text


# ── 窓の会話 ──────────────────────────────────────────────────────────────


def _ip(names):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    agent = _agent(stream_returns=[])
    agent.config.agent_names = names
    ip = InformationProcessing(agent)
    ip._current_speaker_name = lambda: "パパ"
    ip._wake_window().open(time.monotonic())
    return ip


def _talk(ip):
    return [line.split(" ", 2)[2] for line in ip._window_talk().lines]


def _spoken(ip):
    ip._req.trigger_kind = "発話"

    async def go():
        await ip._finish("めだか！", [], "発話")
        await ip.settle_finals()
        await ip.close()

    asyncio.run(go())
    return _talk(ip)


def test_a_spoken_answer_in_the_window_is_named():
    """名前は動いているエージェントの設定から読む（`AgentConfig()` を作り直す道なら空で「わたし」に落ちる）。"""
    assert _spoken(_ip(NAMES)) == ["パジュ（パパへ）：めだか！"]


def test_without_names_the_window_line_stays_watashi():
    """**反証の側**：呼び名が無ければ、いまどおり。"""
    assert _spoken(_ip([])) == ["わたし（パパへ）：めだか！"]


# ── 見たものの在りかの案内 ───────────────────────────────────────────────────


def test_the_pointer_to_what_was_seen_matches_the_heading():
    """案内が指す見出しは、観察の見出しと同じ関数で作る（ずれると、無い行を探させる）。"""
    import inspect

    from familiar_agent.loop.event_loop import InformationProcessing

    src = inspect.getsource(InformationProcessing._intake)
    assert "わたしが見た" not in src
    assert 'subject_line("観察"' in src
    assert subject_line("観察", "パジュ") == "パジュが見た"
