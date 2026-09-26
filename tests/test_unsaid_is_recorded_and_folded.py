"""言いたかったことを、上がってくる形で記録し、伝えたら畳む（出-as 段 7・2026-09-26・`設計方針_話していいかの決まり` §2.6）。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip(*, speaker_id="", speaker_name=""):
    a = _agent(stream_returns=[])
    a._oif = MagicMock()
    a._oif.write = AsyncMock(return_value="obs-new")
    a._oif.set_groundedness = MagicMock(return_value=1)
    a._oif.supersede = MagicMock(return_value=True)
    a.speaker_known = MagicMock(return_value=bool(speaker_id))
    a._pmm.current_speaker_id = speaker_id or None
    ip = InformationProcessing(a)
    ip._current_speaker_name = lambda: speaker_name
    return a, ip


def _written(a):
    call = a._oif.write.await_args
    return call.args[0], call.kwargs


# ── 記録 ───────────────────────────────────────────────────────────────────


def test_blocked_words_go_to_the_listeners_face_with_weight_two():
    a, ip = _ip(speaker_id="p-papa", speaker_name="パパ")
    asyncio.run(ip._finish("明日は雨だよ", [], "独白"))
    mi, kw = _written(a)
    assert mi.content == "パパに言いたかったこと：明日は雨だよ"
    assert mi.direction == "独白"
    assert kw["participants"] == ["p-papa"]  # 相手の面（書き手はパジュ自身）
    a._oif.set_groundedness.assert_called_once_with("obs-new", 2)


def test_an_unknown_listener_is_someone_on_pajus_own_face():
    a, ip = _ip()
    asyncio.run(ip._finish("ねえねえ", [], "独白"))
    mi, kw = _written(a)
    assert mi.content == "誰かに言いたかったこと：ねえねえ"
    assert kw == a._observation_perspective()
    a._oif.set_groundedness.assert_called_once_with("obs-new", 2)


def test_a_plain_monologue_stays_as_it_was():
    """声にしなかっただけの独り言（地の文だけ・結末「沈黙」）は、いまのまま。"""
    a, ip = _ip(speaker_id="p-papa", speaker_name="パパ")
    asyncio.run(ip._finish("静かにしておこう", [], "沈黙"))
    mi, _ = _written(a)
    assert mi.content == "考えたが言わなかった：静かにしておこう"
    a._oif.set_groundedness.assert_not_called()


# ── 畳む ───────────────────────────────────────────────────────────────────


def test_folding_replaces_it_and_resets_the_weight():
    a, ip = _ip()
    asyncio.run(ip._fold_told([("obs-old", "[そばに居た] パパに言いたかったこと：明日は雨だよ")]))
    mi, _ = _written(a)
    assert mi.content == "伝えた：パパに言いたかったこと：明日は雨だよ"
    a._oif.supersede.assert_called_once()
    assert a._oif.supersede.call_args.args[:2] == ("obs-old", "obs-new")
    a._oif.set_groundedness.assert_called_once_with("obs-old", 0, lower=True)


def test_a_spoken_reply_that_used_it_folds_it():
    a, ip = _ip()
    ip._req.told_unsaid = [("obs-old", "パパに言いたかったこと：明日は雨だよ")]
    ip._fold_told = AsyncMock()
    asyncio.run(ip._finish("そういえば明日は雨だよ", [], "発話"))
    ip._fold_told.assert_awaited_once_with([("obs-old", "パパに言いたかったこと：明日は雨だよ")])


def test_an_unspoken_reply_does_not_fold_it():
    a, ip = _ip()
    ip._req.told_unsaid = [("obs-old", "パパに言いたかったこと：明日は雨だよ")]
    ip._fold_told = AsyncMock()
    asyncio.run(ip._finish("そういえば明日は雨だよ", [], "独白"))
    ip._fold_told.assert_not_awaited()


def test_the_light_declaration_folds_what_it_used(monkeypatch):
    a, ip = _ip()
    memories = [
        SimpleNamespace(
            mi=SimpleNamespace(obs_id="aaaaaaaaaaaa-1", content="パパに言いたかったこと：雨")
        )
    ]
    monkeypatch.setattr(
        "familiar_agent.loop.workspace.ask_verdicts",
        AsyncMock(return_value=[{"id": "aaaaaaaaaaaa", "verdict": "referred"}]),
    )
    monkeypatch.setattr("familiar_agent.loop.workspace.apply_memory_verdicts", lambda *a, **k: None)
    ip._fold_told = AsyncMock()

    async def go():
        ip._declare_light_memory_use(
            utterance="おはよう",
            reply="明日は雨だって",
            workspace_ctx="",
            w_id_map={"aaaaaaaaaaaa": "aaaaaaaaaaaa-1"},
            mem=MagicMock(),
            memories=memories,
            spoken=True,
        )
        await asyncio.gather(*list(ip._verdict_tasks))

    asyncio.run(go())
    ip._fold_told.assert_awaited_once_with([("aaaaaaaaaaaa-1", "パパに言いたかったこと：雨")])
