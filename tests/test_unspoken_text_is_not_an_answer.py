"""声にならなかった地の文を「自分が答えた」と書かない（環-e-に の途中で見つけた欠陥）。

主LLM が `say` も調べものも呼ばず**地の文だけ**を返すことがある。そのとき `_iterate` は
画面へ出して（`_emit`）反復を閉じるが、**声にはしていない**——規則が「音になるのは
`say()` だけ。テキストや（ト書き）は誰にも聞こえない」と定めている。

ところが `_finish` は、それを `自分が答えた：…`・`direction="発話"`・役割 `答え` で
記録していた。**相手が聞いていない文が、答えとして記憶に残る。** しかも `_recent_ctx` は
役割 `答え` を「わたし」として並べるので、**パジュは自分が言っていないことを言ったことと
して読み返す**。

**残す価値はある**（旧 `run()` も「考えたが言わなかったこととして記憶には残す価値がある」と
書いていた）。誤っていたのは「区別せず両方を残す」ほうである。区別して残す。

| | 声になった | ならなかった |
|---|---|---|
| 文言 | `自分が答えた：…` | `考えたが言わなかった：…` |
| `direction` | `発話` | `独白` |
| 役割 | `答え` | `独白`（やりとりには入らないが、共起の母集合には入る） |
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.store.relations import HIDDEN_ROLES


def _ip():
    from familiar_agent.loop.event_loop import InformationProcessing
    from familiar_agent.loop.request import Request

    ip = InformationProcessing.__new__(InformationProcessing)
    a = MagicMock()
    a._memory.save_async_with_id = AsyncMock(return_value=("obs-1", True))
    a._observation_perspective = MagicMock(return_value={})
    a._turn_arousal = AsyncMock(return_value=0.5)
    a._spawn_background_task = MagicMock()
    a._run_post_response_pipeline = MagicMock()
    ip._agent = a
    # `__new__` は `__init__` を通らないので、求めの器は自分で置く（に-5-に-1）。
    ip._req = Request()
    ip._req.iterations = 1
    ip._req.iterations_capped = False
    ip._req.request_id = "req-1"
    ip._req.live_version_id = None
    ip._lookups = []
    ip._req.said_fillers = []
    ip._req.speech_to_deliver = []
    ip._req.turn_records = []
    ip._req.exchange_start = 0
    ip._recent_cursor = None
    ip._req.utterance = "こんばんは"
    ip._req.cue = ""
    ip._close_exchange = MagicMock(return_value=[])
    # `_finish` は最後に並びを空にするので、控えた分をここへ写して見る。
    noted: list[tuple[str, str]] = []
    real_note = type(ip)._note_record

    def _note(obs_id, role):
        real_note(ip, obs_id, role)
        if obs_id:
            noted.append((obs_id, role))

    ip._note_record = _note
    return ip, a, noted


def _saved(a) -> dict:
    """`_finish` が O へ書いた1件（内容と direction）。"""
    call = a._memory.save_async_with_id.await_args
    return {"content": call.args[0], "direction": call.kwargs.get("direction")}


# ── 声になった側（変えていないこと）────────────────────────────────────────


def test_a_spoken_answer_is_still_recorded_as_one():
    ip, a, noted = _ip()
    asyncio.run(ip._finish("はい、そうだよ", [], "発話"))
    got = _saved(a)
    assert got["content"].startswith("自分が答えた：")
    assert got["direction"] == "発話"
    assert ("obs-1", "答え") in noted


# ── 声にならなかった側 ─────────────────────────────────────────────────────


def test_unspoken_text_is_not_called_an_answer():
    ip, a, noted = _ip()
    asyncio.run(ip._finish("なるほどと思った", [], "沈黙"))
    got = _saved(a)
    assert not got["content"].startswith("自分が答えた：")
    assert got["content"].startswith("考えたが言わなかった：")


def test_unspoken_text_is_not_recorded_as_speech():
    ip, a, noted = _ip()
    asyncio.run(ip._finish("なるほどと思った", [], "沈黙"))
    assert _saved(a)["direction"] == "独白"


def test_unspoken_text_does_not_take_the_answer_role():
    ip, a, noted = _ip()
    asyncio.run(ip._finish("なるほどと思った", [], "沈黙"))
    roles = [r for _i, r in noted]
    assert "答え" not in roles
    assert "独白" in roles


def test_the_soliloquy_still_reaches_the_cooccurrence_set():
    """**想起からは消さない。** 考えたことも、次に思い出す手がかりになる。"""
    ip, a, noted = _ip()
    asyncio.run(ip._finish("なるほどと思った", [], "沈黙"))
    assert "obs-1" in [i for i, _r in noted]
    assert "独白" not in HIDDEN_ROLES


def test_nothing_is_written_when_there_is_no_text():
    ip, a, noted = _ip()
    asyncio.run(ip._finish("", [], "沈黙"))
    a._memory.save_async_with_id.assert_not_awaited()
