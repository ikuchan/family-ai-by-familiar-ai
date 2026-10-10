"""曲名なしの play_music は、Spotify で止まっていた続きを鳴らす（出-ay 段 4-4e・2026-10-09・`設計方針_判定の段` §2.2.5）。

「音楽かけて」のように名前が無いと、空の名前で探して「「」は見つからなかった」と返していた。いまは Web API で曲を指定せずに
再生を頼み（`PUT /me/player/play`）、鳴り始めたかを確かめる。続きが無い（鳴らない）・Web API が無いときは、かけられなかった
と返す（本人の決定ア）。完了の決まりで、失敗は軽く伝える。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.io import spotify_web
from familiar_agent.tools import music as music_mod
from familiar_agent.tools.music import MusicTool
from tests._arbiter_fakes import decide, prompt_of, writer_says

TABLE = (("ケイマン", "spotify:playlist:aaa", True),)


@pytest.fixture(autouse=True)
def _quick(monkeypatch):
    monkeypatch.setattr(music_mod, "_CONFIRM_SEC", 0.2)
    monkeypatch.setattr(music_mod, "_CONFIRM_TICK", 0.05)


def _tool(playing: bool, *, web=True):
    io = MagicMock()
    io.play = AsyncMock(return_value=True)
    io.set_shuffle = AsyncMock(return_value=True)
    w = MagicMock()
    w.play = MagicMock(return_value=True)
    w.resume = MagicMock(return_value=True)
    w.now_playing = MagicMock(
        return_value={"playing": playing, "context": "spotify:album:old", "item": "spotify:track:x"}
    )
    tool = MusicTool(
        io=io,
        bus=lambda: MagicMock(),
        table=lambda: TABLE,
        web=w if web else None,
        device_name="パジュ",
    )
    return tool, io, w


def _play(tool, **inp):
    return asyncio.run(tool.call("play_music", inp))


def test_no_name_resumes_where_it_stopped():
    tool, io, web = _tool(True)
    text, ok = _play(tool, name="")
    assert ok and "続き" in text
    web.resume.assert_called_once_with("パジュ")
    web.play.assert_not_called()
    io.play.assert_not_awaited()


def test_nothing_to_resume_is_reported():
    tool, _, _ = _tool(False)
    text, ok = _play(tool)
    assert not ok and "かけるものが無かった" in text


def test_without_the_web_it_cannot_resume():
    tool, io, _ = _tool(True, web=False)
    _, ok = _play(tool, name="")
    assert not ok
    io.play.assert_not_awaited()


def test_resume_puts_play_without_naming_anything():
    calls = []

    def http(method, url, *, token, body=None):
        calls.append((method, url, body))
        if url.endswith("/me/player/devices"):
            return {"devices": [{"name": "パジュ", "id": "dev1"}]}
        return {}

    sp = spotify_web.Spotify(token_path="/nonexistent", http=http)
    sp._token = lambda: {"access_token": "t"}  # type: ignore[method-assign]
    assert sp.resume("パジュ") is True
    (put,) = [c for c in calls if c[0] == "PUT"]
    assert put[1].endswith("/me/player/play?device_id=dev1") and put[2] == {}


def test_the_name_is_not_required():
    (play,) = [d for d in music_mod.TOOL_DEFINITIONS if d["name"] == "play_music"]
    assert "name" not in play["input_schema"].get("required", [])


def test_the_arbiter_leaves_the_name_empty_when_none_was_said():
    from familiar_agent.backends.jev import JevAnswer

    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(
        return_value=JevAnswer(
            ok=True,
            answers={
                "meaning": {"choice": "music", "confidence": 0.9},
                "action_music": {"choice": "play_music", "confidence": 0.9},
            },
        )
    )
    writer = writer_says({"tool_input": {"name": ""}})
    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer,
            utterance="パジュ、音楽かけて",
            origin="発話",
            extra_actions=("play_music", "stop_music"),
        )
    )
    assert (d.branch, d.action) == ("action", "play_music")
    assert not (d.tool_input or {}).get("name")
    assert "止まっていた続き" in prompt_of(writer)


# ── 何も指していない言葉は名前ではない（知-al (2)・2026-10-11・本人の決定ア）────────────
# 「音楽をかけて」の「音楽」を曲名として探していた（10/07）。「音楽」「曲」「何か」「なにか」「なんか」「BGM」だけなら
# 空の名前と同じに扱い、止まっていた続きをかける。


@pytest.mark.parametrize(
    "said", ["音楽", "曲", "何か", "なにか", "なんか", "BGM", "音楽を", " 曲 "]
)
def test_a_word_that_names_nothing_resumes(said):
    tool, io, web = _tool(True)
    text, ok = _play(tool, name=said)
    assert ok and "続き" in text
    web.resume.assert_called_once_with("パジュ")
    web.play.assert_not_called()


def test_a_real_name_is_still_searched():
    tool, io, web = _tool(True)
    _play(tool, name="ケイマン")
    web.resume.assert_not_called()


def test_the_description_puts_the_no_name_example_on_the_empty_side():
    (play,) = [d for d in music_mod.TOOL_DEFINITIONS if d["name"] == "play_music"]
    desc = play["description"]
    head, _, tail = desc.partition("名前を言われなければ")
    assert "音楽かけて" not in head and "音楽かけて" in tail
