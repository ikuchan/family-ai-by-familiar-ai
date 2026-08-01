"""記憶接続 OIF の8つの口（`設計方針_OIF` v0.1）。

`ObservationMemory` の本番向け公開面 22 種を8つへまとめる。**挙動は変えない。** 口は
既存の実装へ委譲するだけで、この段では呼び出し側も付け替えない。

通ったものは debug ログに残す。何が通ったか（口の名前・件数・長さ）を残し、内容そのものは
出さない。記憶の本文を出すのは debug に限り、そこでも先頭だけにする。
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import pytest

from familiar_agent.io.oif import (
    MI, OIF, Cue, Health, Recalled, Span, Verdict, View,
)

_LOGGER = "familiar_agent.io.oif"


class _Memory:
    """記憶の中身のふり（DB もモデルも呼ばない）。何を渡されたかを控える。"""

    def __init__(self) -> None:
        self.saved: list[tuple[str, dict]] = []
        self.superseded: list[tuple[str, str]] = []
        self.verdicts: dict[str, str] = {}
        self.appended: list[tuple[str, str]] = []
        self.recall_args: dict = {}
        self.rows: list[dict] = []

    async def save_async_with_id(self, content, **kw):
        self.saved.append((content, kw))
        return "書いた-id", True

    def note_lookup_started(self, obs_id):
        self.appended.append((obs_id, "検索を始めた"))
        return True

    async def recall_async(self, *a, **kw):
        self.recall_args = {"a": a, "kw": kw}
        return self.rows

    async def content_novelty_async(self, content):
        return 0.42

    def mark_superseded(self, old, new):
        self.superseded.append((old, new))
        return True

    def apply_verdicts(self, verdicts):
        self.verdicts = dict(verdicts)
        return len(verdicts)

    async def get_earliest_date_async(self):
        return "2026-06-08"

    def is_embedding_ready(self):
        return True

    def embedding_failed(self):
        return False


def _oif() -> tuple[OIF, _Memory]:
    mem = _Memory()
    return OIF(mem), mem


def _mi(**kw) -> MI:
    base = dict(id="", content="覚えておくこと", timestamp=datetime.now(), direction="観察")
    base.update(kw)
    return MI(**base)          # type: ignore[arg-type]


class TestWrite:
    @pytest.mark.asyncio
    async def test_it_writes_and_returns_the_id(self) -> None:
        oif, mem = _oif()
        got = await oif.write(_mi())
        assert got == "書いた-id"
        assert mem.saved, "書いていない"

    @pytest.mark.asyncio
    async def test_the_mi_becomes_the_stored_fields(self) -> None:
        """MI の属性が、そのまま書き込みへ渡る。"""
        oif, mem = _oif()
        await oif.write(_mi(direction="発話", parent_id="起点", writer_id="書いた人"))
        content, kw = mem.saved[0]
        assert content == "覚えておくこと"
        assert kw["direction"] == "発話"
        assert kw["kind"] == "observation", "kind が direction から作られていない"
        assert kw["parent_id"] == "起点"
        assert kw["writer_id"] == "書いた人"

    @pytest.mark.asyncio
    async def test_now_false_defers_materialization(self) -> None:
        """now=False なら実体化を背景へ回す。"""
        oif, mem = _oif()
        await oif.write(_mi(), now=False)
        assert mem.saved[0][1]["materialize_now"] is False


class TestAppend:
    @pytest.mark.asyncio
    async def test_it_appends_to_an_existing_record(self) -> None:
        oif, mem = _oif()
        assert await oif.append("ある-id", "検索を始めた") is True
        assert mem.appended == [("ある-id", "検索を始めた")]


class TestRecall:
    @pytest.mark.asyncio
    async def test_a_plain_cue_searches_by_text(self) -> None:
        oif, mem = _oif()
        await oif.recall(Cue(text="昨日の天気"))
        assert "昨日の天気" in str(mem.recall_args), "手がかりが渡っていない"

    @pytest.mark.asyncio
    async def test_the_view_carries_the_limit_and_floor(self) -> None:
        oif, mem = _oif()
        await oif.recall(Cue(text="x"), View(k=3, floor=0.2))
        kw = mem.recall_args["kw"]
        assert kw.get("n") == 3, f"件数が渡っていない: {kw}"
        assert kw.get("min_score") == pytest.approx(0.2), f"床が渡っていない: {kw}"

    @pytest.mark.asyncio
    async def test_it_returns_recalled_not_dicts(self) -> None:
        """戻りは `Recalled`（MI ＋ 採点）で、dict ではない。"""
        oif, mem = _oif()
        mem.rows = [{
            "memory_id": "m1", "summary": "本文", "timestamp": datetime.now(),
            "direction": "会話", "emotion": "happy", "fit": 0.8, "groundedness": 0.6,
        }]
        got = await oif.recall(Cue(text="x"))
        assert len(got) == 1
        assert isinstance(got[0], Recalled)
        assert isinstance(got[0].mi, MI)
        assert got[0].mi.id == "m1"
        assert got[0].mi.content == "本文", "summary が content へ移っていない"
        assert got[0].fit == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_excluded_ids_are_passed_through(self) -> None:
        oif, mem = _oif()
        await oif.recall(Cue(text="x", exclude=("a", "b")))
        assert mem.recall_args["kw"].get("exclude_ids") == ["a", "b"]


class TestNovelty:
    @pytest.mark.asyncio
    async def test_it_returns_a_number(self) -> None:
        oif, _ = _oif()
        assert await oif.novelty("なにか") == pytest.approx(0.42)


class TestSupersede:
    def test_it_folds_the_old_into_the_new(self) -> None:
        oif, mem = _oif()
        assert oif.supersede("古い", "新しい") is True
        assert mem.superseded == [("古い", "新しい")]


class TestFeedback:
    def test_verdicts_reach_the_store_as_strings(self) -> None:
        oif, mem = _oif()
        n = oif.feedback({"m1": Verdict.IMPORTANT, "m2": Verdict.UNUSED})
        assert n == 2
        assert mem.verdicts == {"m1": "important", "m2": "unused"}


class TestSpanAndHealth:
    @pytest.mark.asyncio
    async def test_span_returns_the_earliest_date(self) -> None:
        oif, _ = _oif()
        got = await oif.span()
        assert isinstance(got, Span)
        assert got.earliest == date(2026, 6, 8)

    def test_health_reports_the_embedding(self) -> None:
        oif, _ = _oif()
        got = oif.health()
        assert isinstance(got, Health)
        assert got.ready is True and got.failed is False


class TestDebugTrail:
    """通ったものを追えるようにする。"""

    @pytest.mark.asyncio
    async def test_each_gate_leaves_a_debug_line(self, caplog) -> None:
        oif, _ = _oif()
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            await oif.write(_mi())
            await oif.recall(Cue(text="昨日の天気"))
            oif.supersede("a", "b")
        msgs = [r.getMessage() for r in caplog.records]
        assert any("write" in m for m in msgs), f"write が残っていない: {msgs}"
        assert any("recall" in m for m in msgs), f"recall が残っていない: {msgs}"
        assert any("supersede" in m for m in msgs), f"supersede が残っていない: {msgs}"

    @pytest.mark.asyncio
    async def test_the_body_is_not_spelled_out(self, caplog) -> None:
        """内容そのものは出さない（debug でも先頭だけ）。"""
        body = "秘密の話" * 100
        oif, _ = _oif()
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            await oif.write(_mi(content=body))
        for m in (r.getMessage() for r in caplog.records):
            assert len(m) < 200, f"ログが長すぎる（本文を出している）: {len(m)}字"
            assert body not in m, "本文をそのまま出している"

    @pytest.mark.asyncio
    async def test_nothing_leaks_at_info(self, caplog) -> None:
        """INFO 以上には記憶の内容を出さない。"""
        oif, _ = _oif()
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            await oif.write(_mi(content="覚えておくこと"))
        assert not [r for r in caplog.records if r.levelno >= logging.INFO], (
            "INFO 以上に出ている"
        )
