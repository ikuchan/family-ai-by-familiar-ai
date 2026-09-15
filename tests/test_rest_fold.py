"""日次の畳み込み（記-a-ろ-は・2026-09-14）：分け方・LLM への依頼と検査。

対象は機械が決め（`OIF.fold_materials`）、まとめ方は LLM。産物は自己エピソード（日・一人称・
500 字以内）と関係のまとめ（人ごと・300 字以内）。検査に通らなければ書かない（材料は残り、
次の晩に含める）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

from familiar_agent.io.oif import MI
from familiar_agent.loop import rest_fold

_T0 = datetime(2026, 9, 13, 8, 0, tzinfo=timezone.utc)


def _mi(i: int, content: str, *, hours: float = 0.0, direction: str = "発話") -> MI:
    return MI(
        id=f"o{i}",
        obs_id=f"o{i}",
        content=content,
        timestamp=_T0 + timedelta(hours=hours),
        direction=direction,
    )


# ── 分け方 ──────────────────────────────────────────────────────────────────


def test_a_day_is_one_batch_when_small():
    rows = [_mi(i, f"出来事{i}", hours=i) for i in range(10)]
    batches = rest_fold.split_batches(rows, max_items=60)
    assert len(batches) == 1 and batches[0].day == "2026-09-13" and len(batches[0].rows) == 10


def test_days_are_separate_batches_and_big_days_are_split_by_time():
    rows = [_mi(i, f"a{i}", hours=i * 0.2) for i in range(130)]  # 13 日 08:00〜 26 時間ぶん
    batches = rest_fold.split_batches(rows, max_items=60)
    days = sorted({b.day for b in batches})
    assert days == ["2026-09-13", "2026-09-14"]
    assert all(len(b.rows) <= 60 for b in batches)
    assert sum(len(b.rows) for b in batches) == 130
    # 同じ日の分割は時間順で切れている。
    d13 = [b for b in batches if b.day == "2026-09-13"]
    assert len(d13) >= 2 and d13[0].rows[-1].timestamp < d13[1].rows[0].timestamp


# ── 依頼と検査 ────────────────────────────────────────────────────────────────


def test_the_llm_is_asked_for_an_episode_and_per_person_summaries():
    import asyncio

    backend = AsyncMock()
    backend.complete = AsyncMock(
        return_value='{"episode": "今日はこうきとサッカーの話をした。", "persons": {"こうき": "サッカーが好きで、負けると泣く。"}}'
    )
    batch = rest_fold.Batch(
        day="2026-09-13", rows=[_mi(1, "こうき：サッカーしたよ"), _mi(2, "ぼく：いいね")]
    )
    out = asyncio.run(
        rest_fold.ask_summaries(backend, batch, family_names=("パパ", "ママ", "たいき", "こうき"))
    )
    assert out.episode.startswith("今日は") and out.persons == {
        "こうき": "サッカーが好きで、負けると泣く。"
    }
    prompt = backend.complete.call_args.args[0]
    assert "こうき：サッカーしたよ" in prompt and "一人称" in prompt


def test_the_check_rejects_long_empty_or_unknown_names():
    fam = ("パパ", "ママ", "たいき", "こうき")
    ok = rest_fold.Summaries(
        episode="今日はこうきと話した。", persons={"こうき": "サッカーが好き。"}
    )
    assert rest_fold.check(ok, family_names=fam) is None
    assert (
        rest_fold.check(rest_fold.Summaries(episode="", persons={}), family_names=fam) is not None
    )
    assert (
        rest_fold.check(rest_fold.Summaries(episode="あ" * 501, persons={}), family_names=fam)
        is not None
    )
    assert (
        rest_fold.check(
            rest_fold.Summaries(episode="今日", persons={"こうき": "い" * 301}), family_names=fam
        )
        is not None
    )
    assert (
        rest_fold.check(
            rest_fold.Summaries(episode="今日", persons={"泰輝": "…"}), family_names=fam
        )
        is not None
    )


def test_an_unparsable_reply_is_rejected_not_raised():
    import asyncio

    backend = AsyncMock()
    backend.complete = AsyncMock(return_value="よくわからない")
    batch = rest_fold.Batch(day="2026-09-13", rows=[_mi(1, "x")])
    out = asyncio.run(rest_fold.ask_summaries(backend, batch, family_names=()))
    assert out is None


# ── a0（取込の新規性）を効かせる（2026-09-16） ────────────────────────────────


def _mi_a0(i: int, content: str, hours: float, a0: float) -> MI:
    m = _mi(i, content, hours=hours)
    m.groundedness_g0 = a0
    return m


def test_repeats_below_the_novelty_floor_are_left_out_of_the_request_but_folded_with_the_day():
    """a0 < distill_min_a0 の記録は「繰り返し」——依頼には載せず、その日の自己エピソードで畳む。"""
    rows = [
        _mi_a0(1, "新しい話", 1, 0.9),
        _mi_a0(2, "同じ部屋の観察", 2, 0.1),
        _mi_a0(3, "また同じ観察", 3, 0.2),
        _mi_a0(4, "別の話", 4, 0.7),
    ]
    batches = rest_fold.split_batches(rows, max_items=60, min_a0=0.47)
    assert len(batches) == 1
    assert [r.obs_id for r in batches[0].rows] == ["o1", "o4"]
    assert [r.obs_id for r in batches[0].dropped] == ["o2", "o3"]
    # 外したものは 1 本目の束に付く（60 件を超えて割れた日でも、同じ日の最初の要約に畳む）。
    many = [_mi_a0(i, f"話 {i}", i * 0.1, 0.9) for i in range(1, 121)] + [
        _mi_a0(200, "繰り返し", 5, 0.1)
    ]
    batches = rest_fold.split_batches(many, max_items=60, min_a0=0.47)
    assert (
        len(batches) == 2
        and [r.obs_id for r in batches[0].dropped] == ["o200"]
        and batches[1].dropped == []
    )


def test_the_request_shows_novelty_and_asks_to_weigh_by_it():
    import asyncio

    backend = AsyncMock()
    backend.complete = AsyncMock(return_value='{"episode": "x", "persons": {}}')
    batch = rest_fold.Batch(
        day="2026-09-13", rows=[_mi_a0(1, "新しい話", 1, 0.92), _mi_a0(2, "ふつう", 2, 0.55)]
    )
    asyncio.run(rest_fold.ask_summaries(backend, batch, family_names=("パパ",)))
    prompt = backend.complete.call_args.args[0]
    assert "新しさ 0.92" in prompt and "新しさ 0.55" in prompt
    assert "新しさ" in prompt and "中心" in prompt


def test_the_fold_supersedes_dropped_rows_too_and_counts_them():
    import asyncio
    from unittest.mock import MagicMock, patch

    rows = [_mi_a0(1, "新しい話", 1, 0.9), _mi_a0(2, "繰り返し", 2, 0.1)]
    agent = MagicMock()
    agent.config.memory.distill_min_a0 = 0.47
    agent._oif.fold_materials = MagicMock(return_value=rows)
    agent._oif.supersede = MagicMock(return_value=True)
    agent._oif.write = AsyncMock(return_value="ep-1")
    agent._evaluator.emotion_for_turn = AsyncMock(side_effect=Exception("未測定"))
    agent._observation_perspective = MagicMock(return_value={})
    agent._pmm.find_person_id_by_name = MagicMock(return_value="")

    async def _ask(backend, batch, *, family_names):
        assert [r.obs_id for r in batch.rows] == ["o1"]  # 繰り返しは依頼に載らない
        return rest_fold.Summaries(episode="今日は新しい話をした。", persons={})

    with (
        patch("familiar_agent.loop.rest_fold.ask_summaries", new=_ask),
        patch("familiar_agent.loop.rest_fold.family_names_of", return_value=("パパ",)),
    ):
        result = asyncio.run(rest_fold.fold_since_last_rest(agent))
    assert result.folded == 2 and result.left_out == 1
    assert {c.args[0] for c in agent._oif.supersede.call_args_list} == {"o1", "o2"}
