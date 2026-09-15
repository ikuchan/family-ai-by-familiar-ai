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
