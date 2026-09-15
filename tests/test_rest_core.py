"""②核の固め（`loop/rest_core.py`・記-a-ろ-に・`出来事を畳む` §3c）。

同一（LLM なし・上限なし）→ 測り直し → 束ね（機械）→ 平均 u の低い束から超過ぶんだけ
LLM に 1 束 1 文を書かせ、検査に通ったものを `まとめ` として書いて出典を畳む。店はモック。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from familiar_agent.core import measure
from familiar_agent.loop import rest_core
from familiar_agent.loop.rest_fold import Written
from familiar_agent.store.relations import KIND_FOLD

NOW = datetime(2026, 9, 15, 3, 0, tzinfo=timezone.utc)
SELF = "00000000-0000-0000-0000-000000000000"
PAPA = "43161802-ba43-4562-b2d5-8c727aafd729"


def _vec(axis: int, jitter: float = 0.0, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = np.eye(16)[axis] + rng.normal(0.0, jitter, 16)
    return list((v / np.linalg.norm(v)).astype(float))


def _rec(i: int, axis: int, *, days: float, n: int = 1, chars: int = 200, jitter=0.05, who=None):
    return {
        "obs_id": f"o{i}",
        "content": f"記録 {i}：" + "あ" * max(0, chars - 5),
        "direction": "発話",
        "timestamp": NOW - timedelta(days=days),
        "groundedness_g0": 1.0,
        "chars": chars,
        "groundedness_n": n,
        "person_ids": [SELF] + ([who] if who else []),
        "vector": _vec(axis, jitter, seed=i),
    }


def _agent(records, *, target: float, reply=None):
    a = MagicMock()
    a.config.memory.recall_half_life_days = 10.0
    a.config.memory.recall_w_t = 1.0
    a.config.memory.recall_w_g = 1.5
    a.config.memory.info_target_bits = target
    a.config.memory.core_same_cos = 0.98
    a.config.memory.core_bundle_cos = 0.5
    a.config.memory.core_bundle_min = 3
    a.config.memory.core_bundles_per_night = 6
    a._family_md = "## ゆうすけ\n- **名前**：ゆうすけ\n- **呼び方**：パパ\n"
    a._pmm.find_person_id_by_name = MagicMock(return_value=PAPA)
    a._observation_perspective = MagicMock(return_value={"writer_id": SELF})
    a._evaluator.emotion_for_turn = AsyncMock(side_effect=Exception("未測定"))
    a._oif.core_records = MagicMock(return_value=list(records))
    a._oif.supersede = MagicMock(return_value=True)
    a._oif.raise_groundedness = MagicMock(return_value=1)
    a._oif.write = AsyncMock(side_effect=lambda mi, **kw: f"new-{mi.content[:6]}")
    a.backend.complete = AsyncMock(
        side_effect=reply or (lambda *a_, **k: '{"text": "", "sources": []}')
    )
    return a


def _lines(tmp_path):
    return [
        f" {r.kind} " + " ".join(f"{k}={v}" for k, v in r.fields.items())
        for r in measure.read_rows(base_dir=tmp_path)
    ]


# ---- 同一 -----------------------------------------------------------------------


def test_identical_records_fold_into_the_newest_without_the_llm(tmp_path):
    measure.setup(base_dir=tmp_path)
    recs = [_rec(i, 0, days=5 - i, n=1 + i, jitter=0.0) for i in range(4)]  # 完全一致 4 件
    recs.append(_rec(9, 3, days=1.0))
    a = _agent(recs, target=1e12)  # 超過なし → 束ねは走らない
    r = asyncio.run(rest_core.fold_core(a, now=NOW))
    assert r.identical_groups == 1 and r.identical_folded == 3
    newest = "o3"
    folded = {c.args[0]: c.args[1] for c in a._oif.supersede.call_args_list}
    assert folded == {"o0": newest, "o1": newest, "o2": newest}
    assert all(c.kwargs.get("kind") == KIND_FOLD for c in a._oif.supersede.call_args_list)
    a._oif.raise_groundedness.assert_called_once_with(newest, 4)  # 群の最大 n
    a.backend.complete.assert_not_called()
    assert any(" 層1同一 " in ln and "群=1" in ln and "畳んだ=3" in ln for ln in _lines(tmp_path))


# ---- 固め -----------------------------------------------------------------------


def _two_topics():
    old = [_rec(i, 0, days=30 + i, who=PAPA) for i in range(6)]  # 古い丘（平均 u が低い）
    new = [_rec(10 + i, 1, days=1 + i * 0.1) for i in range(6)]  # 新しい丘
    return old + new


def test_no_excess_means_no_llm_and_no_folding(tmp_path):
    measure.setup(base_dir=tmp_path)
    a = _agent(_two_topics(), target=1e12)
    r = asyncio.run(rest_core.fold_core(a, now=NOW))
    assert r.excess_bits == 0 and r.bundles == 0 and r.written == 0
    a.backend.complete.assert_not_called()
    a._oif.write.assert_not_called()


def test_the_oldest_bundle_is_written_and_its_sources_folded(tmp_path):
    measure.setup(base_dir=tmp_path)

    async def reply(prompt, **kw):
        ids = [ln.split(" ")[0] for ln in prompt.splitlines() if ln.startswith("o")]
        return json.dumps(
            {"text": "パパとよく話した。", "sources": ids, "people": ["パパ"]}, ensure_ascii=False
        )

    a = _agent(_two_topics(), target=1.0, reply=reply)  # ほぼ全部が超過
    a.config.memory.core_bundles_per_night = 1
    r = asyncio.run(rest_core.fold_core(a, now=NOW))
    assert r.bundles >= 1 and r.written == 1 and r.folded >= 3
    mi = a._oif.write.call_args.args[0]
    assert mi.direction == "まとめ" and mi.content == "パパとよく話した。"
    kw = a._oif.write.call_args.kwargs
    assert kw["participants"] == [PAPA]  # 出典の面の人（自分は除く）
    folded_from = {c.args[0] for c in a._oif.supersede.call_args_list}
    assert folded_from <= {f"o{i}" for i in range(6)}  # 古い丘が先
    assert mi.timestamp == max(
        _two_topics()[i]["timestamp"] for i in range(6) if f"o{i}" in folded_from
    )
    a._oif.raise_groundedness.assert_called()  # 産物が出典の最大 n を引き継ぐ
    assert r.records and r.records[0].kind == "core_summary" and isinstance(r.records[0], Written)
    line = next(ln for ln in _lines(tmp_path) if " 層1固め " in ln)
    assert "固めた=1" in line and "見送り=0" in line


def test_a_bundle_whose_reply_fails_the_check_is_left_alone(tmp_path):
    measure.setup(base_dir=tmp_path)

    async def reply(prompt, **kw):
        return json.dumps(
            {"text": "たろうと話した。", "sources": ["o0", "zzz"], "people": []}, ensure_ascii=False
        )

    a = _agent(_two_topics(), target=1.0, reply=reply)
    a.config.memory.core_bundles_per_night = 1
    r = asyncio.run(rest_core.fold_core(a, now=NOW))
    assert r.written == 0 and r.skipped == 1
    a._oif.write.assert_not_called()
    # 同一の段で畳んだものはあってよいが、固めの産物へは何も畳んでいない
    assert not any(c.args[1].startswith("new-") for c in a._oif.supersede.call_args_list)


def test_check_rejects_out_of_bundle_sources_long_text_and_unknown_names():
    fam = ("パパ",)
    assert (
        rest_core.check(
            "パパと話した。", ["o1", "o2"], ["パパ"], bundle_ids={"o1", "o2", "o3"}, family=fam
        )
        is None
    )
    assert "出典" in rest_core.check("x", ["o9"], [], bundle_ids={"o1"}, family=fam)
    assert "出典" in rest_core.check("x", [], [], bundle_ids={"o1"}, family=fam)
    assert "字" in rest_core.check("あ" * 301, ["o1"], [], bundle_ids={"o1"}, family=fam)
    assert "名前" in rest_core.check(
        "たろうと話した", ["o1"], ["たろう"], bundle_ids={"o1"}, family=fam
    )
    assert "空" in rest_core.check("", ["o1"], [], bundle_ids={"o1"}, family=fam)
