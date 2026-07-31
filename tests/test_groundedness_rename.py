"""概念1「根づき」（groundedness・記号 g）への改名。

`activation` という語が5つの別の量に相乗りしていた。取込値と参照回数から導き**時間で
減らない**この量を「根づき」と呼び、記号を `g`、英語を `groundedness` にする。英語を
`entrenchment` にしないのは、頭文字 `e` が想起スコアの `e` 軸（感情一致）と衝突し、
文中で取り違えるためである。

ここでは (1) DB 列が新しい名前になっていること、(2) 旧名がコードに残っていないこと、
(3) 導出関数・重み・Config・返り値キーが新しい名前で動くこと、を見る。
"""

from __future__ import annotations

import os
import pathlib

import psycopg2
import pytest

from familiar_agent.config import MemoryConfig, RecallWeights


def _columns(table: str) -> set[str]:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (table,),
        )
        cols = {r[0] for r in cur.fetchall()}
    conn.close()
    return cols


def test_observations_has_the_new_columns() -> None:
    """列は `groundedness_g0` と `groundedness_n`。"""
    cols = _columns("observations")
    assert "groundedness_g0" in cols, "groundedness_g0 が無い（マイグレーション未適用）"
    assert "groundedness_n" in cols, "groundedness_n が無い（マイグレーション未適用）"


def test_observations_no_longer_has_the_old_columns() -> None:
    """旧列は残っていない（両方あると、どちらが正か分からなくなる）。"""
    cols = _columns("observations")
    assert "activation_a0" not in cols, "旧列 activation_a0 が残っている"
    assert "activation_n" not in cols, "旧列 activation_n が残っている"


def test_derive_groundedness_is_the_public_name() -> None:
    """導出関数は `_derive_groundedness`。n=0 なら初期値そのもの。"""
    from familiar_agent.tools.memory import _derive_groundedness

    assert _derive_groundedness(0.75, 0) == pytest.approx(0.75)
    assert _derive_groundedness(0.75, 3) > 0.75, "参照回数で上がらない"


def test_weight_and_config_use_g() -> None:
    """重みは `w_g`、Config は `recall_w_g`／`recall_w_g_jitter`／`recall_g_open`。"""
    cfg = MemoryConfig()
    assert cfg.recall_w_g == 1.5
    assert cfg.recall_w_g_jitter == 0.3
    assert cfg.recall_g_open == 1.0
    w = RecallWeights(1.0, 1.0, 1.0, 1.5, 1.0)
    assert w.w_g == 1.5, "RecallWeights の4つめが w_g になっていない"
    assert cfg.recall_weights("完了").w_g == 1.5


def test_env_names_use_g(monkeypatch) -> None:
    """env は `RECALL_W_G` 系。"""
    monkeypatch.setenv("RECALL_W_G_AFFECT", "0.9")
    assert MemoryConfig().recall_weights("情動").w_g == pytest.approx(0.9)


def test_recall_result_key_is_groundedness() -> None:
    """`recall()` が返す dict のキーは `groundedness`。"""
    from unittest.mock import patch

    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    row = {
        "id": "obs-1", "content": "むかしの話", "timestamp": None,
        "last_recalled_at": None, "recall_count": 0,
        "groundedness_g0": 1.0, "groundedness_n": 0,
        "emotion_p": 0.5, "emotion_pn": 0.5, "emotion_a": 0.5, "emotion_dom": 0.5,
        "direction": "発話", "kind": "observation", "emotion": "neutral",
        "image_path": None, "score": 0.5,
    }
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        mem = ObservationMemory()
        with (
            patch.object(mem._observations, "by_vector", return_value=[row]),
            patch.object(mem._observations, "by_time", return_value=[]),
            patch.object(mem._observations, "by_emotion", return_value=[]),
            patch.object(mem._observations, "situated_cosines", return_value={"obs-1": 0.5}),
        ):
            got = mem.recall("q", n=7)

    assert got, "候補が採点まで届いていない（前提が崩れている）"
    assert "groundedness" in got[0], "返り値キーが groundedness になっていない"
    assert "activation" not in got[0], "旧キー activation が残っている"


def test_old_names_are_gone_from_source() -> None:
    """旧名がソースとテストに残っていない。数え上げでなく grep 0件で示す。"""
    root = pathlib.Path(__file__).resolve().parents[1]
    stale = []
    for sub in ("src", "tests", "migration"):
        for path in (root / sub).rglob("*.py"):
            # 旧名を**検証の対象として**文字列で持つテストは除く。理由を1件ずつ挙げる。
            #   - このテスト自身：改名できたことを旧名で確かめる
            #   - 6概念のガード：撤去した呼び名の一覧を持つ
            if path.name in (pathlib.Path(__file__).name,
                             "test_six_concepts_vocabulary.py"):
                continue
            if sub == "migration":
                continue          # マイグレーションは過去の実行を再現する凍結物
            text = path.read_text(encoding="utf-8")
            for old in ("activation_a0", "activation_n", "_derive_activation",
                        "recall_w_a", "recall_a_open"):
                if old in text:
                    stale.append(f"{path.relative_to(root)}: {old}")
    assert not stale, "旧名が残っている:\n" + "\n".join(sorted(set(stale)))
