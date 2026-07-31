"""概念3「勢い」（dynamism・記号 d）への改名。

ワークスペース競合の候補（`Coalition`）が持つ基礎強度である。提案元が付ける値で、
競合スコアは `勢い × (0.4×urgency + 0.3×novelty + 0.3)` になる。

`activation` という語が5つの別の量に相乗りしていた。この値は取込値と参照回数から導く
「根づき」とは別物で、DB にも載らない実行時だけの量である。英語を `drive_strength` に
しないのは、提案元が欲求（drive）だけでなく情景・記憶・予測・探索・語りもあり、情景の
平均確信度を「drive の強さ」と呼ぶのが誤りになるためである。

ここでは (1) フィールド名、(2) 競合スコアが勢いに比例すること、(3) 各提案元が新しい名前
で組み立てること、(4) 旧名が残っていないこと、を見る。
"""

from __future__ import annotations

import pathlib

import pytest

from familiar_agent.workspace import Coalition


def _coalition(**kw) -> Coalition:
    base = dict(source="test", summary="s", dynamism=0.8, urgency=0.5,
                novelty=0.5, context_block="")
    base.update(kw)
    return Coalition(**base)  # type: ignore[arg-type]


def test_field_is_dynamism() -> None:
    """フィールドは `dynamism`。旧名は持たない。"""
    c = _coalition()
    assert c.dynamism == pytest.approx(0.8)
    assert not hasattr(c, "activation"), "旧名 activation が残っている"


def test_score_is_proportional_to_dynamism() -> None:
    """競合スコアは勢いに比例する（0 なら 0）。"""
    assert _coalition(dynamism=0.0).score() == pytest.approx(0.0)
    weak = _coalition(dynamism=0.4).score()
    strong = _coalition(dynamism=0.8).score()
    assert strong == pytest.approx(weak * 2.0)


def test_score_formula_unchanged() -> None:
    """式そのものは変えていない（改名だけ）。"""
    c = _coalition(dynamism=0.8, urgency=0.5, novelty=0.5)
    assert c.score() == pytest.approx(0.8 * (0.4 * 0.5 + 0.3 * 0.5 + 0.3))


def test_old_name_is_gone_from_source() -> None:
    """`Coalition` を組み立てる側と読む側に旧名が残っていない。

    数え上げでなく grep 0件で示す。除外は理由を1件ずつ挙げる。
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    # 概念3 を扱うファイルだけを見る。ほかの `activation` は別概念（根づき・顕著性）で、
    # それぞれの改名で別に扱う。
    targets = [
        "src/familiar_agent/workspace.py",
        "src/familiar_agent/desires.py",
        "src/familiar_agent/scene.py",
        "src/familiar_agent/prediction.py",
        "src/familiar_agent/exploration.py",
        "src/familiar_agent/self_narrative.py",
    ]
    stale = [t for t in targets if "activation" in (root / t).read_text(encoding="utf-8")]
    assert not stale, "旧名 activation が残っている:\n" + "\n".join(stale)
