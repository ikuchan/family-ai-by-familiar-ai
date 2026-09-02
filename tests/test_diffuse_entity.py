"""拡散想起スライス2 (B) エンティティ辺：関係の面から種 person を選び person 中心で再想起。

段4 で視点列（`subject_id`／`participants_json`／`writer_id`）から situated の面へ移した。
面から引く側の確かめは `test_diffuse_situated.py` にある。ここは純関数の並べ替えと、
有界再帰の骨（`diffuse_ids`）を見る。
"""

from __future__ import annotations



from familiar_agent.core.diffuse import select_entity_seeds


# ── select_entity_seeds（純関数・about 優先・除外・重複除去） ─────────────────

def test_seeds_about_first_then_present_then_actor():
    """種の優先順は「話題の主体 → そばに居た → やった人」（047 の役割名）。"""
    relations = [
        {"person_id": "PA", "relation_key": "about"},
        {"person_id": "PB", "relation_key": "present"},
        {"person_id": "AGENT", "relation_key": "actor"},
        {"person_id": "PC", "relation_key": "about"},
        {"person_id": "PA", "relation_key": "actor"},
    ]
    out = select_entity_seeds(relations, exclude={"AGENT", "SPEAKER"})
    assert out == ["PA", "PC", "PB"]  # about PA,PC → present PB → actor は既出/除外


def test_seeds_excludes_and_dedups_and_skips_none():
    relations = [
        {"person_id": None, "relation_key": "about"},
        {"person_id": "X", "relation_key": "present"},
        {"person_id": "X", "relation_key": "present"},
        {"person_id": "SELF", "relation_key": "actor"},
    ]
    assert select_entity_seeds(relations, exclude={"SELF"}) == ["X"]


def test_seeds_ignore_roles_outside_the_order():
    """種にするのは3つの役割だけ。`addressee` などは種にしない。"""
    relations = [{"person_id": "Y", "relation_key": "addressee"}]
    assert select_entity_seeds(relations, set()) == []


def test_seeds_empty():
    assert select_entity_seeds([], set()) == []


# ── diffuse_ids（有界再帰・純関数・候補取得はコールバック注入） ───────────────

from familiar_agent.core.diffuse import diffuse_ids  # noqa: E402


def _grapher(graph):
    def get(known):
        out = []
        for k in known:
            out += graph.get(k, [])
        return out
    return get


def test_diffuse_recurses_until_dry():
    # a→x, x→y, y→(なし)。深く再帰して x,y を足す
    get = _grapher({"a": ["x"], "x": ["y"]})
    assert diffuse_ids(["a"], get, max_add=5, max_depth=3) == ["x", "y"]


def test_diffuse_respects_max_add():
    get = _grapher({"a": ["x"], "x": ["y"]})
    assert diffuse_ids(["a"], get, max_add=1, max_depth=3) == ["x"]


def test_diffuse_respects_max_depth():
    get = _grapher({"a": ["x"], "x": ["y"]})
    assert diffuse_ids(["a"], get, max_add=5, max_depth=1) == ["x"]


def test_diffuse_excludes_seed_and_dedups():
    get = _grapher({"a": ["a", "x", "x"]})  # 自分・重複を返しても
    assert diffuse_ids(["a"], get, max_add=5, max_depth=2) == ["x"]


def test_diffuse_empty_when_no_candidates():
    assert diffuse_ids(["a"], _grapher({}), max_add=5, max_depth=3) == []
