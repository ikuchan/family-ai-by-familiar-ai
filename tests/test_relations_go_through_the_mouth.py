"""**関係も記憶の口を通る**（環-e-い・関係の面）。

`設計図` ③-2 は「口は4つだけ。この4つ以外に、コンポーネントどうしが直接つながる線は
置かない」と定める。OIF は**記憶ストア O との唯一の出入り口**である。

ところが関係（`relations`／`relation_members`）は 058〜060（2026-09-06〜07）で入った機構で、
**2026-08-01 に設計した OIF に口が無かった**。呼び手は7箇所あり、うち1つは

    RelationStore(self._memory._ctx).record_cooccurrence(mi_ids)

と、**記憶の私的属性（`_ctx`）を掴んで**記憶の内部構造を外から組み立てていた。`ObservationMemory`
すら通っていない。

**書く口は1つにする。** `設計方針_MI間の関係` が「一つの関係が何個でも項を持ち、改訂・継起・
やりとり・共起が**同じ器に載る**」と定めており、書く口も1つが筋である。種類は `kind` が言う。
"""

from __future__ import annotations

import ast
import pathlib

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "familiar_agent"


def _calls(name: str) -> list[str]:
    """本番コードでその名を呼んでいる箇所（file:行）。"""
    out = []
    for f in sorted(_SRC.rglob("*.py")):
        if f.name in ("oif.py", "relations.py", "memory.py"):
            continue  # 口の内側と store 自身
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == name:
                out.append(f"{f.name}:{node.lineno}")
    return out


def _private_context_touches() -> list[str]:
    """**他人の** `_ctx` を掴んでいる箇所（`self._ctx` は持ち主なので数えない）。

    `store/` は `StoreContext` の持ち主、`legacy/` は旧経路なので外す。**外す理由を
    1件ずつ書く**（無言で対象から落とさない）。文字列の中の言及（docstring・コメント）は
    `ast` で見るので入らない。
    """
    leaks = []
    for f in sorted(_SRC.rglob("*.py")):
        if f.parent.name == "store" or "legacy" in f.parts or f.name == "memory.py":
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr != "_ctx":
                continue
            owner = ast.unparse(node.value)
            if owner != "self":
                leaks.append(f"{f.name}:{node.lineno} ({owner}._ctx)")
    return leaks


def test_nobody_reaches_into_the_memorys_private_context():
    """**私的属性を掴まない。** `_ctx` は記憶の内側で、外から組み立てるものではない。

    以前は `RelationStore(self._memory._ctx).record_cooccurrence(...)` と、記憶の内部構造を
    外から組み立てていた。`ObservationMemory` すら通っていなかった。
    """
    assert not _private_context_touches(), "記憶の私的属性を掴んでいる"


def test_relations_are_not_written_outside_the_mouth():
    left = {
        n: _calls(n)
        for n in (
            "record_exchange",
            "record_succession",
            "record_cooccurrence",
            "recent_exchanges",
            "latest_exchange_origin",
        )
    }
    left = {k: v for k, v in left.items() if v}
    assert not left, f"関係を口の外で触っている: {left}"


def test_the_mouth_has_one_way_to_write_a_relation():
    """**反証側。** 呼び出しが消えただけで口を通っていなければ、何も守っていない。"""
    from familiar_agent.io.oif import OIF

    for name in ("link", "exchanges", "latest_origin"):
        assert hasattr(OIF, name), name
    src = "".join(
        (_SRC / p).read_text(encoding="utf-8")
        for p in ("agent.py", "loop/workspace.py", "loop/event_loop.py")
    )
    assert src.count("_oif.link(") == 4  # やりとり2・継起1・共起1
    assert src.count("_oif.exchanges(") == 1
    assert src.count("_oif.latest_origin()") == 1


def test_the_kinds_say_what_the_relation_is():
    """種類は `kind` が言う（口は1つ）。使う語は `store/relations.py` の定数から取る。"""
    from familiar_agent.store.relations import (
        KIND_COOCCURRENCE,
        KIND_EXCHANGE,
        KIND_SUCCESSION,
    )

    src = (_SRC / "loop" / "event_loop.py").read_text(encoding="utf-8")
    assert "KIND_EXCHANGE" in src
    assert (KIND_EXCHANGE, KIND_SUCCESSION, KIND_COOCCURRENCE) == ("やりとり", "継起", "共起")


# ── ベクトル埋め込みは口の内側 ──────────────────────────────────────────────


def test_the_embedding_is_asked_through_the_mouth():
    """**ベクトル埋め込みは OIF の内側にある**（`設計図` ③-2）。

    `agent` が記憶の `is_embedding_ready()`／`embedding_failed()` を直に見ていた。UI・
    `main`・`errors` は `agent` 経由なので外へは漏れていなかったが、`agent` の2行が
    設計と食い違っていた。使える状態かは口が答える（`OIF.health`）。
    """
    from unittest.mock import MagicMock

    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.io.oif import Health

    a = EmbodiedAgent.__new__(EmbodiedAgent)
    a._memory = MagicMock()
    a._oif = MagicMock(health=MagicMock(return_value=Health(ready=True, failed=False)))
    assert a.is_embedding_ready is True
    assert a.embedding_failed() is False
    a._memory.is_embedding_ready.assert_not_called()
    a._memory.embedding_failed.assert_not_called()
