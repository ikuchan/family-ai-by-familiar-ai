"""**O へ書くときは、誰の記録かを必ず言う**（記-f の網羅）。

「誰がやったか」は `situated_memories` の `actor` の面が持ち、その面は `writer_id` から
立つ。渡さないと `writer = writer_id or self._ctx.person_id` に落ち、**書く側のインスタンス
次第**になる。`好奇心` と `記憶` はそれで `__self__` に立っていた——2つの既定の一致に
支えられた正しさで、読んでも分からず、記憶を差し替えれば黙って変わる。

**渡し忘れたら落ちるほうが、黙って別の人の記録になるより良い**（`apply_memory_verdicts` の
対応表で採ったのと同じ判断）。

このテストは**数え上げたリストで代えない**。`save`／`save_async`／`save_async_with_id` の
呼び出しと、OIF の書く口（`_oif.write`）の呼び出しを AST で走査し、1件ずつ見る。書く口は OIF に
一本化した（記-u 段 3）ので、記憶の口を呼ぶのは OIF の中だけになり、書き手を言うかは OIF の呼び手で見る。
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

_CALLS = {"save", "save_async", "save_async_with_id"}
_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "familiar_agent"

#: O へ書かない同名の呼び出し。**1件ずつ理由を書く**（無言で対象から落とさない）。
_NOT_A_MEMORY_WRITE = {
    ("occupancy_sensor.py", "save"): "見えの普通（pose_norms）の保存。O ではない",
    ("observations.py", "save"): "PIL の画像保存（サムネイル作成）",
}


def _is_oif_write(node: ast.Call) -> bool:
    """OIF の書く口か（`agent._oif.write`・`self._oif.write`・`oif.write`）。ファイルの write は数えない。"""
    func = node.func
    if getattr(func, "attr", None) != "write" or not isinstance(func, ast.Attribute):
        return False
    recv = ast.unparse(func.value)
    return recv == "oif" or recv.endswith("_oif") or recv.endswith(".oif")


def _perspective_names(fn: ast.AST) -> set[str]:
    """関数の中で、視点（`_observation_perspective()` など）から作った変数の名前。

    `kw = dict(agent._observation_perspective())` と作ってから顔ぶれを足し、`**kw` で渡す書き方を、
    書き手を渡したと数えるため。
    """
    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and "perspective" in ast.unparse(node.value):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


def _writes() -> list[tuple[str, int, str, bool]]:
    """(file, 行, 口, 書き手が渡っているか) の並び。"""
    out = []
    for f in sorted(_SRC.rglob("*.py")):
        if "legacy" in f.parts:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for fn in fns:
            built = _perspective_names(fn)
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", None)
                if name not in _CALLS and not _is_oif_write(node):
                    continue
                if (f.name, name) in _NOT_A_MEMORY_WRITE:
                    continue
                named = {k.arg for k in node.keywords if k.arg}
                spread = [ast.unparse(k.value) for k in node.keywords if k.arg is None]
                said = "writer_id" in named or any("perspective" in s or s in built for s in spread)
                out.append((f.name, node.lineno, str(name), said))
    # 入れ子の関数は外の関数の走査でも拾うので、同じ呼び出しを 1 件にする。
    return sorted(set(out))


def test_every_write_to_o_says_who():
    silent = [(f, n, c) for f, n, c, said in _writes() if not said]
    assert not silent, f"書き手を言わずに O へ書いている: {silent}"


def test_there_are_writes_to_find():
    """**反証側。** 走査が空振りしていれば、上のテストは何も守っていない。

    件数は 環-e-い で 15 → 6 に減った。ループと `agent` の9箇所が OIF を通るようになり、
    その先は `io/oif.py` の1箇所（既に数えている）へ集まったためである。記-u 段 3 で REST の 4 か所も
    OIF を通るようにしたので、OIF の書く口の呼び手も数える（記憶の口だけなら 1 件になる）。
    """
    assert len(_writes()) >= 5


def test_the_memory_mouth_requires_a_writer():
    """`OIF.write` は書き手を**必須**で受け取る。既定を置くと、渡し忘れが静かに通る。

    OIF はまだ本番コードから呼ばれていない（環-e-い で呼ぶ側を移す）。**通し始めた
    瞬間に効く穴**なので、通す前に塞ぐ。
    """
    from familiar_agent.io.oif import OIF

    p = inspect.signature(OIF.write).parameters["writer_id"]
    assert p.default is inspect.Parameter.empty, "既定を置いている"
    assert p.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_memory_mouth_refuses_an_empty_writer():
    """空文字も受け取らない。`None` を弾いても空で素通りできれば同じことになる。"""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from familiar_agent.io.oif import MI, OIF

    oif = OIF(MagicMock(save_async_with_id=AsyncMock(return_value=("obs", None))))
    with pytest.raises(ValueError):
        asyncio.run(
            oif.write(MI(id="", content="あ", timestamp=None, direction="観察"), writer_id="")
        )
