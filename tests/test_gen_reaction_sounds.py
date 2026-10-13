"""作り置きの声を、指定した言葉だけ作る（`scripts/gen_reaction_sounds.py --only`・出-ce・2026-10-14）。

全部を作り直すと、聞いて決めた「はい」「えっと」「うーん」の声まで変わる。新しく足した言葉だけを作る。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _script():
    spec = importlib.util.spec_from_file_location(
        "gen_reaction_sounds", Path("scripts/gen_reaction_sounds.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_only_the_named_words_are_made():
    gen = _script()
    got = [text for text, _ in gen._pairs({"はーい！", "オッケー"})]
    assert got == ["はーい！", "オッケー"]


def test_without_only_every_word_is_made():
    """**反証の側**：指定が無ければ、今までどおり全部。"""
    gen = _script()
    got = [text for text, _ in gen._pairs(None)]
    assert (
        got[0] == "はい"
        and "えっと" in got
        and len(got) == len(gen.rc.TOOL_WORDS) + len(gen.rc.THINK_WORDS)
    )
