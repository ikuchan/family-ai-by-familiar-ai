"""紛らわしい名前を直した（環-g・段ろ）。

`loop/event_loop.py` には、名前と実体が食い違うものが並んでいた。

- `_loop` は asyncio のイベントループを指すが、**このクラス自体がイベントループ**である
- `_parent_id` は「親」を連想させるが、実体は**求めの O の id**
- `_chain` は「連鎖」だが、実体は**反復の回数**（「連鎖」は記録の鎖・求めの意味でも使われていた）
- `_open_intent()` は「O に残す」と読めるが、**O へ何も書かない**
- `_write_intent_and_dispatch()` が書くのは意図ではなく**版**
- `run_iteration()` は「1反復を回す」だが、実体は**求めを始める**（反復は中で複数回りうる）

語の決めは `用語_略語一覧`、対応表は `設計方針_ループの語を束ねる` §3 にある。
**挙動は変えない。** 証明は数え上げでなく**旧名で引いて0件**で置く。
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).parent.parent

# 旧名。語境界で引く（`_chain` が `_chain_head_id` に、`_generation` が
# `_lookup_generation` に当たらないようにするため）。
_OLD = (
    "_inbox",
    "_tasks",
    "_origin_kind",
    "_origin_text",
    "_parent_id",
    "_chain",
    "_capped_hit",
    "_generation",
    "_version_id",
    "_exclude_from_lookup",
    "_w_index",
    "_exchange_from",
    "_show_from",
    "_progress_pending",
    "_released_speech",
    "run_iteration",
    "_begin_origin",
    "_abort_investigation",
    "_action_of",
    "_open_intent",
    "_write_intent_and_dispatch",
    "_settled",
    "_when",
)

# `_loop` は**ファイルごとに意味が違う**ので、`loop/event_loop.py` の中だけを見る。
#   `loop/event_loop.py`  … asyncio のイベントループ（→ `_asyncio_loop`）
#   `io/aif.py`・`io/dif.py` … I（`InformationProcessing`）そのもの。**正反対**
#   `tools/mic.py`・`realtime_stt_session.py` … それぞれの asyncio ループ
#   `recognition/presence_watcher.py` … 見張りの周回メソッド
# 後ろの4つはこの課題の対象ではない（`設計方針_ループの語を束ねる` §3）。
_LOOP_ONLY_IN = "src/familiar_agent/loop/event_loop.py"


def _hits() -> list[str]:
    import re

    pat = re.compile(r"\b(" + "|".join(_OLD) + r")\b")
    out: list[str] = []
    for base in (_ROOT / "src", _ROOT / "tests"):
        for f in sorted(base.rglob("*.py")):
            if f.name == Path(__file__).name:
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if pat.search(line):
                    out.append(f"{f.relative_to(_ROOT)}:{i}: {line.strip()}")
    return out


def test_no_misleading_name_survives():
    assert _hits() == [], "旧名が残っている:\n" + "\n".join(_hits())


def test_the_asyncio_loop_is_named_so_in_the_loop_itself():
    """このクラス自体がイベントループなので、`_loop` ではどちらか読めなかった。"""
    src = (_ROOT / _LOOP_ONLY_IN).read_text(encoding="utf-8")
    import re

    assert not re.search(r"self\._loop\b", src)
    assert "self._asyncio_loop" in src


def test_the_new_names_are_there():
    from familiar_agent.loop.event_loop import InformationProcessing

    for name in (
        "begin_request",
        "_note_origin",
        "_abort_lookups",
        "_action_of_query",
        "_start_lookup",
        "_dispatch_and_write_version",
    ):
        assert hasattr(InformationProcessing, name), name


def test_the_kept_names_are_untouched():
    """あとの段で消えるものは改名しない（`設計方針_ループの語を束ねる` §3「対象外」）。"""
    src = (_ROOT / "src/familiar_agent/loop/event_loop.py").read_text(encoding="utf-8")
    # 段は（調べものの器）で消えたものは、ここから外している。
    for kept in ("_chain_head_id", "_chain_head_content", "_advance_chain", "_show_seeded"):
        assert kept in src, kept


def test_the_ports_do_not_call_the_information_processing_a_loop():
    """`_loop` が正反対の2つを指していた（環-g・段ろ の続き）。

    `loop/event_loop.py` の `_loop` は **asyncio のイベントループ**だが、
    `io/aif.py` と `io/dif.py` の `_loop` は **I（`InformationProcessing`）そのもの**
    である。`loop/tonic.py` は同じものを `_ip` と呼んでおり、呼び方が3通りに割れていた。

    口は `_ip` に揃える（T の側の呼び方に合わせる）。
    """
    import re

    for name in ("io/aif.py", "io/dif.py"):
        src = (_ROOT / "src/familiar_agent" / name).read_text(encoding="utf-8")
        assert not re.search(r"self\._loop\b", src), name
        assert "self._ip" in src, name
