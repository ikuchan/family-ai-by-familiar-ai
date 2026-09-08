"""撤去済みの略語 `WR` を名前から落とした（環-g・段い）。

用語一覧は **WRDB を撤去した**（060）と記している。WR は種類 `共起` の関係として
`relations` に載るようになった。**にもかかわらず `WR` を名前に持つコードが5ファイルに
残っていた。**

しかも `_wr_ids` の実体は「このターンが作った記録と、その役割」で、拡散想起の母集合にも
使い、やりとりの関係の項にもなる。**WR は2つの用のうち片方の、しかも撤去された呼び名**
である。

証明は数え上げでなく**旧名で引いて0件**で置く。
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).parent.parent
_OLD = ("wr_ids", "_note_wr", "_record_wr")


def _hits() -> list[str]:
    """`src/` と `tests/` から旧名を引く。この file 自身は除く（旧名を語るため）。"""
    out: list[str] = []
    for base in (_ROOT / "src", _ROOT / "tests"):
        for f in sorted(base.rglob("*.py")):
            if f.name == Path(__file__).name:
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if any(old in line for old in _OLD):
                    out.append(f"{f.relative_to(_ROOT)}:{i}: {line.strip()}")
    return out


def test_no_old_wr_name_survives():
    assert _hits() == [], "旧名が残っている:\n" + "\n".join(_hits())


def test_the_new_names_are_there():
    from familiar_agent.loop.event_loop import InformationProcessing
    from familiar_agent.store.relations import combine_cooccurring_ids

    assert callable(combine_cooccurring_ids)
    assert callable(InformationProcessing._note_record)


def test_the_design_id_is_left_alone():
    """`[D-WR拡散想起]` は設計文書の固定 ID である。**名前ではないので残す。**"""
    src = (_ROOT / "src/familiar_agent/config.py").read_text(encoding="utf-8")
    assert "[D-WR拡散想起]" in src
