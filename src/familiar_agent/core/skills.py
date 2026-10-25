"""パジュ用のスキル（やり方）を読む（環-au・2026-10-25・本人の決定）。純関数。

スキルは、遊びの決まりのような「やり方」を人が書いたもの。`skills/` に 1 つずつ md で置く。

- 1 行目：見出し（`# しりとり`）。スキルの名前。
- 2 行目：「いつ要る：…」。調停の Jev には、この行だけを見せて「いま要るか」を聞く。
- 3 行目から：本文。要るときに、書き手（軽量LLM）と主LLM に載せる。

判断のたびに読み直す（直したらすぐ効く・ファイルは小さい）。家族の固有名詞は書かない（git に入れる）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: スキルの置き場（リポジトリ直下の `skills/`）。
SKILLS_DIR = Path(__file__).resolve().parents[3] / "skills"

_WHEN = ("いつ要る：", "いつ要る:")


@dataclass(frozen=True)
class Skill:
    """1 つのスキル。`when` は Jev に見せる行、`body` は書き手と主LLM に載せる本文。"""

    name: str
    when: str
    body: str


#: 一度出した警告（ファイル名と更新時刻）。判断のたびに読み直すので、同じ壊れたファイルの警告は 1 度だけ出す。
_WARNED: "set[tuple[str, float]]" = set()


def _warn_once(path: Path, message: str, *args, exc_info: bool = False) -> None:
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        key = (str(path), 0.0)
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message, *args, exc_info=exc_info)


def _parse(path: Path) -> "Skill | None":
    """見出し・「いつ要る：」・本文に分ける。空行は飛ばし、コロンは全角・半角のどちらでもよい。形が違えば警告して None。"""
    lines = [ln for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    head = lines[0].strip() if lines else ""
    when_line = lines[1].strip() if len(lines) > 1 else ""
    name = head.lstrip("#").strip() if head.startswith("#") else ""
    when = next((when_line[len(w) :].strip() for w in _WHEN if when_line.startswith(w)), "")
    body = "\n".join(lines[2:]).strip()
    if not (name and when and body):
        _warn_once(
            path,
            "スキルを読み飛ばした（見出し・「いつ要る：」の行・本文のどれかが無い）：%s",
            path.name,
        )
        return None
    return Skill(name, when, body)


def load_skills(root: Path = SKILLS_DIR) -> "list[Skill]":
    """`root` の `*.md` を名前の順に読む。無い・読めない・形が違うものは警告して飛ばす。同じ名前は先のものを使う。"""
    try:
        if not root.is_dir():
            return []
        # `glob` は読めないフォルダで黙って空を返すので、読めなければ例外を出す `scandir` で並べる。
        with os.scandir(root) as entries:
            paths = sorted(Path(e.path) for e in entries if e.name.endswith(".md") and e.is_file())
    except OSError:
        _warn_once(
            root, "スキルの置き場を読めなかった（スキル無しで続ける）：%s", root, exc_info=True
        )
        return []
    skills: "list[Skill]" = []
    for path in paths:
        try:
            got = _parse(path)
        except (OSError, ValueError):  # ValueError：UTF-8 で読めない
            _warn_once(path, "スキルを読めなかった：%s", path.name, exc_info=True)
            continue
        if got is None:
            continue
        if any(s.name == got.name for s in skills):
            _warn_once(
                path, "同じ名前のスキルがあるので読み飛ばした：%s（%s）", got.name, path.name
            )
            continue
        skills.append(got)
    return skills


def merged(active: "tuple[str, ...]", chosen: "tuple[str, ...] | list[str]") -> "tuple[str, ...]":
    """載っているスキルに、新しく選ばれたものを足す（順は先のまま・重ねない・外さない）。書き手とループで同じ足し方。"""
    return (*active, *(n for n in chosen if n not in active))


def skills_frame(
    skills: "tuple[Skill, ...] | list[Skill]", names: "tuple[str, ...] | list[str]"
) -> str:
    """名前の挙がったスキルの本文を、[やり方] の枠にする（書き手と主LLM で同じ文）。無ければ空。"""
    wanted = set(names)
    picked = [s for s in skills if s.name in wanted]
    if not picked:
        return ""
    return "[やり方]\n" + "\n\n".join(f"### {s.name}\n{s.body}" for s in picked)
