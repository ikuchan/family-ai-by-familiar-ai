"""パジュ用のスキル（やり方）を読む（環-au・2026-10-25・本人の決定）。純関数。

スキルは、遊びの決まりのような「やり方」を人が書いたもの。`skills/` に 1 つずつ md で置く。

- 1 行目：見出し（`# しりとり`）。スキルの名前。
- 2 行目：「いつ要る：…」。調停の Jev には、この行だけを見せて「いま要るか」を聞く。
- 3 行目から：本文。要るときに、書き手（軽量LLM）と主LLM に載せる。

判断のたびに読み直す（直したらすぐ効く・ファイルは小さい）。家族の固有名詞は書かない（git に入れる）。
"""

from __future__ import annotations

import logging
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


def _parse(path: Path) -> "Skill | None":
    """見出し・「いつ要る：」・本文に分ける。空行は飛ばし、コロンは全角・半角のどちらでもよい。形が違えば警告して None。"""
    lines = [ln for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    head = lines[0].strip() if lines else ""
    when_line = lines[1].strip() if len(lines) > 1 else ""
    name = head.lstrip("#").strip() if head.startswith("#") else ""
    when = next((when_line[len(w) :].strip() for w in _WHEN if when_line.startswith(w)), "")
    body = "\n".join(lines[2:]).strip()
    if not (name and when and body):
        logger.warning(
            "スキルを読み飛ばした（見出し・「いつ要る：」の行・本文のどれかが無い）：%s", path.name
        )
        return None
    return Skill(name, when, body)


def load_skills(root: Path = SKILLS_DIR) -> "list[Skill]":
    """`root` の `*.md` を名前の順に読む。無い・読めない・形が違うものは警告して飛ばす。同じ名前は先のものを使う。"""
    if not root.is_dir():
        return []
    skills: "list[Skill]" = []
    for path in sorted(root.glob("*.md")):
        try:
            got = _parse(path)
        except (OSError, ValueError):  # ValueError：UTF-8 で読めない
            logger.warning("スキルを読めなかった：%s", path.name, exc_info=True)
            continue
        if got is None:
            continue
        if any(s.name == got.name for s in skills):
            logger.warning("同じ名前のスキルがあるので読み飛ばした：%s（%s）", got.name, path.name)
            continue
        skills.append(got)
    return skills
