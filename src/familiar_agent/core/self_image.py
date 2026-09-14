"""自己像——REST 内省の層 2（記-a-へ・2026-09-14・`設計方針_REST内省_自己像` v0.2）。

**いま自分が何を望み、何を気にかけ、何を大事にしているか。** 出来事（層 1）と能力（層 4）のあいだに
立つ抽象で、字数に厳密な上限を持ち、大きくしない。記憶ではない——W には入らず、システム文の
`[守っている決まり]` と同じ層に `[いまの自分]` として載る。

- 現在値は DB（`agent_state` の鍵 `self_image`）。無ければ repo の種 `defaults/self_image.yaml` を
  読んで DB に置く（状態はすべて DB、ファイルは既定と人の入力だけ）。
- 3 欄：望み 8 行 × 60 字・気がかり 8 行 × 60 字・価値 10 行 × 50 字・合計 1,500 字以内。
- 各行は本文・入った日（`since`）・出典（材料にした自己エピソードの id）を持つ。文にするときは
  数値も id も出さず、古さだけ「（9/10 から）」の印で添える。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_KEY = "self_image"
SEED_PATH = Path(__file__).resolve().parents[3] / "defaults" / "self_image.yaml"

#: 欄の名前と（行数の上限, 1 行の字数の上限）。合計の上限は `TOTAL_MAX_CHARS`。
FIELDS: dict[str, tuple[int, int]] = {"望み": (8, 60), "気がかり": (8, 60), "価値": (10, 50)}
TOTAL_MAX_CHARS = 1500
_ATTR = {"望み": "hopes", "気がかり": "concerns", "価値": "values"}


@dataclass(frozen=True)
class Line:
    text: str
    since: date
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelfImage:
    hopes: tuple[Line, ...] = ()
    concerns: tuple[Line, ...] = ()
    values: tuple[Line, ...] = ()

    def field(self, name: str) -> tuple[Line, ...]:
        return getattr(self, _ATTR[name])

    def replace_field(self, name: str, lines: "list[Line] | tuple[Line, ...]") -> "SelfImage":
        return replace(self, **{_ATTR[name]: tuple(lines)})

    def replace_line(self, name: str, index: int, line: Line) -> "SelfImage":
        lines = list(self.field(name))
        lines[index] = line
        return self.replace_field(name, lines)

    def total_chars(self) -> int:
        return sum(len(x.text) for f in FIELDS for x in self.field(f))


# ── 検査 ──────────────────────────────────────────────────────────────────────


def check(image: SelfImage) -> "str | None":
    """字数と行数の検査。通れば None、通らなければ理由（通らない更新は保存しない）。"""
    for name, (max_rows, max_chars) in FIELDS.items():
        lines = image.field(name)
        if not lines:
            return f"{name}が空"
        if len(lines) > max_rows:
            return f"{name}が {max_rows} 行を超えた（{len(lines)} 行）"
        for x in lines:
            if not x.text.strip():
                return f"{name}に空の行"
            if len(x.text) > max_chars:
                return f"{name}の行が {max_chars} 字を超えた（{len(x.text)} 字）"
    if image.total_chars() > TOTAL_MAX_CHARS:
        return f"合計が {TOTAL_MAX_CHARS} 字を超えた（{image.total_chars()} 字）"
    return None


# ── 文 ────────────────────────────────────────────────────────────────────────


def render(image: SelfImage, *, today: "date | None" = None) -> str:
    """システム文の枠 `[いまの自分]`。数値も id も出さない。古さは「（9/10 から）」で添える。"""
    today = today or datetime.now(timezone.utc).date()
    parts = ["[いまの自分]"]
    for name in FIELDS:
        parts.append(f"{name}：")
        for x in image.field(name):
            age = "" if x.since >= today else f"（{x.since.month}/{x.since.day} から）"
            parts.append(f"- {x.text}{age}")
    return "\n".join(parts)


# ── 器（DB と種） ─────────────────────────────────────────────────────────────


def _from_json(data: dict) -> SelfImage:
    def lines(name: str) -> tuple[Line, ...]:
        out = []
        for item in data.get(name) or []:
            if isinstance(item, str):
                out.append(Line(text=item, since=date.today(), sources=()))
            else:
                out.append(
                    Line(
                        text=str(item.get("text", "")),
                        since=date.fromisoformat(str(item.get("since"))),
                        sources=tuple(str(s) for s in (item.get("sources") or [])),
                    )
                )
        return tuple(out)

    return SelfImage(hopes=lines("望み"), concerns=lines("気がかり"), values=lines("価値"))


def _to_json(image: SelfImage) -> dict:
    return {
        name: [
            {"text": x.text, "since": x.since.isoformat(), "sources": list(x.sources)}
            for x in image.field(name)
        ]
        for name in FIELDS
    }


def seed(path: "Path | None" = None, *, today: "date | None" = None) -> SelfImage:
    """種を読む。行の本文だけの YAML を、出典なし・今日の日づけの行にする。"""
    import yaml  # type: ignore[import-untyped]

    data = yaml.safe_load((path or SEED_PATH).read_text(encoding="utf-8")) or {}
    today = today or datetime.now(timezone.utc).date()
    return SelfImage(
        **{
            _ATTR[name]: tuple(
                Line(text=str(t), since=today, sources=()) for t in (data.get(name) or [])
            )
            for name in FIELDS
        }
    )


def stored() -> "SelfImage | None":
    """DB にある現在値。無ければ None。"""
    try:
        from ..db import get_db

        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor() as cur:
                cur.execute("SELECT value_json FROM agent_state WHERE state_key = %s", (STATE_KEY,))
                row = cur.fetchone()
        if not row:
            return None
        raw = row["value_json"] if isinstance(row, dict) else row[0]
        return _from_json(json.loads(raw))
    except Exception as e:  # noqa: BLE001
        logger.warning("自己像を読めなかった: %s", e)
        return None


def store(image: SelfImage) -> bool:
    """現在値を DB（`agent_state`）に置く。O への書き込みではない（履歴は `内省` の記録が持つ）。"""
    try:
        from ..db import get_db

        now = datetime.now(timezone.utc).isoformat()
        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_state (state_key, value_json, updated_at) VALUES (%s, %s, %s) "
                    "ON CONFLICT (state_key) DO UPDATE SET value_json = EXCLUDED.value_json, "
                    "updated_at = EXCLUDED.updated_at",
                    (STATE_KEY, json.dumps(_to_json(image), ensure_ascii=False), now),
                )
            conn.commit()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("自己像を保存できなかった: %s", e)
        return False


def load() -> SelfImage:
    """現在値。DB に無ければ種を読んで DB に置き、それを返す。"""
    got = stored()
    if got is not None:
        return got
    image = seed()
    store(image)
    logger.info("自己像を種から置いた（%d 字）", image.total_chars())
    return image
