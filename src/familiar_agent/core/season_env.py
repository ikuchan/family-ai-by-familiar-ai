"""いまの季節とまわり——REST 内省の季節の層が書く中身（知-ac・2026-09-26・`設計方針_季節の層` v0.1）。

日付は毎ターン渡っているが、「もう寒い」「金木犀が咲いた」のような**いまの様子**は日付からは出てこない。
季節を `ME.md` に書くと書いた瞬間から古くなるので、機械が晩に 1 回書き直す。

- 現在値は DB（`agent_state` の鍵 `season_env`）。ファイルは作らない——機械が書き換えるファイルは
  「ファイルは既定値と人の入力だけ」（`CLAUDE.md`）に反するので、09-20 の `SeasonAndEnv.md` を改めた。
- 4 つの欄：暦・天気・まわり・家の話題。各欄 1〜2 行・1 行 40 字まで。
- **暦は保存しない。** 渡すたびにその日の日付から計算する（いつ渡しても正しい）。
- 天気とまわりは検索結果から**そのまま引いた文**、家の話題は出来事の id を出典に持つ。文にするときは
  出典を出さない。
- 古さで渡し方を変える：7 日までそのまま・8〜30 日は「（N 日前の様子）」・それより古い（か無い）と暦だけ。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)

STATE_KEY = "season_env"
HEADING = "[いまの季節とまわり]"
#: LLM が書く欄（暦は機械が書くので入らない）。並びが渡す順。
FIELDS = ("天気", "まわり", "家の話題")
MAX_ROWS = 2
MAX_CHARS = 40
FRESH_DAYS = 7  # ここまではそのまま渡す
STALE_DAYS = 30  # これを過ぎたら暦だけ

#: 二十四節気（決まった日付の表・本人の承認 2026-09-26）。年によって前後 1 日ずれるが、
#: 「〜を過ぎたころ」と書くので嘘にならない。
SOLAR_TERMS: tuple[tuple[int, int, str], ...] = (
    (1, 6, "小寒"),
    (1, 20, "大寒"),
    (2, 4, "立春"),
    (2, 19, "雨水"),
    (3, 6, "啓蟄"),
    (3, 21, "春分"),
    (4, 5, "清明"),
    (4, 20, "穀雨"),
    (5, 6, "立夏"),
    (5, 21, "小満"),
    (6, 6, "芒種"),
    (6, 21, "夏至"),
    (7, 7, "小暑"),
    (7, 23, "大暑"),
    (8, 8, "立秋"),
    (8, 23, "処暑"),
    (9, 8, "白露"),
    (9, 23, "秋分"),
    (10, 8, "寒露"),
    (10, 24, "霜降"),
    (11, 7, "立冬"),
    (11, 22, "小雪"),
    (12, 7, "大雪"),
    (12, 22, "冬至"),
)


@dataclass(frozen=True)
class Row:
    text: str
    #: 天気・まわりは検索結果からそのまま引いた文、家の話題は出来事の id。
    source: str = ""


@dataclass(frozen=True)
class SeasonEnv:
    written_on: date
    rows: "dict[str, tuple[Row, ...]]" = field(default_factory=dict)


def _term(month: int, day: int) -> str:
    name = next(n for m, d, n in SOLAR_TERMS if (m, d) == (month, day))
    return f"{name}（{month}/{day}）"


def calendar_line(today: date) -> str:
    """いまの暦の行。直前の節気と次の節気を言う。"""
    key = (today.month, today.day)
    past = [(m, d) for m, d, _ in SOLAR_TERMS if (m, d) <= key]
    last = past[-1] if past else (SOLAR_TERMS[-1][0], SOLAR_TERMS[-1][1])  # 年明け前は冬至
    upcoming = [(m, d) for m, d, _ in SOLAR_TERMS if (m, d) > key]
    nxt = upcoming[0] if upcoming else (SOLAR_TERMS[0][0], SOLAR_TERMS[0][1])  # 年末は小寒
    now = f"今日は{_term(*last)}" if last == key else f"{_term(*last)}を過ぎたころ"
    return f"{now}。次は{_term(*nxt)}。"


def _squash(text: str) -> str:
    return "".join(str(text).split())


def check(env: SeasonEnv, *, search_text: str, material_ids: "set[str]") -> "str | None":
    """通さない理由。通るなら None。まとめ方は LLM、通すかどうかは機械。"""
    found = _squash(search_text)
    for name, rows in env.rows.items():
        if name not in FIELDS:
            return f"欄「{name}」は書けない（暦は機械が書く）"
        if len(rows) > MAX_ROWS:
            return f"{name}が {len(rows)} 行（{MAX_ROWS} 行まで）"
        for r in rows:
            if not r.text.strip():
                return f"{name}に空の行"
            if len(r.text) > MAX_CHARS:
                return f"{name}の行が {len(r.text)} 字（{MAX_CHARS} 字まで）"
            if name == "家の話題":
                if r.source not in material_ids:
                    return f"家の話題の出典「{r.source[:12]}」が今夜の出来事に無い"
            elif not r.source or len(r.source) > MAX_CHARS or _squash(r.source) not in found:
                return f"{name}の出典が検索結果に無い：「{r.source[:20]}」"
    return None


def render(env: "SeasonEnv | None", *, today: "date | None" = None) -> str:
    """システム文の `[いまの季節とまわり]`。出典は出さない。"""
    today = today or date.today()
    lines = [f"- 暦：{calendar_line(today)}"]
    age = (today - env.written_on).days if env is not None else None
    if env is None or age is None or age > STALE_DAYS:
        return "\n".join([HEADING, *lines])
    head = f"{HEADING}（{env.written_on.isoformat()} の晩に書いた）"
    if age > FRESH_DAYS:
        head += f"（{age} 日前の様子）"
    for name in FIELDS:
        lines += [f"- {name}：{r.text}" for r in env.rows.get(name, ())]
    return "\n".join([head, *lines])


def _to_json(env: SeasonEnv) -> dict:
    return {
        "written_on": env.written_on.isoformat(),
        "rows": {k: [{"text": r.text, "source": r.source} for r in v] for k, v in env.rows.items()},
    }


def _from_json(data: dict) -> SeasonEnv:
    return SeasonEnv(
        written_on=date.fromisoformat(str(data["written_on"])),
        rows={
            str(k): tuple(Row(str(r.get("text", "")), str(r.get("source", ""))) for r in v)
            for k, v in (data.get("rows") or {}).items()
        },
    )


def stored() -> "SeasonEnv | None":
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
        return _from_json(json.loads(raw) if isinstance(raw, str) else raw)
    except Exception as e:  # noqa: BLE001
        logger.warning("季節とまわりを読めなかった: %s", e)
        return None


def store(env: SeasonEnv) -> bool:
    """現在値を DB（`agent_state`）に置く。履歴は `内省` の記録が持つ。"""
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
                    (STATE_KEY, json.dumps(_to_json(env), ensure_ascii=False), now),
                )
            conn.commit()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("季節とまわりを保存できなかった: %s", e)
        return False


def clear() -> bool:
    """消す（`/season clear`）。消したあとは暦だけが渡る。"""
    try:
        from ..db import get_db

        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM agent_state WHERE state_key = %s", (STATE_KEY,))
            conn.commit()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("季節とまわりを消せなかった: %s", e)
        return False
