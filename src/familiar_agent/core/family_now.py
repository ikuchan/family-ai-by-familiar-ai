"""家族のいまの様子（知-ad 段 2・2026-10-01・本人の決定イ）。

性格や好みは変わるのに、`FAMILY.md` は人が書き直さなければ古いままになる。`FAMILY.md` は人の入力のまま機械は
書き換えず（開発ルール「ファイルは既定値と人の入力だけ」・知-ac の季節の層と同じ判断）、REST が記憶をもとに
人ごとに書いた「いまの様子」を DB（`agent_state` の鍵 `family_now`）に置いて、システム文の別枠に載せる。

人ごとに持つのは 3 つ：本文、前の版（1 つだけ・比べられるように）、数え始めの時刻（最後に読んだ関係のまとめの
時刻。この先の関係のまとめを古い順に 10 本ずつ読んで書き直す・`loop/rest_family_now.py`）。履歴は `内省` の記録が持つ。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from . import state_json

logger = logging.getLogger(__name__)

STATE_KEY = "family_now"
HEADING = "[家族のいまの様子]"


@dataclass(frozen=True)
class Now:
    text: str
    before: str
    counted_from: datetime


def _to_json(data: "dict[str, Now]") -> dict:
    return {
        name: {"text": n.text, "before": n.before, "counted_from": n.counted_from.isoformat()}
        for name, n in data.items()
    }


def _from_json(raw: object) -> "dict[str, Now]":
    out: dict[str, Now] = {}
    if not isinstance(raw, dict):
        return out
    for name, v in raw.items():
        try:
            out[str(name)] = Now(
                text=str(v.get("text", "")),
                before=str(v.get("before", "")),
                counted_from=datetime.fromisoformat(str(v["counted_from"])),
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return out


def stored() -> "dict[str, Now]":
    """DB にある人ごとのいまの様子。無ければ空。"""
    return _from_json(state_json.read(STATE_KEY))


def _store(data: "dict[str, Now]") -> bool:
    return state_json.write(STATE_KEY, _to_json(data))


def update(name: str, text: str, *, counted_from: datetime) -> bool:
    """その人のいまの様子を書き直す。前の版を 1 つ控え、数え始めを `counted_from`（最後に読んだ時刻）にする。"""
    data = stored()
    old = data.get(name)
    data[name] = Now(text=text, before=old.text if old else "", counted_from=counted_from)
    return _store(data)


def clear() -> bool:
    """消す（試験と、やり直したいとき）。"""
    return state_json.clear(STATE_KEY)


def render(data: "dict[str, Now]") -> str:
    """`[家族のいまの様子]` の枠。誰も書かれていなければ空。前の版は載せない。"""
    lines = [f"{name}：{n.text}" for name, n in data.items() if n.text.strip()]
    return HEADING + "\n" + "\n".join(lines) if lines else ""


def current_text() -> str:
    """いま DB にある分を `[家族のいまの様子]` の枠にする。読めなければ空（システム文を組む 3 か所が呼ぶ）。"""
    return render(stored())
