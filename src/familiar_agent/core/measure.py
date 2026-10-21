"""計測ログ——REST 内省が読む専用の記録（記-i・2026-09-13）。

`app.log` は起動ごとに `logs/` へ回転し、人が読むためのものである。ここは**機械（REST 内省・
記-a-に）が数えるための行**を溜める別の file で、`logs/` の兄弟 `rest_logs/measure.log` に置く。
起動時に回転しない。REST が読み終えたら同じ dir 内で `measure.log.<読んだ時刻>` へ改名する
（回転するのは REST）。書き手は `WatchedFileHandler` なので、改名の次の行で新しい file を作る
（logrotate と同じ約束）。`app.log` には流さない（`propagate=False`）。

行は `時刻 種別 key=value …` の 1 行。機械が拾いやすく、人が見ても読める。値に空白と
改行は入れない。累計は持たない——並びのまま渡せば傾向が出る（累計値は傾向を渡せない）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import WatchedFileHandler
from pathlib import Path

LOGGER_NAME = "familiar_agent.measure"
DIR_NAME = "rest_logs"
FILE_NAME = "measure.log"

_logger = logging.getLogger(LOGGER_NAME)


def default_base_dir() -> Path:
    return Path.home() / ".cache" / "familiar-ai"


def setup(base_dir: "Path | None" = None) -> Path:
    """計測ログの書き手を 1 つだけ付け、file の場所を返す（`main.setup_logging` から 1 回）。"""
    path = (base_dir or default_base_dir()) / DIR_NAME / FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    for h in list(_logger.handlers):
        _logger.removeHandler(h)
        h.close()
    handler = WatchedFileHandler(str(path), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    return path


def record(kind: str, **fields: object) -> None:
    """1 行書く。`kind` は種別（`調停` など）、続きは `key=value` の並び（渡した順）。"""
    if not _logger.handlers:
        return  # 書き手が無い（試験や CLI の一部）。黙って捨てる——計測は主の仕事ではない
    body = " ".join(f"{k}={_clean(v)}" for k, v in fields.items())
    _logger.info("%s %s", kind, body)


def _clean(value: object) -> str:
    text = str(value if value is not None else "-")
    return text.replace("\n", " ").replace(" ", "_") or "-"


# ── 読み手と集計（層 3 が使う・記-a-に） ──────────────────────────────────────


@dataclass(frozen=True)
class Row:
    when: datetime
    kind: str
    fields: dict


def _path(base_dir: "Path | None") -> Path:
    return (base_dir or default_base_dir()) / DIR_NAME / FILE_NAME


def read_rows(base_dir: "Path | None" = None) -> list[Row]:
    """`measure.log` の行を種別ごとの key=value に分けて返す（前回の改名以降＝file の全行）。"""
    p = _path(base_dir)
    if not p.exists():
        return []
    return _parse(p)


def _parse(p: Path) -> list[Row]:
    """計測ログ 1 つを行に分ける。時刻の読めない行は飛ばす。"""
    out: list[Row] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split(" ")
        if len(parts) < 2:
            continue
        try:
            when = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
        kind = parts[1]
        fields: dict = {}
        for kv in parts[2:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                fields[k] = v
        out.append(Row(when=when, kind=kind, fields=fields))
    return out


def read_rows_since(since: datetime, base_dir: "Path | None" = None) -> list[Row]:
    """`since` 以後の行を、改名済みの計測ログもまたいで返す（出-bs 段 2・主LLM のキャッシュは 7 日分を見る）。

    `since` より前に改名したファイルには `since` 以後の行が無いので読まない。読んだファイルの中でも行の時刻で絞る。
    ほかの規則は今までどおり `read_rows`（前の晩からの分）を使う。
    """
    d = _path(base_dir).parent
    if not d.exists():
        return []
    out: list[Row] = []
    for p in sorted(d.glob(f"{FILE_NAME}*")):
        stamp = p.name[len(FILE_NAME) + 1 :]
        if stamp:
            try:
                if datetime.strptime(stamp, "%Y%m%dT%H%M%S") < since:
                    continue
            except ValueError:
                continue  # 名前の読めないファイルは計測ログではない
        out.extend(r for r in _parse(p) if r.when >= since)
    return out


def rotate(base_dir: "Path | None" = None) -> "Path | None":
    """読み終えた `measure.log` を同じ dir で `measure.log.<読んだ時刻>` へ改名する（回転するのは REST）。

    書き手は `WatchedFileHandler` なので、次の行で新しい file を作る。無ければ何もしない。
    """
    p = _path(base_dir)
    if not p.exists():
        return None
    target = p.with_name(f"{FILE_NAME}.{datetime.now().strftime('%Y%m%dT%H%M%S')}")
    p.rename(target)
    return target


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]


def summarize_arbiter(rows: list[Row]) -> dict:
    """`調停` の行から、件数・中央・p90・最大・時間切れの回数と割合（`arbiter_timeout_sec` の材料）。"""
    secs = [float(r.fields.get("秒", 0)) for r in rows if r.kind == "調停"]
    timeouts = sum(1 for r in rows if r.kind == "調停" and r.fields.get("時間切れ") == "yes")
    n = len(secs)
    return {
        "件数": n,
        "中央": _quantile(secs, 0.5),
        "p90": _quantile(secs, 0.9),
        "最大": max(secs) if secs else 0.0,
        "時間切れ": timeouts,
        "時間切れの割合": (timeouts / n) if n else 0.0,
    }


def summarize_inner_state(rows: list[Row]) -> dict:
    """`気分`（P・Pn・A・Dom）と `欲求`（5 軸）の行から、軸ごとの分位（境目の材料）。"""
    out: dict = {"件数": {"気分": 0, "欲求": 0}}
    for kind in ("気分", "欲求"):
        cols: dict[str, list[float]] = {}
        for r in rows:
            if r.kind != kind:
                continue
            out["件数"][kind] += 1
            for k, v in r.fields.items():
                try:
                    cols.setdefault(k, []).append(float(v))
                except ValueError:
                    continue
        for k, vals in cols.items():
            out[k] = {
                q: _quantile(vals, p)
                for q, p in (("p10", 0.1), ("p30", 0.3), ("p70", 0.7), ("p90", 0.9))
            }
    return out


def summarize_relation(rows: list[Row]) -> dict:
    """`関連`（どちらの並びから載せたか）と `申告`（参照された id）を突き合わせ、参照された数を並びごとに。"""
    counts = {"遠い": 0, "掘り": 0, "件数": 0}
    last: "dict | None" = None
    for r in rows:
        if r.kind == "関連":
            last = r.fields
        elif r.kind == "申告" and last is not None:
            counts["件数"] += 1
            used = set()
            for k in ("important", "referred"):
                used |= {i for i in r.fields.get(k, "-").split(",") if i and i != "-"}
            for side in counts:
                ids = {i for i in last.get(side, "-").split(",") if i and i != "-"}
                counts[side] += len(ids & used)
            last = None
    return counts


#: 主LLM のキャッシュが残る長さ（Anthropic の ephemeral・5 分）。前の反復からこれ以内なら当たりとみなす。
CACHE_TTL_SEC = 300.0


def summarize_main_llm(rows: list[Row]) -> dict:
    """主LLM の反復の件数と、時刻で数えた当たり（出-bs 段 2・本人の決定ア）。

    当たりは「前の反復から 5 分以内・道具の数が同じ」。キャッシュを使わない週は本物の当たりが記録されないので、
    使う週も使わない週も同じ物差しで数える。最初の 1 回は外れ（前が無い）。
    """
    calls = sorted((r for r in rows if r.kind == "主LLM"), key=lambda r: r.when)
    hits = sum(
        1
        for prev, cur in zip(calls, calls[1:])
        if (cur.when - prev.when).total_seconds() <= CACHE_TTL_SEC
        and cur.fields.get("道具") == prev.fields.get("道具")
    )
    n = len(calls)
    return {"件数": n, "当たり": hits, "割合": hits / n if n else 0.0}
