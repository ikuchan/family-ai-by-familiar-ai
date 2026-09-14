"""REST 内省・層 3「設定値を調整する」（記-a-に・2026-09-14・`設計方針_REST内省_設定値を調整する`）。

計測ログ（`rest_logs/measure.log`）を機械が集計し、**数字**にしてから、登録した規則で値を動かす。
1 晩に 1 値につき 1 刻み、範囲の端で止まる。機械で決まる規則は機械が、傾向の読みが要る値だけ
LLM が数字を見て提案する（生の行は渡さない）。動かしたら `内省` の記録と計測ログに残し、読み終えた
計測ログは改名する（回転するのは REST）。

- 窓 $n$（`recent_exchanges_*`）：続きの相手が窓の端か外に 20% 以上なら +1、続きがあるのに端も外も
  参照されなければ −1（`課題5` D 章）。
- 内部状態の境目：計測した分位へ 1 刻みずつ近づける（等頻度を保つ・`課題5` C2 節）。
- `diffuse_far_share`：参照された側へ 0.1 寄せる。
- `arbiter_timeout_sec`：件数・中央・p90・最大・時間切れの割合を LLM に渡し、提案を 1 刻み・範囲内に丸める。
"""

from __future__ import annotations

import json
import logging
import re

from .. import config_overrides as co
from ..core import measure, settings

logger = logging.getLogger(__name__)

Change = tuple[str, float, float]  # (完全名, 前, 後)

EDGE_SHARE = 0.20  # 窓の端か外の参照がこの割合以上なら広げる〔仮・課題5 D 章〕


def _current(field: str, default: float) -> float:
    v = co.load_overrides().get(field)
    try:
        return float(v) if v is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _step_toward(field: str, current: float, target: float) -> "float | None":
    """登録の刻みで 1 段だけ目標へ近づける。刻み未満の差は動かさない。範囲の端で止まる。"""
    s = settings.get(field)
    if s is None:
        return None
    if abs(target - current) < s.step - 1e-9:
        return None
    nxt = current + s.step if target > current else current - s.step
    nxt = max(s.lo, min(s.hi, nxt))
    if abs(nxt - current) < 1e-9:
        return None
    return round(nxt, 6)


def _defaults() -> dict[str, float]:
    from ..config import InnerStateConfig, MemoryConfig

    m = MemoryConfig()
    inner = InnerStateConfig()
    out: dict[str, float] = {
        "MemoryConfig.recent_exchanges_main": m.recent_exchanges_main,
        "MemoryConfig.recent_exchanges_arbiter": m.recent_exchanges_arbiter,
        "MemoryConfig.diffuse_far_share": m.diffuse_far_share,
        "AgentConfig.arbiter_timeout_sec": 5.0,
    }
    for axis in ("p", "pn", "a", "dom"):
        bands = getattr(inner, f"mood_{axis}")
        for q, v in zip(("p10", "p30", "p70", "p90"), bands):
            out[f"InnerStateConfig.mood_{axis}_{q}"] = float(v)
    for axis in ("seeking", "rest", "bond", "safety", "esteem"):
        out[f"InnerStateConfig.drive_p70_{axis}"] = float(getattr(inner, f"drive_p70_{axis}"))
    return out


def adjust_window(summary: dict) -> list[Change]:
    """窓 $n$（主LLM と軽量LLM の両方を同じ向きに 1 刻み）。"""
    if not summary.get("続き"):
        return []
    share = float(summary.get("端か外の割合", 0.0))
    direction = 1 if share >= EDGE_SHARE else (-1 if share == 0.0 else 0)
    if direction == 0:
        return []
    out: list[Change] = []
    for field in ("MemoryConfig.recent_exchanges_main", "MemoryConfig.recent_exchanges_arbiter"):
        cur = _current(field, _defaults()[field])
        s = settings.get(field)
        assert s is not None
        nxt = max(s.lo, min(s.hi, cur + direction * s.step))
        if nxt != cur:
            out.append((field, cur, nxt))
    return out


def adjust_inner_state(summary: dict) -> list[Change]:
    """気分（P・Pn・A・Dom の p10/p30/p70/p90）と欲求（5 軸の p70）の境目を、計測した分位へ 1 刻み近づける。"""
    defaults = _defaults()
    out: list[Change] = []
    axis_of = {"P": "p", "Pn": "pn", "A": "a", "Dom": "dom"}
    for col, qs in summary.items():
        if col in axis_of:
            for q in ("p10", "p30", "p70", "p90"):
                field = f"InnerStateConfig.mood_{axis_of[col]}_{q}"
                if q in qs:
                    cur = _current(field, defaults[field])
                    nxt = _step_toward(field, cur, float(qs[q]))
                    if nxt is not None:
                        out.append((field, cur, nxt))
        elif col.upper() in ("SEEKING", "REST", "BOND", "SAFETY", "ESTEEM") and "p70" in qs:
            field = f"InnerStateConfig.drive_p70_{col.lower()}"
            cur = _current(field, defaults[field])
            nxt = _step_toward(field, cur, float(qs["p70"]))
            if nxt is not None:
                out.append((field, cur, nxt))
    return out


def adjust_far_share(summary: dict) -> list[Change]:
    far, stale = int(summary.get("遠い", 0)), int(summary.get("掘り", 0))
    if far == stale:
        return []
    field = "MemoryConfig.diffuse_far_share"
    cur = _current(field, _defaults()[field])
    nxt = _step_toward(field, cur, 1.0 if far > stale else 0.0)
    return [(field, cur, nxt)] if nxt is not None else []


_TIMEOUT_PROMPT = """\
調停（軽量LLM）の時間切れの秒数を見直す。いまの値は {current} 秒（範囲 {lo}〜{hi}・一晩に {step} 秒まで）。
前回以降の実測（数字だけ）：件数 {n}・中央 {median} 秒・p90 {p90} 秒・最大 {max} 秒・時間切れ {timeouts} 回（{rate:.0%}）。
時間切れが多ければ伸ばし、p90 が十分小さければ縮める。変えないなら同じ値を返す。
出力は JSON だけ：{{"arbiter_timeout_sec": <数>, "reason": "<一文>"}}
"""


async def adjust_timeout(agent, summary: dict) -> list[Change]:
    """`arbiter_timeout_sec` は LLM が数字を見て提案する。提案は 1 刻み・範囲内に丸める。"""
    field = "AgentConfig.arbiter_timeout_sec"
    s = settings.get(field)
    assert s is not None
    if not summary.get("件数"):
        return []
    cur = _current(field, _defaults()[field])
    prompt = _TIMEOUT_PROMPT.format(
        current=cur,
        lo=s.lo,
        hi=s.hi,
        step=s.step,
        n=summary["件数"],
        median=summary["中央"],
        p90=summary["p90"],
        max=summary["最大"],
        timeouts=summary["時間切れ"],
        rate=float(summary["時間切れの割合"]),
    )
    try:
        raw = str(await agent.backend.complete(prompt, max_tokens=120) or "")
    except Exception as e:  # noqa: BLE001
        logger.warning("rest 設定値：時間切れの提案に失敗: %s", e)
        return []
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return []
    try:
        proposed = float(json.loads(m.group(0)).get("arbiter_timeout_sec"))
    except (ValueError, TypeError, json.JSONDecodeError, AttributeError):
        return []
    nxt = _step_toward(field, cur, proposed)
    return [(field, cur, nxt)] if nxt is not None else []


async def apply(agent, changes: list[Change]) -> int:
    """動かす：`config_overrides` に書き、`内省` の記録と計測ログに残す。書けた数を返す。"""
    done: list[str] = []
    for field, before, after in changes:
        if co.save_override(field, after):
            done.append(f"{field}：{before} → {after}")
            measure.record("設定値", 名前=field, 前=before, 後=after)
    co.clear_cache()
    if done:
        content = "設定値を見直した：" + "／".join(done)
        await agent._memory.save_async_with_id(
            content[:500],
            direction="内省",
            kind="observation",
            materialize_now=True,
            **agent._observation_perspective(),
        )
        logger.info("rest 設定値：%d 件を動かした", len(done))
    return len(done)


async def adjust_settings(agent) -> int:
    """層 3 の 1 段：読む → 集計 → 規則で動かす → 記録 → 計測ログを改名する。"""
    rows = measure.read_rows()
    changes: list[Change] = []
    changes += adjust_window(measure.summarize_window(rows))
    changes += adjust_inner_state(measure.summarize_inner_state(rows))
    changes += adjust_far_share(measure.summarize_relation(rows))
    changes += await adjust_timeout(agent, measure.summarize_arbiter(rows))
    n = await apply(agent, changes)
    rotated = measure.rotate()
    if rotated is not None:
        logger.info("rest 計測ログを改名した：%s", rotated.name)
    return n
