"""Config の自己調整の器（記-a）。

設計（`直近の進め方と進捗` v0.14）は、内省の1パスに「Config 自己調整（範囲内・人の設定は
変えない）」を含める。ここはその器で、**内省が値を提案する部分は記-a-に で足す**。

値の優先順位は3段。

    env（人が明示） > agent_state（内省が調整） > Config の既定

**人が env に書いた値が always 勝つ。** 設計の「人の設定は変えない」をこの順序で満たす。

**範囲を登録した値だけを調整できる。** 登録が無ければ拒否する（安全側）。`rate` や
`theta_fire` のような式の骨格を決める値を内省が動かすと、蓄積や発火が壊れる。

**接続情報は常に拒否する。** 書き換えられると機器へ繋がらなくなり、しかも `agent_state`
（DB）に入るので `.env` を見ても原因が分からない。復旧に人手が要る。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_STATE_KEY = "config_overrides"

# 調整してよい値と、その範囲（下限, 上限）。**ここに無い値は変えられない。**
# 範囲の根拠は `計測・設定値 根拠台帳` に書く。
RANGES: dict[str, tuple[float, float]] = {
    # 蒸留の材料から外す新規性の下限。実測の分布は 最小 0.143・p10 0.469・p25 0.604 で、
    # 0.20 は「ほぼ何も外さない」、0.70 は「4割近く外す」に当たる。この外側を選ぶ理由が無い。
    "MemoryConfig.distill_min_a0": (0.20, 0.70),
}

# 接続情報を表す語。フィールド名がこれで終わる／これを含むものは調整できない。
_PROTECTED_PARTS = ("key", "secret", "password", "username", "host", "token", "url", "id")

# プロセス内のキャッシュ。Config は使うたびに生成されるので、毎回 DB を触ると重い。
# 内省が書いたときと、テストが明示したときに捨てる。
_cache: dict[str, Any] | None = None


def is_protected(field: str) -> bool:
    """接続情報かどうか。`CameraConfig.password` のような完全名で渡す。"""
    name = field.rsplit(".", 1)[-1].lower()
    return any(part in name.split("_") for part in _PROTECTED_PARTS)


def clear_cache() -> None:
    """次に読むとき DB から取り直す。"""
    global _cache
    _cache = None


def load_overrides() -> dict[str, Any]:
    """内省が調整した値を返す。読めなければ空（Config の既定へ落ちる）。"""
    global _cache
    if _cache is not None:
        return _cache
    result: dict[str, Any] = {}
    try:
        import psycopg2.extras

        from .db import get_db

        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT value_json FROM agent_state WHERE state_key = %s", (_STATE_KEY,)
                )
                row = cur.fetchone()
        if row:
            loaded = json.loads(row["value_json"])
            if isinstance(loaded, dict):
                result = loaded
    except Exception as e:  # noqa: BLE001
        # 読めないだけで動作は止めない。Config の既定が使われる。
        logger.debug("Config の調整値を読めなかったので既定を使う: %s", e)
    _cache = result
    return result


def save_override(field: str, value: Any) -> bool:
    """内省の提案を保存する。受け付けたら True。

    拒否する理由は3つ（接続情報・未登録・範囲外）で、いずれもログに残す。黙って捨てると、
    内省が「調整したつもり」のまま値が変わらない状態になり、原因を追えない。
    """
    if is_protected(field):
        logger.warning("Config の調整を拒否（接続情報は変えられない）: %s", field)
        return False
    span = RANGES.get(field)
    if span is None:
        logger.warning("Config の調整を拒否（範囲が登録されていない）: %s", field)
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        logger.warning("Config の調整を拒否（数値でない）: %s=%r", field, value)
        return False
    low, high = span
    if not (low <= number <= high):
        logger.warning("Config の調整を拒否（範囲外 %s〜%s）: %s=%s", low, high, field, number)
        return False

    current = dict(load_overrides())
    current[field] = number
    try:
        from .db import get_db

        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_state (state_key, value_json, updated_at)"
                    " VALUES (%s, %s, %s)"
                    " ON CONFLICT (state_key) DO UPDATE"
                    "   SET value_json = EXCLUDED.value_json,"
                    "       updated_at = EXCLUDED.updated_at",
                    (_STATE_KEY, json.dumps(current), datetime.now(timezone.utc).isoformat()),
                )
            conn.commit()
    except Exception as e:  # noqa: BLE001
        logger.exception("Config の調整を保存できなかった: %s", e)
        return False
    clear_cache()
    logger.info("Config を調整した：%s=%s", field, number)
    return True


def resolve_float(field: str, env_name: str, default: float) -> float:
    """3段の優先順位で値を決める。`env > agent_state > 既定`。

    `field` は `MemoryConfig.distill_min_a0` のような完全名。
    """
    if os.environ.get(env_name) is not None:
        try:
            return float(os.environ[env_name])
        except ValueError:
            return default
    value = load_overrides().get(field)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _delete_all() -> None:
    """保存済みの調整をすべて消す（テスト用）。"""
    try:
        from .db import get_db

        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM agent_state WHERE state_key = %s", (_STATE_KEY,))
            conn.commit()
    except Exception as e:  # noqa: BLE001
        logger.debug("Config の調整を消せなかった: %s", e)
    clear_cache()
