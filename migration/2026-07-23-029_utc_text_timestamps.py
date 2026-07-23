"""TEXT 時刻列の既存ローカル値を UTC（aware ISO）へ寄せる（DB=UTC 統一）。

`clock.now_local_iso()`（ローカル・tz なし）で書かれていた TEXT 時刻列を、
書き込み側は `clock.now_utc_iso()`（UTC・`+00:00`）へ直した。既存行はローカル
時計のままなので、ここで同じ UTC 時計へ寄せる。対象は保存時刻列のみで、
`observations.timestamp`（timestamptz・`now_utc()` で正しく UTC）と、
`observations.py` のローカル暦日境界計算（`now_local_iso()` を維持）は対象外。

変換：ローカル naive ISO 文字列 → `(値 − ローカルオフセット)` に `+00:00` を付す。
ローカルオフセットは移行時点の OS 値（JST は UTC+9・夏時間なし）を凍結して使う。

二重適用の防止は二段。
1. ランナー（`db_migrations`）が `schema_migrations` で適用済みを記録し二度実行しない（主）。
2. 本体でも tz サフィックスの無い行（`position('+' in col)=0`）だけ変換するので、
   既に UTC 化された行（`+00:00` 付き）は再実行してもスキップされる（保険）。

全行を一律に寄せてよいのは、移行時点の TEXT 時刻列がすべてローカル書き込み側で
作られたものだからである（書き込みの修正とこの移行が同時に入る）。
"""

from __future__ import annotations

# 対象 table → 時刻列（マッピング調査で確定）。available_at/last_seen_at/created_at は
# 比較・ORDER BY に効くが、全行を同一変換するので辞書順の整合は保たれる。
_TARGETS: dict[str, list[str]] = {
    "memory_jobs": ["available_at", "created_at", "updated_at"],
    "memory_events": ["created_at"],
    "semantic_facts": ["last_seen_at", "created_at", "updated_at"],
    "behavior_policies": ["last_seen_at", "created_at", "updated_at"],
    "memory_revisions": ["created_at"],
    "memory_links": ["created_at"],
    "persons": ["created_at", "updated_at"],
    "episodes": ["created_at", "updated_at"],
    "episode_memories": ["added_at"],
    "memory_activation": ["activated_at"],
    "unfinished_business": ["created_at"],
}


def _offset_seconds() -> int:
    from datetime import datetime

    from familiar_agent.store.clock import local_tz

    off = datetime.now(local_tz()).utcoffset()
    return int(off.total_seconds()) if off else 0


def upgrade(conn) -> None:
    off_seconds = _offset_seconds()
    with conn.cursor() as cur:
        for table, cols in _TARGETS.items():
            for col in cols:
                # naive（tz サフィックス無し）だけ変換。ローカル→UTC は値からオフセットを引く。
                cur.execute(
                    f"UPDATE {table} SET {col} = "
                    f"to_char(({col}::timestamp - make_interval(secs => %s)), "
                    f"'YYYY-MM-DD\"T\"HH24:MI:SS.US') || '+00:00' "
                    f"WHERE {col} IS NOT NULL AND position('+' in {col}) = 0",
                    (off_seconds,),
                )
    conn.commit()
