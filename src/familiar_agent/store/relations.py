"""MI 間の関係（`relations` と `relation_members`）。

記録どうしのつながりを、二項の列ではなく**多項の関係**として持つ
（`設計方針_MI間の関係` v0.1）。一つの関係が何個でも項を持ち、`position` で順序を
表す。改訂（旧と新）、継起（前と後）、やりとり（問いと版と答えと要約）、共起
（一緒に活性した記録の集まり）が、同じ器に載る。

この2テーブルを触るのはこのモジュールだけにする。

段 1 では、既存の経路からこの口を呼ばない。想起の絞り（改訂の旧として現れるか）と
連なりの辿りは、それを使う段で問い合わせの形が決まってから足す。ここに先回りして
置くと、実際には要らない形の口が残る。

使うものは文脈（`StoreContext`）から受け取る。
"""

from __future__ import annotations

import logging

from .context import StoreContext

logger = logging.getLogger(__name__)

# 項の並び。`(観測 id, 役割, 位置)`。位置は順序を持たない関係で None を取る。
Member = tuple[str, str, "int | None"]

# 「もう現行ではない」を表す役割。改訂も畳み込みも解決も前進も、理由は違うが帰結は同じ
# なので、帰結をこの役割に担わせ、種類は理由だけを言う（`設計方針_MI間の関係` v0.3）。
ROLE_OLD = "旧"

# 「口に出したが、覚えておく中身がない」を表す役割。つなぎの一言がこれにあたる。
ROLE_FILLER = "つなぎ"

# **想起に出さない役割。** 理由は二つあるが、帰結は同じである。
HIDDEN_ROLES = (ROLE_OLD, ROLE_FILLER)

# 種類は、隠す理由を言うだけである。隠すかどうかは役割 `旧` が決める。
KIND_REVISION = "改訂"  # 版が進み、前の版が現在の状態としては誤りになった
KIND_FOLD = "畳み込み"  # 逐語が要約に吸われた。畳まれた側は誤りではない
KIND_RESOLVE = "解決"  # 保留していたことが果たされた
KIND_ADVANCE = "前進"  # 記録の鎖が一つ進んだ
KIND_EXCHANGE = "やりとり"  # 一つのターンの記録（起点・版・見た・答え・要約）
KIND_SUCCESSION = "継起"  # ターンどうしの前後
KIND_UNCLASSIFIED = "未分類"  # 059 が移した既存の辺。どの書き手が作ったか判別できない


def not_hidden(alias: str = "o") -> str:
    """その観測がまだ現行であることを表す SQL の述語を返す。

    **隠すかどうかは役割だけで決まる。** 種類を見ないので `relations` との結合が要らず、
    `idx_relation_members_obs(obs_id, role)` を一度引くだけで済む。

    想起の絞りはこの一箇所から取る。17 箇所へ同じ文を書き写すと、意味を変えたときに
    書き換え漏れが出る。
    """
    return "NOT " + hidden(alias)


def hidden(alias: str = "o") -> str:
    """その観測がもう現行でないことを表す SQL の述語を返す。`not_hidden` の裏。"""
    _roles = ", ".join(f"'{r}'" for r in HIDDEN_ROLES)
    return (
        "EXISTS (SELECT 1 FROM relation_members _rm "
        f"WHERE _rm.obs_id = {alias}.id AND _rm.role IN ({_roles}))"
    )


class RelationStore:
    """関係の持ち主。"""

    def __init__(self, ctx: StoreContext) -> None:
        self._ctx = ctx

    def add(self, kind: str, members: "list[Member]") -> "int | None":
        """関係を1つ書き、その id を返す。項が空なら書かず None を返す。

        ヘッダと項は同じトランザクションで書く。途中で落ちたときにヘッダだけが残ると、
        項の無い関係が溜まり、関係の数が実際のつながりの数と合わなくなる。
        """
        rows = [(str(o), str(r), p) for o, r, p in members if o]
        if not rows:
            return None
        with self._ctx.lock:
            conn = self._ctx.conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO relations (kind) VALUES (%s) RETURNING id", (kind,))
                    rid = int(cur.fetchone()["id"])
                    cur.executemany(
                        "INSERT INTO relation_members "
                        "(relation_id, obs_id, role, position) VALUES (%s, %s, %s, %s)",
                        [(rid, o, r, p) for o, r, p in rows],
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return rid

    def hide(self, old_id: str, new_id: str, kind: str = KIND_UNCLASSIFIED) -> bool:
        """`old_id` を現行から外し、`new_id` を新しい側として関係で結ぶ。

        **先着が勝つ。** 既に `旧` になっている観測は、部分一意索引
        （`idx_relation_members_old`）が二本目を落とす。張り替えると「どの記録が解決
        したか」のつながりが失われるので、二度目は何もせず `False` を返す。
        """
        from psycopg2.errors import UniqueViolation

        try:
            rid = self.add(kind, [(old_id, ROLE_OLD, 0), (new_id, "新", 1)])
        except UniqueViolation:
            return False
        return rid is not None

    def latest_member(self, kind: str, role: str) -> "str | None":
        """その種類・その役割の項のうち、いちばん新しい関係のもの。

        **繋ぐためではなく、どこから見せるかのカーソルである。** 起動直後は持ち回りが
        空なので、ここから直近のやりとりの表示を始める。無ければ None。
        """
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT m.obs_id FROM relation_members m "
                    "JOIN relations r ON r.id = m.relation_id AND r.kind = %s "
                    "WHERE m.role = %s ORDER BY r.id DESC LIMIT 1",
                    (kind, role),
                )
                row = cur.fetchone()
        return None if row is None else str(row["obs_id"])

    def recent_exchanges(
        self, origin_id: str, *, roles: "tuple[str, ...]" = ("起点", "つなぎ", "答え")
    ) -> list[dict]:
        """`origin_id` から継起をさかのぼり、各やりとりの項を古い順に返す。

        **上限を持たせない。** 会話はどこかで始まってどこかで終わるので、根で自然に
        止まる。何歩まで、と決めるのは、鎖の長さを機構の側で決めることになる。

        返すのは口に出したものだけである（既定は 起点・つなぎ・答え）。`版` と `見た` は
        内部の作業記録で、会話ではない。混ぜると、調べている途中の文字列が履歴として
        読まれる。

        返りは `content`・`role`・`direction`・`timestamp`・`depth` を持つ dict の並び。
        `depth` は 0 が渡した起点で、さかのぼるほど大きい。並びは古い順である。
        """
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "WITH RECURSIVE walk AS ("
                    "  SELECT %s::text AS obs_id, 0 AS depth"
                    "  UNION ALL"
                    "  SELECT prev.obs_id, w.depth + 1 FROM walk w"
                    "    JOIN relation_members nxt"
                    "      ON nxt.obs_id = w.obs_id AND nxt.role = '後'"
                    "    JOIN relations nr"
                    "      ON nr.id = nxt.relation_id AND nr.kind = %s"
                    "    JOIN relation_members prev"
                    "      ON prev.relation_id = nxt.relation_id AND prev.role = '前'"
                    ") "
                    "SELECT o.content, m.role, o.direction, o.timestamp, w.depth "
                    "FROM walk w "
                    "JOIN relation_members head"
                    "  ON head.obs_id = w.obs_id AND head.role = '起点' "
                    "JOIN relations r"
                    "  ON r.id = head.relation_id AND r.kind = %s "
                    "JOIN relation_members m ON m.relation_id = r.id "
                    "JOIN observations o ON o.id = m.obs_id "
                    "WHERE m.role = ANY(%s) "
                    "ORDER BY w.depth DESC, m.position",
                    (origin_id, KIND_SUCCESSION, KIND_EXCHANGE, list(roles)),
                )
                return [dict(r) for r in cur.fetchall()]

    def members_of(self, relation_id: int) -> list[dict]:
        """関係の項を位置の昇順で返す。位置を持たない項は末尾に置く。"""
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT obs_id, role, position FROM relation_members "
                    "WHERE relation_id = %s ORDER BY position ASC NULLS LAST, obs_id",
                    (relation_id,),
                )
                return [dict(r) for r in cur.fetchall()]

    def relations_for(
        self, obs_id: str, kind: "str | None" = None, role: "str | None" = None
    ) -> list[int]:
        """その観測が項として入る関係の id を、古い順に返す。"""
        sql = [
            "SELECT r.id FROM relation_members m",
            "JOIN relations r ON r.id = m.relation_id",
            "WHERE m.obs_id = %s",
        ]
        args: list[object] = [str(obs_id)]
        if role is not None:
            sql.append("AND m.role = %s")
            args.append(role)
        if kind is not None:
            sql.append("AND r.kind = %s")
            args.append(kind)
        sql.append("ORDER BY r.id")
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(" ".join(sql), tuple(args))
                return [int(r["id"]) for r in cur.fetchall()]
