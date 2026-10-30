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
from datetime import datetime

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
KIND_COOCCURRENCE = "共起"  # 求めの起点と、その答えに使った記憶の集まり（順序を持たない・記-s）
KIND_UNCLASSIFIED = "未分類"  # 059 が移した既存の辺。どの書き手が作ったか判別できない
KIND_EXPERIENCE = "経験繋がり"  # 1 日の会話の塊で、意味の似ていない飛び 3 件（REST が書く・出-bq）
ROLE_TERM = "項"  # 経験繋がりの項。3 つとも同じ役割で、位置は記録の時刻の古い順


def _first(row):
    """1列だけの行から値を取り出す。

    呼び手が渡してくる接続は、経路によって tuple を返したり dict を返したりする
    （層は `RealDictCursor` で包むが、拡散想起は素の接続を持ち回している）。
    """
    if hasattr(row, "values"):
        return list(row.values())[0]
    return row[0]


#: 共起に入れる申告（答えに使った・見たが価値が無かった）。unused は入れない（記-s）。
COOCCURRING_VERDICTS = ("important", "referred", "useless")


def cooccurring_members(origin: str, verdicts: "dict[str, str]") -> "list[str] | None":
    """共起の項＝そのターンの起点と、申告が important・referred・useless の記憶（記-s・2026-10-14・本人の決定）。

    `verdicts` は完全な id → 申告。起点は判定せずに入れる。使った記憶が 2 件未満なら None（2 つにしか
    かからない網は作らない）。W の全部を入れていたころは 1 本が十数件を結び、1 日が 1 つの塊になった（出-bq の計測）。
    """
    if not origin:
        return None
    used = [i for i, v in verdicts.items() if v in COOCCURRING_VERDICTS and i != origin]
    if len(used) < 2:
        return None
    return [origin, *used]


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

    def add(
        self, kind: str, members: "list[Member]", *, at: "datetime | None" = None
    ) -> "int | None":
        """関係を1つ書き、その id を返す。項が空なら書かず None を返す。

        `at` を渡すと関係の時刻にする（経験繋がりは持ち主の塊の日・出-bq）。渡さなければ書いた時刻。

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
                    cur.execute(
                        "INSERT INTO relations (kind, created_at) "
                        "VALUES (%s, COALESCE(%s, now())) RETURNING id",
                        (kind, at),
                    )
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

    def extend(self, relation_id: int, members: "list[Member]") -> int:
        """既にある関係の末尾へ項を足す。足した数を返す。

        やりとりの関係は反復を閉じるときに書くが、会話要約は背景で遅れて来る。関係を
        要約待ちにすると、閉じた直後の反復から直近のやりとりが見えない（2026-09-13 実機）。
        位置を持たない項は、いまの最大位置の次から順に置く。
        """
        rows = [(str(o), str(r), p) for o, r, p in members if o]
        if not rows:
            return 0
        with self._ctx.lock:
            conn = self._ctx.conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COALESCE(MAX(position), -1) AS last FROM relation_members "
                        "WHERE relation_id = %s",
                        (relation_id,),
                    )
                    nxt = int(cur.fetchone()["last"]) + 1
                    placed = []
                    for o, r, p in rows:
                        if p is None:
                            p, nxt = nxt, nxt + 1
                        placed.append((relation_id, o, r, p))
                    cur.executemany(
                        "INSERT INTO relation_members "
                        "(relation_id, obs_id, role, position) VALUES (%s, %s, %s, %s)",
                        placed,
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return len(rows)

    def replace_members(self, kind: str, relation_id: int, members: "list[Member]") -> int:
        """種類が `kind` の関係の項をまるごと差し替える。id はそのまま（経験繋がりの入れ替え・出-bq）。

        種類が違う・項が空なら何もせず 0 を返す（id の取り違えで改訂の旧を消さない・項の無い関係を残さない）。
        消すのと書くのは同じトランザクションで行う。途中で落ちたら元の項のまま残る。
        """
        rows = [(str(o), str(r), p) for o, r, p in members if o]
        if not rows:
            return 0
        with self._ctx.lock:
            conn = self._ctx.conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM relations WHERE id = %s AND kind = %s FOR UPDATE",
                        (relation_id, kind),
                    )
                    if cur.fetchone() is None:
                        conn.rollback()
                        return 0
                    cur.execute(
                        "DELETE FROM relation_members WHERE relation_id = %s", (relation_id,)
                    )
                    cur.executemany(
                        "INSERT INTO relation_members "
                        "(relation_id, obs_id, role, position) VALUES (%s, %s, %s, %s)",
                        [(relation_id, o, r, p) for o, r, p in rows],
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return len(rows)

    def remove(self, kind: str, relation_id: int) -> bool:
        """種類が `kind` の関係を外す（項は `ON DELETE CASCADE` で消える）。種類が違えば外さない。

        いま外すのは、塊が育って 2 本を含んだ経験繋がりの残り（出-bq）だけである。
        """
        with self._ctx.lock:
            conn = self._ctx.conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM relations WHERE id = %s AND kind = %s", (relation_id, kind)
                    )
                    got = cur.rowcount
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return bool(got)

    def experience_material(self, since: datetime, *, final_direction: str) -> dict:
        """経験繋がりの材料（出-bq）。

        - `edges`：`since` より後に書かれた共起・継起・やりとり。`id`・`kind`・`created_at`・
          `members`（`(記録の id, 役割)` の位置順）を持つ dict の並び。
        - `parents`：その項のうち、向き `final_direction`（最終版）の記録 → 親（起点）。
        - `links`：既存の経験繋がり全部。関係の id → `at`（関係の時刻＝持ち主の塊の日）と `ids`（項の記録の id・位置順）。

        畳まれた（役割 `旧` を持つ）記録も除かない。項が後で畳まれても関係は残し、除くのは想起の側である。
        """
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT r.id, r.kind, r.created_at, m.obs_id, m.role FROM relations r "
                    "JOIN relation_members m ON m.relation_id = r.id "
                    "WHERE r.kind = ANY(%s) AND r.created_at >= %s "
                    "ORDER BY r.id, m.position NULLS LAST",
                    ([KIND_COOCCURRENCE, KIND_SUCCESSION, KIND_EXCHANGE], since),
                )
                edges: "dict[int, dict]" = {}
                for r in cur.fetchall():
                    e = edges.setdefault(
                        int(r["id"]),
                        {
                            "id": int(r["id"]),
                            "kind": r["kind"],
                            "created_at": r["created_at"],
                            "members": [],
                        },
                    )
                    e["members"].append((str(r["obs_id"]), str(r["role"])))
                ids = sorted({o for e in edges.values() for o, _ in e["members"]})
                cur.execute(
                    "SELECT id, parent_id FROM observations "
                    "WHERE id = ANY(%s) AND direction = %s AND parent_id IS NOT NULL",
                    (ids, final_direction),
                )
                parents = {str(r["id"]): str(r["parent_id"]) for r in cur.fetchall()}
                cur.execute(
                    "SELECT r.id, r.created_at, m.obs_id FROM relations r "
                    "JOIN relation_members m ON m.relation_id = r.id "
                    "WHERE r.kind = %s ORDER BY r.id, m.position",
                    (KIND_EXPERIENCE,),
                )
                links: "dict[int, dict]" = {}
                for r in cur.fetchall():
                    lk = links.setdefault(int(r["id"]), {"at": r["created_at"], "ids": []})
                    lk["ids"].append(str(r["obs_id"]))
        return {"edges": list(edges.values()), "parents": parents, "links": links}

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

    def latest_origins(self, n: int, kind: str = KIND_EXCHANGE) -> list[str]:
        """時系列で新しい順に、やりとりの起点を n 件返す（記-h）。

        **辺は見ない。** 直近のやりとりは「最近何があったか」で、継起の辺があるかどうかに
        関係なく要る（辺だけを頼ると、話題が切り替わった瞬間に直近が空になる）。順序は
        起点 O の時刻で決める（関係 id の順だと、遅れて書かれた関係が先に来る）。
        """
        if n <= 0:
            return []
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT m.obs_id FROM relation_members m "
                    "JOIN relations r ON r.id = m.relation_id AND r.kind = %s "
                    "JOIN observations o ON o.id = m.obs_id "
                    "WHERE m.role = '起点' ORDER BY o.timestamp DESC, r.id DESC LIMIT %s",
                    (kind, int(n)),
                )
                return [str(r["obs_id"]) for r in cur.fetchall()]

    def record_cooccurrence(self, obs_ids: "list[str]") -> "int | None":
        """ある反復で一緒に活性した記録を、一つの共起として残す。"""
        return self.add(KIND_COOCCURRENCE, [(o, "項", None) for o in obs_ids])

    def cooccurring(
        self,
        obs_ids: "list[str]",
        min_shared: int = 2,
        limit: int = 20,
        *,
        conn: "object | None" = None,
    ) -> list[str]:
        """渡した id と `min_shared` 件以上重なる過去の共起から、残りの id を返す。

        **種類で絞る。** 一つの表に全部の関係が載っているので、絞りを落とすと同じターンの
        問いと答え（やりとり）が共起として返ってくる。

        渡した id 自身と、自己認識の記録（向き「内省」）は除く（記-u 段 4 で種類から向きへ移した）。

        **既に錠を持っている呼び手は `conn` を渡す。** 錠は再入できないので、拡散想起の
        ように外側で錠を取ったまま呼ぶ経路が、ここで取り直すと止まる。
        """
        ids = [str(o) for o in obs_ids if o]
        if len(ids) < min_shared:
            return []
        if conn is not None:
            return self._cooccurring(conn, ids, min_shared, limit)
        with self._ctx.lock:
            return self._cooccurring(self._ctx.conn(), ids, min_shared, limit)

    def _cooccurring(self, conn, ids: list, min_shared: int, limit: int) -> list[str]:
        """共起の本体。錠は呼び手が持つ。"""
        with conn.cursor() as cur:
            cur.execute(
                "WITH matched AS ("
                "  SELECT m.relation_id FROM relation_members m"
                "    JOIN relations r ON r.id = m.relation_id AND r.kind = %s"
                "  WHERE m.obs_id = ANY(%s)"
                "  GROUP BY m.relation_id"
                "  HAVING count(DISTINCT m.obs_id) >= %s"
                ") "
                "SELECT DISTINCT i.obs_id FROM relation_members i "
                "JOIN matched t ON t.relation_id = i.relation_id "
                "JOIN observations o ON o.id::text = i.obs_id "
                "WHERE NOT (i.obs_id = ANY(%s)) AND o.direction <> '内省' "
                "LIMIT %s",
                (KIND_COOCCURRENCE, ids, min_shared, ids, limit),
            )
            return [str(_first(r)) for r in cur.fetchall()]

    def recent_finals(
        self, since: datetime, until: datetime, *, exclude_id: str, direction: str
    ) -> list[dict]:
        """`since` から `until` までに閉じた最終版（記-t 段 2）を、新しい順に返す。

        最終版は向き（`direction`）で見分ける（記-u 段 1・それまでは向き `求め` と文の頭の印で見分けていた）。時刻は閉じた時刻なので、
        時刻で絞れば「その前に終わった塊」になる。改訂で畳まれた最終版と、`exclude_id`（いま書いた自分）は除く。
        返りは `id`・`content`・`timestamp` を持つ dict の並び。
        """
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT o.id, o.content, o.timestamp FROM observations o "
                    "WHERE o.direction = %s "
                    "AND o.timestamp >= %s AND o.timestamp <= %s AND o.id <> %s "
                    f"AND {not_hidden('o')} "
                    "ORDER BY o.timestamp DESC",
                    (direction, since, until, exclude_id),
                )
                return [dict(r) for r in cur.fetchall()]

    def thread(self, obs_id: str, *, before: int, after: int) -> list[dict]:
        """`obs_id` から継起を前へ `before` 歩・後へ `after` 歩たどり、古い順に返す（記-q・2026-10-15）。

        前へさかのぼるだけの `recent_exchanges` と違い、ある記録の**あと**どう続いたかも引く。たどるのは継起の辺だけで、
        項の役割は見ない（最終版どうしも、昔の起点どうしも同じに歩く）。改訂で畳まれた記録は返さない（歩くのは通り抜ける）。
        返りは `id`・`content`・`timestamp`・`offset`（前は負・自分は 0・後は正）を持つ dict の並び。
        """
        walk = (
            "WITH RECURSIVE back AS ("
            "  SELECT %s::text AS obs_id, 0 AS depth"
            "  UNION ALL"
            "  SELECT p.obs_id, b.depth + 1 FROM back b"
            "    JOIN relation_members n ON n.obs_id = b.obs_id AND n.role = '後'"
            "    JOIN relations r ON r.id = n.relation_id AND r.kind = %s"
            "    JOIN relation_members p ON p.relation_id = n.relation_id AND p.role = '前'"
            "  WHERE b.depth < %s"
            "), fwd AS ("
            "  SELECT %s::text AS obs_id, 0 AS depth"
            "  UNION ALL"
            "  SELECT n.obs_id, f.depth + 1 FROM fwd f"
            "    JOIN relation_members p ON p.obs_id = f.obs_id AND p.role = '前'"
            "    JOIN relations r ON r.id = p.relation_id AND r.kind = %s"
            "    JOIN relation_members n ON n.relation_id = p.relation_id AND n.role = '後'"
            "  WHERE f.depth < %s"
            "), steps AS ("
            "  SELECT obs_id, -depth AS off FROM back UNION SELECT obs_id, depth FROM fwd"
            ") "
            "SELECT o.id, o.content, o.timestamp, min(s.off) AS pos FROM steps s "
            "JOIN observations o ON o.id::text = s.obs_id "
            f"WHERE {not_hidden('o')} "
            "GROUP BY o.id, o.content, o.timestamp ORDER BY pos"
        )
        args = (obs_id, KIND_SUCCESSION, max(0, before), obs_id, KIND_SUCCESSION, max(0, after))
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(walk, args)
                rows = [dict(r) for r in cur.fetchall()]
        # `offset` は SQL の予約語なので、列は `pos` で受けて詰め替える。
        return [
            {**{k: v for k, v in r.items() if k != "pos"}, "offset": int(r["pos"])} for r in rows
        ]

    def id_from_short(self, short: str) -> "str | None":
        """W に印字した 12 桁の id（`-` を除いた先頭 12 字）から、記録の完全な id を引く（記-q）。

        12 桁の 16 進でなければ引かない。当たりが 1 件でなければ（無い・重なる）None。
        """
        sid = (short or "").strip().lower()
        if len(sid) != 12 or any(c not in "0123456789abcdef" for c in sid):
            return None
        prefix = f"{sid[:8]}-{sid[8:]}%"
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM observations WHERE id::text LIKE %s LIMIT 2", (prefix,))
                rows = cur.fetchall()
        return str(_first(rows[0])) if len(rows) == 1 else None

    def succession_neighbors(
        self, obs_ids: "list[str]", *, conn: "object | None" = None
    ) -> list[str]:
        """渡した記録それぞれと継起で直接つながる記録（前後 1 つずつ）の id を返す（記-q・関連想起の出どころ）。

        渡した id 自身と、改訂で畳まれた記録は除く。**既に錠を持っている呼び手は `conn` を渡す**（`cooccurring` と同じ）。
        """
        ids = [str(o) for o in obs_ids if o]
        if not ids:
            return []
        if conn is not None:
            return self._succession_neighbors(conn, ids)
        with self._ctx.lock:
            return self._succession_neighbors(self._ctx.conn(), ids)

    def _succession_neighbors(self, conn, ids: list) -> list[str]:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT other.obs_id FROM relation_members me "
                "JOIN relations r ON r.id = me.relation_id AND r.kind = %s "
                "JOIN relation_members other ON other.relation_id = me.relation_id "
                " AND other.obs_id <> me.obs_id "
                "JOIN observations o ON o.id::text = other.obs_id "
                f"WHERE me.obs_id = ANY(%s) AND NOT (other.obs_id = ANY(%s)) AND {not_hidden('o')}",
                (KIND_SUCCESSION, ids, ids),
            )
            return [str(_first(r)) for r in cur.fetchall()]

    def recent_exchanges(
        self, origin_id: str, *, roles: "tuple[str, ...]" = ("起点", "添え", "つなぎ", "答え")
    ) -> list[dict]:
        """`origin_id` から継起をさかのぼり、各やりとりの項を古い順に返す。

        **上限を持たせない。** 会話はどこかで始まってどこかで終わるので、根で自然に
        止まる。何歩まで、と決めるのは、鎖の長さを機構の側で決めることになる。

        返すのは口に出したものだけである（既定は 起点・添え・つなぎ・答え。添えは出-au 段 3）。`版` と `見た` は
        内部の作業記録で、会話ではない。混ぜると、調べている途中の文字列が履歴として
        読まれる。

        返りは `obs_id`・`content`・`role`・`direction`・`timestamp`・`depth` を持つ dict の並び。
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
                    "SELECT o.id AS obs_id, o.content, m.role, o.direction, o.timestamp, w.depth "
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

    def roles_of(self, obs_ids: "list[str]", kind: str = KIND_EXCHANGE) -> dict[str, str]:
        """その記録たちが、その種類の関係で取っている役割を `obs_id → 役割` で返す。

        **種類で絞る。** 1つの記録は何本もの関係に入る（やりとり・継起・共起・改訂）。
        `やりとり` に絞れば、1つの記録の役割は1つに決まる（区間で切っているため）。
        """
        ids = [str(i) for i in obs_ids if i]
        if not ids:
            return {}
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT m.obs_id, m.role FROM relation_members m "
                    "JOIN relations r ON r.id = m.relation_id AND r.kind = %s "
                    "WHERE m.obs_id = ANY(%s)",
                    (kind, ids),
                )
                return {str(row["obs_id"]): str(row["role"]) for row in cur.fetchall()}
