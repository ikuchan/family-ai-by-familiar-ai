"""`observations.person_id`（所有者絞り）を落とす。

**所有者絞りは、そもそも人を分けていなかった。** 010 が「人ごとの記憶空間」を意図して
入れた列だが、2026-08-03 のダンプでは 5080 行のうち 4904 行（96.5%）が既定値 `default`
のままで、家族4人のうち2人（いくながこうき・いくながたえこ）は所有行を1件も持たない。
`NOT NULL DEFAULT 'default'` の列を、書き込み側が文脈の person のまま埋め続けた結果である。

**人と記憶の結びつきは situated が担う**（[D-在席相関/V2]）。047 で `actor`（誰がやったか）
と `present`（誰が居たか）の面が立ったので、設計が定めた順序——**関係生成が立ってから列を
落とす**——の条件が満たされた。設計は「所有者フィルタは廃し、p 軸（在席者相関）は在席関係の
行を使う」と定めている（`gap分析` §4）。

**重複判定の30秒窓だけは絞りを保ち、`writer_id` へ移す。** 重複とは「同じ書き手が同じ内容を
同じ kind で窓の内に」であって、家族の二人が同じ挨拶をしたものは重複ではない。過去データでは
別書き手の衝突が 0件・同書き手の重複も 0件で、この付け替えで見える挙動は変わらない
（反証側も確認済み）。将来、話者識別が働いたときに正しく分かれる。

**撤去で本番の不具合が1件直る。** `recall_self_model` と `recall_curiosities` は
`person_id=AGENT_SELF_ID`（`...0000`）で絞っていたが、書き込みは文脈の person で入るため、
8月3日時点の `self_model` 958 行・`curiosity` 238 行の**すべて**が `default`（`...0001`）
だった。一致する行が無く、**両関数は本番で常に空を返していた**。所有者絞りごと外れるので
この食い違いも消える。

落とすのは `observations.person_id`（所有者）だけである。`situated_memories.person_id`
（誰と関係する面か）と、想起の視点（`store.context.viewpoint_of`）は残る。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の
`schema_migrations` には既に記録されているので、ここでは流れない。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE observations DROP COLUMN IF EXISTS person_id")
