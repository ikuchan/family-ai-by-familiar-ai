"""`persons.perspective_vec`（人ごとの視点ベクトル）を落とす。

**ベクトルの差を「人」でなく「関係」で作る形へ移す前段である。**

044 までの situated ベクトルは `normalise(mem_vec + ALPHA·p_vec − mu)` で、`p_vec` は
人ごとに育てた視点ベクトルだった（`ALPHA=0.30`・書き込みごとに `lr=0.05` で更新）。
2026-08-03 のダンプでは persons 6 行のうち 3 行が非NULL で、この項は生きていた。

差は 047 が足す**関係項**（`relation_concept`）が担う（[D-在席相関/V2]「関係の種別は
列でなく vector で表す」）。実物（2026-08-21）でも、**同じ観測・同じ関係なら人が違っても
コサインは 1.000000** で一致する（`addressee` 186 対・`about` 105 対・`companion`／
`beneficiary`／`experiencer`／`owner` 18 対）。ベクトルは「観測 × 関係」だけで決まり、
人には依らない。面の言葉（役割の接頭辞＋出来事の本文）から作るので、誰かは入らない。

**視点シフト検索（役割1）の絞り（`s.person_id = ?`）は残る。** 変わるのは「どの行が
母集合か」ではなく「その行のベクトルが人によって違うか」である。

045 の直後・047 の前は、生成が `relation_key='presence'` 固定なので関係による差もまだ
無い。したがって同じ観測に対する全員のベクトルが同一になる。これは通過点である。

視点を育てる口（`update_perspective_vec`・`lr=0.05`）も一緒に落とす。育てる先が関係
ベクトルへ移るためで、残すと書き込み先の無い学習になる。係数 `ALPHA`（0.30・計測台帳 §10
で「仮値・未検証」）も、視点項の係数なので一緒に落とす。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の
`schema_migrations` には既に記録されているので、ここでは流れない。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE persons DROP COLUMN IF EXISTS perspective_vec")
