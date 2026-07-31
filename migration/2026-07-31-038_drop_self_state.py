"""自己状態（`self_state`）の保存行を落とす。

`SelfState` は6軸（arousal／fatigue／social_pull／sensor_confidence／
unresolved_tension／focus_stability）を毎ターン更新して `agent_state` へ保存していたが、
**読み出す経路が2つとも死んでいた**（旧 ReAct のプロンプト組み立てと、テストからしか
呼ばれない文脈生成）。機構ごと撤去したので、保存行も残さない。

読み手が居なくなった値を DB に置いたままにすると、次に見た人が「まだ使っている」と読む。
ソースを読むたびのノイズと同じ理由で落とす。

`arousal` という名前が PAD の A 軸（高ぶり）と二重になっていた問題も、これで消える。

**値は復元できない。** ただし誰も読んでいなかったので、失われる情報は無い。同じことが
要るなら MI の想起で実現する方針である。

src を import せず自前完結させる（マイグレーションは過去の一度きりの実行を再現する凍結物）。
"""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM agent_state WHERE state_key = 'self_state'")
