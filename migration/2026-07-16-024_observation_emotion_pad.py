"""Add emotion PAD columns (P/Pn/A/Dom) to observations（書き込み PAD 化 W1a・案B）。

課題5 v0.23：観測 emotion を4軸 PAD（快 P／不快 Pn／喚起 A／支配 Dom・
各 [0,1]・中立0.5）で持つ。この段（W1a）は軸ごとの数値列を追加するだけで、
既存行・新規行とも既定0.5。文字列 emotion 列は残す（ラベル読み出し経路のため）。
評価器・スコア・recall は無変更で列は誰も読まない（外部挙動不変）。

CHECK は各列にインラインで持たせ、`ADD COLUMN IF NOT EXISTS` による列の存在
チェックで制約の再作成も同時に防ぐ（冪等・別名制約の二重作成を避ける）。

ラベル→PAD の一回限りの写像（既存行の値埋め・W1b）と、評価器が P/Pn/Dom を
直接出力する書き込み（W2）は後続。
"""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        for axis in ("emotion_p", "emotion_pn", "emotion_a", "emotion_dom"):
            cur.execute(
                f"ALTER TABLE observations ADD COLUMN IF NOT EXISTS {axis} "
                f"double precision NOT NULL DEFAULT 0.5 "
                f"CHECK ({axis} >= 0 AND {axis} <= 1)"
            )
