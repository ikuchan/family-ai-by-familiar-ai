"""致命的な起動エラー（#10）。DB・埋め込み等、これ無しでは動けない依存の失敗を表す。

起動経路（main.py の CLI／gui.py の GUI）がこれを捕らえ、原因＋対処を1行で示して
クリーン終了する（生 traceback を出さない）。
"""

from __future__ import annotations


class FatalStartupError(Exception):
    """動作継続が不可能な致命的失敗。message は利用者向けの1行（原因＋対処）。"""


def check_embedding_fatal(agent) -> None:
    """埋め込みモデルの読込失敗を致命として扱う（#10）。記憶が死ぬので黙って劣化させない。"""
    if getattr(agent, "embedding_failed", None) and agent.embedding_failed():
        raise FatalStartupError(
            "埋め込みモデルを読み込めません。記憶（想起・保存）が機能しないため起動を中止します。"
            "モデルの取得・依存（sentence-transformers）を確認してください。"
        )
