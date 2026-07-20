"""会話履歴スライスの読み取り専用フラット化。

tool 結果はネスト list として履歴に格納され、API 送信時にのみ実メッセージへ
展開される。要約や整合性チェックなど生の履歴を走査する読み取り側は、先に
flatten しないと list 要素で msg.get(...) が AttributeError になる。
"""

from __future__ import annotations


def _flatten_history(messages: list) -> list[dict]:
    """履歴スライスを「dictのみのフラット列」にして返す（読み取り専用走査向け）。

    tool結果はネストlistとして履歴に格納され（make_tool_results /
    _flatten_messages 参照）、API送信時にのみ実メッセージへ展開される。
    要約トランスクリプトや整合性チェックなど、生の履歴を走査する読み取り側は
    先にflattenしないと list 要素で msg.get(...) が AttributeError になる。

    backend._flatten_messages を再利用しないのは、そのシグネチャがbackend間で
    不統一（OpenAI互換backendは (system, messages)）なため。
    """
    flat: list[dict] = []
    for msg in messages:
        if isinstance(msg, list):
            flat.extend(m for m in msg if isinstance(m, dict))
        elif isinstance(msg, dict):
            flat.append(msg)
    return flat
