"""ログに家族の言葉と名前を出さないための 2 つの口（出-dm・出-dn・出-dp・2026-11-21）。

ログに出してよいのは、時刻、決まった文、件数だけである。家族の言葉（聞いた言葉、調べものの問い、メモや予定の中身）には
名前や曲名が入りうるので、`chars` で字数だけにする。名前と曲名は `tag` で短い印にする。印には起動ごとに変わる
ランダムな値を混ぜるので、同じ起動のあいだは同じ名前が同じ印になり（ログの流れは追える）、よくある名前の一覧と
照らしても元の名前は当たらない。
"""

from __future__ import annotations

import hashlib
import hmac
import os

#: 起動ごとに変わる値（印から名前を戻されないため）。
_SALT = os.urandom(16)


def chars(text: object) -> str:
    """言葉を字数だけにする（例「12 字」）。"""
    return f"{len(str(text or ''))} 字"


def tag(value: object) -> str:
    """名前や曲名を短い印にする（例「#1a2b」）。空なら「-」。"""
    s = str(value or "")
    if not s:
        return "-"
    return "#" + hmac.new(_SALT, s.encode("utf-8"), hashlib.sha256).hexdigest()[:4]
