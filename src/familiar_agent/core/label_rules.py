"""タイマー・ストップウォッチの名前の検め（知-u・2026-09-19）。純関数。

`label` は主LLM か調停が書く文字列で、09-16 に「何のために時間を測るか不明」が名前として入り、枠・返事・
O の記録にそのまま出た。名前は**物の名**（短い名詞句）であるべきなので、機械で検めて既定名に置き換える：
前後の空白・引用符・句点を落とす／`MAX_CHARS`（20 字〔仮〕）を超える／「不明」「わからない」などを含む。
置き換えたことは INFO に残す（返りには添えない）。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

MAX_CHARS = (
    20  # 〔仮〕「パパがコーヒーを淹れる時間」（13 字）は通り、「…を測る」の文（20 字超）は落ちる
)
_UNKNOWN = re.compile(r"不明|わからない|分からない|わかりません|分かりません|未定|特になし|なし$")
_TRIM = " \t　「」『』\"'。．,.、"


def clean_label(text: object, fallback: str) -> str:
    """名前として使える形に整える。使えなければ `fallback`（「測る」「タイマー」）。"""
    s = str(text or "").strip(_TRIM)
    if not s or len(s) > MAX_CHARS or _UNKNOWN.search(s):
        if s:
            logger.info("名前を検めて「%s」にした：%r", fallback, s[:40])
        return fallback
    return s
