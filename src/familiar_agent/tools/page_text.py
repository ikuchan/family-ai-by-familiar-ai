"""読んだページを、主LLM と照らしに渡す文にする（知-bp・知-br）。

fetch はリンクを `[文字](URL)` にしてから切るので、全文で読み、URL を消してから切る。
"""

from __future__ import annotations

import re

FETCH_ALL = 999_999  # fetch が受ける上限
PAGE_CHARS = 8000
# URL と題（`(URL "題")`）。URL の中の空白と 1 段の括弧（…/X_(Y)）まで。行はまたがない
_URL = r"\((?:[^()\n]|\([^()\n]*\))*\)"
_IMAGE = re.compile(r"!\[([^\]\n]*)\]" + _URL)
# 見える文字の中の 1 段の角括弧（Wikipedia の注 `[[1]](…)`）まで
_LINK = re.compile(r"\[((?:[^\[\]\n]|\[[^\[\]\n]*\])*)\]" + _URL)


def visible(text: str) -> str:
    """画像とリンクを見える文字だけにする（`![説明](URL)`→説明・`[文字](URL)`→文字）。"""
    return _LINK.sub(r"\1", _IMAGE.sub(r"\1", text))
