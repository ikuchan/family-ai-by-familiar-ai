"""pyopenjtalk に渡す文の口（環-ad・2026-10-07 実機）。

OpenJTalk は文を約 8KB の決まった大きさの入れ物に写すので、それを越える文を渡すとスタックを壊してプロセスごと落ちる。
実機 12:49・21:45・22:06 の 3 回、想起の語の軸が天気の検索結果（全角の URL を含む）をそのまま渡して落ちた。別の
プロセスでは 8,002 バイトまでは通り、約 8,150 バイトで `stack smashing detected` になった。ここで文を
`OPENJTALK_MAX_BYTES`（4,000 バイト〔仮〕）以下の塊に切り、塊ごとに呼んで結果をつなぐ。

**大きさは、中で全角に直した後で数える**（環-af・2026-10-08）。OpenJTalk は半角を全角（1 字 3 バイト）に直してから写す
ので、決まるのは UTF-8 のバイト数ではなく、ほぼ字数 × 3 である（半角だけで 2,720 字が通り 2,740 字で落ちた）。UTF-8 で
数えていたので、半角の多い文（実機 21:26・Tavily の英語まじりの結果・3,990 バイト／2,873 字）が 1 塊のまま渡って落ちた。
1 字を `max(3, UTF-8 のバイト数)` として数える（`_size`）。

pyopenjtalk は中で 1 つの共有の作業物を使う。想起（作業用スレッド）と声の読み（別のスレッド）が同時に呼ぶことがある
ので、鍵を取って 1 本ずつ呼ぶ。pyopenjtalk が無ければ、呼び手が読み込みの失敗を受け止める（ここは読み込みを
呼び手と同じ形で行う）。
"""

from __future__ import annotations

import os
import threading
from typing import Any

#: 塊の大きさ（バイト）の既定。落ちる境目（約 8,150）の半分ほど。〔仮・本人の決定〕
DEFAULT_MAX_BYTES = 4000
#: 切りたい位置（このあとで切る）。見つからなければ文字の切れ目で切る。
_BREAKS = "。！？!?\n、，, "

_LOCK = threading.Lock()


def max_bytes() -> int:
    try:
        return max(16, int(os.environ.get("OPENJTALK_MAX_BYTES", DEFAULT_MAX_BYTES)))
    except ValueError:
        return DEFAULT_MAX_BYTES


def _size(ch: str) -> int:
    """1 字の大きさ。OpenJTalk の中で全角に直した後（半角も 3 バイト・4 バイトの字は 4）。"""
    return max(3, len(ch.encode("utf-8")))


def chunks(text: str, max_bytes: int) -> "list[str]":
    """文を `max_bytes` 以下（全角に直した後の大きさ・`_size`）の塊に切る。なるべく句読点・改行のあとで切り、文字は割らない。"""
    out: list[str] = []
    rest = text or ""
    while rest:
        if sum(_size(ch) for ch in rest) <= max_bytes:
            out.append(rest)
            break
        # 入るところまでの文字数を数える
        size, end = 0, 0
        for i, ch in enumerate(rest):
            size += _size(ch)
            if size > max_bytes:
                break
            end = i + 1
        cut = end
        for j in range(end, 0, -1):
            if rest[j - 1] in _BREAKS:
                cut = j
                break
        cut = max(cut, 1)
        out.append(rest[:cut])
        rest = rest[cut:]
    return out


def run_frontend(text: str, max_bytes: "int | None" = None) -> "list[Any]":
    """`pyopenjtalk.run_frontend` を塊ごとに呼び、結果をつなぐ。"""
    import pyopenjtalk

    feats: list[Any] = []
    for part in chunks(text, max_bytes or globals()["max_bytes"]()):
        with _LOCK:
            feats.extend(pyopenjtalk.run_frontend(part))
    return feats


def g2p_kana(text: str, max_bytes: "int | None" = None) -> str:
    """`pyopenjtalk.g2p(…, kana=True)` を塊ごとに呼び、読みをつなぐ。"""
    import pyopenjtalk

    out: list[str] = []
    for part in chunks(text, max_bytes or globals()["max_bytes"]()):
        with _LOCK:
            out.append(str(pyopenjtalk.g2p(part, kana=True)))
    return "".join(out)
