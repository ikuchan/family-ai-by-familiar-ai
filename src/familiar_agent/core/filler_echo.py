"""つなぎと重なった冒頭を落とす（出-aj #4・2026-09-24）。純関数。

実機 15:49、つなぎで「こんにちは。」と言った 6 秒後、本応答が「**こんにちは！**さっき
タイマーが鳴っていたので…」と始めた。相手はもう聞いている。

**言葉では止まらなかった。** 9 通り試して最良が 8 回中 2 回で、基準は 8/8 で繰り返す
（`根拠台帳` §47）。

**どちらも自分が言った言葉である。** 意味を判断する必要はなく、2 秒差で同じことを 2 回
言っているのを数えて落とせばよい。**挨拶の一覧は持たない**——表を持てば、表に無い挨拶を
見逃し、表にある語を別の意味で使ったときに消す（`loop/coherence` の「語の表で文を落とせば、
意味を読めない機械がパジュの普通の発話を黙って消す」と同じ危うさ）。つなぎに現れたか
どうかだけで決めれば、その危うさが無い。

**落とすのは冒頭だけ。** 途中の挨拶（`タイマー止めましたよ。こんにちは、どなたですか？`）は
残す。そこを落とすには「もう言ったか」を文脈で判断することになり、機械の仕事ではない。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: 冒頭の一節をどこで切るか。読点まで見るのは、「こんにちは、どなたですか？」のように
#: 読点で続く形があるため。
_BREAK = "。．！!？?、，,・…\n"

#: 照らすときに外す記号（つなぎは `。`、本応答は `！` のように、記号だけが違う）。
_TRIM = "。．！!？?、，,・… 　\t"

#: これより短い冒頭は落とさない。「あ」「え」のような相槌で誤爆させないため。
MIN_HEAD = 2


def _head(text: str) -> "tuple[str, int]":
    """冒頭の一節と、それに続く区切りまでの長さ。区切りが無ければ全部。"""
    for i, ch in enumerate(text):
        if ch in _BREAK:
            j = i
            while j < len(text) and text[j] in _TRIM:
                j += 1
            return text[:i], j
    return text, len(text)


def drop_echo(text: str, fillers: "list[str] | None") -> str:
    """本応答の冒頭が、直前に言ったつなぎと重なっていたら、その分を落とす。

    `fillers` は**実際に声に出した**つなぎ（`Request.said_fillers`）。出せずに落ちた
    ものは入らないので、聞こえていない言葉を「もう言った」とは数えない。
    """
    if not text or not fillers:
        return text
    head, cut = _head(text)
    key = head.strip(_TRIM)
    if len(key) < MIN_HEAD:
        return text
    rest = text[cut:].strip()
    if not rest:
        return text  # 落とすと何も残らない。**黙らせない**
    for filler in fillers:
        if key in (filler or "").strip(_TRIM) or key in (filler or ""):
            logger.info("event-loop つなぎと重なる冒頭を落とした：%r", head[:24])
            return rest
    return text
