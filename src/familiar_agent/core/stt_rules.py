"""書き起こしの検め（知-y・2026-09-19）。純関数。

Whisper は無音や雑音で、学習元（動画の字幕）の定型文「ご視聴ありがとうございました」を出す（幻聴）。実機 13:32 に
会話入力として届いた（タイマーの門で落ちたが、普段なら求めが立つ）。**完全一致**（前後の空白と句読点は無視）で
捨てる。「教えてくれてありがとうございました」のような普通の文は落とさない。表は実機で出たものだけ足す。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_BOILERPLATE = {
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
    "ありがとうございました",
}
_TRIM = " \t　。．、,.!！?？"


def is_hallucination(text: str) -> bool:
    s = (text or "").strip(_TRIM)
    return s in _BOILERPLATE


# ── 語の組（2026-09-20）──────────────────────────────────────────────────────
#
# 表は **(直すべき語, (あり得る語…))** の並び。STT へ渡す語の列（`hotwords`）も、書き起こしの
# 直しも、この 1 つの表から導く。直すべき語が `_DELETE`（`-`）の組は「消す」。
#
# 渡した語の列は、音に情報が無いとき**そのまま書き起こされる**。faster-whisper は `hotwords` を
# `initial_prompt` と同じ枠（`<|startofprev|>`＝直前に出てきた言葉）へ入れるので、音が効かない
# 区間では渡した言葉だけで決まる。実機 2026-09-20 17:22〜17:26 に、渡した例文がそのまま会話入力
# として積まれ、GUI に並び、タイマーまで掛かった。だから列そのものを「消す組」として足す
# （`with_hint`）。語 1 つは消さない——人が名前を呼んだ言葉だからである。

#: 直すべき語がこれなら「消す」。空にすると書き忘れと見分けが付かないので印を置く（本人の決定）。
_DELETE = "-"
_GROUP_SEP = re.compile(r"[;；]")
_TARGET_SEP = re.compile(r"[:：]")
_WORD_SEP = re.compile(r"[、,]")
_LEADING = " \t　、。．，"


def parse_word_groups(text: str) -> "tuple[tuple[str, tuple[str, ...]], ...]":
    """設定値の 1 行を語の組へ。`直すべき語：あり得る語、あり得る語; -：消す語` の形。

    組は `;`、直すべき語と配列は `：`、語は読点で区切る。**空白は区切りに使わない**ので、
    語の中の空白（列そのもの）はそのまま残る。読めない組は飛ばす。
    """
    out: list[tuple[str, tuple[str, ...]]] = []
    for chunk in _GROUP_SEP.split(text or ""):
        if not chunk.strip():
            continue
        parts = _TARGET_SEP.split(chunk, maxsplit=1)
        if len(parts) != 2:
            logger.debug("語の組として読めないので飛ばす：%r", chunk)
            continue
        target = parts[0].strip()
        words = tuple(w.strip() for w in _WORD_SEP.split(parts[1]) if w.strip())
        if not target or not words:
            logger.debug("語の組が欠けているので飛ばす：%r", chunk)
            continue
        out.append((target, words))
    return tuple(out)


#: 語の列の区切り（知-ah・2026-09-24）。**空白で区切らない。**
#: `hotwords` は `<|startofprev|>`（直前に出てきた言葉）の枠へ入るので、渡した列の
#: 書き方が書き起こしの書き方になる。空白の列を渡すと書き起こしも空白で切れた
#: （実機 15:57「しょうめん を み て」）。同じ音で測ると、空白は名前 3/4・歪み 1/3、
#: 読点は 4/4・0/3（`根拠台帳` §46）。読点にしたのは `ME.md` の「名前：」と同じ形で、
#: **人が書いた区切りをそのまま渡す**ことになるからである。
_HINT_SEP = "、"


def hotwords_for(groups: "tuple[tuple[str, tuple[str, ...]], ...]") -> str:
    """STT へ渡す語の列。表の語を読点でつなぐ（消す組の `-` は入れない）。"""
    words: list[str] = []
    for target, samples in groups:
        for w in ((target,) if target != _DELETE else ()) + tuple(samples):
            if w and w not in words:
                words.append(w)
    return _HINT_SEP.join(words)


def with_hint(
    groups: "tuple[tuple[str, tuple[str, ...]], ...]", hint: str
) -> "tuple[tuple[str, tuple[str, ...]], ...]":
    """渡した語の列そのものを「消す組」として足す。列が無ければそのまま。

    **語 1 つは消さない**（知-ah・2026-09-24）。消す組は「渡した列がまるごと書き起こされた」
    ときのためのもので、語 1 つでは人が名前を呼んだ言葉と見分けが付かない。守りが無かった
    ため、正しく取れた名前が消えていた（`ねえ、パジュ、聞こえる?` → `ねえ、、聞こえる?`）。
    """
    if not hint or _HINT_SEP not in hint:
        return groups
    return groups + ((_DELETE, (hint,)),)


def fix_words(text: str, groups: "tuple[tuple[str, tuple[str, ...]], ...]") -> str:
    """書き起こしに表を当てる。あり得る語を直すべき語へ、`-` の組は消す。

    長い語から当てる（「パジュー」を先に直さないと「ー」が残る）。消したあとに残る空白と
    行頭の読点は詰める。
    """
    if not text or not groups:
        return text
    out = text
    pairs: list[tuple[str, str]] = []
    for target, samples in groups:
        for w in samples:
            pairs.append((w, "" if target == _DELETE else target))
    for word, to in sorted(pairs, key=lambda kv: -len(kv[0])):
        if word and word != to:
            out = out.replace(word, to)
    out = re.sub(r"[ \t　]+", " ", out).strip()
    return out.lstrip(_LEADING).strip()
