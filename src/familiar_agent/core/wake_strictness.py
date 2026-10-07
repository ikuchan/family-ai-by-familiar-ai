"""名前で会話の窓を開ける基準（2026-10-05 実機・本人の決定・`設計方針_話していいかの決まり` §2.3）。純関数。

実機 2026-10-05 23:34、誰も居ない部屋の音（1.1 秒）を書き起こしが「パジュー」と聞き違え（無音らしさ 0.150・確かさ
-0.723）、文頭の名前で窓が開いて返事をした。基準を「ふだん」と「厳しい」の 2 段にする（値は `STTConfig.wake_*`・仮）。

- **名前だけ**（名前 1 つと、伸ばす音・句読点だけ）：無音らしさと確かさが届かなければ捨てる（ふだん 0.10・-0.65、
  厳しい 0.05・-0.60）。
- **名前だけの繰り返し**（「パジュ、パジュー」）は想定外として、声なら値に関わらず捨てる（`is_name_repeat`・
  2026-10-07 本人の決定）。誰も呼んでいないのに「パジュ、パジュ、パジュー」と書き起こされ（無音らしさ 0.08〜0.19）、
  パジュが返事をした。以前は名前＋言葉の側に入れていた。
- **名前＋言葉**：ふだんはいまのまま（1 字違いを許す・確かさを見ない）。厳しいときは 1 字違いを許さず、
  無音らしさ 0.20 以下・確かさ -0.90 以上（09-26 の本物 8 件がすべて通る値）。
- 確かさが無い入力（ElevenLabs の書き起こし・TUI の録音）は確かさを見ずに通す（測りようがない）。

厳しい段になる条件（居ない・誰の声か分からない）は呼び手（入口の門）が決める。
"""

from __future__ import annotations

import unicodedata

from .silence_rules import _head, _loose, names_me

#: 名前の後ろに続いても「名前だけ」とみなす伸ばしの音（ゆるい読みでは長音「ー」は落ちている）。
_TAIL = set("うぅ")


def is_name_only(text: str, names: "list[str] | tuple[str, ...]") -> bool:
    """書き起こしが名前 1 つと、伸ばす音・句読点・空白だけか。"""
    head = _head(_loose(text))
    for raw in names:
        n = _loose(raw)
        if n and head.startswith(n):
            rest = head[len(n) :]
            if all(
                ch in _TAIL or ch.isspace() or unicodedata.category(ch)[0] in "PSZ" for ch in rest
            ):
                return True
    return False


def is_name_repeat(text: str, names: "list[str] | tuple[str, ...]") -> bool:
    """書き起こしが名前だけを 2 回以上繰り返したものか（間の伸ばす音・句読点・空白は飛ばす）。"""
    s = _loose(text)
    loosed = sorted({_loose(n) for n in names if n and _loose(n)}, key=len, reverse=True)
    count, i = 0, 0
    while i < len(s):
        ch = s[i]
        if ch in _TAIL or ch.isspace() or unicodedata.category(ch)[0] in "PSZ":
            i += 1
            continue
        hit = next((n for n in loosed if s.startswith(n, i)), None)
        if hit is None:
            return False
        count += 1
        i += len(hit)
    return count >= 2


def admits(
    text: str,
    names: "list[str] | tuple[str, ...]",
    *,
    strict: bool,
    no_speech: "float | None" = None,
    logprob: "float | None" = None,
    cfg=None,
) -> "tuple[bool, str]":
    """名前で窓を開けてよいか。（受けるか, 捨てた理由）。"""
    if cfg is None:
        from ..config import STTConfig

        cfg = STTConfig()
    if not names_me(text, names, fuzzy=not strict):
        return False, "名前が無い"
    measured = no_speech is not None and logprob is not None
    if is_name_only(text, names):
        if not measured:
            return True, ""
        ns_max = cfg.wake_strict_name_no_speech if strict else cfg.wake_name_no_speech
        lp_min = cfg.wake_strict_name_logprob if strict else cfg.wake_name_logprob
        if no_speech > ns_max or logprob < lp_min:  # type: ignore[operator]
            return False, f"名前だけで確かさが足りない（{'厳しい' if strict else 'ふだん'}）"
        return True, ""
    if strict and measured:
        if no_speech > cfg.wake_strict_no_speech or logprob < cfg.wake_strict_logprob:  # type: ignore[operator]
            return False, "名前＋言葉で確かさが足りない（厳しい）"
    return True, ""
