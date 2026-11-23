"""使われる情報量 $I$ と根づきの減り $\\Delta$（`設計方針_REST内省_出来事を畳む` §1・§4・記-a-ろ-ろ）。

$I=\\sum_i b_i\\,u(t_i,n_i)$——生の総量ではなく**想起されうる分**。$b_i$ は字数から見積もり
（日本語の自然文は 1 字 ≈ 5〜6 bit・`BITS_PER_CHAR`〔仮〕）、$u$ は採点の合成式 $m$ を
時間 $t$ と根づき $g$ の 2 軸だけで組んで $m_{\\max}$ で割ったもの（0〜1）。式は `tools/memory.py` の
`_score_breakdown`／`_derive_groundedness` と同じにする——ここは**読む**だけで採点には触らない。

$\\Delta=\\max(0,\\lceil\\log_2(I/I^\\*)\\rceil)$〔仮〕。$I \\le I^\\*$ なら 0（減らさない晩もある）。
純関数。DB も設定も読まない（読むのは `loop/rest_info.py`）。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

BITS_PER_CHAR = 5.5  # 日本語の自然文の見積もり〔仮・出来事を畳む §1〕
G_CEILING = 2.0  # 根づき g の上限（`_derive_groundedness` の c）


def bits(text: str) -> float:
    return bits_of_chars(len(text or ""))


def bits_of_chars(chars: int) -> float:
    return float(max(0, int(chars))) * BITS_PER_CHAR


def _derive_groundedness(
    a0: float,
    n: int,
    *,
    floor: float = 0.0,
    c: float = 2.0,
    epsilon: float = 0.001,
    step: float = 0.33,
) -> float:
    """初期値 a0 と正味デルタ回数 n から活性 a を導出する。

    a0 を [floor, c] で正規化し、ε で両端に寄せてロジットで無限区間へ写し、
    n·step を足してロジスティックで [floor, c] へ戻す。n=0 なら g=g0。
    floor・c・ε・step は設定値（値の確定は課題5・Config から差し替え可）。
    """
    span = c - floor
    x0 = (a0 - floor) / span
    x0 = min(max(x0, epsilon), 1.0 - epsilon)
    u = math.log(x0 / (1.0 - x0)) + n * step
    y = 1.0 / (1.0 + math.exp(-u))
    return floor + span * y


def _groundedness(g0: float, n: int) -> float:
    """`_derive_groundedness` の既定（floor 0・c 2・ε 0.001・step 0.33）。"""
    return _derive_groundedness(float(g0), int(n))


def usability(*, age_days: float, g0: float, n: int, hl: float, w_t: float, w_g: float) -> float:
    """想起されやすさ $u\\in[0,1]$。$t=2^{-d/HL}$・$g=g(n)$ の加重平均を最大値で割る。"""
    t = 2.0 ** (-max(0.0, float(age_days)) / max(1e-9, float(hl)))
    g = _groundedness(g0, n)
    denom = float(w_t) + float(w_g)
    if denom <= 0.0:
        return 1.0
    m = (float(w_t) * t + float(w_g) * g) / denom
    m_max = (float(w_t) * 1.0 + float(w_g) * G_CEILING) / denom
    return max(0.0, min(1.0, m / m_max))


def _age_days(ts, now: datetime) -> float:
    if ts is None:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (now - ts).total_seconds() / 86400.0)


def total(rows: list[dict], *, now: datetime, hl: float, w_t: float, w_g: float) -> float:
    """$\\sum b_i u_i$。行は `chars`・`timestamp`・`groundedness_g0`・`groundedness_n` を持つ。"""
    out = 0.0
    for r in rows:
        u = usability(
            age_days=_age_days(r.get("timestamp"), now),
            g0=float(r.get("groundedness_g0", 1.0) or 1.0),
            n=int(r.get("groundedness_n", 0) or 0),
            hl=hl,
            w_t=w_t,
            w_g=w_g,
        )
        out += bits_of_chars(int(r.get("chars", 0) or 0)) * u
    return out


def delta_for(total_bits: float, target_bits: float) -> int:
    """$\\Delta=\\max(0,\\lceil\\log_2(I/I^\\*)\\rceil)$。目標が正でなければ 0。"""
    if target_bits <= 0.0 or total_bits <= target_bits:
        return 0
    return max(0, int(math.ceil(math.log2(total_bits / target_bits))))
