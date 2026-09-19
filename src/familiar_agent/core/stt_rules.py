"""書き起こしの検め（知-y・2026-09-19）。純関数。

Whisper は無音や雑音で、学習元（動画の字幕）の定型文「ご視聴ありがとうございました」を出す（幻聴）。実機 13:32 に
会話入力として届いた（タイマーの門で落ちたが、普段なら求めが立つ）。**完全一致**（前後の空白と句読点は無視）で
捨てる。「教えてくれてありがとうございました」のような普通の文は落とさない。表は実機で出たものだけ足す。
"""

from __future__ import annotations

_BOILERPLATE = {
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
    "ありがとうございました",
}
_TRIM = " \t　。．、,.!！?？"


def is_hallucination(text: str) -> bool:
    s = (text or "").strip(_TRIM)
    return s in _BOILERPLATE


# ── 名前の手がかりと直し（知-z・2026-09-19）────────────────────────────────


def initial_prompt_for(name: str) -> str:
    """faster-whisper の `initial_prompt` に渡す、名前入りの短い例文。

    `hotwords` だけでは large-v3 が語彙に無い「パジュ」を落とす・化かす（今日 2 回「パパだよ」に）。
    直前の文脈として名前を 2 度見せる。長くすると無音でその文を反響しやすいので 2 文にとどめる。
    """
    n = (name or "").strip()
    if not n:
        return ""
    return f"{n}、こんにちは。{n}、3 分測って。"


def normalize_name(text: str, names: "tuple[str, ...] | list[str]") -> str:
    """書き起こしに上がった聞き違いの綴り（`ME.md` の 2 語目以降）を、先頭の綴り（正しい名前）へ直す。

    `hotwords` に綴りを全部渡すと STT はそちらの字で書く。完全一致だけ当て、表に無い語
    （「体重」）は触らない。長い綴りから当てる（「パジュー」を先に直さないと「ー」が残る）。
    """
    if not text or len(names) < 2:
        return text
    canon = names[0]
    for alt in sorted(names[1:], key=len, reverse=True):
        if alt and alt != canon:
            text = text.replace(alt, canon)
    return text
