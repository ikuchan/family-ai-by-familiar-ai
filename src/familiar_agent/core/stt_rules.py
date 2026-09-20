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


# ── 名前の直し（知-z・2026-09-19）──────────────────────────────────────────
#
# 名前入りの例文を `initial_prompt` に渡す手もあったが、2026-09-20 の実機でやめた。はっきり
# しない音に対して Whisper がその例文をそのまま書き出し、話していないのに「パジュ、3 分測って。」
# が繰り返し上がった。名前の手がかりは `hotwords`（語の並び・文ではないので反響しない）だけにする。


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


def is_echo_of_hint(text: str, hints: "tuple[str, ...] | list[str]") -> bool:
    """道具へ渡した手がかりが、そのまま書き起こされて返ってきたか（2026-09-20）。

    Whisper は手がかり（`hotwords`・`initial_prompt`）を「これから出てくる言葉」として使うので、
    はっきりしない音に当たると**渡した文をそのまま書き出す**。実機で、渡した例文の後半 11 字が
    10.7 秒の音や 30 秒の窓から繰り返し上がり、人の言葉として GUI に並んだ（17:22〜17:26）。
    例文は撤去したが、手がかりが人の言葉として返ること自体を門で止める。

    **丸ごと同じときだけ**捨てる（前後の空白と句読点は無視）。手がかりの一部（綴り 1 語の
    「パジュ」）は、人が名前を呼んだ場合があるので捨てない。
    """
    s = (text or "").strip(_TRIM)
    if not s:
        return False
    return any(s == (h or "").strip(_TRIM) for h in hints if h)
