"""つなぎの長さを検める（出-cj・2026-10-14・本人の決定ア・15 字）。純関数。

軽量LLM には「短い一言」とだけ頼んでいて、10/14 10:13 に 26 字（声で 4.3 秒）のつなぎが流れた。頼む文に字数を書いた
うえで、超えたものは機械が作り置きの短い言葉に替える。言ったつなぎや鳴った合図の言葉と同じとみなされる言葉は使わない
（同じかは `filler_echo.repeats`）。どちらも使えなければ空（言わない）。
"""

from __future__ import annotations

from .filler_echo import repeats

#: つなぎの字数の上限（句読点も数える）〔仮〕。
FILLER_MAX_CHARS = 15
#: 上限を超えたときに替える言葉〔仮〕。上から、使えるものを使う。
SHORT_FILLERS = ("ちょっと待ってね", "少し待っててね")


def fit(text: str, *, said: "list[str]", cue_words: "list[str]") -> str:
    """`text` が上限以内ならそのまま、超えていれば作り置きの短い言葉。使えるものが無ければ空。"""
    text = (text or "").strip()
    if len(text) <= FILLER_MAX_CHARS:
        return text
    used = [*said, *cue_words]
    return next((s for s in SHORT_FILLERS if not repeats(s, used)), "")
