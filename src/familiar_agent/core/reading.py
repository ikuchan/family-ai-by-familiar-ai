"""声に出すときの読み（出-ac・2026-09-19）。純関数。

ElevenLabs が「出入口」を「しゅつにゅうくち」と読んだ（定点の名前は発話の文にそのまま入る）。声にする直前に
読みの表で置き換える。画面の文字は変えない。表はコードに持ち、定点名・家族の呼び名など、読み違いが見つかったら足す。
"""

from __future__ import annotations

#: 表記 → 読み（声だけ）。長いものから当てる。
READINGS: dict[str, str] = {
    "出入口": "でいりぐち",
}


def for_speech(text: str) -> str:
    out = text or ""
    for word, kana in sorted(READINGS.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(word, kana)
    return out
