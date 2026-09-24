"""声に出すときの読み（出-ac・環-t・2026-09-19）。

ElevenLabs（flash・multilingual_v2）は漢字の読みが使えない（出入口・押入・金木犀・九月十九日・三分が全部 ×・
`language_code=ja` でも同じ）。v3 は読めるが遅い。**pyopenjtalk（Style-Bert-VITS2 と同じ辞書）で読みをひらがな
にしてから渡す**と、いまの声と速さのまま自然に読めた（本人が聴いて確認）。順は、読みの表（固有名詞の例外）→
pyopenjtalk。表はカタカナ・ひらがなで書く（辞書に載らない読み違いが見つかったら足す）。画面の文字は変えない。
pyopenjtalk が無ければ表の分だけ（degrade）。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: 読みに直さない部分：角括弧タグ（eleven_v3 の話し方の指示）と、英字の語（pyopenjtalk は綴りを 1 字ずつ読む）。
_KEEP = re.compile(r"\[[^\]]*\]|[A-Za-z][A-Za-z0-9 .,'\-]*")
_KANJI = re.compile(r"[\u4e00-\u9fff\u3005]")

#: 表記 → 読み（声だけ）。辞書より先に当てる。長いものから。
READINGS: dict[str, str] = {
    "出入口": "でいりぐち",
    # pyopenjtalk は「その間」を ソノカン と読む（環-v・2026-09-24 実測）。SBV2 も同じ辞書を
    # 使うので、ひらがな化を通さない担い手にも当てる必要がある。
    "その間": "そのあいだ",
}

_warned = False


def _g2p(text: str) -> "str | None":
    """pyopenjtalk の読み（カタカナ）。無ければ None。"""
    global _warned
    try:
        import pyopenjtalk
    except Exception:  # noqa: BLE001
        if not _warned:
            logger.warning("pyopenjtalk が無いので声の読みは表の分だけ")
            _warned = True
        return None
    try:
        return str(pyopenjtalk.g2p(text, kana=True))
    except Exception as e:  # noqa: BLE001
        logger.warning("読みに失敗したので元の文のまま：%s", e)
        return None


def _hira(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def to_hiragana(text: str) -> str:
    """漢字まじりの文を読み（ひらがな）に。句読点は残る。pyopenjtalk が無ければそのまま。

    漢字を含む部分だけ直す（かなだけの文は ElevenLabs がそのまま読める・「こんばんは」を「こんばんわ」にしない）。
    角括弧タグと英字の語は触らない。
    """
    if not text:
        return ""
    out: list[str] = []
    pos = 0
    for m in _KEEP.finditer(text):
        out.append(_convert(text[pos : m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_convert(text[pos:]))
    return "".join(out)


def _convert(segment: str) -> str:
    if not segment or not _KANJI.search(segment):
        return segment
    kana = _g2p(segment)
    if kana is None:
        return segment
    return _hira(kana).replace("　", "")


def fix_readings(text: str) -> str:
    """表の分だけ当てる（**ひらがな化はしない**・環-v）。長いものから。

    自前で読む担い手（SBV2）にも要る。`pyopenjtalk` の辞書は `その間` を ソノカン と読み、
    SBV2 は同じ辞書を使うからである。全文をひらがなにすると、読めている担い手の読みまで
    崩しかねないので、**直す語だけ**を置き換える。
    """
    out = text or ""
    for word, kana in sorted(READINGS.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(word, kana)
    return out


def for_speech(text: str) -> str:
    return to_hiragana(fix_readings(text))
