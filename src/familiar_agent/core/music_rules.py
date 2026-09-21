"""音楽の決まり（知-aa 段 1・2026-09-21）。純関数。

鳴らす先は `spotifyd`（MPRIS・`io/music.py`）。ここが持つのは 4 つだけである。

- **表の読み取り**：`MUSIC.md` の `名前：URI` の並び。人が書く file で、機外に出さない
  （`ME.md`・`FAMILY.md` と同じ扱い）。
- **通す言葉**：鳴っているあいだは、音楽の操作とプレイリストの変更だけを通す（本人の決定・
  2026-09-21）。タイマーの「聞かないの門」と同じ形で、**通すかを決めるだけ**——何をするかは
  道具が決める。時間の道具（タイマー・アラーム）は通さない。
- **寿命**：鳴らし始めてから 30 分で止め、止めたことを一言言う（本人の決定）。
- **減音**：パジュが声を出しているあいだ（と、その後 10 秒）は 4 分の 1 に絞る。基準は人が
  変えた値を覚える。機械の反射で、主LLM は通らない。
"""

from __future__ import annotations

import re

#: 鳴らし続けられる上限（秒）。30 分で止めて通常状態へ戻る。
MUSIC_MAX_SEC = 1800
#: 会話中に絞る割合と、声が止まってから戻すまでの秒数。
DUCK_RATIO = 0.25
RESTORE_AFTER_SEC = 10

# 名前に**改行を含めない**（否定の文字クラスは改行にも当たるので、前の行から繋がって
# 「```\n朝の曲」のような名前になる・雛形の例で踏んだ 2026-09-21）。
_ROW = re.compile(
    r"^[ \t　]*([^：:（(#`\n][^：:\n]*)[：:][ \t　]*(spotify:[A-Za-z0-9:]+)[ \t　]*$",
    re.MULTILINE,
)
#: 音楽の操作の言葉。長い文の中の「止めて」は通さない（テレビの台詞でありうる）。
_CONTROL = re.compile(r"(止め|ストップ|中止|やめ|次の曲|次へ|スキップ|大きく|小さく|音量)")
#: 操作の言葉として受ける長さ（字）。プレイリスト名はこれと別に表で当てる。
CONTROL_MAX_CHARS = 14


def parse_music_md(text: str) -> "tuple[tuple[str, str], ...]":
    """`MUSIC.md` の `名前：URI` を並びで取る。URI が無い行と雛形の括弧書きは飛ばす。"""
    return tuple((m.group(1).strip(), m.group(2).strip()) for m in _ROW.finditer(text or ""))


def find_playlist(text: str, table: "tuple[tuple[str, str], ...]") -> "str | None":
    """言われた言葉から、表のプレイリストを当てる。長い名前から見る。"""
    s = (text or "").strip()
    if not s:
        return None
    for name, uri in sorted(table, key=lambda kv: -len(kv[0])):
        if name and name in s:
            return uri
    return None


def is_music_word(text: str, table: "tuple[tuple[str, str], ...]") -> bool:
    """鳴っているあいだに通す言葉か（音楽の操作か、表にあるプレイリストの名前）。"""
    s = (text or "").strip()
    if not s:
        return False
    if find_playlist(s, table) is not None:
        return True
    return len(s) <= CONTROL_MAX_CHARS and bool(_CONTROL.search(s))


def expired(started_at: float, *, now: float) -> bool:
    """鳴らし始めてから `MUSIC_MAX_SEC` を過ぎたか。"""
    return (now - started_at) >= MUSIC_MAX_SEC


def duck(base: float) -> float:
    """会話中の音量。基準の 4 分の 1（0.0〜1.0 に収める）。"""
    return max(0.0, min(1.0, base)) * DUCK_RATIO
