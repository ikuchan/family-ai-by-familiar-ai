"""音楽の決まり（知-aa 段 1・2026-09-21）。純関数。

鳴らす先は `spotifyd`（MPRIS・`io/music.py`）。ここが持つのは 4 つだけである。

- **表の読み取り**：`MUSIC.md` の `名前：URI` の並び。人が書く file で、機外に出さない
  （`ME.md`・`FAMILY.md` と同じ扱い）。
- **通す言葉**：鳴っているあいだは、音楽の操作とプレイリストの変更だけを通す（本人の決定・
  2026-09-21）。タイマーの「聞かないの門」と同じ形で、**通すかを決めるだけ**——何をするかは
  道具が決める。時間の道具（タイマー・アラーム）は通さない。**ただし文頭に名前があれば何でも
  通す**（2026-09-30・本人の決定ア。「パジュ、いま何の曲？」「パジュ、3 分測って」）。
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
    r"^[ \t　]*([^：:（(#`\n][^：:\n]*)[：:][ \t　]*(spotify:[A-Za-z0-9:]+)"
    r"(?:[：:][ \t　]*([^\n]*))?[ \t　]*$",
    re.MULTILINE,
)
#: 順番の指し方。3 つ目の欄（`名前：URI：ランダム`）と、言葉の中の言い方に同じ表を使う。
_SHUFFLE = re.compile(r"(ランダム|しゃっふる|シャッフル|順不同)")
_IN_ORDER = re.compile(r"(順番|順に|そのまま)")
#: 音楽の操作の言葉。長い文の中の「止めて」は通さない（テレビの台詞でありうる）。
_CONTROL = re.compile(
    r"(止め|ストップ|中止|やめ|次の曲|次へ|スキップ|大きく|小さく|音量|ランダム|シャッフル|順番)"
)
#: 操作の言葉として受ける長さ（字）。プレイリスト名はこれと別に表で当てる。
CONTROL_MAX_CHARS = 14


def parse_music_md(text: str) -> "tuple[tuple[str, str, bool], ...]":
    """`MUSIC.md` の `名前：URI[：ランダム]` を並びで取る（名前・URI・ランダムか）。

    3 つ目の欄は**その表の既定**で、言葉で上書きできる（`wants_shuffle`・本人の決定）。
    URI が無い行と雛形の括弧書きは飛ばす。
    """
    return tuple(
        (m.group(1).strip(), m.group(2).strip(), bool(_SHUFFLE.search(m.group(3) or "")))
        for m in _ROW.finditer(text or "")
    )


def find_playlist(
    text: str, table: "tuple[tuple[str, str, bool], ...]"
) -> "tuple[str, str, bool] | None":
    """言われた言葉から、表の行を当てる。長い名前から見る。"""
    s = (text or "").strip()
    if not s:
        return None
    for row in sorted(table, key=lambda r: -len(r[0])):
        if row[0] and row[0] in s:
            return row
    return None


def wants_shuffle(text: str, default: bool) -> bool:
    """順番か、ランダムか。言葉で言われていればそれに従い、無ければ表の既定のまま。"""
    s = text or ""
    if _SHUFFLE.search(s):
        return True
    if _IN_ORDER.search(s):
        return False
    return default


def is_music_word(
    text: str,
    table: "tuple[tuple[str, str, bool], ...]",
    *,
    names: "list[str] | tuple[str, ...]" = (),
) -> bool:
    """鳴っているあいだに通す言葉か（文頭に名前があるか、音楽の操作か、表にあるプレイリストの名前）。

    名前の判定は窓の門と同じ（文頭・`heard_name`）。名前つきなら時間の道具も通す（本人の決定ア）。
    """
    from .wake_window import heard_name

    s = (text or "").strip()
    if not s:
        return False
    if heard_name(s, list(names)):
        return True
    if find_playlist(s, table) is not None:
        return True
    return len(s) <= CONTROL_MAX_CHARS and bool(_CONTROL.search(s))


def expired(started_at: float, *, now: float) -> bool:
    """鳴らし始めてから `MUSIC_MAX_SEC` を過ぎたか。"""
    return (now - started_at) >= MUSIC_MAX_SEC


def duck(base: float) -> float:
    """会話中の音量。基準の 4 分の 1（0.0〜1.0 に収める）。"""
    return max(0.0, min(1.0, base)) * DUCK_RATIO


# ── 近いところから順に探す（知-aa 段 3・2026-10-02・本人の決定）──────────────────

#: 主LLM が言葉で指す種類 → Spotify の検索の種類。指さなければ曲。
KINDS = {"曲": "track", "アーティスト": "artist", "アルバム": "album", "プレイリスト": "playlist"}


def _norm(text: str) -> str:
    """表記ゆれを均す（カタカナとひらがな・長音・濁点・大小・空白）。名前の判定と同じゆるい読みを使う。"""
    from .silence_rules import _loose

    return re.sub(r"[\s　・]", "", _loose(text or ""))


def _hits(name: str, key: str) -> bool:
    n = _norm(name)
    return len(n) >= 2 and n in key


def find_local(said: str, table, catalog) -> "tuple[str, str, str, bool] | None":
    """手元から探す（1〜4 のうち検索の要らない分）。(どこから, 名前, URI, ランダムの既定) か None。

    1 `MUSIC.md` → 2 自分のプレイリスト → 3 ライブラリ（保存したアルバムと曲）→ 4 プレイリストに入っている曲。
    長い名前から当てる（「夜のドライブ」を「ドライブ」より先に）。
    """
    key = _norm(said)
    if not key:
        return None
    for name, uri, shuffle in sorted(table or (), key=lambda r: -len(r[0])):
        if _hits(name, key):
            return "MUSIC.md", name, uri, shuffle
    for p in sorted(catalog.playlists, key=lambda p: -len(p.name)):
        if _hits(p.name, key):
            return "プレイリスト", p.name, p.uri, False
    for t in list(catalog.albums) + list(catalog.tracks):
        if _hits(str(t.get("title") or ""), key):
            return "ライブラリ", _label(t), str(t["uri"]), False
    for t in catalog.playlist_tracks():
        if _hits(str(t.get("title") or ""), key):
            return "プレイリストの曲", _label(t), str(t["uri"]), False
    return None


def playlist_artist_in(said: str, catalog) -> "str | None":
    """言われた言葉に、プレイリストに入っているアーティストの名前があるか（多く入っている人から）。"""
    key = _norm(said)
    for artist, _n in catalog.playlist_artists():
        if _hits(artist, key):
            return artist
    return None


def same_name(a: str, b: str) -> bool:
    return bool(_norm(a)) and _norm(a) == _norm(b)


def my_artists(catalog) -> "set[str]":
    """プレイリストとライブラリに居るアーティスト（均した名前）。全体の検索で先に選ぶ。"""
    names = {a for a, _ in catalog.playlist_artists()}
    names |= {str(t.get("artist") or "") for t in list(catalog.albums) + list(catalog.tracks)}
    return {_norm(n) for n in names if n}


def _label(t: dict) -> str:
    artist = str(t.get("artist") or "")
    return f"「{t.get('title', '')}」" + (f"（{artist}）" if artist else "")
