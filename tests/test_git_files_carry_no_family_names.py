"""出-do：git に入れるファイルに、家族の名前を書かない（2026-11-21）。

手元に `FAMILY.md`（git の外）があるときだけ、それを読み、名前（漢字）、英字（名と姓）、呼び方（ひらがなとカタカナ）が
git に入っているファイルに無いかを見る。落ちたときに出すのはファイルの名前と件数だけで、名前そのものは出さない。
2 人以上に共通する呼び方と、パパ・ママ・パジュは見ない。
"""

from __future__ import annotations

import collections
import pathlib
import re
import subprocess

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_OUTSIDE = {"FAMILY.md", "ME.md", "PEOPLE.md"}
_ROLES = {"パパ", "ママ", "パジュ"}


def _kata(t: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in t)


_KANA = {
    "あ": "a",
    "い": "i",
    "う": "u",
    "え": "e",
    "お": "o",
    "か": "ka",
    "き": "ki",
    "く": "ku",
    "け": "ke",
    "こ": "ko",
    "さ": "sa",
    "し": "shi",
    "す": "su",
    "せ": "se",
    "そ": "so",
    "た": "ta",
    "ち": "chi",
    "つ": "tsu",
    "て": "te",
    "と": "to",
    "な": "na",
    "に": "ni",
    "ぬ": "nu",
    "ね": "ne",
    "の": "no",
    "は": "ha",
    "ひ": "hi",
    "ふ": "fu",
    "へ": "he",
    "ほ": "ho",
    "ま": "ma",
    "み": "mi",
    "む": "mu",
    "め": "me",
    "も": "mo",
    "や": "ya",
    "ゆ": "yu",
    "よ": "yo",
    "ら": "ra",
    "り": "ri",
    "る": "ru",
    "れ": "re",
    "ろ": "ro",
    "わ": "wa",
    "を": "wo",
    "ん": "n",
    "が": "ga",
    "ぎ": "gi",
    "ぐ": "gu",
    "げ": "ge",
    "ご": "go",
    "ざ": "za",
    "じ": "ji",
    "ず": "zu",
    "ぜ": "ze",
    "ぞ": "zo",
    "だ": "da",
    "ぢ": "ji",
    "づ": "zu",
    "で": "de",
    "ど": "do",
    "ば": "ba",
    "び": "bi",
    "ぶ": "bu",
    "べ": "be",
    "ぼ": "bo",
    "ぱ": "pa",
    "ぴ": "pi",
    "ぷ": "pu",
    "ぺ": "pe",
    "ぽ": "po",
}
_YOON = {"ゃ": "ya", "ゅ": "yu", "ょ": "yo"}


def romaji(h: str) -> "str | None":
    out, i = "", 0
    while i < len(h):
        c = h[i]
        if c == "っ" and i + 1 < len(h) and h[i + 1] in _KANA:
            out += _KANA[h[i + 1]][0]
            i += 1
            continue
        if c in _KANA:
            r = _KANA[c]
            if i + 1 < len(h) and h[i + 1] in _YOON:
                y = _YOON[h[i + 1]]
                r = r[:-1] + (y[-1] if r in ("shi", "chi", "ji") else y)
                i += 1
            out += r
        elif c == "ー" and out:
            out += out[-1]
        else:
            return None
        i += 1
    return out


def _words(text: str) -> set[str]:
    """名前（漢字）、英字（名と姓、大文字小文字の 3 形）、呼び方（「〜さん」は「〜」）と、そのカタカナとローマ字。"""
    people: list[dict[str, list[str]]] = []
    for line in text.splitlines():
        m = re.match(r"^\s*[-*]?\s*\*\*(名前|英字|呼び方)\*\*\s*[：:]\s*(.+)$", line)
        if not m:
            continue
        if m.group(1) == "名前":
            people.append({"名前": [], "英字": [], "呼び方": []})
        if not people:
            continue
        toks = [t.strip("「」『』（）() ") for t in re.split(r"[、,，・/／]+", m.group(2))]
        people[-1][m.group(1)] += [t for t in toks if t]
    calls = collections.Counter(t for p in people for t in set(p["呼び方"]))
    words: set[str] = set()
    for p in people:
        words |= set(p["名前"])
        for t in p["英字"]:
            for w in (t, *t.split()):
                words |= {w, w.lower(), w.upper()}
        for t in p["呼び方"]:
            if t in _ROLES or calls[t] >= 2 or len(t) < 2:
                continue
            base = t[:-2] if t.endswith("さん") and len(t) > 3 else t
            if base in _ROLES:
                continue
            words |= {base, _kata(base)}
            rk = romaji(base)
            if rk and len(rk) >= 3:
                for a in (rk, rk.replace("ou", "o"), rk.replace("ou", "oo")):
                    words |= {a, a.capitalize(), a.upper()}
    return {w for w in words if len(w) >= 2}


def test_no_git_file_carries_a_family_name() -> None:
    family = _ROOT / "FAMILY.md"
    if not family.exists():
        pytest.skip("FAMILY.md が無い機体（名前が分からない）")
    words = _words(family.read_text(encoding="utf-8"))
    pats = [
        re.compile(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])")
        if w.isascii()
        else re.compile(re.escape(w))
        for w in words
    ]
    # `-z` で受ける（日本語の名前のファイルが引用符の形で返り、読み飛ばされないように）。
    files = [
        f
        for f in subprocess.run(
            ["git", "ls-files", "-z"], cwd=_ROOT, capture_output=True, text=True, check=True
        ).stdout.split("\0")
        if f
    ]
    found: dict[str, int] = {}
    for f in files:
        if f in _OUTSIDE:
            continue
        try:
            text = (_ROOT / f).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        n = sum(len(p.findall(text)) for p in pats)
        if n:
            found[f] = n
    assert not found, f"家族の名前が git のファイルにある（名前は出さない）：{found}"
