"""ME.md / FAMILY.md / 話者接頭辞の純粋パーサ（loop 非依存・境界R B3）。

いずれも文字列を受けて文字列や構造を返す純関数で、EmbodiedAgent の状態にも
制御流れにも依存しない。agent.py の staticmethod から切り出した（挙動不変）。
"""

from __future__ import annotations

import re

_SPEAKER_PREFIX_RE = re.compile(
    r"^[\[［]([^\]］]+)[\]］]\s*(.*)$|^@([^\s:：]+)[:\s：]\s*(.*)$",
    re.DOTALL,
)


ME_MD_CANDIDATES = ("ME.md", "~/.familiar_ai/ME.md")


def read_me_md() -> str:
    """`ME.md` を読む（作業 dir → `~/.familiar_ai/`）。無ければ空。

    名前の正本は `ME.md` なので、集音（`hotwords`）も agent もここから読む。
    """
    from pathlib import Path

    for c in ME_MD_CANDIDATES:
        path = Path(c).expanduser()
        if path.exists():
            try:
                return path.read_text(encoding="utf-8").strip()
            except Exception:  # noqa: BLE001
                continue
    return ""


def parse_me_name(text: str) -> str:
    """Extract the AI's name from ME.md. Returns empty string if not found."""
    m = re.search(r"名前\s*[：:]\s*(.+)", text)
    if not m:
        return ""
    return m.group(1).strip()


def parse_me_names(text: str) -> list[str]:
    """`ME.md` の「名前： …」から、名前として使える言葉を並びで取る。

    沈黙依頼は名前で呼ばれたときだけ受けるので、どう呼ばれても通る必要がある。呼び方は
    一つとは限らないので、読点かカンマで区切って並べられるようにする。
    """
    line = parse_me_name(text)
    if not line:
        return []
    parts = re.split(r"[、,]", line)
    return [p.strip() for p in parts if p.strip()]


def parse_family_md(text: str) -> list[dict]:
    """Parse FAMILY.md into a list of {name, display_name, latin} dicts.

    Supports the FAMILY-template.md format:
      ## Section heading
      - **名前**：田中太郎
      - **呼び方**：お父さん
      - **英字**：Taro Tanaka   → latin="taro"（無ければ ""）
    """
    if not text:
        return []

    _NAME_RE = re.compile(r"[-*]\s*\*{0,2}名前\*{0,2}\s*[：:]\s*(.+)", re.MULTILINE)
    _CALL_RE = re.compile(r"[-*]\s*\*{0,2}呼び方\*{0,2}\s*[：:]\s*(.+)", re.MULTILINE)
    # 英字（`Yusuke Ikunaga`）。先頭の語を小文字にした `yusuke` が個人ティアの道具名の鍵
    # （`ask_vault_yusuke`・知-f）。書いていない人は個人ティアの道具を持たない。
    _LATIN_RE = re.compile(r"[-*]\s*\*{0,2}英字\*{0,2}\s*[：:]\s*(.+)", re.MULTILINE)
    _TEMPLATE_SKIP = re.compile(r"^[（(].*[）)]$")

    members: list[dict] = []
    # Split on level-2 headings; each section describes one person
    sections = re.split(r"\n(?=##\s)", "\n" + text)
    for section in sections:
        name_m = _NAME_RE.search(section)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        if not name or _TEMPLATE_SKIP.match(name):
            continue
        call_m = _CALL_RE.search(section)
        display_name = call_m.group(1).strip() if call_m else ""
        if display_name and _TEMPLATE_SKIP.match(display_name):
            display_name = ""
        latin_m = _LATIN_RE.search(section)
        latin = latin_m.group(1).strip() if latin_m else ""
        if _TEMPLATE_SKIP.match(latin):
            latin = ""
        latin = latin.split()[0].lower() if latin else ""
        members.append({"name": name, "display_name": display_name or name, "latin": latin})
    return members


def extract_speaker_prefix(user_input: str) -> tuple[str, str | None]:
    """Parse [name] or @name: prefix. Return (stripped_text, speaker_name | None)."""
    m = _SPEAKER_PREFIX_RE.match(user_input)
    if not m:
        return user_input, None
    if m.group(1) is not None:
        # [name] format
        return (m.group(2) or "").strip(), m.group(1).strip()
    # @name: format
    return (m.group(4) or "").strip(), m.group(3).strip()
