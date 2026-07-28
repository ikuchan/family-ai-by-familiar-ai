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
    """Parse FAMILY.md into a list of {name, display_name} dicts.

    Supports the FAMILY-template.md format:
      ## Section heading
      - **名前**：田中太郎
      - **呼び方**：お父さん
    """
    if not text:
        return []

    _NAME_RE = re.compile(r"[-*]\s*\*{0,2}名前\*{0,2}\s*[：:]\s*(.+)", re.MULTILINE)
    _CALL_RE = re.compile(r"[-*]\s*\*{0,2}呼び方\*{0,2}\s*[：:]\s*(.+)", re.MULTILINE)
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
        members.append({"name": name, "display_name": display_name or name})
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
