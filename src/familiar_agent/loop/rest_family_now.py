"""REST が家族のいまの様子を書き直す（知-ad 段 3・2026-10-01・本人の決定）。

材料は日々の蒸留（日次の畳み込み）が書いた**関係のまとめ（`person_summary`）だけ**で、生の記録は使わない。
家族ひとりずつ、まだ読んでいない関係のまとめを**古い順に 10 本ずつ**読み（10 本＝その人と関わった 10 日ぶん）、
そのたびに前の版へ重ねて書き直す。読み終えた位置は最後に読んだ関係のまとめの時刻として控え（`family_now`
の数え始め）、次はその先から読む。1 晩のうちに 10 本そろうかぎり繰り返し、届かない残りは次の晩へ回す。

書き手はフルLLM（日々の蒸留と同じ担い手）。人が書いた `FAMILY.md` のその人の節も渡すが、書き換えるのは DB の
いまの様子だけで、`FAMILY.md` には書かない（開発ルール・本人の決定イ）。返りが空・読めない・300 字の超過なら
書かず、位置も進めない（その人はその晩やめ、次の晩にもう一度試す）。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..core import family_now, parsing
from ..core.speaker_claim import call_name_of
from .rest_fold import PERSON_MAX_CHARS

logger = logging.getLogger(__name__)

#: 一度に読む関係のまとめの本数（その人と関わった日数・本人の決定 2026-10-01）。
CHUNK = 10
#: まだ一度も書いていない人の数え始め（日々の蒸留がいちばん古くに書いたものから読む）。
_BEGINNING = datetime(1970, 1, 1, tzinfo=timezone.utc)

_PROMPT = """\
あなたはパジュ（この家で家族と暮らす伴侶）で、{who}の**いまの様子**を書き直す。
下に、前に書いた{who}の様子と、そのあと{who}と関わった {n} 日ぶんの記録（その日に{who}について分かったこと）を
古い順に並べる。前の様子に、新しい日々の記録を重ねて、いまの{who}を書く——性格・好み・いま夢中なこと・
気がかり・続いていること。古くなったことは書き換え、続いていることは残す。

{limit} 字以内。記録に無いことは書かない。人が書いた紹介（下の [家族が書いた{who}]）と違ってきていれば、
記録のほうに合わせてよい（紹介そのものは書き換えない）。

出力は次の JSON だけ（ほかには何も書かない）：
{{"now": "…"}}

[家族が書いた{who}]
{section}

[前に書いた様子]
{before}

[そのあと {n} 日ぶんの記録]
{notes}
"""


@dataclass
class FamilyNowResult:
    rewrites: "dict[str, int]" = field(default_factory=dict)  # 呼び方 → その晩に書き直した回数
    skipped: int = 0  # 返りが使えず、その人をその晩やめた数


def _members(agent) -> "list[tuple[str, str, str]]":
    """家族ひとりずつ（呼び方の先頭, 人物 id, `FAMILY.md` のその人の節）。人物表に無い人は除く。"""
    text = str(getattr(agent, "_family_md", "") or "")
    sections = re.split(r"\n(?=##\s)", "\n" + text)
    out: list[tuple[str, str, str]] = []
    for m in parsing.parse_family_md(text):
        name = str(m.get("name") or "").strip()
        pid = agent._pmm.find_person_id_by_name(name) if name else None
        if not (pid and isinstance(pid, str)):
            continue
        who = call_name_of(m)
        section = next((s.strip() for s in sections if f"：{name}" in s or f":{name}" in s), "")
        out.append((who, pid, section))
    return out


def _parse(raw: str) -> "str | None":
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    text = str(data.get("now", "") if isinstance(data, dict) else "").strip()
    if not text or len(text) > PERSON_MAX_CHARS:
        return None
    return text


async def update_family_now(agent) -> FamilyNowResult:
    """家族ひとりずつ、10 本そろうかぎり古い順に読んで、いまの様子を書き直す。"""
    result = FamilyNowResult()
    current = family_now.stored()
    for who, pid, section in _members(agent):
        cur = current.get(who)
        since = cur.counted_from if cur else _BEGINNING
        before = cur.text if cur else ""
        notes = list(agent._oif.person_notes_after(pid, since) or [])
        while len(notes) >= CHUNK:
            chunk, notes = notes[:CHUNK], notes[CHUNK:]
            lines = "\n".join(
                f"- {r['timestamp']:%Y-%m-%d} {r.get('content', '')}"
                if isinstance(r.get("timestamp"), datetime)
                else f"- {r.get('content', '')}"
                for r in chunk
            )
            prompt = _PROMPT.format(
                who=who,
                n=len(chunk),
                limit=PERSON_MAX_CHARS,
                section=section or "（なし）",
                before=before or "（まだ書いていない）",
                notes=lines,
            )
            try:
                raw = await agent.backend.complete(prompt, max_tokens=800)
            except Exception as e:  # noqa: BLE001
                logger.warning("rest いまの様子の依頼に失敗（%s・次の晩に持ち越す）: %s", who, e)
                raw = ""
            text = _parse(str(raw or ""))
            if text is None:
                logger.warning("rest いまの様子を見送った（%s・位置は進めない）", who)
                result.skipped += 1
                break
            family_now.update(who, text, counted_from=chunk[-1]["timestamp"])
            before = text
            result.rewrites[who] = result.rewrites.get(who, 0) + 1
    logger.info(
        "rest いまの様子：%s（見送り %d）", result.rewrites or "書き直しなし", result.skipped
    )
    return result
