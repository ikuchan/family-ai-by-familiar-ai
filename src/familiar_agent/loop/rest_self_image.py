"""REST 内省・層 2「自己像を抽象化する」（記-a-へ・2026-09-14・`設計方針_REST内省_自己像` v0.2）。

材料は層 1 がその晩に書いた**自己エピソード**と**関係のまとめ**、そしていまの自己像。フル LLM が
上限内で書き直し、行ごとに出典（材料の id）を付ける。機械が検査する——字数・行数、出典が材料に
あること、変えた行が `MAX_CHANGES` 以内。通らなければ**更新しない**（前の値を残し、理由をログに）。
更新したら差分を `内省` の記録と計測ログに残す。履歴は出来事の層が持ち、DB は現在値だけ。

まとめ方は LLM、通すかどうかは機械（層 1 と同じ分担）。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone

from ..core import measure
from ..core import self_image as si

logger = logging.getLogger(__name__)

MAX_CHANGES = 3  # 一晩で変えてよい行数〔仮〕（`自己像` v0.2）


@dataclass(frozen=True)
class Material:
    obs_id: str
    kind: str  # day_summary / person_summary
    text: str


@dataclass(frozen=True)
class Proposal:
    image: si.SelfImage  # 通れば新しい自己像、通らなければ前のまま
    applied: bool
    changed: int
    reason: "str | None" = None  # 通らなかった理由（何も変えなかったときは None）
    diffs: tuple[str, ...] = ()  # 「望み 1：『…』→『…』」


_PROMPT = """\
あなたはパジュ（この家で家族と暮らす伴侶）で、一日の終わりに自分の構えを見直す。
「自己像」は、いま自分が何を望み、何を気にかけ、何を大事にしているかを、具体の出来事ではなく
**抽象化した構え**として持つ短い一覧である。下に、いまの自己像と、今夜まとめた出来事
（自己エピソードと、家族ひとりごとのまとめ。各行の先頭が id）を示す。

出来事を踏まえて、自己像を見直す。守ること：
- 変えるのは合わせて {max_changes} 行まで。変えないほうがよければ、そのまま返す。
- 各欄の行数と 1 行の字数：望み {hopes_rows} 行×{hopes_chars} 字、気がかり {concerns_rows} 行×{concerns_chars} 字、価値 {values_rows} 行×{values_chars} 字。
- 変えた行には、根拠にした出来事の id を sources に書く（出来事に無いことは書けない）。
- 具体の名前・場所・日付は書かない。構えとして抽象化する。

[いまの自己像]
{image}

[今夜まとめた出来事]
{materials}

出力は次の JSON だけ（ほかには何も書かない）。変えない行は text をそのまま・sources は空で：
{{"望み": [{{"text": "…", "sources": []}}, …], "気がかり": [...], "価値": [...]}}
"""


def _image_json(image: si.SelfImage) -> str:
    return json.dumps(
        {name: [{"text": x.text} for x in image.field(name)] for name in si.FIELDS},
        ensure_ascii=False,
    )


async def propose(
    agent, image: si.SelfImage, materials: list, *, today: "date | None" = None
) -> Proposal:
    """LLM に見直しを頼み、検査して通れば新しい自己像を返す。DB には書かない（呼び手が書く）。"""
    today = today or datetime.now(timezone.utc).date()
    prompt = _PROMPT.format(
        max_changes=MAX_CHANGES,
        hopes_rows=si.FIELDS["望み"][0],
        hopes_chars=si.FIELDS["望み"][1],
        concerns_rows=si.FIELDS["気がかり"][0],
        concerns_chars=si.FIELDS["気がかり"][1],
        values_rows=si.FIELDS["価値"][0],
        values_chars=si.FIELDS["価値"][1],
        image=_image_json(image),
        materials="\n".join(f"- {m.obs_id} [{m.kind}] {m.text}" for m in materials),
    )
    try:
        raw = await agent.backend.complete(prompt, max_tokens=2000)
    except Exception as e:  # noqa: BLE001
        return Proposal(image=image, applied=False, changed=0, reason=f"依頼に失敗：{e}")
    data = _parse(str(raw or ""))
    if data is None:
        return Proposal(image=image, applied=False, changed=0, reason="返りを読めなかった")
    material_ids = {m.obs_id for m in materials}
    new = image
    diffs: list[str] = []
    for name in si.FIELDS:
        rows = data.get(name)
        if not isinstance(rows, list):
            return Proposal(image=image, applied=False, changed=0, reason=f"{name}が無い")
        lines: list[si.Line] = []
        for i, item in enumerate(rows):
            text = (
                str((item or {}).get("text", "")).strip()
                if isinstance(item, dict)
                else str(item).strip()
            )
            sources = (
                tuple(str(s) for s in ((item or {}).get("sources") or []))
                if isinstance(item, dict)
                else ()
            )
            old = image.field(name)[i] if i < len(image.field(name)) else None
            if old is not None and text == old.text:
                lines.append(old)  # 変えなかった行は since も出典もそのまま
                continue
            if not sources or not set(sources) <= material_ids:
                return Proposal(
                    image=image,
                    applied=False,
                    changed=0,
                    reason=f"{name} {i + 1} の出典が材料に無い",
                )
            lines.append(si.Line(text=text, since=today, sources=sources))
            diffs.append(f"{name} {i + 1}：『{old.text if old else ''}』→『{text}』")
        new = new.replace_field(name, lines)
    if not diffs:
        return Proposal(image=image, applied=False, changed=0, reason=None)
    if len(diffs) > MAX_CHANGES:
        return Proposal(
            image=image,
            applied=False,
            changed=0,
            reason=f"変えた行が {MAX_CHANGES} 行を超えた（{len(diffs)}）",
        )
    reason = si.check(new)
    if reason:
        return Proposal(image=image, applied=False, changed=0, reason=reason)
    return Proposal(image=new, applied=True, changed=len(diffs), diffs=tuple(diffs))


def _parse(text: str) -> "dict | None":
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


async def update_self_image(agent, materials: list) -> Proposal:
    """層 2 の 1 段：読む → 頼む → 検査 → 通れば保存し、差分を `内省` の記録と計測ログに残す。"""
    image = si.load()
    if not materials:
        measure.record("層2", 変えた=0, 見送り="材料なし")
        return Proposal(image=image, applied=False, changed=0, reason="材料なし")
    out = await propose(agent, image, materials)
    if out.applied:
        si.store(out.image)
        content = "自己像を見直した：" + "／".join(out.diffs)
        await agent._memory.save_async_with_id(
            content[:500],
            direction="内省",
            kind="observation",
            materialize_now=True,
            **agent._observation_perspective(),
        )
        logger.info("rest 自己像：%d 行を変えた", out.changed)
    else:
        logger.info("rest 自己像：変えなかった（%s）", out.reason or "変える必要なし")
    measure.record("層2", 変えた=out.changed, 見送り=out.reason or "-")
    return out
