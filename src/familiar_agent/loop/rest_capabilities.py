"""REST 内省・層 4「能力を再定義する」（記-a-と・2026-09-14・`用語一覧` v0.70）。

能力の器は 2 つ（`capability_state.py`）——**一覧**（`capabilities.yaml` は既定・現在値は DB の
`agent_state.capabilities`）と**要約**（`agent_state.capability_summary`・システム文の
`[あなたは誰か]` に載る 1 枚）。この層は、

- 一覧を `MANIFEST_EVERY_DAYS` 日〔仮〕に 1 度、実装の docstring・`.env`・MCP 設定から LLM に
  書き直させ、形の検査（YAML として読める・`id` が揃って重複せず・有効条件がある・件数が半分未満に
  減っていない）に通ったものだけ DB に置く。**file は実行時に書かない**（既定は git が持つ）。
- 要約を、一覧か自己像（層 2）が変わった晩、または要約がまだ無いときだけ作り直す。`ME.md`
  （人が書いた人格）が先頭にそのまま残り、`SUMMARY_MAX_CHARS` 字〔仮〕に収まるものだけ保存する。

どちらも検査に落ちたら前の値を残す（層 2 と同じ作法）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import yaml  # type: ignore[import-untyped]

from ..capability_state import (
    build_generation_prompt,
    build_self_understanding_prompt,
    capabilities_updated_at,
    collect_manifest_context,
    filter_enabled,
    load_capabilities,
    live_tool_names,
    load_summary,
    save_summary,
    store_capabilities,
)
from ..core import measure
from ..core.helpers import strip_code_fence

logger = logging.getLogger(__name__)

MANIFEST_EVERY_DAYS = 7  # 一覧を書き直す間隔〔仮〕
#: 要約の上限（字）。`ME.md`（人が書くぶん・実測 1,223 字）＋「私にできること」20 行
#: （実測 35 字／行）を容れる（本人の決定・2026-09-23）。1,000 字だった頃は `ME.md` だけで
#: 超えており、**要約を作っても必ず捨てられていた**（出-ao）。`[あなたは誰か]` は毎ターン
#: 主LLM へ届くが、system の安定部に入るのでキャッシュ越しである。
SUMMARY_MAX_CHARS = 2000
MIN_KEEP_RATIO = 0.5  # 件数がこの割合未満に減った一覧は、途中で切れた出力とみなして置かない


@dataclass(frozen=True)
class ManifestResult:
    changed: bool
    reason: "str | None" = None  # 置かなかった理由（None なら検査に通った・または変化なし）


def due_for_manifest(last_at: "datetime | None", now: datetime) -> bool:
    if last_at is None:
        return True
    if last_at.tzinfo is None:
        last_at = last_at.replace(tzinfo=timezone.utc)
    return now - last_at >= timedelta(days=MANIFEST_EVERY_DAYS)


def summary_max_tokens() -> int:
    """要約を頼むときの上限トークン。**上限の字数から決める**（出-ao・2026-09-23）。

    固定の 1,000 で頼んでいたため生成が途中で尽き、末尾が `- ` のまま保存されていた。
    日本語は 1 字およそ 1〜1.5 トークンなので、字数の 2 倍を渡す。
    """
    return SUMMARY_MAX_CHARS * 2


def due_for_summary(
    *,
    manifest_changed: bool,
    self_image_changed: bool,
    exists: bool = True,
    summary: "str | None" = None,
    me_md: str = "",
) -> bool:
    """要約を作り直す晩か。

    引き金は 4 つ——一覧が変わった・自己像が変わった・要約がまだ無い・**`ME.md` が
    書き換わった**（出-ao・2026-09-23）。4 つ目が無かったため、`ME.md` を書き直しても
    `[あなたは誰か]` には届かず、本番では 10 日前の要約が残り続けていた。

    `ME.md` が変わったかは、**保存済みの要約そのもの**で分かる。要約は `ME.md` を先頭に
    逐語で含むので、先頭が合わなければ古い。更新時刻を持ち回らなくてよい。
    """
    if summary is not None:
        exists = bool(summary)
        if exists and me_md.strip() and not summary.startswith(me_md.strip()):
            return True
    return manifest_changed or self_image_changed or not exists


def _cut_off(text: str) -> bool:
    """生成が途中で尽きた形か。箇条書きの頭だけ・読点で終わる・閉じていない括弧。"""
    tail = (text or "").rstrip()
    if not tail:
        return True
    last = tail.splitlines()[-1].strip()
    if last in ("-", "*", "#", "##", "###") or last.endswith(("、", "，", "：", ":")):
        return True
    return tail.count("（") != tail.count("）") or tail.count("「") != tail.count("」")


def check_manifest(text: str, current: str) -> "str | None":
    """形の検査。通れば None、落ちれば理由。"""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return f"YAML として読めない：{type(e).__name__}"
    caps = (data or {}).get("capabilities") if isinstance(data, dict) else None
    if not isinstance(caps, list) or not caps:
        return "capabilities が空"
    ids: list[str] = []
    for i, c in enumerate(caps, 1):
        if not isinstance(c, dict) or not c.get("id"):
            return f"{i} 件目に id が無い"
        if not c.get("summary"):
            return f"{c['id']} に summary が無い"
        if "enabled" not in c and "enabled_env" not in c:
            return f"{c['id']} に enabled／enabled_env が無い"
        ids.append(str(c["id"]))
    if len(set(ids)) != len(ids):
        return "id が重複している"
    try:
        before = len(((yaml.safe_load(current) or {}).get("capabilities") or []))
    except yaml.YAMLError:
        before = 0
    if before and len(ids) < before * MIN_KEEP_RATIO:
        return f"件数が {before} → {len(ids)} に減りすぎ"
    return None


async def regenerate_manifest(agent) -> ManifestResult:
    current = load_capabilities()
    prompt = build_generation_prompt(collect_manifest_context(live_tool_names(agent)), current)
    try:
        raw = str(await agent.backend.complete(prompt, max_tokens=2500) or "")
    except Exception as e:  # noqa: BLE001
        return ManifestResult(changed=False, reason=f"依頼に失敗：{e}")
    text = strip_code_fence(raw).strip() + "\n"
    reason = check_manifest(text, current)
    if reason:
        logger.warning("rest 層 4：一覧を置かなかった（%s）", reason)
        return ManifestResult(changed=False, reason=reason)
    if text.strip() == current.strip():
        return ManifestResult(changed=False)
    store_capabilities(text)
    logger.info("rest 層 4：一覧を書き直した（%d 字）", len(text))
    return ManifestResult(changed=True)


async def refresh_summary(agent, manifest: str) -> "str | None":
    """要約を作り直して保存する。保存しなかったら理由を返す。"""
    me_md = str(getattr(agent, "_me_md", "") or "")
    prompt = build_self_understanding_prompt(
        me_md=me_md, manifest=filter_enabled(manifest, tools=set(live_tool_names(agent)))
    )
    try:
        raw = str(await agent.backend.complete(prompt, max_tokens=summary_max_tokens()) or "")
    except Exception as e:  # noqa: BLE001
        return f"依頼に失敗：{e}"
    text = strip_code_fence(raw).strip()
    if me_md.strip() and not text.startswith(me_md.strip()):
        return "ME.md がそのまま残っていない"
    if _cut_off(text):
        # 生成が尽きた出力は、上限の検査を通り抜けてしまう（保存済みの 749 字は末尾が
        # `- ` のままだった・出-ao）。**途中で切れたものを置かない。**
        return "途中で切れている"
    if len(text) > SUMMARY_MAX_CHARS:
        return f"{SUMMARY_MAX_CHARS} 字を超えた（{len(text)}）"
    save_summary(text)
    logger.info("rest 層 4：要約を作り直した（%d 字）", len(text))
    return None


async def redefine_capabilities(
    agent, *, self_image_changed: bool, now: "datetime | None" = None
) -> str:
    """層 4 を 1 回。何をしたかの短い文を返す（`内省` の記録に載る）。"""
    now = now or datetime.now(timezone.utc)
    parts: list[str] = []
    manifest_changed = False
    if due_for_manifest(capabilities_updated_at(), now):
        r = await regenerate_manifest(agent)
        manifest_changed = r.changed
        if r.reason:
            parts.append(f"能力の一覧は置かなかった（{r.reason}）")
        else:
            parts.append("能力の一覧を書き直した" if r.changed else "能力の一覧は変わらなかった")
    if due_for_summary(
        manifest_changed=manifest_changed,
        self_image_changed=self_image_changed,
        summary=load_summary(),
        me_md=str(getattr(agent, "_me_md", "") or ""),
    ):
        reason = await refresh_summary(agent, load_capabilities())
        parts.append(f"要約は作り直さなかった（{reason}）" if reason else "要約を作り直した")
    measure.record(
        "層4",
        一覧=("書き直した" if manifest_changed else "-"),
        要約=("作り直した" if any(p == "要約を作り直した" for p in parts) else "-"),
        見送り=next((p for p in parts if "なかった" in p), "-"),
    )
    return "。".join(parts) if parts else "能力は見送った"
