"""REST 内省・層 4「能力を再定義する」（記-a-と・2026-09-14・`用語一覧` v0.70）。

能力の器は 2 つ（`capability_state.py`）——**一覧**（`capabilities.yaml`）と**要約**
（`agent_state.capability_summary`・システム文の `[あなたは誰か]` に載る 1 枚）。
**この層が作り直すのは要約だけ**である。

要約は、自己像（層 2）が変わった晩・要約がまだ無いとき・`ME.md` が書き換わったときに
作り直す。`ME.md`（人が書いた人格）が先頭にそのまま残り、`SUMMARY_MAX_CHARS` 字〔仮〕に
収まるものだけ保存する。検査に落ちたら前の値を残す（層 2 と同じ作法）。

**一覧を機械に書き直させる道は 環-y（2026-09-24）で撤去した。** 7 日に 1 度 LLM に書かせて
DB へ置く作りだったが、10 日たっても一度も走らず、DB の行は空のままだった。一覧を書くのは
機能を作る人である。書き忘れは番人テスト（`test_capabilities_match_reality`）が捕まえる。
"""

from __future__ import annotations

import logging
from datetime import datetime

from ..capability_state import (
    build_self_understanding_prompt,
    filter_enabled,
    live_tool_names,
    load_capabilities,
    load_summary,
    save_summary,
)
from ..core import measure
from ..core.helpers import strip_code_fence

logger = logging.getLogger(__name__)

#: 要約の上限（字）。`ME.md`（人が書くぶん・実測 1,223 字）＋「私にできること」20 行
#: （実測 35 字／行）を容れる（本人の決定・2026-09-23）。1,000 字だった頃は `ME.md` だけで
#: 超えており、**要約を作っても必ず捨てられていた**（出-ao）。`[あなたは誰か]` は毎ターン
#: 主LLM へ届くが、system の安定部に入るのでキャッシュ越しである。
SUMMARY_MAX_CHARS = 2000


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
    """層 4 を 1 回。何をしたかの短い文を返す（`内省` の記録に載る）。

    やるのは**要約の作り直しだけ**である（環-y）。一覧は `capabilities.yaml` が正で、
    機械は触らない。`now` は呼び手の作法を揃えるために残す（いまは使わない）。
    """
    parts: list[str] = []
    if due_for_summary(
        manifest_changed=False,
        self_image_changed=self_image_changed,
        summary=load_summary(),
        me_md=str(getattr(agent, "_me_md", "") or ""),
    ):
        reason = await refresh_summary(agent, load_capabilities())
        parts.append(f"要約は作り直さなかった（{reason}）" if reason else "要約を作り直した")
    measure.record(
        "層4",
        要約=("作り直した" if any(p == "要約を作り直した" for p in parts) else "-"),
        見送り=next((p for p in parts if "なかった" in p), "-"),
    )
    return "。".join(parts) if parts else "能力は見送った"
