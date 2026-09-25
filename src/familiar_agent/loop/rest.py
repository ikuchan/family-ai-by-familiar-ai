"""REST 内省の 1 パス（記-a）。

目的は自己認識の変革で、仕事は 4 層を順に更新すること（`用語一覧` v0.70）——
出来事（畳む）→ 自己像（抽象化する）→ 設定値（調整する）→ 能力（再定義する）。
層 1 と層 2 のあいだで**季節の層**（`rest_season.py`・知-ac）が、いまの季節と家のまわりを書き直す。
起動は T の純粋欠乏発火で、誰も居ないときだけ回る（`loop/tonic.py`）。

**いま動いているのは層 1 の計測と減り（`rest_info.py`）・②核の固め（`rest_core.py`）・①日次の畳み込み（`rest_fold.py`）、層 2（自己像の見直し・
`rest_self_image.py`）、層 3（設定値の調整と計測ログの改名・`rest_settings.py`）、層 4（能力の
再定義と要約の作り直し・`rest_capabilities.py`）**。層 1 は 4 つの段が揃った（2026-09-15）。回ったことは `direction='内省'` の記録に残す——
ログだけだと、起動しなかったのか、起動したが何もしなかったのかを区別できない。
"""

from __future__ import annotations

import logging

from .rest_capabilities import redefine_capabilities
from .rest_core import fold_core
from .rest_fold import fold_since_last_rest
from .rest_info import measure_and_decay
from .rest_season import update_season
from .rest_self_image import Material, update_self_image
from .rest_settings import adjust_settings

logger = logging.getLogger(__name__)


async def run_rest_pass(agent) -> str:
    """内省を 1 パス回して、何をしたかを返す（同じ文を `内省` の記録にも書く）。

    層 1 の畳み込みが落ちても例外を外へ出さない——材料は残り、次の晩に持ち越す。
    """
    parts: list[str] = []
    # 層 1 の前段：使われる情報量 I を測り、超えていれば参照されなかった核の根づきを減らす（記-a-ろ-ろ）。
    try:
        parts.append(await measure_and_decay(agent))
    except Exception as e:  # noqa: BLE001
        logger.exception("rest 層 1 の計測に失敗: %s", e)
        parts.append("使われる情報量を測れなかった")
    # 層 1 の②：同一を 1 件に、核が I* を超えていれば意味の近い束をまとめ知識に固める（記-a-ろ-に）。
    records: tuple = ()
    try:
        c = await fold_core(agent)
        records = c.records
        bits = []
        if c.identical_folded:
            bits.append(f"同じ記録 {c.identical_folded} 件を 1 件に")
        if c.written:
            bits.append(f"核を {c.written} 束に固めて {c.folded} 件を畳んだ（見送り {c.skipped}）")
        elif c.excess_bits > 0:
            bits.append(
                f"核は目標を {int(c.excess_bits)} bit 超えていたが固めなかった（見送り {c.skipped}）"
            )
        parts.append("。".join(bits) if bits else "核は固めるものが無かった")
    except Exception as e:  # noqa: BLE001
        logger.exception("rest 層 1 の核の固めに失敗: %s", e)
        parts.append("核を固められなかった")
    try:
        r = await fold_since_last_rest(agent)
        records = tuple(records) + tuple(r.records)
        if r.materials == 0:
            parts.append("畳むものが無かった")
        else:
            parts.append(
                f"出来事 {r.materials} 件を畳み、自己エピソードと関係のまとめを {r.written} 件書いた。"
                f"見送り {r.skipped} 回"
            )
    except Exception as e:  # noqa: BLE001
        logger.exception("rest 層 1 の畳み込みに失敗（次の晩に持ち越す）: %s", e)
        parts.append("出来事を畳めなかった・次の晩に持ち越す")
    # 季節の層：層 1 の後・層 2 の前。いまの季節と家のまわりを晩に 1 回書き直す（知-ac）。
    # 層の番号は振り直さず名前で呼ぶ（層 2〜4 の番号・ログ・計測の種別はそのまま）。
    try:
        s = await update_season(agent, list(records))
        parts.append(
            "季節とまわりを書いた" if s.applied else f"季節とまわりは書かなかった（{s.reason}）"
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("rest 季節の層に失敗: %s", e)
        parts.append("季節とまわりを書けなかった")
    # 層 2：層 1 が書いたものを材料に、自己像を見直す（記-a-へ）。
    self_image_changed = False
    try:
        p = await update_self_image(agent, [Material(w.obs_id, w.kind, w.text) for w in records])
        self_image_changed = p.applied
        parts.append(
            f"自己像を {p.changed} 行変えた"
            if p.applied
            else f"自己像は変えなかった（{p.reason or '変える必要なし'}）"
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("rest 層 2 の見直しに失敗: %s", e)
        parts.append("自己像を見直せなかった")
    # 層 3：計測ログを集計して設定値を動かし、読み終えた計測ログを改名する（記-a-に）。
    try:
        n = await adjust_settings(agent)
        parts.append(f"設定値を {n} 件動かした" if n else "設定値は動かさなかった")
    except Exception as e:  # noqa: BLE001
        logger.exception("rest 層 3 の調整に失敗: %s", e)
        parts.append("設定値を見直せなかった")
    # 層 4：能力の一覧を 7 日に 1 度書き直し、一覧か自己像が変わった晩は要約を作り直す（記-a-と）。
    try:
        parts.append(await redefine_capabilities(agent, self_image_changed=self_image_changed))
    except Exception as e:  # noqa: BLE001
        logger.exception("rest 層 4 の再定義に失敗: %s", e)
        parts.append("能力を見直せなかった")
    content = "内省を回した（" + "。".join(parts) + "）"
    logger.info("rest 内省パス：%s", content)
    await agent._memory.save_async_with_id(
        content[:500],
        direction="内省",
        kind="observation",
        materialize_now=True,
        **agent._observation_perspective(),
    )
    return content
