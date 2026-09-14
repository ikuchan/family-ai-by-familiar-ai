"""REST 内省の 1 パス（記-a）。

目的は自己認識の変革で、仕事は 4 層を順に更新すること（`用語一覧` v0.70）——
出来事（畳む）→ 自己像（抽象化する）→ 設定値（調整する）→ 能力（再定義する）。
起動は T の純粋欠乏発火で、誰も居ないときだけ回る（`loop/tonic.py`）。

**いま動いているのは層 1 の①（日次の畳み込み・`rest_fold.py`）、層 2（自己像の見直し・
`rest_self_image.py`）、層 3（設定値の調整と計測ログの改名・`rest_settings.py`）**。層 1 の②
（核の固め）と $n$ の減り、層 4 はこれから（`課題8` 記-a）。回ったことは `direction='内省'` の記録に残す——
ログだけだと、起動しなかったのか、起動したが何もしなかったのかを区別できない。
"""

from __future__ import annotations

import logging

from .rest_fold import fold_since_last_rest
from .rest_self_image import Material, update_self_image
from .rest_settings import adjust_settings

logger = logging.getLogger(__name__)


async def run_rest_pass(agent) -> str:
    """内省を 1 パス回して、何をしたかを返す（同じ文を `内省` の記録にも書く）。

    層 1 の畳み込みが落ちても例外を外へ出さない——材料は残り、次の晩に持ち越す。
    """
    parts: list[str] = []
    records: tuple = ()
    try:
        r = await fold_since_last_rest(agent)
        records = r.records
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
    # 層 2：層 1 が書いたものを材料に、自己像を見直す（記-a-へ）。
    try:
        p = await update_self_image(agent, [Material(w.obs_id, w.kind, w.text) for w in records])
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
