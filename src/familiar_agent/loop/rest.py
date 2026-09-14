"""REST 内省の 1 パス（記-a）。

目的は自己認識の変革で、仕事は 4 層を順に更新すること（`用語一覧` v0.70）——
出来事（畳む）→ 自己像（抽象化する）→ 設定値（調整する）→ 能力（再定義する）。
起動は T の純粋欠乏発火で、誰も居ないときだけ回る（`loop/tonic.py`）。

**いま動いているのは層 1 の①（日次の畳み込み・`rest_fold.py`）だけ**。層 1 の②（核の固め）と
$n$ の減り、層 2〜4 はこれから（`課題8` 記-a）。回ったことは `direction='内省'` の記録に残す——
ログだけだと、起動しなかったのか、起動したが何もしなかったのかを区別できない。
"""

from __future__ import annotations

import logging

from .rest_fold import fold_since_last_rest

logger = logging.getLogger(__name__)


async def run_rest_pass(agent) -> str:
    """内省を 1 パス回して、何をしたかを返す（同じ文を `内省` の記録にも書く）。

    層 1 の畳み込みが落ちても例外を外へ出さない——材料は残り、次の晩に持ち越す。
    """
    try:
        r = await fold_since_last_rest(agent)
        if r.materials == 0:
            content = "内省を回した（畳むものが無かった）"
        else:
            content = (
                f"内省を回した（出来事 {r.materials} 件を畳み、自己エピソードと関係のまとめを "
                f"{r.written} 件書いた。見送り {r.skipped} 回）"
            )
    except Exception as e:  # noqa: BLE001
        logger.exception("rest 層 1 の畳み込みに失敗（次の晩に持ち越す）: %s", e)
        content = "内省を回した（出来事を畳めなかった・次の晩に持ち越す）"
    logger.info("rest 内省パス：%s", content)
    await agent._memory.save_async_with_id(
        content[:500],
        direction="内省",
        kind="observation",
        materialize_now=True,
        **agent._observation_perspective(),
    )
    return content
