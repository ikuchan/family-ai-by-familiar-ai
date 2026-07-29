"""REST 内省の1パス（#2・順1＝骨格）。

設計の正本は `直近の進め方と進捗` v0.14（折衷型・設計図 Phase 5）。起動は T の純粋欠乏
発火で、誰も居ないときだけ回る。1パスは次の順で進める。

    読み込み → 蒸留（自己エピソード・per-person 関係サマリ）
             → open 棚卸し（孤児は Warn・消さない）
             → Config 自己調整（範囲内・人の設定は変えない）

圧縮系（near-dup 統合・situated の relation_key 語彙の増減）は同じパスの中で量ベースに
行い、平均ベクトルの再推定はさらに低い頻度で回す。すべて版履歴で可逆にし、機械が決める
こと（距離・冗長度・Warn）と LLM が決めること（蒸留・棚卸し・命名・値の提案）を分ける。

**いまあるのは骨格だけで、上の仕事はどれも実装していない。** 先に起動条件が正しく掛かる
ことを確かめるためである。この機構はこれまで一度も動いたことがなく、中身を作り込んでから
起動の誤りが見つかると切り分けが難しい。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def run_rest_pass(agent) -> str:
    """内省を1パス回して、何をしたかを返す。

    記録を O に残すのは、回ったこと自体を後から確かめるためである。ログだけだと、
    起動しなかったのか、起動したが何もしなかったのかを区別できない。
    """
    content = "内省を回した（骨格のみ・蒸留と棚卸しは未実装）"
    logger.info("rest 内省パス：%s", content)
    await agent._memory.save_async_with_id(
        content[:500],
        direction="内省",
        kind="observation",
        materialize_now=True,
        **agent._observation_perspective(),
    )
    return content
