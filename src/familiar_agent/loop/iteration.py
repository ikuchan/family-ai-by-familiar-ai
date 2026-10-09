"""1 反復（環-ab E・2026-09-29）。取込 → 想起 → 調停 → 出力（発話 or ツール投げ）で終わる。

`InformationProcessing._iterate` の本体をメソッドオブジェクトにした。局所変数を属性にし、段ごとの
メソッドに分ける（`run` が上から順に呼ぶ）。道具（調停・発話・記録）は `InformationProcessing` のものを
呼ぶので、試験が差し替えた道具はそのまま効く。挙動は変えていない。
"""

from __future__ import annotations

import asyncio
import logging

from ..config import MemoryConfig
from ..person_memory_manager import AGENT_SELF_ID
from . import reply_budget, workspace
from .generator import _companion_ctx, _iter_ctx, _present_ctx

logger = logging.getLogger(
    "familiar_agent.loop.event_loop"
)  # ログの名前はループのまま（読み手の検索を変えない）


def _log_recall_weights(trigger, base, used, memories) -> None:
    """採用した5軸重みと、その重みで出た上位のスコアを残す（INFO）。記憶の内容は出さない。"""

    def _fmt(w):
        return "(%.2f,%.2f,%.2f,%.2f,%.2f)" % (w.w_r, w.w_t, w.w_e, w.w_g, w.w_p)

    top = "/".join("%.3f" % r.fit for r in memories[:3])
    logger.info(
        "event-loop 想起 trigger=%s w=%s 基底=%s 上位=%s %d件",
        trigger, _fmt(used), _fmt(base), top or "なし", len(memories),
    )  # fmt: skip


def closes_silently(decision, *, trigger_kind: str, returned: "frozenset[str]") -> bool:
    """調停が黙ると決めた反復を、主LLM を呼ばずに沈黙で閉じるか（出-w・出-ay 段 4-2・4-4b）。

    light で文が無いとき。出-ay 段 4-4b から人の言葉への最初の反復でも「黙る」を選ぶ（成立しないもの・文脈に合わない
    言葉）。書けなかったときは調停が full へ倒すので、light で文が無いのは黙ると決めたときだけになった。
    """
    del trigger_kind, returned  # 起点を問わない（呼び手の形は残す）
    return decision.branch == "light" and not decision.text


class Iteration:
    """1 反復の流れと、段どうしで渡す状態。`ip` は `InformationProcessing`。"""

    def __init__(self, ip) -> None:
        self.ip = ip
        self.agent = ip._agent
        self.utterance = ip._req.utterance
        # この反復が属する世代。打ち切られたら（世代が進んだら）、フルLLM の返りを待って
        # いる最中でも、出力せずに畳む。実機で、打ち切った直後に走っていた反復が
        # fetch_deferred を投げ、返事も1つ余計に出た。
        self.gen = ip._request_generation
        self.max_chain = max(1, self.agent.config.event_max_iterations)

    async def run(self) -> str:
        decided = await self._intake()
        if decided is not None:
            # **出す反復。** 想起も調停も回さない——回すと軽量LLM が主LLM の決定を覆せて
            # しまい、「そのまま出す」と矛盾する（`設計方針_主LLMを投げっぱなしにする`）。
            return await self.ip._act_on_decision(decided, utterance=self.utterance, gen=self.gen)
        self._count()
        await self._recall()
        if (
            self.ip._slow_notice_received
        ):  # 待ちの一言だけ言って閉じない（分岐は決めない・出-aq 段 6）
            return await self.ip._say_waiting_filler(
                self.utterance, self.ws.for_arbiter, self.chain
            )
        self._cap()
        await self._decide()
        closed = await self._close_early()
        return closed if closed is not None else await self._throw_main_llm()

    async def _intake(self):
        """1. 取込：駆動体が受けた完了を O に書き、open 意図を解決する。"""
        ip = self.ip
        self.drained, decided = await ip._intake()
        if decided is not None:
            # **出す反復は数えない。** 数えるのは軽量LLM が司る反復だけである。さらに
            # **主LLM の返りで 0 へ戻す**——主LLM が調査結果を見て「足りない」と判断した
            # なら、それは新しい一巡である。上限は暴走防止の安全弁であって、材料を見た
            # うえで再度調べることを止めるためのものではない
            # （`設計方針_主LLMを投げっぱなしにする` ⑤）。
            ip._req.iterations = 0
            ip._req.iterations_capped = False
        return decided

    def _count(self) -> None:
        # ここから先は**決める反復**である。数えるのはここだけ。「まだかかっている」で起きた反復
        # （つなぎだけ出す）は数えない（出-au 段 2）。20 秒ごとに繰り返すと、数えれば上限
        # （`event_max_iterations`）に届き、答えを考える前に打ち切りになる。
        ip = self.ip
        if not ip._slow_notice_received:
            ip._req.iterations += 1
        self.chain = ip._req.iterations
        if self.drained:
            logger.debug(
                "event-loop iter=%d/%d QC取込=%d件", self.chain, self.max_chain, self.drained
            )

    async def _recall(self) -> None:
        """2. REC（想起）：O（＋現入力）→ W。W は派生なので反復末に捨てる。

        手がかりは「取り込んだもの」＝鎖の先頭（反復1なら人の発話、反復2以降なら完了 O）。最初の発話で
        探し続けると、いま届いた完了とは無関係な検索になる（④ の想起クエリ）。**取込 O を候補から外さない**：
        いま届いた結果を全文で見せる必要があるので、1位に来るのが正しい順位である（[D-想起起動] の1本の流れ）。
        """
        ip, agent = self.ip, self.agent
        # **誰の面から引くか。** 話者が居なければパジュ自身。想起は口を通すので、面は `View.viewpoint`
        # で言う（環-e-い）。申告も同じ面へ当てる。`situated_memories` は人ごとなので、ずれると 0 行に
        # 当たる（出-h-ろ ③）。以前は `_active_memory()` の選び方で、話者の指定が切れた直後にずれていた（環-ab）。
        self.viewpoint = agent._pmm.current_speaker_id or AGENT_SELF_ID
        self.verdict_view = self.viewpoint
        self.cue = ip._req.cue or self.utterance
        _mcfg = MemoryConfig()
        # 5軸の重みは trigger 種別で決める（`課題5_パラメータ仮案` §280）。基準は「この反復を何を手がかりに
        # 動くか」。`ip._req.trigger_kind` を書き換えないのは、出口の門（窓と在席・`_speak`・
        # `_delivery_block_reason`）が使っており、人に話しかけられて始まった求めを止めてしまうためである。
        trigger = "完了" if self.drained else ip._req.trigger_kind
        w_base = _mcfg.recall_weights(trigger)
        self.weights = _mcfg.jitter_weights(w_base)
        self.ws = await workspace.recall(
            agent._oif, self.cue, viewpoint=self.viewpoint, weights=self.weights, req=ip._req
        )
        _log_recall_weights(trigger, w_base, self.weights, self.ws.memories)
        ip._returned_now = self.ws.returned_actions  # 声の選び方が読む（環-u）
        ip._req.just_returned.clear()  # 「いま道具から返った」はこの反復の W にだけ載せる
        if ip._slow_notice_received:
            return  # 待ちの一言だけの反復は、続き先を判定しない（出-aq 段 6）
        # 続き先の判定を投げる。**待たずに先へ進む。** 調停と並行して走らせれば、実測 0.72 秒
        # （`根拠台帳` §29）はほぼ隠れる。受け取るのはシステム文を組む直前で、そこは待つ。
        self.follows_task = asyncio.ensure_future(
            ip._judge_follows(self.ws.for_main, self.utterance or "")
        )

    def _cap(self) -> None:
        ip = self.ip
        # 誰と話していると思って喋ったかを残す（口調がおかしいとき、話者が渡っていないのか従って
        # いないのかを切り分ける）。**この求めで何回目に考えるか**は1度だけ数え、ログ・調停・主LLM へ
        # 同じ値を渡す（別々に数えると食い違う）。
        self.present_ctx = _present_ctx(self.agent)
        self.round_ = ip._thinking_round
        logger.debug(
            "event-loop iter=%d/%d 考え=%d回目 顔ぶれ=%s",
            self.chain, self.max_chain, self.round_, self.present_ctx,
        )  # fmt: skip
        # **上限は2つ。** 反復（1回の一巡の長さ）と、考えた回数（求め全体で主LLM を呼んだ数）。
        # 反復は主LLM の返りで 0 へ戻るので、輪が閉じたときは考えた回数だけが効く。
        self.capped = self.chain >= self.max_chain or ip._thinking_capped
        if self.capped:
            # 上限で打ち切ったことは、後からログだけで判別できる必要がある。
            what, now, cap = (
                ("考えた回数", self.round_, self.agent.config.max_thinking_rounds)
                if ip._thinking_capped
                else ("反復", self.chain, self.max_chain)
            )
            logger.info("event-loop %s %d/%d 上限に達したため探索を打ち切る", what, now, cap)
            ip._req.iterations_capped = True

    async def _decide(self) -> None:
        ip, said = self.ip, self.utterance or self.ip._req.cue
        self.decision = await ip._decide(
            utterance=said,
            workspace_ctx=self.ws.for_arbiter,  # 同じ W・狭い窓（記-h）
            present_ctx=self.present_ctx,
            capped=self.capped,
            round_=self.round_,
            memories=self.ws.memories,
            returned=self.ws.returned_actions,
            returned_lookups=self.ws.returned_lookups,
        )
        # 「いまは話しかけないで」と読めたら、その人が居るあいだ黙る（次の反復から）。解くのも同じ口。
        await ip._apply_requests(self.decision, utterance=said)
        # 調停が時期を指した（「去年の夏の話」）なら、その基準で想起し直して W を組み直す。
        self.ws = await ip._recall_at(
            self.decision, self.ws, cue=self.cue, viewpoint=self.viewpoint, weights=self.weights
        )

    async def _close_early(self) -> "str | None":
        """主LLM を起こさずに閉じる道。閉じたら出したものを、続けるなら None を返す。"""
        ip, decision, memories = self.ip, self.decision, self.ws.memories
        if self.gen != ip._request_generation:
            logger.info("event-loop 打ち切られた求めの反復なので畳む（調停後）")
            return ""
        # (a') 調停が「黙る」（light・text 空）と決めた：沈黙で閉じる。情動の求め（出-w）と、結果が届いた反復
        # （出-ay 段 4-2：音楽をかけた・次の曲などは黙る）。
        if closes_silently(
            decision, trigger_kind=ip._req.trigger_kind, returned=self.ws.returned_actions
        ):
            logger.info("event-loop 調停が黙ると決めたので沈黙で閉じる（%s）", ip._req.trigger_kind)
            await ip._finish("", memories, "沈黙", gen=self.gen)
            return ""
        # (a) 軽量で閉じる。**記憶が育つ経路は申告1本しかない。** 主LLM を起こさない反復もそこを通す
        # （出-h-ろ）。聞くのは背景で、閉じるのは待たない。
        if decision.branch == "light" and decision.text:
            spoken, outcome = await ip._speak(decision.text, branch="light")
            ip._declare_light_memory_use(
                utterance=self.utterance or ip._req.cue,
                reply=spoken or decision.text,
                workspace_ctx=self.ws.for_main,
                w_id_map=self.ws.verdict_map,
                verdict_view=self.verdict_view,
                memories=memories,
                spoken=outcome == "発話",
            )
            await ip._finish(spoken, memories, outcome, gen=self.gen)
            return spoken
        # (c) 定型：探すと決まっている反復も、フルLLM を起こさず投げて閉じる。
        if decision.branch == "action" and decision.query and not self.capped:
            await ip._dispatch_arbiter_action(decision, utterance=self.utterance or ip._req.cue)
            logger.info(
                "event-loop 反復 %d/%d 出力=%s（調停・続きは完了で起きる）",
                self.chain, self.max_chain, decision.action,
            )  # fmt: skip
            return ""
        return None

    async def _throw_main_llm(self) -> str:
        """(b) 主LLM を投げる。前につなぎは挟まない——待たせたら待ちの知らせが言う（出-aq 段 7）。"""
        ip, agent, decision, ws = self.ip, self.agent, self.decision, self.ws
        memories, workspace_ctx, w_id_map = ws.memories, ws.for_main, ws.id_map
        mood = asyncio.ensure_future(
            ip._judge_companion(self.utterance or "")
        )  # 相手の気分（出-av）
        # 判定は辺を書くだけで W は変えない（記-h）。結末は計測ログへ（記-i）。
        await ip._note_follows(workspace_ctx, self.utterance or "", w_id_map, self.follows_task)
        companion = _companion_ctx(await mood, ip._current_speaker_name())
        # 返事の予算（出-k-ろ）：長さは数字で渡し、`max_tokens` はそこから固定する。
        budget = reply_budget.decide(
            effort=decision.effort,
            researched=ip._researched(),
            w_count=len(memories),
            origin=ip._req.trigger_kind,
            talking=ip._talking(),
        )
        system = ip._build_system(
            present_ctx="\n".join(p for p in (self.present_ctx, companion) if p),
            workspace_ctx=workspace_ctx,
            iter_ctx=_iter_ctx(
                chain=self.chain,
                max_chain=self.max_chain,
                thinking_round=self.round_,
                capped=self.capped,
                budget=budget,
                missing=ip._missing_tools(),
                tone=ip._tone_note(),
            ),
            music=await ip._music_now(),
        )
        # 起点が人の発話ならそのまま、情動・機器なら内的な出来事として渡す（空文字は API が受けない）。
        user_msg = agent.backend.make_user_message(
            ip._user_content(self.utterance or ip._req.cue, memories)
        )
        # **投げて終わる。** 返りは QC を通り、次の反復（出す反復）が実行する。
        ip._dispatch_main_llm(
            messages=[user_msg],
            system=system,
            effort=decision.effort,
            capped=self.capped,
            memories=memories,
            w_id_map=dict(ws.verdict_map),  # 申告の母数は過去の列だけ（出-n 4）
            verdict_view=self.verdict_view,
            recent_frame=ws.recent_text(ws.n_main),
            max_tokens=budget.max_tokens,
        )
        await ip._write_version()
        logger.info(
            "event-loop 反復 %d/%d 考え=%d回目 出力=主LLM（続きは返りで起きる）",
            self.chain, self.max_chain, self.round_,
        )  # fmt: skip
        return ""
