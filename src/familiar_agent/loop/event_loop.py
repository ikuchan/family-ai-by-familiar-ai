"""イベント駆動ループ（#11 段階1）：I（情報処理機構）の最小縦切り。

設計正本＝`設計図_Mermaid` ③ I 詳細図。ここでは I の中の **LPM（ループ核）** と
**QC（完了キュー）** だけを実体化する。反復は QC を drain（取込→O 書込）→ REC（想起→W）→
GEN（生成）で進み、say で1出力して終わる／内部ツール（recall）は結果を QC へ積んで次反復へ
連鎖する（[D-単一想起]：相関ID を使わず結果は O→W 経由で再会）。

現行 run() と排他（`EVENT_LOOP` on の user turn のみ）。AIF/DIF/QA/QD、ARB/APR/ACT/MNT の
クラス分離は後続段階（ここでは stub しない）。永続化は既存 `_run_post_response_pipeline`
（utility LLM のみ）を流用し、消化した完了 O はターン観察 id で supersede する。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from .arbiter import arbitrate
from .prompt import build_event_system_prompt

logger = logging.getLogger(__name__)


def _present_ctx(agent) -> str:
    from ..core.helpers import format_present_ctx

    pmm = getattr(agent, "_pmm", None)
    if pmm is None:
        return ""
    try:
        rows = pmm.presence_status()
    except Exception:  # noqa: BLE001
        return ""
    if not rows:
        return ""
    speaker = next((r["name"] for r in rows if r.get("is_speaker")), "")
    others = [r["name"] for r in rows if not r.get("is_speaker")]
    return format_present_ctx(speaker, others)


def _pi_ctx() -> str:
    """mood/drive を PI として定性注入する（生値は出さない）。DB 失敗は空で degrade。"""
    try:
        from ..config import DriveConfig
        from ..core.drive_autonomy import drive_snapshot
        from ..drive_register import load_current_drives
        from ..emotion_pad import label_from_pad
        from ..mood_register import load_current_mood

        mood = load_current_mood()
        drives = load_current_drives()
        return f"[内部状態(PI)] 気分: {label_from_pad(mood)} / 欲求: {drive_snapshot(drives, DriveConfig())}"
    except Exception as e:  # noqa: BLE001
        logger.debug("PI ctx unavailable: %s", e)
        return ""


class InformationProcessing:
    """I：情報処理機構（Information-processing）。③ I 詳細図の器。

    段階1で実体化するのは **QC（完了キュー）** と **LPM（ループ核）＝`run_iteration`** のみ。
    O・C（Config）・W・RH 相当のツール実行は既存実体を持つ `agent` を当面参照する。
    """

    def __init__(self, agent):
        self._agent = agent
        # QC：完了キュー（Completion Queue）。RH（資源ハンドラ）が書き、LPM が drain する。
        # 要素＝(何を探したか, 結果, 起点の open 意図 id)。意図 id は完了が再会して解決するのに使う。
        self._completion_queue: asyncio.Queue[tuple[str, str, str | None]] = asyncio.Queue()
        # ループ記録は1本の鎖にする：トリガO → 意図O → 完了O → 意図O2 → …。新しい記録を
        # 書くたび直前の生きた記録を supersede するので、生き残るのは常に鎖の先頭1件だけ。
        # これで前の記録が想起に出てこなくなり、除外は「その検索を出した意図自身」で足りる。
        self._chain_head_id: str | None = None
        self._chain_head_content: str = ""
        # RH（実行担当）が走らせている投げっぱなしの呼び出し。QC が空でもこれが残っていれば
        # 結果が届くまで待つ（イベント駆動＝キュー到来で起きる）。
        self._inflight = 0
        self._tasks: set[asyncio.Task] = set()
        # 駆動体（QC 到来で次の反復を起こす）と、そこへ渡す取込待ちの完了。
        self._driver: asyncio.Task | None = None
        self._inbox: list[tuple[str, str, str | None]] = []
        # 発話が出るまでの連鎖長（発話でリセット）。上限に達した反復は recall を渡さない。
        self._chain = 0
        self._utterance = ""
        self._on_text = None
        self._pending_intent: tuple[str, dict] = ("", {})


    def _advance_chain(self, new_id: str | None, content: str = "") -> None:
        """ループ記録の鎖を1つ進める（直前の生きた記録を新しい記録で supersede）。

        内容も持つのは、この先頭（取込の起点）を W へ決定的に加えるため。
        """
        if not new_id:
            return
        if self._chain_head_id and self._chain_head_id != new_id:
            self._agent._memory.mark_superseded(self._chain_head_id, new_id)
            logger.debug("event-loop 鎖を進める %.8s → %.8s", self._chain_head_id, new_id)
        self._chain_head_id = new_id
        self._chain_head_content = content

    def _dispatch_recall(self, tool_input: dict, query: str, intent_id: str | None) -> None:
        """RH：recall を非同期に実行し、結果を QC へ積む（投げっぱなし・待たない）。"""
        self._inflight += 1
        task = asyncio.create_task(self._run_recall(tool_input, query, intent_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_recall(self, tool_input: dict, query: str, intent_id: str | None) -> None:
        try:
            out, _ = await self._agent._memory_tool.call(
                "recall", tool_input, exclude_ids=[intent_id] if intent_id else None
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("event-loop recall の実行に失敗: %s", e)
            out = f"（recall を実行できなかった：{e}）"
        self._completion_queue.put_nowait((query, str(out), intent_id))
        logger.debug(
            "event-loop RH 完了をQCへ（id=%s qsize=%d 意図=%.8s）",
            id(self), self._completion_queue.qsize(), intent_id or "-",
        )

    async def _intake(self) -> int:
        """取込：駆動体が受けた完了（と QC の残り）を O に書き、open 意図を解決する。"""
        agent = self._agent
        # `_inbox` は作り直さず中身だけ移す。駆動体は `self._inbox.append(await get())` の
        # append を await の前に束縛するので、ここで差し替えると駆動体が捨てられた古い
        # リストへ積み、完了が黙って失われる（実機で観測）。
        items = list(self._inbox)
        self._inbox.clear()
        while not self._completion_queue.empty():
            items.append(self._completion_queue.get_nowait())
        logger.debug(
            "event-loop 取込（id=%s items=%d inflight=%d qsize=%d）",
            id(self), len(items), self._inflight, self._completion_queue.qsize(),
        )

        for query, result_text, intent_id in items:
            self._inflight = max(0, self._inflight - 1)
            # 探した事実と結果を1件に残す。open 意図と入れ替わるので W には結果つきが載る。
            obs_id, _ = await agent._memory.save_async_with_id(
                f"「{query}」を探した結果：{result_text}"[:500],
                direction="完了",
                kind="observation",
                materialize_now=True,
                **agent._observation_perspective(),
            )
            # 完了が open 意図に再会して解決（[D-単一想起]）＝鎖を1つ進める。
            self._advance_chain(obs_id, f"「{query}」を探した結果：{result_text}"[:500])
        return len(items)

    def _tools(self, *, with_recall: bool = True) -> list[dict]:
        """渡すツール＝say（発話）＋recall。連鎖上限の反復では recall を外す。"""
        agent = self._agent
        say = agent._tts.get_tool_definitions() if agent._tts else []
        if not with_recall:
            return say
        recall = [
            d for d in agent._memory_tool.get_tool_definitions() if d.get("name") == "recall"
        ]
        return say + recall

    async def run_iteration(self, utterance: str, on_text=None) -> str:
        """人の発話で1反復を起こす。1反復＝1出力（発話 or ツール投げ）で終わる。

        ツールを投げた反復は発話を持たないので空文字を返す。続きは、完了が QC に届いて
        駆動体が起こす次の反復が担う。`on_text` は出力先（駆動体が起こす反復も使う）。
        """
        agent = self._agent
        self._on_text = on_text or self._on_text
        self._utterance = utterance
        self._chain = 0
        self._ensure_driver()

        # 取込：来た事実（人の発話）を O に書く（④シーケンス）。
        trigger_id, _ = await agent._memory.save_async_with_id(
            utterance[:500],
            direction="発話",
            kind="observation",
            materialize_now=True,
            **agent._observation_perspective(),
        )
        self._advance_chain(trigger_id, utterance[:500])
        return await self._iterate()

    def _ensure_driver(self) -> None:
        """駆動体：QC 到来で次の反復を起こす（イベント駆動・時計は見ない）。"""
        if self._driver is None or self._driver.done():
            self._driver = asyncio.create_task(self._drive())

    async def _drive(self) -> None:
        while True:
            try:
                # 属性の参照は await の後に行う（束縛のすり替わりを避ける・上の注記）。
                item = await self._completion_queue.get()
                self._inbox.append(item)
                while not self._completion_queue.empty():
                    self._inbox.append(self._completion_queue.get_nowait())
                logger.debug(
                    "event-loop 駆動体が完了を受領（id=%s inbox=%d）", id(self), len(self._inbox)
                )
                await self._iterate()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("event-loop 駆動体で例外: %s", e)

    async def close(self) -> None:
        if self._driver is not None:
            self._driver.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._driver
            self._driver = None

    async def _iterate(self) -> str:
        """1反復：取込 → W 構築 → 生成 → 出力（発話 or ツール投げ）で終わる。"""
        from ..capability_state import load_summary
        from ..config import MemoryConfig

        agent = self._agent
        utterance = self._utterance
        max_chain = max(1, agent.config.event_max_iterations)
        self._chain += 1
        chain = self._chain
        logger.debug("event-loop iter=%d/%d 開始", chain, max_chain)

        # 1. 取込：駆動体が受けた完了を O に書き、open 意図を解決する。
        drained = await self._intake()
        if drained:
            logger.debug("event-loop iter=%d/%d QC取込=%d件", chain, max_chain, drained)

        # 2. REC（想起）：O（＋現入力）→ W。W は派生なので反復末に捨てる。
        # 一律の規則：取込で書いた記録（＝鎖の先頭）は検索から外し、W へは決定的に加える。
        # 素通しだと問いと同一文の記録が必ず上位に来て、限られた枠から本物の記憶を押し出す。
        # 手がかりは「取り込んだもの」＝鎖の先頭（反復1なら人の発話、反復2以降なら完了 O）。
        # 最初の発話で探し続けると、いま届いた完了とは無関係な検索になる（④ の想起クエリ）。
        mem = agent._active_memory()
        origin_ids = [self._chain_head_id] if self._chain_head_id else None
        cue = self._chain_head_content or utterance
        memories = await mem.recall_async(
            cue,
            n=MemoryConfig().recall_n,
            recall_mode="conversation",
            exclude_ids=origin_ids,
        )
        origin_ctx = f"[取込] {self._chain_head_content}" if self._chain_head_content else ""
        workspace_ctx = "\n\n".join(
            p for p in [origin_ctx, mem.format_for_context(memories)] if p and p.strip()
        )

        # 3. ARB（調停）：軽量LLM が会話の重さを自己判断し、出し方を3つへ振り分ける（段4）。
        #    フルLLM は「言語生成が要るとき」だけ起こす。実測で1ターン 10.5 秒のうち LLM が
        #    10.2 秒を占め、recall を投げるだけの反復にもフルを使っていた。
        capped = chain >= max_chain
        decision = await arbitrate(
            agent._utility_backend, utterance=utterance, workspace_ctx=workspace_ctx,
        )
        logger.debug("event-loop iter=%d/%d 調停=%s effort=%s",
                     chain, max_chain, decision.branch, decision.effort)

        # (a) 軽量で閉じる：フルLLM を起こさず、軽量LLM の応答で反復を終える。
        if decision.branch == "light" and decision.text:
            text = decision.text
            if agent._tts is not None:
                with contextlib.suppress(Exception):
                    await agent._tts.call("say", {"text": text})
            if self._on_text is not None:
                self._on_text(text)
            await self._finish(text, memories, "発話")
            return text

        # (c) 定型：探すと決まっている反復も、フルLLM を起こさず投げて閉じる。
        if decision.branch == "action" and decision.query and not capped:
            self._open_intent(utterance, {"query": decision.query})
            logger.info("event-loop 反復 %d/%d 出力=ツール投げ（調停・続きは完了で起きる）",
                        chain, max_chain)
            return ""
        system = build_event_system_prompt(
            me_md=getattr(agent, "_me_md", ""),
            family_md=getattr(agent, "_family_md", ""),
            capabilities=load_summary(),
            present_ctx=_present_ctx(agent),
            pi_ctx=_pi_ctx(),
            iter_ctx=(
                f"[反復] {chain}/{max_chain}"
                + ("（これ以上は探せない。いまある材料で答える）" if capped else "")
            ),
            workspace_ctx=workspace_ctx,
        )
        # 生成中はストリームしない：ツールを選ぶ反復で出る前置きの地の文が表示され重複するため。
        user_msg = agent.backend.make_user_message(utterance)
        result, _raw = await agent.backend.stream_turn(
            system=system,
            messages=[user_msg],
            tools=self._tools(with_recall=not capped),
            max_tokens=agent.config.max_tokens,
            on_text=None,
            effort=decision.effort,
        )

        # 出力その1＝発話（run() と同じ「先頭 say 採用」）。ここで反復は終わる。
        say_tc = next((tc for tc in result.tool_calls if tc.name == "say"), None)
        if say_tc is not None:
            logger.debug("event-loop iter=%d/%d 決定=say", chain, max_chain)
            text = str(say_tc.input.get("text", "")).strip()
            if text and agent._tts is not None:
                with contextlib.suppress(Exception):
                    await agent._tts.call("say", {"text": text})
            if text and self._on_text is not None:
                self._on_text(text)
            await self._finish(text, memories, "発話")
            return text

        # 出力その2＝ツール投げ。投げた時点でこの反復は終わり、続きは完了が起こす次の反復。
        # 上限の反復では recall を渡していないので、返ってきても投げない（連鎖を必ず閉じる）。
        recall_tc = next((tc for tc in result.tool_calls if tc.name == "recall"), None)
        if recall_tc is not None and not capped:
            logger.debug("event-loop iter=%d/%d 決定=recall", chain, max_chain)
            self._open_intent(utterance, dict(recall_tc.input))
            logger.info("event-loop 反復 %d/%d 出力=ツール投げ（続きは完了で起きる）",
                        chain, max_chain)
            return ""

        # どちらも無ければ素テキストへフォールバック（表示はここで1回）。
        logger.debug("event-loop iter=%d/%d 決定=none", chain, max_chain)
        text = (result.text or "").strip()
        if text and self._on_text is not None:
            self._on_text(text)
        await self._finish(text, memories, "沈黙")
        return text

    def _open_intent(self, utterance: str, tool_input: dict) -> None:
        """open 意図を O に残し、RH へ投げる（待たない）。意図は常に高々1件に保つ。"""
        self._pending_intent = (utterance, tool_input)
        self._tasks.add(t := asyncio.create_task(self._write_intent_and_dispatch()))
        t.add_done_callback(self._tasks.discard)

    async def _write_intent_and_dispatch(self) -> None:
        agent = self._agent
        utterance, tool_input = self._pending_intent
        query = str(tool_input.get("query", "")).strip()
        content = f"「{utterance}」について recall（query={query}）を要求した。結果はまだ無い。"[:500]
        intent_id, _ = await agent._memory.save_async_with_id(
            content,
            direction="意図",
            kind="observation",
            materialize_now=True,
            **agent._observation_perspective(),
        )
        # 意図を書いた時点でトリガ（や前回の完了）は死ぬ＝この検索には出てこない。
        self._advance_chain(intent_id, content)
        # 自分が出した検索が自分自身を拾わないよう、意図 O の id だけ狭く除外する。
        self._dispatch_recall(tool_input, query, intent_id)

    async def _finish(self, text: str, memories: list[dict], outcome: str) -> None:
        """発話で連鎖が閉じた反復の後始末：総括ログと永続化（ループ中 O を supersede）。"""
        agent = self._agent
        logger.info("event-loop 終了: 反復=%d 結末=%s text_len=%d",
                    self._chain, outcome, len(text))
        obs_ids = [self._chain_head_id] if self._chain_head_id else []
        self._chain_head_id = None
        self._chain = 0
        try:
            arousal = await agent._turn_arousal(self._utterance, text)
            agent._spawn_background_task(
                agent._run_post_response_pipeline(
                    user_input=self._utterance, final_text=text,
                    camera_used=False, camera_image=None,
                    observation_action_name=None, observation_action_input=None,
                    companion_mood="engaged", is_desire_turn=False, desires=None,
                    arousal=arousal, memories=memories,
                    superseded_ids=obs_ids or None,
                ),
                name="event-post-response",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("event-loop persistence spawn failed: %s", e)
