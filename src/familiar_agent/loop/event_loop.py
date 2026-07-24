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
        self._completion_queue: asyncio.Queue[str] = asyncio.Queue()

    def _tools(self) -> list[dict]:
        """段階1スライス2で渡すツール＝say（発話）＋recall（内部・外部I/Oなし）のみ。"""
        agent = self._agent
        say = agent._tts.get_tool_definitions() if agent._tts else []
        recall = [
            d for d in agent._memory_tool.get_tool_definitions() if d.get("name") == "recall"
        ]
        return say + recall

    async def run_iteration(self, utterance: str, on_text=None) -> str:
        """LPM：QC を drain して回す反復ループ。1反復＝1出力（say で終了）。

        `on_text` は生成のストリーミング出力先（CUI/GUI へ逐次表示）。
        """
        from ..capability_state import load_summary

        agent = self._agent
        consumed_completion_ids: list[str] = []
        memories: list[dict] = []
        text = ""

        for _ in range(max(1, agent.config.event_max_iterations)):
            # 1. 取込：QC を drain し、完了結果を O に書く（consumed を控え末尾で supersede）。
            while not self._completion_queue.empty():
                result_text = self._completion_queue.get_nowait()
                obs_id, _ = await agent._memory.save_async_with_id(
                    result_text[:500],
                    direction="完了",
                    kind="observation",
                    materialize_now=True,
                    **agent._observation_perspective(),
                )
                if obs_id:
                    consumed_completion_ids.append(obs_id)

            # 2. REC（想起）：O（＋現入力）→ W。完了 O があれば関連で W に上がる。
            mem = agent._active_memory()
            memories = await mem.recall_async(utterance, recall_mode="conversation")
            workspace_ctx = mem.format_for_context(memories)

            system = build_event_system_prompt(
                me_md=getattr(agent, "_me_md", ""),
                family_md=getattr(agent, "_family_md", ""),
                capabilities=load_summary(),
                present_ctx=_present_ctx(agent),
                pi_ctx=_pi_ctx(),
                workspace_ctx=workspace_ctx,
            )

            # 3. GEN（生成）：say＋recall のみ渡す。1回の stream_turn で多段はしない。
            user_msg = agent.backend.make_user_message(utterance)
            result, _raw = await agent.backend.stream_turn(
                system=system,
                messages=[user_msg],
                tools=self._tools(),
                max_tokens=agent.config.max_tokens,
                on_text=on_text,
            )

            # say → 発話して終了（run() と同じ「先頭 say 採用」）。
            say_tc = next((tc for tc in result.tool_calls if tc.name == "say"), None)
            if say_tc is not None:
                text = str(say_tc.input.get("text", "")).strip()
                if text and agent._tts is not None:
                    with contextlib.suppress(Exception):
                        await agent._tts.call("say", {"text": text})
                # say tool_call の text はストリームされないので表示のため明示的に流す。
                if text and on_text is not None:
                    on_text(text)
                break

            # recall → RH が実行し結果を QC へ積んで次反復へ連鎖（O→W 経由で再会）。
            recall_tc = next((tc for tc in result.tool_calls if tc.name == "recall"), None)
            if recall_tc is not None:
                out, _ = await agent._memory_tool.call("recall", dict(recall_tc.input))
                self._completion_queue.put_nowait(out)
                continue

            # どちらも無ければ result.text へフォールバック（既にストリーム済み）して終了。
            text = (result.text or "").strip()
            break

        # 永続化＝既存 pipeline（utility LLM のみ）。消化した完了 O をターン観察で supersede。
        try:
            arousal = await agent._turn_arousal(utterance, text)
            agent._spawn_background_task(
                agent._run_post_response_pipeline(
                    user_input=utterance, final_text=text,
                    camera_used=False, camera_image=None,
                    observation_action_name=None, observation_action_input=None,
                    companion_mood="engaged", is_desire_turn=False, desires=None,
                    arousal=arousal, memories=memories,
                    superseded_ids=consumed_completion_ids or None,
                ),
                name="event-post-response",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("event-loop persistence spawn failed: %s", e)

        return text
