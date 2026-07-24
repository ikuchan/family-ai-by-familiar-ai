"""イベント駆動ループ（#11 段階1）：人の発言→拡散込み想起→1反復1出力（発話）。

現行 run() と排他（`EVENT_LOOP` on の user turn のみ）。段階1は発話のみ（ツールを渡さず
1出力を保証）。永続化は既存 `_run_post_response_pipeline`（utility LLM のみ）を流用する。
"""

from __future__ import annotations

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


async def run_iteration(agent, utterance: str) -> str:
    """1反復＝1出力：想起（拡散込み）で W を作り、フルLLM で1発話を生成して返す。"""
    from ..capability_state import load_summary

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

    # 生成：ツールを渡さない＝多段 ReAct を構造的に禁止し1出力を保証（発話のみ）。
    user_msg = agent.backend.make_user_message(utterance)
    result, _raw = await agent.backend.stream_turn(system=system, messages=[user_msg], tools=[])
    text = (result.text or "").strip()

    # 永続化＝既存 pipeline（utility LLM のみ）を応答クリティカルパス外で回す。
    try:
        arousal = await agent._turn_arousal(utterance, text)
        agent._spawn_background_task(
            agent._run_post_response_pipeline(
                user_input=utterance, final_text=text,
                camera_used=False, camera_image=None,
                observation_action_name=None, observation_action_input=None,
                companion_mood="engaged", is_desire_turn=False, desires=None,
                arousal=arousal, memories=memories,
            ),
            name="event-post-response",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("event-loop persistence spawn failed: %s", e)

    return text
