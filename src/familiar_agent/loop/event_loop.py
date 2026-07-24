"""イベント駆動ループ（#11 段階1）：人の発言→拡散込み想起→1反復1出力（発話）。

現行 run() と排他（`EVENT_LOOP` on の user turn のみ）。段階1は発話のみ（ツールを渡さず
1出力を保証）。永続化は既存 `_run_post_response_pipeline`（utility LLM のみ）を流用する。
"""

from __future__ import annotations

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


async def run_iteration(agent, utterance: str, on_text=None) -> str:
    """1反復＝1出力：想起（拡散込み）で W を作り、フルLLM で1発話を生成して返す。

    `on_text` は生成のストリーミング出力先（CUI/GUI へ逐次表示）。
    """
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

    # 生成：say ツールだけを渡す＝発話のみ（多段 ReAct を構造的に禁止）。say は body-tool
    # （voice part）であり、slice1 は「1反復で say を1回」＝1反復1出力。
    say_tools = agent._tts.get_tool_definitions() if agent._tts else []
    user_msg = agent.backend.make_user_message(utterance)
    result, _raw = await agent.backend.stream_turn(
        system=system,
        messages=[user_msg],
        tools=say_tools,
        max_tokens=agent.config.max_tokens,
        on_text=on_text,
    )

    # 発話の取り出し：run() と同じ「先頭 say を採用・重複 say は抑制」に揃える。
    # say が無ければ result.text へフォールバック（保険）。
    text = ""
    say_tc = next((tc for tc in result.tool_calls if tc.name == "say"), None)
    if say_tc is not None:
        text = str(say_tc.input.get("text", "")).strip()
        if text and agent._tts is not None:
            with contextlib.suppress(Exception):
                await agent._tts.call("say", {"text": text})
        # say tool_call の text はストリームされないので、CUI/GUI 表示のため明示的に流す。
        if text and on_text is not None:
            on_text(text)
    else:
        text = (result.text or "").strip()  # フォールバックは stream_turn が既にストリーム済み

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
