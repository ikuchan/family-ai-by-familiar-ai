"""Person identification and presence tools for the ReAct agent."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..person_memory_manager import PersonMemoryManager

logger = logging.getLogger(__name__)


class PersonTool:
    """LLM-facing tools for declaring who is present and who is speaking."""

    def __init__(self, manager: "PersonMemoryManager") -> None:
        self._manager = manager

    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "declare_speaker",
                "description": (
                    "今この発言をしているのが誰かを宣言する。"
                    "名前の自己紹介・話し方・記憶との照合で判断したときに呼ぶ。"
                    "発言者が変わったら毎回呼ぶこと。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name":       {"type": "string", "description": "persons.name に登録された名前"},
                        "confidence": {"type": "number", "description": "0.0〜1.0"},
                        "reason":     {"type": "string"},
                    },
                    "required": ["name", "confidence"],
                },
            },
            {
                "name": "note_person_arrived",
                "description": (
                    "カメラや声で誰かがその場に来たことを検知したとき。"
                    "まだ発言していなくてもよい。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name":       {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["name", "confidence"],
                },
            },
            {
                "name": "note_person_left",
                "description": "誰かがその場を離れたことを検知したとき。",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "who_is_present",
                "description": "現在その場にいる人物と話者の一覧を返す。",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "ask_who_is_speaking",
                "description": (
                    "誰と話しているか不明のとき、相手に名前を尋ねる文章を生成する。"
                    "戻り値を発話すること。"
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    async def call(self, tool_name: str, tool_input: dict) -> tuple[str, str | None]:
        if tool_name == "declare_speaker":
            return await self._declare_speaker(tool_input)
        if tool_name == "note_person_arrived":
            return await self._arrived(tool_input)
        if tool_name == "note_person_left":
            return await self._left(tool_input)
        if tool_name == "who_is_present":
            return await self._who_is_present()
        if tool_name == "ask_who_is_speaking":
            return await self._ask_who()
        return f"Unknown person tool: {tool_name}", None

    async def _declare_speaker(self, inp: dict) -> tuple[str, None]:
        from ..person_memory_manager import RecognitionHint
        name = str(inp["name"])
        confidence = float(inp.get("confidence", 0.9))
        reason = str(inp.get("reason", ""))
        pid = await self._resolve(name)
        hint = RecognitionHint(person_id=pid, confidence=confidence, source="llm", reason=reason)
        switched = await self._manager.apply_hint(hint)
        status = "切り替えました" if switched else "既にアクティブです"
        return f"話者を {name} に設定しました（{status}）。理由: {reason}", None

    async def _arrived(self, inp: dict) -> tuple[str, None]:
        name = str(inp["name"])
        conf = float(inp.get("confidence", 0.8))
        pid = await self._resolve(name)
        await self._manager.person_arrived(pid, conf)
        return f"{name} がその場に来ました（conf={conf:.0%}）。", None

    async def _left(self, inp: dict) -> tuple[str, None]:
        name = str(inp["name"])
        persons = {p["name"]: p for p in self._manager.list_persons()}
        if name in persons:
            await self._manager.person_left(persons[name]["id"])
        return f"{name} がその場を離れました。", None

    async def _who_is_present(self) -> tuple[str, None]:
        present = self._manager.get_present_ids()
        speaker = self._manager.current_speaker_id
        if not present:
            return "その場には誰もいません。", None
        lines = []
        for pid in present:
            name = self._manager.get_person_name(pid)
            tag = " ← 話者" if pid == speaker else ""
            lines.append(f"- {name}{tag}")
        return "現在その場にいる人:\n" + "\n".join(lines), None

    async def _ask_who(self) -> tuple[str, None]:
        info = self._manager.get_active_person_info()
        current = info.get("display_name", "不明")
        return (
            f"（現在の話者: {current}）\n"
            "「あなたのお名前を教えていただけますか？」"
        ), None

    async def _resolve(self, name: str) -> str:
        """Return person_id, auto-registering if unknown."""
        persons = {p["name"]: p for p in self._manager.list_persons()}
        if name in persons:
            return persons[name]["id"]
        pid = self._manager.register_person(name)
        logger.info("Auto-registered person: %s (%s)", name, pid)
        return pid
