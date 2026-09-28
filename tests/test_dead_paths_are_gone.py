"""呼び手の無い仕組みを外した（環-ab の A・2026-09-28）。

- 保留の発話の書く口：`note_to_share`（主LLM に出していない記憶の道具）→ `PendingSpeechStore` →
  `pending_speech` テーブル。配る口は出-as 段 9b で外し、読む口は 0 件だった。
- 立ち位置に規則を添える口：`with_rules` と `build_context` の `rules` の欄。呼び手は発話前の検査だけで、
  検査は出-au 段 5-3 で Jev へ移った（規則は機械で読む）。

名前が残ると、次に読む者が動いている仕組みだと信じるので、src から消えたことを確かめる。
"""

from __future__ import annotations

import inspect
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "familiar_agent"

GONE = (
    "note_to_share",
    "PendingSpeechStore",
    "pending_speech_store",
    "_notes_registered_this_turn",
    "_pending_store",
    "PendingSpeechConfig",
    "PENDING_SPEECH_",
    "with_rules",
)


def test_the_old_names_are_gone_from_src():
    hits = []
    for p in SRC.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for name in GONE:
            if name in text:
                hits.append(f"{p.relative_to(SRC)}: {name}")
    assert hits == []


def test_the_pending_speech_store_module_is_gone():
    assert not (SRC / "tools" / "pending_speech_store.py").exists()


def test_the_context_has_no_slot_for_rules():
    from familiar_agent.core.context_parts import build_context

    assert "rules" not in inspect.signature(build_context).parameters


# ── 使われていないモジュール（環-ab の残り・vulture と grep で確かめた・2026-09-28）───────────


DEAD_MODULES = (
    "event_bus.py",  # どこからも import されない。JSONL に記録する作り（保存は PostgreSQL だけの決まりにも反する）
    "tools/person.py",  # `PersonTool`。どこからも import されない
)


def test_the_dead_modules_are_gone():
    assert [m for m in DEAD_MODULES if (SRC / m).exists()] == []


# ── 旧 `run()` の残り（ループが呼ばない `agent.py` のメソッドと、書くだけの属性・環-ab R-2）────────


OLD_RUN_LEFTOVERS = (
    "_execute_tool",
    "_boost_from_internal_result",
    "_anniversary_context",
    "_infer_companion_mood",
    "_backup_status_note",
    "_should_compact",
    "_compact_messages",
    "_last_tool_error",
    "_tool_failure_streak",
)


def test_the_old_run_leftovers_are_gone_from_the_agent():
    import re

    text = (SRC / "agent.py").read_text(encoding="utf-8")
    assert [n for n in OLD_RUN_LEFTOVERS if re.search(rf"\b{n}\b", text)] == []
