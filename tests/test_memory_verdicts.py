"""フルLLM が「想起した記憶をどう扱ったか」を申告する（課題5 E節 段2）。

設計は「**フルLLM が参照した MI だけ**再評価（上げ下げ）＋freshness 更新」と定めている。
その更新契機がこの申告である。想起しただけで更新すると、一度上がった記録が自分を押し上げ
続ける（実機で 47日前の挨拶が t=1.000 で居座り、5秒前の自分の発話を押し出した）。

判定は4つ。**W に出た記憶すべて**について返させる。

- important（大事）　　 n += 1 ＋ 時間の起点を更新
- useless（不要）　　　 n -= 1 ＋ 時間の起点を更新
- referred（参照）　　　時間の起点だけ更新
- unused（使わなかった）何もしない

**照合できたものだけ適用する。** 落とされた分を「使わなかった」と決めつけると、申告漏れと
本当に使わなかったことを混同する。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.backends import ToolCall
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.loop.prompt import EVENT_SYSTEM_PROMPT
from tests.test_event_loop import _agent, _run, _turn


def test_the_say_tool_accepts_verdicts():
    from familiar_agent.tools.tts import TTSTool

    schema = TTSTool.get_tool_definitions(MagicMock())[0]["input_schema"]
    verdicts = schema["properties"]["memory_verdicts"]
    assert verdicts["items"]["properties"]["verdict"]["enum"] == [
        "important", "useless", "referred", "unused"]
    assert "memory_verdicts" not in schema["required"]      # 省略可


def test_the_prompt_asks_for_every_recalled_memory():
    assert "memory_verdicts" in EVENT_SYSTEM_PROMPT
    assert "important" in EVENT_SYSTEM_PROMPT and "unused" in EVENT_SYSTEM_PROMPT


def test_verdicts_are_matched_through_the_index_not_by_prefix_guessing():
    # 写し間違いは一致せず、黙って別の記憶へ適用されない。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    ip = InformationProcessing(a)
    ip._w_index = {"aaaaaaaaaaaa": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"}
    ip._apply_memory_verdicts([
        {"id": "aaaaaaaaaaaa", "verdict": "important"},
        {"id": "zzzzzzzzzzzz", "verdict": "useless"},      # W に無い id
    ])
    applied = a._memory.apply_verdicts.call_args.args[0]
    assert applied == {"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa": "important"}


def test_nothing_is_applied_when_the_workspace_had_no_memories():
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    ip = InformationProcessing(a)
    ip._apply_memory_verdicts([{"id": "aaaaaaaaaaaa", "verdict": "important"}])
    a._memory.apply_verdicts.assert_not_called()


def test_the_workspace_prints_twelve_digit_ids():
    # 8桁だと記録が10万件規模でほぼ確実に衝突する。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a, utterance="おはよう")
    ip = InformationProcessing(a)
    ip._compose_workspace(a._active_memory(), [{"memory_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}])
    assert list(ip._w_index) == ["aaaaaaaabbbb"]
