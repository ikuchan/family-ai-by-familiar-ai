"""LLM の返答に付くコードフェンスを剥がす（知-a S1 の積み残し）。

`課題8 §7` の知覚スライス S1 は「見た画像を VLM で言葉にする」を含み**完了**と記録されて
いるが、実物は成立していなかった。`extract_entities` が `json.loads(raw)` を直接呼ぶため、
返答が Markdown のコードフェンスで包まれていると `Expecting value: line 1 column 1 (char 0)`
で落ち、`[]` を返す。`see` の完了は「撮った。以上」で終わり、何が写っていたかが載らない。

**フェンスは気まぐれに付く。** 実機（llava:7b・同じ画像・同じプロンプト）で、1回目は素の
JSON（`chair`・`person`）、2回目はフェンス付き（`child`・`adult`・`chair`）だった。手元の
合成画像では6回とも付いた。通る回と通らない回が混ざるので、症状が間欠的に見えていた。

同じ処理は `capability_state.save_manifest` に既にある（YAML のフェンス剥がし）。3箇所目を
作らないよう、剥がす処理を1つに集めて両方から呼ぶ。
"""

from __future__ import annotations

import pytest

from familiar_agent.core.helpers import strip_code_fence


class TestStripCodeFence:
    def test_a_json_fence_is_stripped(self) -> None:
        got = strip_code_fence('```json\n{"a": 1}\n```')
        assert got == '{"a": 1}', f"剥がせていない: {got!r}"

    def test_a_yaml_fence_is_stripped(self) -> None:
        assert strip_code_fence("```yaml\ncapabilities:\n```") == "capabilities:"

    def test_a_bare_fence_is_stripped(self) -> None:
        assert strip_code_fence("```\nplain\n```") == "plain"

    def test_text_without_a_fence_is_untouched(self) -> None:
        """フェンスが無ければ何もしない（素の JSON も通る）。"""
        assert strip_code_fence('{"a": 1}') == '{"a": 1}'

    def test_surrounding_whitespace_is_trimmed(self) -> None:
        assert strip_code_fence('  \n```json\n{"a": 1}\n```  \n') == '{"a": 1}'

    def test_an_unclosed_fence_is_still_stripped(self) -> None:
        """閉じが欠けた返答でも、開きだけは剥がす（打ち切られた生成）。"""
        assert strip_code_fence('```json\n{"a": 1}') == '{"a": 1}'

    def test_a_fence_inside_the_body_survives(self) -> None:
        """本文の途中にある ``` は消さない（先頭と末尾だけを見る）。"""
        got = strip_code_fence("```\n見出し\n```python\ncode\n```")
        assert "```python" in got, f"本文が壊れている: {got!r}"

    def test_empty_input(self) -> None:
        assert strip_code_fence("") == ""


@pytest.mark.asyncio
async def test_extract_entities_reads_a_fenced_reply() -> None:
    """フェンス付きの返答でもエンティティを読める（これが直したい症状）。"""
    from familiar_agent.scene import extract_entities

    class _Backend:
        async def complete(self, _prompt, *_a, **_kw):
            return ('```json\n{"entities": ['
                    '{"label": "chair", "category": "object", "confidence": 0.9},'
                    '{"label": "person", "category": "person", "confidence": 0.8}'
                    ']}\n```')

    got = await extract_entities("居間を見た", _Backend())
    assert [e["label"] for e in got] == ["chair", "person"], f"読めていない: {got}"


@pytest.mark.asyncio
async def test_extract_entities_still_reads_a_bare_reply() -> None:
    """素の JSON も従来どおり読める（実機では回ごとに形が変わる）。"""
    from familiar_agent.scene import extract_entities

    class _Backend:
        async def complete(self, _prompt, *_a, **_kw):
            return '{"entities": [{"label": "window", "category": "location"}]}'

    got = await extract_entities("窓を見た", _Backend())
    assert [e["label"] for e in got] == ["window"]


@pytest.mark.asyncio
async def test_an_unreadable_reply_is_still_logged() -> None:
    """剥がしても読めない返答は、今までどおり残す（黙って握り潰さない）。"""
    import logging

    from familiar_agent.scene import extract_entities

    class _Backend:
        async def complete(self, _prompt, *_a, **_kw):
            return "I can see a living room with a sofa."

    import _pytest.logging  # noqa: F401

    caplog = logging.getLogger("familiar_agent.scene")
    records = []

    class _Catch(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = _Catch()
    caplog.addHandler(h)
    try:
        assert await extract_entities("説明", _Backend()) == []
    finally:
        caplog.removeHandler(h)
    assert records, "読めない返答が記録されていない"


def test_save_manifest_uses_the_shared_stripper() -> None:
    """YAML 側の剥がしも同じ処理へ寄せる（同じ実装を2つ持たない）。"""
    import inspect

    from familiar_agent import capability_state

    src = inspect.getsource(capability_state.save_manifest)
    assert "strip_code_fence" in src, "共通の剥がしを使っていない"
    assert '"```yaml"' not in src, "独自のフェンス剥がしが残っている"
