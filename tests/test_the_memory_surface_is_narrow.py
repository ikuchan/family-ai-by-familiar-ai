"""**口の公開面を絞る**（環-e-い・`設計方針_OIF` の役割3）。

`ObservationMemory` は `store/` 切り出しのときにファサードになったが、**公開面が 73 種**
あり、口として働いていない。記憶の実体（pgvector・BYTEA・面・次元）を機構から隠すには、
外へ出す面を絞る必要がある。

**落とすのは「役割が終わったもの」だけ。** まだ繋いでいないものは残し、**なぜ残すかを
1件ずつ書く**（無言で残さない・無言で落とさない）。
"""

from __future__ import annotations

import ast
import pathlib

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "familiar_agent"
_MEM = _SRC / "tools" / "memory.py"

#: **役割が終わったので落とした面。** 1件ずつ理由を書く。
_RETIRED = {
    "record_exchange": "関係の書く口は1つにした（`OIF.link`・2026-09-11）",
    "record_succession": "同上",
    "format_feelings_for_context": "旧 ReAct 経路の文脈組み立て。経路は 環-c で撤去した",
    "format_self_model_for_context": "同上",
    "format_curiosities_for_context": "同上",
    "format_day_summaries_for_context": "同上",
    "format_semantic_facts_for_context": "同上",
    "format_behavior_policies_for_context": "同上",
    # 環-d（2026-09-14）：`legacy/semantic_layer.py` ごと落とした（読む呼び手が 0 件・表も 061 で落とす）
    "recall_semantic_facts": "legacy 表（semantic_facts）と一緒に撤去",
    "recall_semantic_facts_async": "同上",
    "recall_behavior_policies": "legacy 表（behavior_policies）と一緒に撤去",
    "recall_behavior_policies_async": "同上",
    "recall_revisions": "legacy 表（memory_revisions）と一緒に撤去",
    "adjust_semantic_fact_confidence": "同上",
    "adjust_semantic_fact_confidence_async": "同上",
    "adjust_behavior_policy_confidence": "同上",
    "adjust_behavior_policy_confidence_async": "同上",
    "link_memories": "legacy 表（memory_links）と一緒に撤去",
    "link_memories_async": "同上",
    "get_linked_memories": "同上",
    "get_linked_memories_async": "同上",
}

#: **まだ繋いでいないので残す面。** 落とすと、使うときに作り直しになる。
_NOT_YET = {
    "recall_self_model_async": "記-a（REST 内省）が使う",
    "recall_curiosities_async": "同上",
    "recent_feelings_async": "同上",
    "append_memory_event_async": "記-b（蒸留）のジョブ投入",
    "mark_job_done": "同上（ジョブの完了）",
    "get_dates_with_summaries": "同上",
}


def _public_faces() -> set[str]:
    tree = ast.parse(_MEM.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ObservationMemory"
    )
    return {
        m.name
        for m in cls.body
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_")
    }


def test_the_retired_faces_are_gone():
    """**旧名の grep が0件**であることを完了条件にする（数え上げで代えない）。"""
    left = sorted(set(_RETIRED) & _public_faces())
    assert not left, f"落としたはずの面が残っている: {left}"


def test_nothing_calls_the_retired_faces():
    """呼び手も残っていない。**テストも含めて**見る。"""
    calls = []
    for f in list(_SRC.rglob("*.py")) + list(pathlib.Path(__file__).resolve().parent.rglob("*.py")):
        if f == _MEM or f.name == pathlib.Path(__file__).name:
            continue
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) in _RETIRED:
                calls.append(f"{f.name}:{node.lineno} {node.func.attr}")
    assert not calls, f"落とした面をまだ呼んでいる: {calls}"


def test_the_faces_we_keep_are_still_there():
    """**反証側。** まだ繋いでいないものまで落としていないか。"""
    faces = _public_faces()
    gone = sorted(n for n in _NOT_YET if n not in faces)
    assert not gone, f"理由をつけて残すはずの面が消えている: {gone}"


def test_the_surface_got_narrower():
    """絞った結果を数で残す。**動いたら書き換える**（数は隠さない）。

    73 → **65 種**（役割の終わった8面を落とした）。記憶へ直につながる線は
    **24 → 12 箇所**（`agent.py` は 16 → 4）。

    2026-09-12 に **66 種**へ1つ足した（`exchange_roles_of`——誰が言ったかを役割から
    引く口。相手の言葉が「わたしが言った」と印字されていたのを直すため）。
    2026-09-13 に **67 種**へ1つ足した（`extend`——既にある関係の末尾へ項を足す口。
    やりとりの関係を閉じるときに書き、会話要約を背景で足すため）。
    2026-09-14 に `delete_day_summaries_for_date` を落とし（終了時の日次要約の撤去・記-a-ろ-は）、
    `observations_since_last_rest` を足した（畳み込みの材料）。数は据え置き。
    2026-09-14 に環-d で legacy の包み 13 面を落とし **54 種**。
    2026-09-15 に記-a-ろ-ろ で層 1 の計測と減りの口 3 面（`core_faces`・`fresh_since_last_rest`・
    `decay_groundedness`）を足し **57 種**。同日 記-a-ろ-に で `core_records`・`raise_groundedness` を足し **59 種**。
    """
    assert len(_public_faces()) <= 59
