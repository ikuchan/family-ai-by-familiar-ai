"""作業記憶 W を、殻から出す（環-e-に・に-5-に-2）。

`_iterate`（202行）と `_act_on_decision`（97行）は、`_dispatch_main_llm`・`_finish`・
`_speak`・`_write_version` など**11個と9個を呼び返す**。この2つは**殻そのもの**なので、
別 file へ出せば `InformationProcessing` への逆参照が要り、に-4 で断ったばかりの
「核が殻を呼び返す」形が戻る。**殻は残す。**

一方、W をめぐる 174 行は**何も呼び返さない**。`_w_id_map` は W の索引、`compose` が組み、
`recall` が中身を入れ、`link_follows` と `apply_memory_verdicts` が W の id を使う。
これが核である（Functional core / Imperative shell）。

**class を作らずモジュール関数にする**（`loop/generator.py` の前例）。W は反復ごとに
作り直すもので、長生きの持ち主に抱えさせると寿命が混ざる（に-2「ニ．反復の寿命は
束にしない」）。**求めは引数で受け取り、保持しない。**
"""

from __future__ import annotations

import inspect

from familiar_agent.loop import workspace
from familiar_agent.loop.request import Request


# ── 核であること ───────────────────────────────────────────────────────────


def test_the_core_never_calls_back_into_the_shell():
    """**核は殻を呼び返さない。** 呼び返せば、出した意味が無くなる。"""
    # **呼び出しを見る**（語ではない）。docstring がなぜ殻を残すかを述べているので、
    # 語で探すと自分の説明文に当たる。
    src = inspect.getsource(workspace)
    for shell in (
        "dispatch_main_llm(",
        "_speak(",
        "_finish(",
        "_write_version(",
        "_start_lookup(",
    ):
        assert shell not in src, f"殻を呼び返している：{shell}"


def test_the_core_keeps_no_state_of_its_own():
    """モジュール関数だけで、状態を持たない（`self` も、module 変数の書き換えも無い）。"""
    for name in ("open_ids", "compose", "recall", "link_follows", "apply_memory_verdicts"):
        fn = getattr(workspace, name)
        assert "self" not in inspect.signature(fn).parameters, name


def test_the_request_is_passed_in_not_held():
    """求めは引数で受け取る。抱えると、W（反復の寿命）と求めの寿命が混ざる。"""
    assert "req" in inspect.signature(workspace.open_ids).parameters
    assert "req" in inspect.signature(workspace.compose).parameters


# ── 対応表を返す ───────────────────────────────────────────────────────────


def test_compose_returns_both_the_text_and_the_index():
    """W と、12桁 → 完全な id の対応表。**対応表は W から導かれる**ので一緒に返す。"""
    from unittest.mock import MagicMock

    mem = MagicMock()
    mem.format_for_context = MagicMock(return_value="[想起]昔の話")
    text, id_map = workspace.compose(
        mem, [{"memory_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "summary": "昔の話"}], Request()
    )
    assert "[想起]昔の話" in text
    assert id_map == {"aaaaaaaaaaaa": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"}


def test_the_open_ids_come_from_the_request():
    r = Request()
    r.request_id = "obs1"
    r.live_version_id = "ver1"
    assert workspace.open_ids(r) == ["obs1", "ver1"]


# ── 申告の対応表を省略できない ─────────────────────────────────────────────


def test_the_verdict_table_has_no_default():
    """**渡し忘れたら落ちる**ほうが、黙って別の記憶へ当たるより良い。

    以前は `w_id_map=None` でループのいまの対応表へ落ちた。主LLM は投げっぱなしなので、
    返るまでに別の完了が届けばその表は作り直されている（環-h ②）。落とし先を無くす。
    """
    params = inspect.signature(workspace.apply_memory_verdicts).parameters
    assert params["w_id_map"].default is inspect.Parameter.empty


# ── 対応表は属性でなく、引数と返り値で回る（に-5-に-2-2）─────────────────


def test_the_index_is_not_an_attribute():
    """W は反復ごとに作り直す。**属性に置く理由がない**（に-2「ニ．反復の寿命」）。

    属性のままだと、主LLM が飛行中に別の完了が届いたとき表が作り直され、12桁が当たれば
    申告が黙って別の記憶へ当たる。落とし先そのものを無くす。
    """
    from familiar_agent.loop.event_loop import InformationProcessing

    src = inspect.getsource(InformationProcessing.__init__)
    assert "_w_id_map" not in src


def test_the_recent_context_takes_the_index():
    """`_recent_ctx` は続き先の辺を張るので、その反復の対応表を受け取る。"""
    from familiar_agent.loop.event_loop import InformationProcessing

    assert "w_id_map" in inspect.signature(InformationProcessing._recent_ctx).parameters


def test_the_abort_has_nothing_to_clear():
    """属性が無いので、打ち切りが表を片付ける行も無くなる。"""
    from familiar_agent.loop.event_loop import InformationProcessing

    assert "w_id_map" not in inspect.getsource(InformationProcessing._abort_lookups)


# ── 殻の側 ─────────────────────────────────────────────────────────────────


def test_the_shell_delegates_instead_of_composing():
    from familiar_agent.loop.event_loop import InformationProcessing

    src = inspect.getsource(InformationProcessing)
    for call in (
        "workspace.recall(",
        "workspace.link_follows(",
        "workspace.apply_memory_verdicts(",
    ):
        assert call in src, call
    # 中身は核の側にある。殻は組み立てない。
    assert "def _compose_workspace" not in src
    assert "def _apply_memory_verdicts" not in src


def test_the_shell_keeps_the_iteration():
    """`_iterate` と `_act_on_decision` は殻なので残る（出せば逆参照が要る）。"""
    from familiar_agent.loop.event_loop import InformationProcessing

    assert hasattr(InformationProcessing, "_iterate")
    assert hasattr(InformationProcessing, "_act_on_decision")
