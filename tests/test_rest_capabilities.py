"""REST 内省・層 4「能力を再定義する」（記-a-と）。

一覧（`capabilities.yaml` は既定・現在値は DB）を 7 日〔仮〕に 1 度、実装の docstring から
LLM に書き直させ、形の検査に通ったものだけ DB に置く。要約（`[あなたは誰か]` に載る 1 枚）は
一覧か自己像が変わった晩だけ作り直す。**file は実行時に書かない。**
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.loop import rest_capabilities as rc

NOW = datetime(2026, 9, 14, 3, 0, tzinfo=timezone.utc)

GOOD_YAML = (
    "capabilities:\n"
    "  - id: memory\n    summary: 記憶を探せる\n    enabled: true\n"
    "  - id: family_schedule\n    summary: 家の予定を引ける\n    enabled_env: NO_SUCH_ENV_FOR_TEST\n"
)


def _agent(reply: str, me_md: str = "名前： パジュ\n一人称：ぼく"):
    a = MagicMock()
    a.backend = MagicMock()
    a.backend.complete = AsyncMock(return_value=reply)
    a._me_md = me_md
    return a


# ---- いつ回すか ---------------------------------------------------------------


def test_the_manifest_is_due_when_never_made_or_older_than_the_interval():
    assert rc.due_for_manifest(None, NOW)
    assert rc.due_for_manifest(NOW - timedelta(days=rc.MANIFEST_EVERY_DAYS), NOW)
    assert not rc.due_for_manifest(NOW - timedelta(days=rc.MANIFEST_EVERY_DAYS - 1), NOW)


def test_the_summary_is_due_only_when_something_it_is_made_from_changed():
    assert not rc.due_for_summary(manifest_changed=False, self_image_changed=False)
    assert rc.due_for_summary(manifest_changed=True, self_image_changed=False)
    assert rc.due_for_summary(manifest_changed=False, self_image_changed=True)
    # まだ 1 枚も無ければ、変化が無くても作る（初回）。
    assert rc.due_for_summary(manifest_changed=False, self_image_changed=False, exists=False)


# ---- 一覧の再定義 -------------------------------------------------------------


def test_a_well_formed_manifest_goes_to_the_db_not_the_file():
    a = _agent("```yaml\n" + GOOD_YAML + "```")
    with (
        patch.object(rc, "load_capabilities", return_value="capabilities:\n  - id: memory\n"),
        patch.object(rc, "collect_manifest_context", return_value="## Built-in tools"),
        patch.object(rc, "store_capabilities") as store,
    ):
        r = asyncio.run(rc.regenerate_manifest(a))
    assert r.changed and r.reason is None
    stored = store.call_args.args[0]
    assert stored.startswith("capabilities:")  # フェンスは剥がれている
    assert "family_schedule" in stored


def test_an_unchanged_manifest_is_not_stored_again():
    a = _agent(GOOD_YAML)
    with (
        patch.object(rc, "load_capabilities", return_value=GOOD_YAML),
        patch.object(rc, "collect_manifest_context", return_value=""),
        patch.object(rc, "store_capabilities") as store,
    ):
        r = asyncio.run(rc.regenerate_manifest(a))
    assert not r.changed and r.reason is None
    store.assert_not_called()


def test_a_broken_manifest_is_refused_and_the_reason_is_named():
    cases = {
        "not: yaml: [": "読めない",
        "capabilities: []": "空",
        "capabilities:\n  - summary: id が無い\n    enabled: true\n": "id",
        "capabilities:\n  - id: a\n    summary: x\n    enabled: true\n  - id: a\n    summary: y\n    enabled: true\n": "重複",
        "capabilities:\n  - id: a\n    summary: 有効条件が無い\n": "enabled",
    }
    for reply, needle in cases.items():
        a = _agent(reply)
        with (
            patch.object(rc, "load_capabilities", return_value=GOOD_YAML),
            patch.object(rc, "collect_manifest_context", return_value=""),
            patch.object(rc, "store_capabilities") as store,
        ):
            r = asyncio.run(rc.regenerate_manifest(a))
        assert not r.changed and r.reason and needle in r.reason, (reply, r.reason)
        store.assert_not_called()


def test_a_manifest_that_lost_too_many_entries_is_refused():
    # 途中で切れた出力（27 件 → 3 件）をそのまま置くと、能力が一晩で消える。
    current = "capabilities:\n" + "".join(
        f"  - id: c{i}\n    summary: s\n    enabled: true\n" for i in range(10)
    )
    a = _agent(GOOD_YAML)  # 2 件
    with (
        patch.object(rc, "load_capabilities", return_value=current),
        patch.object(rc, "collect_manifest_context", return_value=""),
        patch.object(rc, "store_capabilities") as store,
    ):
        r = asyncio.run(rc.regenerate_manifest(a))
    assert not r.changed and r.reason and "減り" in r.reason
    store.assert_not_called()


# ---- 要約の作り直し -----------------------------------------------------------


def test_the_summary_keeps_me_md_verbatim_and_is_saved():
    me = "名前： パジュ\n一人称：ぼく"
    a = _agent(me + "\n\n## 私にできること\n- 記憶を探せる\n")
    with patch.object(rc, "save_summary") as save:
        ok = asyncio.run(rc.refresh_summary(a, GOOD_YAML))
    assert ok is None
    assert save.call_args.args[0].startswith(me)
    prompt = a.backend.complete.call_args.args[0]
    assert "family_schedule" not in prompt  # 有効条件を満たさない行は渡さない（env 無し）


def test_a_summary_that_rewrote_me_md_or_grew_too_long_is_refused():
    me = "名前： パジュ\n一人称：ぼく"
    a = _agent("名前： パジュ（要約された）\n## 私にできること\n- x")
    with patch.object(rc, "save_summary") as save:
        reason = asyncio.run(rc.refresh_summary(a, GOOD_YAML))
    assert reason and "そのまま" in reason
    save.assert_not_called()
    a = _agent(me + "\n## 私にできること\n" + "- あ" * rc.SUMMARY_MAX_CHARS)
    with patch.object(rc, "save_summary") as save:
        reason = asyncio.run(rc.refresh_summary(a, GOOD_YAML))
    assert reason and "字" in reason
    save.assert_not_called()


# ---- 層としての 1 回 ---------------------------------------------------------


def test_the_layer_skips_everything_when_nothing_is_due():
    a = _agent(GOOD_YAML)
    # 「何も変わっていない」は、要約が**いまの `ME.md` を先頭に含む**こと（出-ao）。
    # 含まなければ `ME.md` が書き換わったとみなして作り直す。
    fresh = a._me_md + "\n\n## 私にできること\n\n- 記憶を探せるよ。"
    with (
        patch.object(rc, "capabilities_updated_at", return_value=NOW - timedelta(days=1)),
        patch.object(rc, "load_summary", return_value=fresh),
        patch.object(rc, "store_capabilities") as store,
        patch.object(rc, "save_summary") as save,
    ):
        text = asyncio.run(rc.redefine_capabilities(a, self_image_changed=False, now=NOW))
    assert "見送" in text
    store.assert_not_called()
    save.assert_not_called()
    a.backend.complete.assert_not_called()


def test_the_layer_refreshes_the_summary_when_the_self_image_changed():
    me = "名前： パジュ\n一人称：ぼく"
    a = _agent(me + "\n## 私にできること\n- 記憶を探せる")
    with (
        patch.object(rc, "capabilities_updated_at", return_value=NOW - timedelta(days=1)),
        patch.object(rc, "load_capabilities", return_value=GOOD_YAML),
        patch.object(rc, "load_summary", return_value="ある"),
        patch.object(rc, "save_summary") as save,
    ):
        text = asyncio.run(rc.redefine_capabilities(a, self_image_changed=True, now=NOW))
    assert "要約を作り直した" in text
    save.assert_called_once()


# ---- ME.md が書き換わったら作り直す（出-ao・2026-09-23） ----------------------


def test_the_summary_is_due_when_me_md_changed():
    """`ME.md` を書き換えても作り直す引き金が無かった（実機：要約は 09-13 のまま）。

    要約は `ME.md` を**先頭に逐語で**含むので、先頭が合わなければ古い。更新時刻を持ち回る
    必要はなく、保存済みの要約そのものが材料になる。
    """
    me = "名前： パジュ\n一人称：ぼく"
    assert (
        rc.due_for_summary(
            manifest_changed=False,
            self_image_changed=False,
            summary=me + "\n\n## 私にできること\n- x",
            me_md=me,
        )
        is False
    )
    assert (
        rc.due_for_summary(
            manifest_changed=False,
            self_image_changed=False,
            summary="名前： パジュ（古い）\n\n## 私にできること\n- x",
            me_md=me,
        )
        is True
    )


def test_no_summary_is_still_due():
    assert (
        rc.due_for_summary(
            manifest_changed=False, self_image_changed=False, summary="", me_md="名前： パジュ"
        )
        is True
    )


def test_no_me_md_does_not_force_a_rebuild():
    """`ME.md` が無い機体で、毎晩作り直させない。"""
    assert (
        rc.due_for_summary(
            manifest_changed=False, self_image_changed=False, summary="なにか", me_md=""
        )
        is False
    )


# ---- 途中で切れた要約を置かない ----------------------------------------------


def test_a_summary_cut_off_midway_is_refused():
    """`max_tokens` で尽きた出力は、上限の検査を通り抜けていた（保存済みは末尾が `- `）。"""
    me = "名前： パジュ\n一人称：ぼく"
    a = _agent(me + "\n\n## 私にできること\n\n- 記憶を探せるよ。\n-")
    with patch.object(rc, "save_summary") as save:
        reason = asyncio.run(rc.refresh_summary(a, GOOD_YAML))
    assert reason and "切れて" in reason
    save.assert_not_called()


def test_the_cap_fits_me_md_plus_twenty_lines():
    """上限は `ME.md`（実測 1,223 字）＋ 20 行（35 字／行）を容れる（本人の決定・2,000 字）。"""
    assert rc.SUMMARY_MAX_CHARS >= 1223 + 20 * 35


def test_the_ask_is_big_enough_for_the_cap():
    """`max_tokens` は上限から決める。足りないと途中で切れる（日本語 1 字 ≈ 2 トークン）。"""
    assert rc.summary_max_tokens() >= rc.SUMMARY_MAX_CHARS
