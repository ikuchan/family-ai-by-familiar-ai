"""REST 内省・層 4「能力を再定義する」（記-a-と）。

この層が作り直すのは**要約だけ**である（`[あなたは誰か]` に載る 1 枚）。自己像が変わった晩・
要約がまだ無いとき・`ME.md` が書き換わったときに作り直し、`ME.md` が先頭にそのまま残って
上限に収まるものだけ保存する。

一覧（`capabilities.yaml`）を機械に書き直させる道は 環-y（2026-09-24）で撤去した。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
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


# ---- 要約を作り直す ---------------------------------------------------------


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
        patch.object(rc, "load_summary", return_value=fresh),
        patch.object(rc, "save_summary") as save,
    ):
        text = asyncio.run(rc.redefine_capabilities(a, self_image_changed=False, now=NOW))
    assert "見送" in text
    save.assert_not_called()
    a.backend.complete.assert_not_called()


def test_the_layer_refreshes_the_summary_when_the_self_image_changed():
    me = "名前： パジュ\n一人称：ぼく"
    a = _agent(me + "\n## 私にできること\n- 記憶を探せる")
    with (
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
