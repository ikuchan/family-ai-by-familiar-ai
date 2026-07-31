"""自己状態（`SelfState`）の撤去。

6軸（arousal／fatigue／social_pull／sensor_confidence／unresolved_tension／
focus_stability）を毎ターン更新して `agent_state.self_state` へ保存していたが、**読み出す
経路が2つとも死んでいた**。

- `_system_prompt` → `_interoception` → プロンプト（どちらも後に撤去）：`_system_prompt` を呼ぶ生きた経路が
  無い。実行中のプロンプトは `build_event_system_prompt` が組み、自己状態を受け取らない。
- `_online_temporal_context` の `unresolved_tension`：テストからしか呼ばれない。

書き込みだけが `_run_post_response_pipeline` 経由で毎ターン走っていた。読まれない値の
ために計算と DB 書き込みを続けるのは、ソースを読むたびのノイズにもなる。

`arousal` という名前が PAD の A 軸（高ぶり）と二重になっていた問題も、これで消える。
"""

from __future__ import annotations

import importlib
import pathlib

import pytest


def test_module_is_gone() -> None:
    """`self_state` モジュールは無い。"""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("familiar_agent.self_state")


def test_agent_has_no_self_state() -> None:
    """エージェントは自己状態を持たない。"""
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "src/familiar_agent/agent.py").read_text(encoding="utf-8")
    assert "SelfState" not in src, "agent.py に SelfState が残っている"
    assert "_self_state" not in src, "agent.py に _self_state が残っている"


def test_arousal_is_only_the_pad_axis() -> None:
    """`arousal` は PAD の A 軸だけを指す（自己状態の軸としては残っていない）。

    同じ名前が2つの量に付いていると、読む側が取り違える。撤去でこの二重化が消える。
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    stale = []
    for path in (root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # 自己状態の軸としての arousal は、基準値の辞書か snapshot 経由で現れていた。
        if '"arousal": 0.35' in text or 'get("arousal", 0.35)' in text:
            stale.append(str(path.relative_to(root)))
    assert not stale, "自己状態の arousal が残っている:\n" + "\n".join(stale)


def test_migration_drops_the_agent_state_row() -> None:
    """`agent_state` の `self_state` 行をマイグレーションで消す。

    読み手が居なくなった値を DB に残すと、次に見た人が「まだ使っている」と読む。
    場当たりの DELETE だと他の環境に効かないので、マイグレーションで通す。
    """
    import importlib.util
    import os
    import pathlib

    import psycopg2

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO agent_state (state_key, value_json, updated_at)"
            " VALUES ('self_state', %s, now())"
            " ON CONFLICT (state_key) DO UPDATE SET value_json = EXCLUDED.value_json",
            ('{"arousal": 0.35}',),
        )
        cur.execute("SELECT 1 FROM agent_state WHERE state_key = 'self_state'")
        assert cur.fetchone() is not None, "前提が崩れている（行を置けていない）"

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "migration" / "2026-07-31-038_drop_self_state.py")
    spec = importlib.util.spec_from_file_location("drop_self_state_probe", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.upgrade(conn)

    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM agent_state WHERE state_key = 'self_state'")
        assert cur.fetchone() is None, "self_state 行が消えていない"
    conn.close()
