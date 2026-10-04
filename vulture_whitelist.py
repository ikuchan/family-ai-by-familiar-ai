"""vulture の許可リスト（環-ab R-7c・2026-09-28）。

vulture は構文木だけを見るので、フレームワークが名前で呼ぶもの・試験やスクリプトが使う道具・値から引く
列挙などを「使われていない」と誤って出す。ここに載せたものは、呼び手を 1 つずつ確かめて**残す**と決めた
（`課題8` 環-ab）。新しく使われていないものが出たら、消すか、理由を添えてここへ足す。

使い方：`uv run vulture`（`pyproject.toml` の `[tool.vulture]` が、見る場所とこのファイルを渡す）。
"""

# ruff: noqa
# fmt: off

# ── フレームワークが名前で呼ぶ（Textual・Qt・zeroconf・logging・sounddevice）──────────
_.update_service  # unused method (src/familiar_agent/camera_discovery.py:247)
_.remove_service  # unused method (src/familiar_agent/camera_discovery.py:250)
__init_subclass__  # unused function (src/familiar_agent/gui.py:68)
_.closeEvent  # unused method (src/familiar_agent/gui.py:2441)
_.namer  # unused attribute (src/familiar_agent/main.py:69)
time_info  # unused variable (src/familiar_agent/tools/mic.py:197)
BINDINGS  # unused variable (src/familiar_agent/tui.py:151)
_.on_mount  # unused method (src/familiar_agent/tui.py:204)
_.on_input_submitted  # unused method (src/familiar_agent/tui.py:311)
_.action_restart_realtime_stt  # unused method (src/familiar_agent/tui.py:540)
_.action_clear_history  # unused method (src/familiar_agent/tui.py:602)
_.action_quit  # unused method (src/familiar_agent/tui.py:617)

# ── 設定：`getattr(cfg, f"voice_{axis}")` で引く・登録表の「見る計測」の欄（設計が登録に求める）──
voice_seeking  # unused variable (src/familiar_agent/config.py:860)
voice_rest  # unused variable (src/familiar_agent/config.py:866)
voice_bond  # unused variable (src/familiar_agent/config.py:872)
voice_safety  # unused variable (src/familiar_agent/config.py:878)
voice_esteem  # unused variable (src/familiar_agent/config.py:884)
measure_kind  # unused variable (src/familiar_agent/core/settings.py:20)

# ── 値から引く列挙（`Verdict("important")`）と、DB に残る関係の種類の語彙 ──────────────
IMPORTANT  # unused variable (src/familiar_agent/io/oif.py:187)
USELESS  # unused variable (src/familiar_agent/io/oif.py:188)
REFERRED  # unused variable (src/familiar_agent/io/oif.py:189)
UNUSED  # unused variable (src/familiar_agent/io/oif.py:190)
KIND_RESOLVE  # unused variable (src/familiar_agent/store/relations.py:41)
KIND_ADVANCE  # unused variable (src/familiar_agent/store/relations.py:42)

# ── 試験・スクリプトの道具（状態を戻す・下ごしらえ・外から読む）──────────────────────
_.warm_keys  # unused method (src/familiar_agent/backends/anthropic.py:302)
_delete_all  # unused function (src/familiar_agent/config_overrides.py:137)
_.replace_line  # unused method (src/familiar_agent/core/self_image.py:52)
sql_to_vec  # unused function (src/familiar_agent/db.py:193)
identical_groups  # unused variable (src/familiar_agent/loop/rest_core.py:61)
_.scene_surprise  # unused method (src/familiar_agent/occupancy_sensor.py:104)
_.record_cooccurrence  # unused method (src/familiar_agent/store/relations.py:203)
_.stop_all  # unused method (src/familiar_agent/store/stopwatches.py:64)
_.loop_counter  # unused property (src/familiar_agent/voice_guard.py:74)

# ── 使う予定が決まっている口（記-a の記憶の面・構造化の問いの族）─────────────────────
ask_numbers  # unused function (src/familiar_agent/core/structured_ask.py:60)
ask_choice  # unused function (src/familiar_agent/core/structured_ask.py:85)
_.recent_feelings_async  # unused method (src/familiar_agent/tools/memory.py:1245)
_.recall_self_model_async  # unused method (src/familiar_agent/tools/memory.py:1284)
_.recall_curiosities_async  # unused method (src/familiar_agent/tools/memory.py:1302)

# ── 結線は知-d（顔と声の登録）で行う（いまはアプリに import されていない）──────────────
_._absent  # unused attribute (src/familiar_agent/recognition/presence_watcher.py:41)
_.identify_async  # unused method (src/familiar_agent/recognition/voice.py:91)
