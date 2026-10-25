"""イベント駆動ループ（#11 段階1）：I（情報処理機構）の最小縦切り。

設計正本＝`設計図_Mermaid` ③ I 詳細図。ここでは I の中の **LPM（ループ核）** と
**QC（完了キュー）** だけを実体化する。反復は QC を drain（取込→O 書込）→ REC（想起→W）→
GEN（生成）で進み、say で1出力して終わる／内部ツール（recall）は結果を QC へ積んで次反復へ
連鎖する（[D-単一想起]：相関ID を使わず結果は O→W 経由で再会）。

人の発話はこの経路が処理する。AIF/DIF/QA/QD、ARB/APR/ACT/MNT の
クラス分離は後続段階（ここでは stub しない）。永続化は既存 `_run_post_response_pipeline`
（utility LLM のみ）を流用し、消化した完了 O はターン観察 id で supersede する。
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
import contextlib
from dataclasses import dataclass, replace
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..backends.types import TurnResult

from ..poses import nearest_pose
from ..scene import extract_entities, read_photo
from ..store import clock
from .arbiter import CONFIRM_QUERY, Arbiter, ArbiterInput, Decision as ArbiterDecision
from ..store.relations import KIND_EXCHANGE, KIND_REVISION, KIND_SUCCESSION
from ..io.dif import DIF
from ..core import (
    family_now,
    final_version,
    jev_judges,
    parsing,
    unsaid,
)
from ..core.drive_autonomy import TALKING_AXES
from ..core.silence_hold import Heard
from ..core.tool_gate import gate_personal_tools
from ..core.tool_text import tool_calls_from_text
from ..io.oif import MI, Recalled
from ..person_memory_manager import AGENT_SELF_ID
from .speech_check import PAST_CHARS, facts_ctx, fitting_cap, past_rows
from ..core.wake_window import WakeWindow
from ..core.window_talk import WindowTalk, result_text as _talk_result
from ..backends.types import system_text
from ..core import skills as skills_mod
from ..core.measure import record as measure_record
from ..config import main_llm_cache_on
from .generator import SILENCE_MARK, SILENCE_NOTE, _pi_ctx, _present_ctx
from . import workspace
from ..store import intents as _intents
from .intent_ledger import IntentLedger
from .request import Lookup, Request
from .prompt import build_event_system_prompt

logger = logging.getLogger(__name__)

#: 起点の記録を O へ書くときの上限（字）。会話・独白・情動・知らせの起点。
_ORIGIN_MAX = 500
#: 記録だけする機器の知らせ（予定・メモ）は 1,500 字まで（情-s・2026-10-17・本人の決定ア）。入口が 500 字で
#: 切っていて、予定 67 件中 54 件が後ろの日を失っていた。
_DEVICE_MAX = 1500

# 連鎖が続けられる反復で渡す動作。上限に達した反復では say だけにして必ず閉じる。
_FULL_ACTIONS = (
    "say",
    # 主LLM のつなぎの道具（`filler`）は撤去した（出-cl 段 3・2026-10-15）。待ちは合図の声と電子音で知らせる。
    "recall",
    # 思い出し方を変えて引き直す道具（出-ah・2026-09-21）。W に載る情報が足りないとき、
    # 面・件数と思い出し方・時期と幅・直近の窓を、主LLM 自身が動かせる。効き目はその 1 回だけ。
    "recall_as",
    "recall_deeper",
    "recall_when",
    "recall_recent",
    "recall_tree",  # 記憶の木の節を指して要約を読む（記-m）
    "recall_thread",  # 記録の前後の話をたどる（記-q）
    "search_deferred",
    "fetch_deferred",
    "see",
    "look",
    "house_rules",
    "family_schedule",
    "notion_search",
    "journal",
    "vault",
    # タイマー（知-n）。結果はその場で返り、それを見て何を言うかは次の反復が決める（`recall` と同じ）。
    "set_timer",
    "cancel_timer",
    "pause_timer",
    "resume_timer",
    "set_alarm",
    "cancel_alarm",
    # 音楽（知-aa 段 1・2026-09-21）。器（`agent._music_tool`）が無ければ渡らない。
    "play_music",
    "stop_music",
    "next_track",
    "music_volume",
    "music_suggestion_reply",  # すすめた曲への返事（知-aa 段 4）
    "start_stopwatch",  # ストップウォッチ（知-u・別物）
    "stop_stopwatch",
)
#: 思い出し方を変える道具（出-ah）。`recall` と同じく、その場で引いて完了として返る。
_WIDEN_ACTIONS = (
    "recall_as",
    "recall_deeper",
    "recall_when",
    "recall_recent",
    "recall_tree",
    "recall_thread",
)

#: `recall_thread` で前後それぞれにたどる件数の上限（〔仮〕・記-q・2026-10-15 本人の決定メ）。道具の説明でも教える。
THREAD_MAX_STEPS = 10

#: 音楽の道具（知-aa）。`recall` と同じく、その場で返る。
_MUSIC_ACTIONS = (
    "play_music",
    "stop_music",
    "next_track",
    "music_volume",
    "music_suggestion_reply",
)

_TIMER_ACTIONS = ("set_timer", "cancel_timer", "pause_timer", "resume_timer")
# ストップウォッチ（知-u・2026-09-18）。タイマーとは別物・別の道具（`agent._stopwatch_tool`）。
_STOPWATCH_ACTIONS = ("start_stopwatch", "stop_stopwatch")
# アラーム（知-q・2026-09-18）。タイマーとは別物・別の道具（`agent._alarm_tool`）。
_ALARM_ACTIONS = ("set_alarm", "cancel_alarm")
# 確認待ちへの答え（出-y・2026-09-18）。預かりが生きているあいだだけ候補と道具に載る（`agent.confirm_alive`）。
# 機械が預かった入力で掛ける／捨てる。返りは道具の帰りと同じ道（想起なし・`workspace.RETURN_WITHOUT_RECALL`）。
_CONFIRM_ACTIONS = ("confirm", "decline")
# MCP の同期の道具（結果がその場で返る）。動作名（調停が使う）と道具名（主LLM が呼ぶ）の
# 両方から、(道具名, 求めの見出し) を引く。ここに無い MCP の道具は動作の表に載らない。
_MCP_LOOKUPS: dict[str, tuple[str, str]] = {
    "house_rules": ("get_house_rules", "家の決まりを見る"),
    "get_house_rules": ("get_house_rules", "家の決まりを見る"),
    # 期間を持つ道具は見出しに期間を入れる（`{days}`）。固定だと `days=2` の呼び直しが
    # 同じ求めに見えて止まる（2026-09-14 実機・出-p）。
    "family_schedule": ("get_family_schedule", "家族の予定を見る（{days} 日ぶん）"),
    "get_family_schedule": ("get_family_schedule", "家族の予定を見る（{days} 日ぶん）"),
    # Notion（知-k）。検索は語を持つので見出しに語を入れる（`_query_label` が組む）。
    "notion_search": ("search_notion", "Notion で「{query}」を探す"),
    "search_notion": ("search_notion", "Notion で「{query}」を探す"),
    # 日次記録は 1 人の体の記録なので、持ち主があれば名前に英字が付く（`get_journal_<latin>`・知-az）。
    # 付いた名前は `_PERSONAL_PREFIXES` が `journal` に戻す。道具名は `_personal_tool_name()` が解く。
    "journal": ("get_journal", "日次記録を見る"),
    "get_journal": ("get_journal", "日次記録を見る"),
    # 個人ティアの記録（`obsidian-memo`・知-g-い）。道具名は話者の英字で決まる（`ask_vault_<latin>`）
    # ので、表には動作名だけを置き、道具名は `_vault_tool_name()` が解く。数十秒かかるが、
    # 調べものは投げっぱなしなので別経路は要らない（MCP の時間切れだけ長い・`call_timeout_for`）。
    "vault": ("ask_vault_", "記録に「{query}」を聞く"),
}
# 名前に持ち主の英字が付く道具（個人ティア）の頭と、その動作名。話者ゲート（`_gated`）が本人以外を落とす。
_PERSONAL_PREFIXES = {"ask_vault_": "vault", "get_journal_": "journal"}


def _personal_action(name: str) -> str:
    """持ち主の英字が付いた道具名なら動作名（`ask_vault_yusuke` → `vault`）。違えば空。"""
    return next((a for p, a in _PERSONAL_PREFIXES.items() if name.startswith(p)), "")


# 調べる動作＝結果が後の反復に届くもの。投げたらその反復は終わる。
# `see`・`look` も含める。結果はその場で返るが、それを見て何を言うかは次の反復が決める
# （`recall` と同じ）。ここに入れないと 1反復1出力 が崩れる。
_LOOKUP_ACTIONS = (
    "recall",
    "search_deferred",
    "fetch_deferred",
    "see",
    "look",
    # MCP の同期の道具は即座に返るが、それを見て何を言うかは次の反復が決める（`recall` と
    # 同じ）。**主LLM は道具名で呼ぶ**ので、動作名だけでなく道具名も並べる（2026-09-13 まで
    # 動作名 `house_rules` しか無く、主LLM が `get_house_rules` を呼んでも捨てられていた）。
    *_MCP_LOOKUPS.keys(),
    *_TIMER_ACTIONS,
    *_ALARM_ACTIONS,
    *_STOPWATCH_ACTIONS,
    *_CONFIRM_ACTIONS,
    # 音楽も主LLM が道具名で呼ぶ。載っていなかったので `play_music` が捨てられ、求めが沈黙で閉じた（2026-10-07 実機）。
    *_MUSIC_ACTIONS,
)


def _query_label(action: str, tool_input: dict) -> str:
    """その求めの見出し。飛行中の一覧・完了の照合・W の「調べたもの」で鍵になる。

    `see` の入力は空で、`look` は向きしか持たない。`query`／`url` から取ると両方とも
    空文字になり、別々の求めが同じ鍵で衝突する。動作ごとに見出しを作る。
    """
    if action == "see":
        return "目の前を見る"
    if action in _TIMER_ACTIONS:
        # 同じ求めで同じ道具を二度呼ぶこともある（掛けてすぐ止める）ので、入力ごとに別の見出しにする。
        what = str(tool_input.get("label") or tool_input.get("id") or "").strip()
        return {
            "set_timer": "タイマーを掛ける",
            "cancel_timer": "タイマーを止める",
            "pause_timer": "タイマーを一時停止する",
            "resume_timer": "タイマーを再開する",
        }[action] + (f"「{what}」" if what else "")
    if action in _ALARM_ACTIONS:
        what = str(tool_input.get("label") or tool_input.get("id") or "").strip()
        return {"set_alarm": "アラームを掛ける", "cancel_alarm": "アラームを止める"}[action] + (
            f"「{what}」" if what else ""
        )
    if action in _STOPWATCH_ACTIONS:
        what = str(tool_input.get("label") or tool_input.get("id") or "").strip()
        return {"start_stopwatch": "測り始める", "stop_stopwatch": "測るのを止める"}[action] + (
            f"「{what}」" if what else ""
        )
    if action in _CONFIRM_ACTIONS:
        return CONFIRM_QUERY[action]
    if action in _MUSIC_ACTIONS:
        # 音楽にも見出しを付ける（出-be・2026-10-10 実機 11:05:16）。空だと同じ求めの 2 回目の音楽の操作が、名前に
        # 関係なく「すでに調べた」で止まり、かけ直しが投げられなかった。名前が違えば別の操作として投げる。
        name = str(tool_input.get("name") or "").strip()
        return {
            "play_music": f"音楽をかける「{name}」" if name else "音楽をかける（止まっていた続き）",
            "stop_music": "音楽を止める",
            "next_track": "次の曲にする",
            "music_volume": f"音量を変える「{str(tool_input.get('how') or '').strip()}」",
            "music_suggestion_reply": (
                f"すすめた曲に返事する「{str(tool_input.get('reply') or '').strip()}」"
            ),
        }[action]
    if action == "look":
        return f"{tool_input.get('pose') or tool_input.get('direction') or ''}を見に行く"
    action = _personal_action(action) or action
    if action in _MCP_LOOKUPS:
        return _MCP_LOOKUPS[action][1].format(
            query=str(tool_input.get("query") or tool_input.get("question") or "").strip(),
            days=_days_of(tool_input),
        )
    return str(tool_input.get("query") or tool_input.get("url", "")).strip()


SCHEDULE_DAYS_MAX = 14  # カレンダー MCP の `days` の上限（`mcp/calendar_mcp/server.py` と同じ）

#: 待ちの見張りが起きる間隔（秒）。待たせているあいだ、この間隔で窓を延ばす（窓は 10 秒なので 1 秒で足りる）。
WAIT_WATCH_TICK_SEC = 1.0

#: 聞き取りの確認の預かりの寿命（秒・〔仮〕・出-cm・2026-10-15 本人の決定）。聞き返しへの答えはすぐ返るので短くし、
#: あとの別の話の「はい」で見立ての動作を動かさない。
HEARD_CHECK_TTL_SEC = 30.0


def _heard_inputs(heard: "dict | None") -> dict:
    """聞き取りの確認の預かりを調停の材料の欄へ写す（出-cm）。無ければ空。"""
    if not heard:
        return {}
    return {
        "heard_check_action": str(heard.get("action") or ""),
        "heard_check_utterance": str(heard.get("utterance") or ""),
        "heard_check_question": str(heard.get("question") or ""),
    }


def _days_of(tool_input: dict) -> int:
    """入力の `days`（無ければ道具の既定 1）。見出しの期間に使う。"""
    try:
        return max(1, min(SCHEDULE_DAYS_MAX, int(tool_input.get("days", 1))))
    except (TypeError, ValueError):
        return 1


def _season_env_text() -> str:
    """いまの季節とまわり（季節の層・知-ac）の枠。読めなければ暦の行だけ（degrade）。"""
    try:
        from ..core import season_env

        return season_env.render(season_env.stored())
    except Exception as e:  # noqa: BLE001
        logger.debug("季節とまわりを読めなかった（無しで続ける）: %s", e)
        return ""


def _self_image_text() -> str:
    """自己像（層 2）の枠。DB の現在値（無ければ種）を文にする。読めなければ空（degrade）。"""
    try:
        from ..core import self_image

        return self_image.render(self_image.load())
    except Exception as e:  # noqa: BLE001
        logger.debug("自己像を読めなかった（無しで続ける）: %s", e)
        return ""


def _action_family(action: str) -> str:
    """道具名（主LLM が呼ぶ）を動作名（候補の表の鍵）に正規化する（`get_family_schedule` → `family_schedule`）。"""
    if personal := _personal_action(action):
        return personal
    if action in _MCP_LOOKUPS:
        tool = _MCP_LOOKUPS[action][0]
        for name, (t, _label) in _MCP_LOOKUPS.items():
            if t == tool and not name.startswith("get_") and name != "search_notion":
                return name
    return action


def _tool_input_of(decision) -> dict:
    """調停の決定から道具の入力を作る。`tool_input` を書いてきた道具（タイマー）はそれをそのまま。"""
    given = getattr(decision, "tool_input", None)
    if isinstance(given, dict) and given:
        return dict(given)
    return _tool_input_for(decision.action, decision.query)


def _tool_input_for(action: str, query: str) -> dict:
    """調停の決定（動作名と語）を、その道具が受け取る入力へ変える。

    語を受け取るのは見出しに `{query}` を持つ道具だけ。見出しが固定の MCP の道具
    （`house_rules`・`journal`）へ `query` を渡すと、サーバーが
    `unexpected keyword argument 'query'` で落ちる（2026-09-13 実機・J と K）。その結果が
    「届いた」ことになり、主LLM が正しい引数で呼び直しても同語二度投げ禁止に止められ、
    上限まで空回りして「見つかりませんでした」と答えた。

    期間を持つ `family_schedule` は、調停が `query` に書いた**日数**を `days` にする
    （候補の文に範囲 1〜14 を書いてある・`arbiter._EXTRA_ACTIONS`）。数字が読めなければ
    `days` を渡さず道具の既定（今日だけ）に任せる——足りなければ主LLM が別の `days` で
    呼び直せる（見出しが期間を含むので止まらない）。「明日の予定」を今日 1 日ぶんで引いて
    答えられなかった 2026-09-14 の実機（出-p）。
    """
    if action == "vault":
        return {"question": query}  # 記録の道具は `question` を受ける（`ask_vault_<latin>`）
    if action in _MCP_LOOKUPS and "{days}" in _MCP_LOOKUPS[action][1]:
        m = re.search(r"\d+", query)
        return {"days": max(1, min(SCHEDULE_DAYS_MAX, int(m.group())))} if m else {}
    if action in _MCP_LOOKUPS and "{query}" not in _MCP_LOOKUPS[action][1]:
        return {}
    return {"query": query}


def _has_image(messages: list) -> bool:
    """本文に画像ブロックが含まれるか（ログ用）。担い手ごとの形の違いは `type` で吸収する。"""
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list) and any(
            isinstance(c, dict) and c.get("type") == "image" for c in content
        ):
            return True
        parts = m.get("parts") if isinstance(m, dict) else None
        if isinstance(parts, list) and any(
            isinstance(c, dict) and "inline_data" in c for c in parts
        ):
            return True
    return False


def _camera_tool_def(agent, name: str) -> list[dict]:
    """カメラの道具定義から1つだけ取り出す。カメラが無ければ空。"""
    cam = getattr(agent, "_camera", None)
    if cam is None:
        return []
    return [d for d in cam.get_tool_definitions() if d.get("name") == name]


#: 思い出し方を変える道具の定義（出-ah・2026-09-21）。**説明文に使いどころを書く**——
#: 道具があっても、いつ使うかが伝わらなければ使われない。軸の名前（重み・床）は見せず、
#: 人の言葉で指させる。上限を超えた指定は丸めるだけで、断らない。
_WIDEN_DEFS: dict[str, dict] = {
    "recall_as": {
        "name": "recall_as",
        "description": (
            "別の人の面で思い出す。**思い出せないときや、その人との話を探したいとき**に使う。"
            "家族はそれぞれ別の記憶の空間を持っていて、ふだんはいま話している相手の面から思い出している。"
            "返ってくるのは、その人との記憶だと分かる形で返る（目の前の相手に話してよいかは自分で考える）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "person": {"type": "string", "description": "誰の面か（家族の呼び方）"},
                "query": {"type": "string", "description": "探す言葉"},
            },
            "required": ["person", "query"],
        },
    },
    "recall_deeper": {
        "name": "recall_deeper",
        "description": (
            "もっとたくさん、別の重みで思い出す。**思い出せないときや、断片しか出てこないとき**に使う。"
            "ふだんは 7 件だが、最大 20 件まで増やせる。"
            "思い出し方は「新しい順に」「印象に残っていることを」「よく思い出すことを」"
            "「この人との関わりで」「話に近いものを」から選ぶ。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "探す言葉"},
                "n": {"type": "integer", "description": "件数（1〜20・省略で 20）"},
                "way": {"type": "string", "description": "思い出し方（省略でふだんのまま）"},
            },
            "required": ["query"],
        },
    },
    "recall_when": {
        "name": "recall_when",
        "description": (
            "その頃のことを思い出す。**思い出せないときや、いつの話か見当がついているとき**に使う。"
            "ふだんは「いま」を基準に思い出している。日付は 2026-08-15 の形で渡す"
            "（「去年の夏」のような言葉は自分で日付に直す）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "探す言葉"},
                "date": {"type": "string", "description": "基準の日（2026-08-15 の形）"},
                "span_days": {"type": "integer", "description": "前後何日まで見るか（省略で 30）"},
            },
            "required": ["query", "date"],
        },
    },
    "recall_tree": {
        "name": "recall_tree",
        "description": (
            "遠い過去を、記憶の木をたどって思い出す。**思い出せないときや、何か月・何年も前の話のとき**に使う。"
            "記憶は年 → 月 → 日の要約になっている。節は年 2025・月 2025-08・日 2025-08-15 の形で、"
            "混ぜていくつでも指せる。まず年か月の要約を読み、気になるところを月や日で指し直す"
            "（「去年の夏」なら 2025-07・2025-08・2025-09）。人を指すとその人との要約、省けば家族全体。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "節（2025・2025-08・2025-08-15 の形）",
                },
                "person": {"type": "string", "description": "誰との要約か（家族の呼び方・省略可）"},
            },
            "required": ["nodes"],
        },
    },
    "recall_recent": {
        "name": "recall_recent",
        "description": (
            "さっきの話をもっと広く読み直す。**思い出せないときや、少し前のやりとりが必要なとき**に使う。"
            "ふだん見えているのは直近 5 分ぶんだけで、最大 30 分・20 往復まで広げられる。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "何分前まで（1〜30・省略で 30）"},
                "turns": {"type": "integer", "description": "何往復まで（1〜20・省略で 20）"},
            },
        },
    },
    # 記録の前後の話をたどる（記-q・2026-10-15・本人の決定ハ・メ）。件数は主LLM が決める（既定は持たない）。
    "recall_thread": {
        "name": "recall_thread",
        "description": (
            "思い出した話の前後をたどる。**「あのあとどうなった？」「その話の続きは？」「その前は何の話だった？」のとき**"
            "に使う。作業状態に載った記録の id（12 桁）を渡すと、その記録から話の続きの鎖を前と後にたどって、古い順に返す。"
            "前に何件・後に何件たどるかは自分で決める（前だけなら after を 0）。どちらも 10 件まで"
            "（それより多く書いても 10 件で止める）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "作業状態に載った記録の id（12 桁）"},
                "before": {"type": "integer", "description": "前に何件たどるか（0〜10）"},
                "after": {"type": "integer", "description": "後に何件たどるか（0〜10）"},
            },
            "required": ["id", "before", "after"],
        },
    },
}


def _widen_lines(ws, k: int) -> str:
    """引き直した記憶を、`recall` と同じ読み方の行にする（出-ah）。載せるのは `k` 件まで。"""
    from .workspace import _lines

    picked = sorted(ws.memories, key=lambda r: r.fit, reverse=True)[:k]
    if not picked:
        return "その引き方では、思い出せるものが無かった"
    names: dict = {}
    with contextlib.suppress(Exception):
        names = dict(ws.oif.actors([r.mi.obs_id for r in picked]))
    return _lines(picked, names)


def _widen_def(name: str) -> dict:
    """思い出し方を変える道具の定義（出-ah）。表から 1 つ返す。"""
    return _WIDEN_DEFS[name]


def _timer_def(agent, name: str) -> list[dict]:
    """タイマーの道具定義から 1 つだけ。器が無ければ空（知-n）。"""
    tool = getattr(agent, "_timer_tool", None)
    if tool is None:
        return []
    return [d for d in tool.get_tool_definitions() if d.get("name") == name]


def _family_and_people(agent) -> str:
    """写真の読み取りへ渡す人の記述。家族に、家族以外で知っている人（`PEOPLE.md`・知-ab）を添える。

    名前を言えるようにするだけで、見立てた人が家族以外なら人物表に無いので話者にはならない（`_apply_seen_people`）。
    """
    family = str(getattr(agent, "_family_md", "") or "")
    people = str(getattr(agent, "_people_md", "") or "")
    return family + ("\n\n[家族以外で知っている人]\n" + people if people.strip() else "")


def _music_def(agent, name: str) -> list[dict]:
    """音楽の道具定義から 1 つだけ。器が無ければ空（知-aa）。"""
    tool = getattr(agent, "_music_tool", None)
    if tool is None:
        return []
    return [d for d in tool.get_tool_definitions() if d.get("name") == name]


def _suggestion_reply_def(agent) -> list[dict]:
    """すすめた曲への返事の道具定義。返事を待っているとき（今日すすめた）だけ（出-ay 段 4-4f・確認待ちと同じ形）。"""
    from ..core import music_suggestion as ms

    try:
        if not ms.awaiting_reply(ms.stored(), today=ms.today()):
            return []
    except Exception:  # noqa: BLE001
        return []
    return _music_def(agent, "music_suggestion_reply")


def _stopwatch_def(agent, name: str) -> list[dict]:
    """ストップウォッチの道具定義から 1 つだけ。器が無ければ空（知-u）。"""
    tool = getattr(agent, "_stopwatch_tool", None)
    if tool is None:
        return []
    return [d for d in tool.get_tool_definitions() if d.get("name") == name]


def _stopwatch_frame(agent) -> str:
    """`[ストップウォッチ]` の枠。器が無い・読めなければ空。"""
    tool = getattr(agent, "_stopwatch_tool", None)
    if tool is None:
        return ""
    try:
        return str(tool.frame() or "")
    except Exception as e:  # noqa: BLE001
        logger.warning("ストップウォッチの枠を組めなかった: %s", e)
        return ""


def _alarm_def(agent, name: str) -> list[dict]:
    """アラームの道具定義から 1 つだけ。器が無ければ空（知-q）。"""
    tool = getattr(agent, "_alarm_tool", None)
    if tool is None:
        return []
    return [d for d in tool.get_tool_definitions() if d.get("name") == name]


def _confirm_defs(agent, name: str) -> list[dict]:
    """確認待ちへの答えの道具定義（出-y）。預かりが生きていなければ空。"""
    from ..core.confirm_state import TOOL_DEFS

    alive = getattr(agent, "confirm_alive", None)
    if alive is None or not alive():
        return []
    return [dict(d) for d in TOOL_DEFS if d["name"] == name]


def _alarm_frame(agent) -> str:
    """`[アラーム]` の枠。器が無い・読めなければ空。"""
    tool = getattr(agent, "_alarm_tool", None)
    if tool is None:
        return ""
    try:
        return str(tool.frame() or "")
    except Exception as e:  # noqa: BLE001
        logger.warning("アラームの枠を組めなかった: %s", e)
        return ""


def _timer_frame(agent) -> str:
    """`[タイマー]` の枠（動いているもの・直前に鳴ったもの）。器が無い・読めなければ空（degrade）。"""
    tool = getattr(agent, "_timer_tool", None)
    if tool is None:
        return ""
    try:
        return str(tool.frame() or "")
    except Exception as e:  # noqa: BLE001
        logger.warning("タイマーの枠を組めなかった: %s", e)
        return ""


def _looks_like_text_tool_call(text: str) -> bool:
    """文で書かれた道具呼び出しらしいか（`<invoke>` か、先頭が `名前(` で末尾が `)`）。"""
    t = text.strip()
    return "<invoke name=" in t or bool(
        re.match(r"^[A-Za-z_][A-Za-z0-9_]*\(", t) and t.endswith(")")
    )


@dataclass
class Decision:
    """主LLM の返りと、**投げたときに見ていたもの**（環-h・段は）。

    主LLM は投げっぱなしになり、返りは別の反復（出す反復）で実行される。そのあいだに別の
    完了が届けば、W も対応表も次の反復のもので作り直される。だから**返りと一緒に運ぶ**。
    に-5-に-2 で対応表は属性でなくなったので、いまは**ここが唯一の持ち場**である。

    - `memories`：共起は「その反復で一緒に活性した記録」なので、**主LLM が見た W** でなければ
      意味がない
    - `w_id_map`：記憶の申告（Jev が答える・記-r）は W に印字された12桁で返るので、その W を作った
      ときの対応表でないと引けない
    - `verdict_view`：**申告を当てる面**（`OIF.feedback` の viewpoint）。`situated_memories` は人ごとで、
      想起は話者の面を通る。基底の記憶へ書くと視点が `__self__` へ寄り、話者が同定されている場面で
      申告が0行に当たる（出-h-ろ ③）。申告は記憶の口（OIF）を通す（環-ab）
    - `system`・`effort`：発話前の検査の差し戻しで**主LLM をもう一度呼ぶ**のに要る
    - `capped`：上限の反復では調べる動作を渡していないので、返ってきても投げない
    - `retried`：これは言い直しの返りか。真なら**もう検査しない**（1回だけ）
    - `original_text`：言い直しが `say` を返さなかったときの戻り先。差し戻しが同期
      だったころは同じ関数の変数だったので運ぶ必要がなかったが、投げっぱなしにすると
      失われる（段は-2）
    """

    result: "TurnResult"
    memories: "list[Recalled]"
    w_id_map: dict[str, str]
    verdict_view: str  # 申告を当てる面（`OIF.feedback` の viewpoint）
    recent_frame: str  # W の直近の枠（発話前の検査が「さっき何を言ったか」として見る）
    system: object
    effort: "str | None"
    capped: bool
    max_tokens: int = 0  # 返事の予算（出-k-ろ）。言い直しも同じ値で呼ぶ
    retried: bool = False
    original_text: str = ""
    #: say の無い文を、say で言い直すよう差し戻した返りか（出-ck）。真ならもう差し戻さない（1 回だけ）。
    say_asked: bool = False


#: **新しい求めを始めるきっかけ**（ほかは、開いている求めの続きとして取り込む）。
_NEW_REQUEST_KINDS = ("会話入力", "機器", "情動")
#: 複数来ていたらこの順に採る。**人の言葉が最優先**である（環-f-い-2）。機器が情動より
#: 先なのは、3本のキューを `asyncio.wait` の union で待っていたころ、取り出したあとの
#: 分岐がこの順だったため（環-f-い-1 でその規則をここへ移した）。
_TRIGGER_PRIORITY = {"会話入力": 3, "機器": 2, "情動": 1}
#: **調査中でも待たせないきっかけ。** 人が言い直したら前の調査は打ち切る側なので、
#: 保留箱へは入れない（`push_utterance` が積む前に打ち切っている）。
_NEVER_HELD_KINDS = ("会話入力",)
#: 目と首。`look` は首を向けたあとその場で 1 枚撮るので、帰りは `see` と同じ扱い
#: （誰が出したかを控え、出した側が判断し、写真と見た印を残す・2026-09-16）。
_CAMERA_ACTIONS = ("see", "look")
#: 長すぎる最終版の要約を主LLM に頼む文（記-t 段 1・本人の決定：O の最大に入らなければフルLLM が要約する）。
#: 時刻の頭は機械が付けるので、ここでは渡さない。
_FINAL_SUMMARY_PROMPT = (
    "次は、家のロボット（パジュ）の 1 回のやりとりの記録である。{n} 字以内の、一続きの日本語に要約する。"
    "何を聞かれたか（何が起きたか）と、何と答えたか（どう終わったか）は、必ず言葉どおりに残す。"
    "調べた結果と見たものは、要点を残す。要約の文だけを返す。\n\n{text}"
)


#: 窓の最終版の要約（環-ar 段 E）。窓の会話は何往復もあるので、1 回のやりとり向けの指示では大半の往復が落ちる。
_WINDOW_SUMMARY_PROMPT = (
    "次は、家のロボット（パジュ）と家族の、ひと続きの会話の記録である。{n} 字以内の、一続きの日本語に要約する。"
    "何をして遊んだか・何を話したか、その流れと終わり方を残す。最後のやりとりは言葉どおりに残す。"
    "調べた結果と見たものは、要点を残す。要約の文だけを返す。\n\n{text}"
)


@dataclass
class _PendingFinal:
    """背景で書く最終版の材料（`_close_final` → `_spawn_final`）。窓の会話の求めなら `talk` 以下を持つ（環-ar 段 E）。"""

    head: str
    body: str
    course: str
    limit: int
    parent: "str | None"
    live: "str | None"
    end: datetime
    start: datetime
    talk: "WindowTalk | None" = None
    #: 閉じた時点の窓の会話の行（背景で書くあいだに行が足されても、閉じた時点のものを書く）。
    lines: "tuple[str, ...]" = ()
    #: 窓の中で貯めた経緯（調べた結果・見たもの・声にならなかった閉じ方）。
    steps: "tuple[str, ...]" = ()


#: 黙る印のまわりに付きうるもの（かぎ括弧・句読点・空白）。
_MARK_TRIM = "「」『』。．.、,!！ \t\n　"


def _is_silence_mark(text: str) -> bool:
    """主LLM の文が、黙る印（`SILENCE_MARK`）で終わるか（出-db 段 F）。

    かぎ括弧や句点を付けて写す・短い理由を前に添える（「テレビの声なので（黙る）」）こともあるので、括弧の全角・半角、
    まわりのかぎ括弧・句読点・空白は問わず、終わりが印なら黙るとみなす。印を声にも画面にも出さない。
    """
    t = (text or "").replace("(", "（").replace(")", "）").strip(_MARK_TRIM)
    return t.endswith(SILENCE_MARK)


def _caches_window(backend) -> bool:
    """窓の会話の行ごとにキャッシュの印を付ける backend か（環-ar 段 D）。ほかは system を 1 本の文にするので守るものが無い。"""
    from ..backends.anthropic import AnthropicBackend

    return isinstance(backend, AnthropicBackend)


#: 窓の会話の中の say だけの回に、毎回変わる部分の最後へ添える言葉（環-ar 段 D）。道具の一覧は変えないので、言葉で伝える。
SAY_ONLY_NOTE = "[この回] say だけを使う。ほかの道具は呼ばない（呼んでも使われない）。"


def _with_say_only_note(system: Any) -> Any:
    """system の最後（毎回変わる部分）に、say だけの回の言葉を添える。前のまとまりには触らない（キャッシュを外さない）。

    言い直しは前の回の system をそのまま使うので、すでに添えてあれば足さない。
    """
    if not isinstance(system, tuple) or not system or SAY_ONLY_NOTE in system[-1]:
        return system
    return (*system[:-1], f"{system[-1]}\n\n{SAY_ONLY_NOTE}" if system[-1] else SAY_ONLY_NOTE)


def _without_silence_note(system: Any) -> Any:
    """言い直しに渡す system から、黙ってよいの指示を外す（出-db 段 F：伝えるのは最初の返事だけ）。"""
    if isinstance(system, tuple):
        return tuple(_without_silence_note(part) for part in system)
    if isinstance(system, str):
        return system.replace("\n" + SILENCE_NOTE, "").replace(SILENCE_NOTE, "")
    return system


@dataclass
class Trigger:
    """**きっかけ**：W 構築を起動する出来事。IIF の待ち行列に並ぶ1件（環-f-い-1）。

    用語一覧の「きっかけ（trigger）」そのものである——**会話入力・知覚イベント・情動発火・
    完了の4つ**。`Completion`（資源から返ってきたもの）という名前だったが、情動と機器も同じ
    列に並ぶようになったので、中身に合う語へ改めた（2026-09-11）。

    以前は `(語, 結果, 意図id, 種別, 番号)` の5つ組で、**位置で意味が決まっていた**。種別に
    よって埋まる欄が違うのに、位置で運んでいた。待たせている時間の知らせ（種別 `進捗`）は、つなぎと一緒に
    撤去した（出-cl 段 2・2026-10-15）。

    | 種別 | いつ | 埋まる欄 |
    |---|---|---|
    | `完了` | 調べものが終わった | `query`・`result`・`index` |
    | `決定` | 主LLM が返った | `decision`（`Decision`） |
    | `情動` | drive が発火した（AIF 経由） | `query`（drive の名）・`result`（促しの文） |
    | `機器` | 人が出入りした（DIF 経由） | `query`（種別）・`result`（中身） |
    | `会話入力` | 人が話しかけた | `query`（人の言葉）・`future`（呼び手が待っている） |

    **`会話入力` だけが返事を持って帰る。** 呼び手（GUI・CUI）はその反復の出力を待っている
    ので、`future` に入れて返す。待っているのは呼び手だけで、**反復は駆動体の上で回る**
    （環-f-い-2）。

    **種別に `発話` を使わない。** その語はコードの中で3つの別物を指している——人が話しかけた
    こと・パジュがつないだ一言・パジュが答えたこと（`direction="発話"` として DB にも入って
    いる）。用語一覧が定めるきっかけの語は `会話入力` である。

    **`決定` は投げたときの W を一緒に運ぶ。** 共起は「その反復で一緒に活性した記録」なので
    主LLM が実際に見た W でなければ意味がなく、記憶の申告（Jev）は W に印字された
    12桁で返るので、その W を作ったときの対応表でないと引けない（`設計方針_主LLMを投げっぱなし
    にする` ②）。求めの寿命の状態に頼ると、飛行中に別の完了が届いたとき上書きされる。
    """

    kind: str
    query: str = ""
    result: str = ""
    intent_id: "str | None" = None
    index: int = 0
    #: 調べものが**道具の失敗**で終わった（出-o）。結果ではなく「その道具はいま使えない」。
    failed: bool = False
    decision: "Decision | None" = None
    # 確かめて掛けたタイマーが鳴った（知-n）。配信ゲートを通り抜ける。
    passes_gate: bool = False
    # `会話入力` だけが使う。呼び手がここで返事を待っている。
    future: "asyncio.Future[str] | None" = None
    # `会話入力` だけが使う。声（`voice`）かキーボード（`keyboard`）か（出-as 段 2）。
    source: str = ""
    # `会話入力` だけが使う。届いた時刻（`time.monotonic()`・出-au 段 1-1）。窓はこの時刻で判定する。
    arrived: float = 0.0
    # `会話入力` だけが使う。名前で呼ばれたか（出-au 段 1-2）。名前がある入力だけが打ち切り、名前の無い
    # 入力は飛行中の調べものがある間は待つ。門が決める。直に作った会話入力（試験）は名前ありとして扱う。
    named: bool = True
    #: 窓の中で 1 字違いの名前だけが当たった声（出-db 段 D）。名前つきには数えず、聞き違えた名前の直しにだけ使う。
    near_name: bool = False
    #: `会話入力` だけが使う。門で受けたときの窓の会話の番号（環-ar 段 B）。窓で受けていなければ 0。番号は門で 1 回だけ
    #: 決め、宛先の判定と求めはこれを使う（後から届いた時刻で引き直すと、門の決め方とずれる）。
    window_session: int = 0
    # `会話入力` だけが使う。その発話の声の特徴（知-ae）。門を通ったら話者を照らし、求めに持たせる。
    voice: Any = None
    # `会話入力` だけが使う。書き起こしの無音らしさと確かさ（名前で起きる基準・2026-10-05）。無ければ None。
    no_speech: "float | None" = None
    logprob: "float | None" = None
    # `会話入力` だけが使う。声で名前だけを聞き、返事をせずに窓を開けた（出-ch 段 3）。画面には「聞いた」と出す。
    quiet_open: bool = False


class InformationProcessing:
    """I：情報処理機構（Information-processing）。③ I 詳細図の器。

    この class が持つのは**装置の寿命**（起動から終了まで）のものだけである——3つのキュー、
    駆動体、外との口（`set_output`・`push_*`・`start`・`close`）、背景タスク。
    それ以外の寿命は別の持ち主にある（環-e-に の に-5-に）。

    | 寿命 | 持ち主 |
    |---|---|
    | 装置：起動〜終了 | **ここ** |
    | 求め：始まり〜閉じる | `loop/request.py` の `Request`（`self._req`） |
    | 反復：1反復 | 引数と返り値で渡す（`loop/workspace.py`） |

    **`_iterate` と `_act_on_decision` はここに残る。** どちらも `_dispatch_main_llm`・
    `_finish`・`_speak`・`_write_version` を呼び返す**殻**で、別 file へ出せば逆参照が要る
    （Functional core / Imperative shell）。

    O・C（Config）・RH 相当のツール実行は、既存実体を持つ `agent` を当面参照する。
    """

    def __init__(self, agent):
        self._agent = agent
        # QC：完了キュー（Trigger Queue）。RH（資源ハンドラ）が書き、LPM が drain する。
        # 要素＝(何を探したか, 結果, 起点の open 意図 id)。意図 id は完了が再会して解決するのに使う。
        self._triggers: asyncio.Queue[Trigger] = asyncio.Queue()
        # 外の機械（声・調べもの）へはこの口だけを通す（環-e-は）。要るものだけを渡す。
        self._dif = DIF(
            tts=agent._tts,
            search=agent._deferred_search,
            fetch=agent._deferred_fetch,
            mcp=agent._mcp,
            # 機器の出来事をこのループへ積む口も持つ（メモの見回り・タイマーが `device()` を呼ぶ・
            # 実機で `'NoneType' object has no attribute 'push_device'`・2026-09-15）。
            ip=self,
        )
        # 声のあいだ音楽を絞る口（知-aa・機械の反射・主LLM は通らない）。器が無ければ挿さない。
        music_tool = getattr(agent, "_music_tool", None)
        if music_tool is not None:
            from .music_watch import duck_while_speaking

            async def _duck(speak, _tool=music_tool, _agent=agent):
                return await duck_while_speaking(
                    io=_tool._io,
                    bus=_tool._bus(),
                    state=getattr(_agent, "_music_state", None),
                    speak=speak,
                )

            self._dif.set_music_ducker(_duck)

        # 直前に書いた版の id。`recall` ツールが自分自身を拾わないための除外に使う。
        #: この反復で道具から返ったもの（環-u）。W を組むときに写す（`just_returned` は
        #: W を組んだ時点で空になるため）。声の選び方がこれを読む。
        self._returned_now: frozenset[str] = frozenset()
        self._recall_exclude_id: str | None = None
        # 求めの世代。打ち切るたびに1つ進める。**走っている反復と、飛んでいる調査の完了**を
        # 古い世代として捨てるのに使う。打ち切りの時点で外部呼び出しは既に飛んでおり、
        # 反復もフルLLM の返りを待っている最中なので、止めるには番号で見分けるしかない。
        self._request_generation = 0
        # see の帰りで起きた反復か。調停を飛ばして主LLM へ戻す（`_decide`）。
        self._see_returned = False
        # 聞き取りの確認の預かり（出-cm）：見立ての動作・元の言葉・聞き返した文・預けた時刻（monotonic）。
        self._heard_check: "dict | None" = None
        self._background_tasks: set[asyncio.Task] = set()
        #: 裏で書いている版と、版を書く列の最後（出-bz-ろ）。版は 1 本の鎖なので、頼まれた順に 1 つずつ書く。
        self._version_writes: "set[asyncio.Future]" = set()
        self._version_tail: "asyncio.Future[str | None] | None" = None
        self._cue_sound: "asyncio.Future[None] | None" = None  # いま鳴らしている合図（出-bd ②）
        # 申告（軽量LLM）は**打ち切っても消さない**ので、`_background_tasks` とは別に持つ。
        # 主LLM の返りは言い直されれば古くなるが、申告は「実際にその記憶を使った」という
        # 事実で、あとから古くならない（出-h-ろ）。
        self._verdict_tasks: set[asyncio.Task] = set()
        # **保留箱**：調査中に届いた情動・機器を避けておく場所（環-f-い-1）。
        #
        # 待ち行列が3本だったころは「調査中は完了キューだけを待つ」ことで、情動と機器を
        # **キューに残して**待たせていた。1本の列は先頭からしか取れないので、取り出して
        # から避ける形へ改めた。**意味は同じ**——取りこぼしではなく待たせるだけである。
        #
        # 打ち切り（言い直し）でも捨てない。捨てるのは完了だけで、情動と機器は別の
        # きっかけであり、言い直しで無かったことにはならない。
        self._held: list[Trigger] = []
        # 黙っているあいだに届いたもの（情-h）。明けた瞬間の 1 つの求めへ渡す。
        self._muted: list[Heard] = []
        self._muted_since: "float | None" = None
        self._pending_heard: list[Heard] = []
        # ウェイクワードの窓（出-as）。声は窓の中でだけ会話として受ける。
        self._wake = WakeWindow()
        # 駆動体（キュー到来で次の反復を起こす）と、そこへ渡す取込待ちの完了。
        self._driver: asyncio.Task | None = None
        self._asyncio_loop: asyncio.AbstractEventLoop | None = None
        self._drained_completions: list[Trigger] = []
        # 開いている意図の台帳（出-cc）。完了は uuid で器を引く。求めをまたぐので装置の寿命で持つ（`_ledger`）。
        self._ledger_obj: IntentLedger = IntentLedger()
        # 求めの寿命の状態は `Request` が持つ（に-5-に）。**`None` にしない**——求めが無い
        # ことは `request_id is None` が表す約束を、そのまま器の中へ持ち込む。
        self._req = Request()
        self._on_text = None
        # 発話の通知先（GUI は「発話は on_action("say") で来る」前提で作られており、
        # 素テキストは say の前の途中経過としてしか扱わない）。CUI は持たない。
        self._on_action = None
        # 求めが開いているかの通知先（GUI の停止ボタンが従う・環-j）。
        self._on_request_state = None
        # 投げる前に控える1件（`_start_lookup` が置き、背景タスクが読む）。
        self._pending_lookup: tuple[str, dict, str] = ("", {}, "recall")

    def _note_origin(self, obs_id: str | None) -> None:
        """このターンの起点を控える。

        **何に続くかはここで決めない。** 継起は、求めを閉じて最終版を書いた後に、最終版どうしで結ぶ
        （記-t 段 3・`_link_succession`）。
        """
        if not obs_id:
            return
        self._note_record(obs_id, "起点")

    def _close_exchange(self) -> "list[tuple[str, str]] | None":
        """いまのやりとりの区間を切り出し、次の始まりを進める。

        母集合への持ち越し（`self._req.turn_records`）はそのまま残す。打ち切った調査と、言い直した
        問いの共起は、たどる価値があるためである。
        """
        members = self._req.turn_records[self._req.exchange_start :]
        self._req.exchange_start = len(self._req.turn_records)
        return members or None

    def _note_record(self, obs_id: str | None, role: str) -> None:
        """このターンが作った記録を、役割つきで控える。

        **一つの並びが二つの用を賄う。** 拡散想起の母集合（共起の関係）へ渡す id と、
        やりとりの関係の項が、どちらもここから出る。別々に持つと、片方へ足し忘れたときに
        気づけない。

        役割は 起点・版・見た・答え。会話要約は背景で遅れて作られるので、ここには来ない
        （`_run_post_response_pipeline` が末尾に足す）。
        """
        if obs_id and all(obs_id != i for i, _ in self._req.turn_records):
            self._req.turn_records.append((obs_id, role))

    async def _write_version(self, *, aborted: bool = False) -> str | None:
        """求めの新しい版を書き、直前の版を畳む。

        求めは1本の版チェーンとして進む。畳むのは版が進んだからで、種類は `改訂` である
        （`設計方針_MI間の関係`）。親子のファンアウトではないので親子をまとめて畳む操作は
        要らない（撤去済み）。

        人の発話の記録と、自分が答えた記録は**鎖の外**にある。畳まない。
        """
        return await self._queue_version(self._version_content(aborted=aborted))

    def _queue_version(self, content: str) -> "asyncio.Future[str | None]":
        """版を書く列に並べる（出-bz-ろ）。版は 1 本の鎖なので、頼まれた順に、前の書き込みが済んでから書く。

        鍵で順に通す形では、裏の仕事が走り出す前に、あとから来た待つ書き込みが先に鍵を取り、順番が崩れた。
        """
        before = self.__dict__.get("_version_tail")  # `__new__` で組んだ装置（試験）でも落ちない

        async def _write() -> "str | None":
            if before is not None:
                await asyncio.gather(before, return_exceptions=True)
            return await self._write_version_body(content)

        task = asyncio.ensure_future(_write())
        self._version_tail = task
        return task

    async def _write_version_body(self, content: str) -> str | None:
        """版を 1 つ書いて、直前の版を畳む（`_write_version` の中身・鍵の中で呼ぶ）。"""
        agent = self._agent
        started = time.monotonic()  # 版を書く（埋め込み・新しさ）にかかった時間を残す（出-bz）
        version_id = await agent._oif.write(
            MI(
                id="",
                content=content[: agent.config.completion_content_max],
                timestamp=None,
                direction="求め",
                parent_id=self._req.request_id,
            ),
            **agent._observation_perspective(),
        )
        logger.info(
            "event-loop 版を書いた %.2f 秒（%d 字）",
            time.monotonic() - started,
            min(len(content), agent.config.completion_content_max),
        )
        if version_id:
            self._note_record(version_id, "版")
            if self._req.live_version_id and self._req.live_version_id != version_id:
                agent._oif.supersede(self._req.live_version_id, version_id, kind=KIND_REVISION)
            self._req.live_version_id = version_id
            # 手がかり（次の反復の想起クエリ）は、いまの版そのものにする。
            self._req.cue = content
        return version_id

    # ── 最終版（記-t 段 1・2026-10-14・本人の決定）──────────────────────────────────────

    async def _close_final(self, outcome: str, text: str, *, aborted: bool = False):
        """閉じるときに最終版を組み、収まればその場で書く。超えれば背景に回す材料を返す（`_spawn_final`）。

        最終版は「何を聞かれ、何を答えたか」の塊で、版の鎖の最後（親は起点・前の版は改訂で畳む・やりとりの「版」の項）。
        閉じた時刻を残すため、収まるものは閉じたその場で書く。

        窓の会話の中の、人との会話の求め（窓の会話の行を足すのと同じ条件）は、いつも背景の列に回す（環-ar 段 E）。
        窓の最終版は前の窓の最終版を畳んで書き直すので、前が書き終わってから順に書く（前が背景で要約中でも
        同じものを二度畳まない）。行にならない求め（タイマーの知らせなど）は窓の最終版に触らない（記録が消える）。
        """
        req, agent = self._req, self._agent
        end = clock.now_utc()
        start = req.began_at or end
        head = final_version.head(start, end)
        body = final_version.body(
            trigger_kind=req.trigger_kind or "発話",
            asked=req.request_text,
            added=[t for _, t in req.added],
            outcome="打ち切り" if aborted else outcome,
            answer=text,
            block_reason=req.block_reason,
        )
        steps = workspace.lookup_lines(req, aborted=aborted, with_main=False)
        steps += [f"見た：{t}" for t in req.seen_texts.values()]
        course = final_version.course(steps)
        limit = int(agent.config.completion_content_max)
        parent, live = req.request_id, req.live_version_id
        talk = self._talk_of(req.window_session)
        if talk is not None and (req.trigger_kind == "発話" or self._talking()):
            # 窓の中の経緯は貯めて、窓の最終版に全部載せる。声にならなかった閉じ方（打ち切った・言えなかった）は
            # 行にならないので、これも経緯に貯める（次の書き直しで消えないように）。
            talk.steps.extend(steps)
            if aborted or outcome != "発話":
                talk.steps.append(body)
            return _PendingFinal(
                head, body, course, limit, parent, live, end, start,
                talk=talk, lines=tuple(talk.lines), steps=tuple(talk.steps),
            )  # fmt: skip
        if len(head + body + course) > limit:
            return _PendingFinal(head, body, course, limit, parent, live, end, start)
        final_id = None
        with contextlib.suppress(Exception):
            final_id = await self._write_final(head + body + course, parent=parent, live=live)
        if final_id:
            self._note_record(final_id, "版")
            req.live_version_id = final_id
            content = head + body + course
            writes = self.__dict__.setdefault("_final_writes", set())
            writes.add(task := asyncio.create_task(self._link_succession(final_id, content, start)))
            task.add_done_callback(writes.discard)
        return None

    async def _write_final(
        self,
        content: str,
        *,
        parent: "str | None",
        live: "str | None",
        at: "datetime | None" = None,
        fold: str = "",
    ) -> "str | None":
        """最終版を 1 件書き、前の版を改訂で畳む。`at` は閉じた時刻（背景で書くときだけ渡す）。

        `fold` は前の窓の最終版（環-ar 段 E）。これも改訂で畳む。
        """
        agent = self._agent
        extra = {"at": at} if at is not None else {}
        final_id = await agent._oif.write(
            MI(id="", content=content, timestamp=None, direction="求め", parent_id=parent),
            **agent._observation_perspective(),
            **extra,
        )
        folded = [
            old
            for old in ((live, fold) if final_id else ())
            if old and old != final_id and agent._oif.supersede(old, final_id, kind=KIND_REVISION)
        ]
        logger.info(
            "event-loop 最終版を書いた（%d 字%s）",
            len(content),
            "・前の窓の最終版を畳んだ" if fold and fold in folded else "",
        )
        return final_id

    def _spawn_final(self, pending: "_PendingFinal", exchange_id: "int | None") -> None:
        """最終版を背景で書く。長すぎれば主LLM に要約させる。時刻の頭は機械が付け、閉じた時刻で書く。

        要約が空・長すぎ・失敗なら、聞かれたことと答えを残して経緯の末尾を切った文で書く（書かないことはしない）。
        窓の最終版は新しい行を残して古い行から落とす。書いたらやりとりの関係の末尾に「版」として足す（閉じたときの
        関係には、まだ入っていない）。

        窓の会話の求め（`pending.talk`）は、窓ごとの列に並べて順に書く（環-ar 段 E）。前の窓の最終版があれば、窓が
        開いた時刻から始まる窓の会話の全部で書き直して前を畳み、続きの辺は聞かずに、窓の 1 つ目の最終版が持っていた
        「前」を付け直す。無ければ（窓の 1 つ目）、いまどおりその求めの文で書き、続きの辺を聞いて「前」を控える。
        """
        p, agent = pending, self._agent
        talk = p.talk
        before = talk.final_tail if talk is not None else None

        async def _run() -> None:
            if before is not None and not before.done():
                # 前を待つだけで、こちらが取り消されても前は取り消さない（`gather` は前まで取り消す）。
                await asyncio.wait({before})
            fold = talk.final_id if talk is not None else ""
            if fold:
                head = final_version.head(talk.opened_at, p.end)  # type: ignore[union-attr]
                body = final_version.window_body(list(p.lines))
                course, prompt = final_version.course(list(p.steps)), _WINDOW_SUMMARY_PROMPT
            else:
                head, body, course, prompt = p.head, p.body, p.course, _FINAL_SUMMARY_PROMPT
            content = head + body + course
            if len(content) > p.limit:
                content = await self._shorten_final(head, body, course, p.limit, prompt, bool(fold))
            final_id = None
            with contextlib.suppress(Exception):
                final_id = await self._write_final(
                    content, parent=p.parent, live=p.live, at=p.end, fold=fold
                )
            if final_id and exchange_id:
                with contextlib.suppress(Exception):
                    agent._oif.extend(exchange_id, [(final_id, "版", None)])
            if not final_id:
                return
            if talk is None:
                await self._link_succession(final_id, content, p.start)
                return
            talk.final_id = final_id  # 窓の次の求めが畳む
            if not fold:
                talk.before_id = await self._link_succession(final_id, content, p.start) or ""
            elif talk.before_id:
                # 畳んだ窓の 1 つ目の最終版が持っていた「前」を、書き直した最終版にも付ける（窓の前とつなぐ）。
                await self._link_succession(final_id, content, p.start, prev=talk.before_id)

        writes = self.__dict__.setdefault("_final_writes", set())
        writes.add(task := asyncio.create_task(_run()))
        task.add_done_callback(writes.discard)
        if talk is not None:
            talk.final_tail = task
            logger.info(
                "event-loop 窓の最終版を列に並べた（前を待つ=%s）",
                "はい" if before is not None and not before.done() else "いいえ",
            )

    async def _shorten_final(
        self, head: str, body: str, course: str, limit: int, prompt: str, window: bool
    ) -> str:
        """長すぎる最終版を主LLM に要約させる。だめなら切る（窓の最終版は古い行から落とす）。"""
        room = max(1, limit - len(head))
        logger.info(
            "event-loop 最終版を要約する（%s・%d 字 → %d 字まで）",
            "窓の会話" if window else "1 回のやりとり",
            len(body + course),
            room,
        )
        summary = ""
        try:
            summary = str(
                await self._agent.backend.complete(prompt.format(n=room, text=body + course), room)
                or ""
            ).strip()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("event-loop 最終版の要約に失敗した（切って書く）", exc_info=True)
        if summary and len(summary) <= room:
            return head + summary
        if window:
            return final_version.cut_window(head, body, course, limit)
        return final_version.cut(head + body + course, limit)

    async def _link_succession(
        self, final_id: str, content: str, start: datetime, *, prev: str = ""
    ) -> "str | None":
        """いま書いた最終版を、その前に閉じた最終版の続きなら継起（前 → 後）で結ぶ（記-t 段 2・背景）。

        結ぶのは最終版どうし（本人の決定）。起点どうしでは、想起しても人の言葉しか出てこず意味のない網になる。
        候補は開始時刻の前 `succession_window_sec`（5 分〔仮〕）のうちに閉じた最終版で、Jev が「どれの続きか」を
        「どれの続きでもない」と並べて選ぶ。候補が無ければ聞かない。失敗・確信度が低いときは書かない。
        結んだ「前」の id を返す。`prev` を渡されたら聞かずにそれと結ぶ（書き直した窓の最終版に、窓の 1 つ目が
        結んでいた「前」を付け直す・環-ar 段 E）。
        """
        agent = self._agent
        try:
            if not prev:
                window = timedelta(seconds=int(agent.config.memory.succession_window_sec))
                rows = agent._oif.recent_finals(start - window, start, exclude_id=final_id)
                if not rows:
                    logger.info("event-loop 最終版の継起を書かない：候補なし")
                    return None
                limit = int(getattr(agent.config, "jev_state_max_chars", 28_000))
                current, cands = final_version.fit(
                    content, {str(r["id"]): str(r["content"]) for r in rows}, limit
                )
                prev = (
                    await jev_judges.judge_succession(
                        getattr(agent, "_jev", None),
                        current=current,
                        candidates=cands,
                        min_conf=float(agent.config.jev_confidence_min),
                    )
                    or ""
                )
                if not prev:
                    logger.info(
                        "event-loop 最終版の継起を書かない：どれの続きでもない（候補 %d 件）",
                        len(cands),
                    )
                    return None
            agent._oif.link(KIND_SUCCESSION, [(prev, "前", 0), (final_id, "後", 1)])
            logger.info("event-loop 最終版の継起を書いた：前 %.8s → 後 %.8s", prev, final_id)
            return prev
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("event-loop 最終版の継起を書けなかった", exc_info=True)
            return None

    async def settle_finals(self) -> None:
        """背景で書いている最終版が書き終わるのを待つ（試験と終了時）。"""
        writes = self.__dict__.get("_final_writes") or set()
        if writes:
            await asyncio.gather(*list(writes), return_exceptions=True)

    def _write_version_later(self) -> None:
        """版を裏で書く（出-bz-ろ・2026-10-13・本人の決定）。書き終わりを待つものが無いときだけ使う。

        主LLM を投げたあとと、主LLM の答えが届いたときの版は、書き終わりを待っているものが無い。10/13 00:35 には
        それぞれ 0.8 秒、駆動体と答えの声を待たせた。調べものの結果が届いたときの版は、直後の想起で使うので待つ。
        """
        # 版の文は頼んだ時点で組む。書く時点で組むと、そのあいだに増えた調べものが前の版に入る。
        task = self._queue_version(self._version_content())
        writes = self.__dict__.setdefault("_version_writes", set())
        writes.add(task)
        task.add_done_callback(writes.discard)

    async def _settle_versions(self) -> None:
        """裏で書いている版が書き終わるのを待つ（求めを閉じる前・出-bz-ろ）。失敗は書いた側が残す。"""
        writes = self.__dict__.get("_version_writes") or set()
        if writes:
            await asyncio.gather(*list(writes), return_exceptions=True)

    async def _write_seen_mark(self, content: str, *, image_path: str | None = None) -> str | None:
        """見たことを O へ書く（`direction="観察"`・鎖の外・畳まない）。

        `image_path` は撮った画像の在りか。**この求めのあいだ主LLM が画像そのものを見る**
        ための手がかりで（`_seen_image`）、印の文（ラベル列）は想起・埋め込みの材料として
        別に残る——「見て語る」と「覚えて探す」は役目が違う（`イベント駆動ループ` v0.43）。

        旧 `run()` がカメラを使ったターンで書いていた記録の続きである（本番に 256 件
        あり、2026-07-24 で途絶えている）。新しいループへ移るとき `camera_used` が
        渡らなくなって書き込みが到達しなくなり、見た印が失われていた。同じ意味の
        記録なので `direction` は分けない。

        旧との違いは、`see` の完了が**どの定点を見たか**を頭に付けることである
        （`_run_camera`）。定点名が入って初めて、W が「次はここを見る番だ」を選べる。

        **この求めのあいだ、W へ浮かせる。** `see` の帰りは版に載せない（同じ出来事が
        2件になって枠を食う）ので、見えたものを持つのはこの記録だけである。役割 `見た` で
        控えると、`workspace.open_ids` が求めのあいだ活性に下限を課して W へ載せる。
        似ている順の採点に任せていたときは載らず、主LLM が `see` を5回出した
        （2026-09-12 実機）。
        """
        agent = self._agent
        obs_id = await agent._oif.write(
            MI(
                id="",
                content=content[: agent.config.completion_content_max],
                timestamp=None,
                direction="観察",
                parent_id=self._req.request_id,
                image_path=image_path,
            ),
            **agent._observation_perspective(),
        )
        # 役割 `見た` で控える。やりとりの関係と共起の材料になり、**`open_ids` が W へ
        # 浮かせる**（docstring）。
        self._note_record(obs_id, "見た")
        if obs_id:
            self._req.seen_texts[obs_id] = content  # 最終版の経緯（記-t）
        if image_path:
            self._req.seen_image_path = image_path
        return obs_id

    def _build_system(
        self,
        *,
        present_ctx: str,
        workspace_ctx: str,
        iter_ctx: str,
        music: str = "",
        window: "tuple[str, ...]" = (),
        skills: str = "",
    ) -> "tuple[str, ...]":
        """主LLM の system 文（安定部・可変部）を組む。材料の出所はここに集める。

        窓の会話の中なら、あいだに見出しと行（`window`）が 1 つずつ入る（環-ar 段 D）。
        """
        from ..capability_state import load_summary

        agent = self._agent
        # `[タイマー]` の枠は反復の枠の後ろに添える（動いているものが無ければ何も足さない・知-n）。
        timers = _timer_frame(agent)
        if timers:
            iter_ctx = (iter_ctx + "\n\n" + timers) if iter_ctx else timers
        alarms = _alarm_frame(agent)  # `[アラーム]` も同じ場所（別物だが置き方は同じ）
        if alarms:
            iter_ctx = (iter_ctx + "\n\n" + alarms) if iter_ctx else alarms
        watches = _stopwatch_frame(agent)  # `[ストップウォッチ]` も（知-u）
        if watches:
            iter_ctx = (iter_ctx + "\n\n" + watches) if iter_ctx else watches
        # `[音楽]` も（知-aa 段 2）。読むのは非同期なので、呼び手が `_music_now` で読んで渡す。
        if music:
            iter_ctx = (iter_ctx + "\n\n" + music) if iter_ctx else music
        suggestion = self._suggestion_frame()  # `[音楽のおすすめ]`（知-aa 段 4）
        if suggestion:
            iter_ctx = (iter_ctx + "\n\n" + suggestion) if iter_ctx else suggestion
        return build_event_system_prompt(
            self_understanding=load_summary() or getattr(agent, "_me_md", ""),
            family_md=getattr(agent, "_family_md", ""),
            people_md=str(getattr(agent, "_people_md", "") or ""),
            family_now=family_now.current_text(),
            self_image=_self_image_text(),
            season_env=_season_env_text(),
            present_ctx=present_ctx,
            pi_ctx=_pi_ctx(self._req),
            iter_ctx=iter_ctx,
            workspace_ctx=workspace_ctx,
            # 角括弧タグを許すかは合成の担い手が決める（`根拠台帳` §9）。
            allow_tts_tags=self._dif.understands_tags,
            origin=self._req.trigger_kind,
            # 見回りの行き先の材料（知-c）。自発の求めだけ。
            now_note=self._patrol_note() if self._req.trigger_kind == "情動" else "",
            window=window,
            skills=skills,
        )

    def _researched(self) -> bool:
        """この求めでネット調査（`search_deferred`／`fetch_deferred`）を投げたか（返事の予算用）。"""
        return any(lk.action in ("search_deferred", "fetch_deferred") for lk in self._req.lookups)

    def _seen_image(self, memories: "list | None" = None) -> "tuple[str, str] | None":
        """この求めで**最後に見た**画像を (base64, 在りか) で返す。無ければ None。

        在りかは求めが持つ（`Request.seen_image_path`・`_write_seen_mark` が置く）。**W に
        載っているかは見ない**——W で探すと、VLM の差し替え（新しい記録）が検索に載る前の
        瞬間に「写真なし」になり、調停が see を出し直した（2026-09-13 実機）。**この求めの
        最新1枚だけ**である（1枚 ≈ 970 トークン。求めが閉じれば要らないし、見直したなら
        新しいほうが正しい）。過去の記憶の画像は添えない。ファイルが消えていれば文字だけで
        進む（落とさない）。
        """
        path = self._req.seen_image_path
        if not path:
            return None
        try:
            data = Path(path).read_bytes()
        except OSError as e:
            logger.warning("event-loop 見た画像を読めないので文字だけで進む: %s", e)
            return None
        return base64.b64encode(data).decode(), path

    def _user_content(self, text: str, memories: "list | None") -> "str | list":
        """主LLM へ渡す本文。この求めで見た画像があれば**画像ブロックを添える**。

        画像を受け取るのは主LLM だけである（調停・発話前の検査・申告の軽量LLM は文字だけ）。
        見ると決めたのは主LLM 自身で、見たものを語るのも主LLM だからである。
        """
        if self._req.trigger_kind == "情動":
            # 自発の求めは「人の言葉」ではない（情-e）。湧いたこととして渡す。
            text = f"[いま湧いたこと] {text}\n（これは自分がしたくなったことで、誰かに頼まれたのではない。許可は要らない。理由も要らない。誰にも断らず、何をするかだけ決める）"
        found = self._seen_image(memories)
        if found is None:
            return text
        if self._req.trigger_kind == "情動" and self._delivery_block_reason():
            # **不在の独り言には写真を添えない**（情-c）。言わない返事に 1 枚 ≈ 970 トークンを
            # 掛けない（実機では主LLM 費の 4 割がここだった）。見た印と即席ラベルは残る。
            logger.info("event-loop 不在の独り言なので写真を添えない")
            return text
        b64, path = found
        logger.info("event-loop 主LLM に画像を添える：%s", Path(path).name)
        return [
            {"type": "text", "text": text},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            },
        ]

    async def _refine_seen_mark(
        self,
        old_id: str,
        *,
        text: str,
        image_b64: str,
        image_path: str | None,
        prefix: str,
        gen: int,
    ) -> None:
        """VLM の意味づけが返ったら、即席の印（YOLO）を**新しい版で差し替える**（背景）。

        差し替えは要約・内省と同じ型——遅れて来た中身が元の記録を supersede する。
        `turn_records` の `見た` の id も新へ差し替えるので、`open_ids` は新を浮かせ、
        `_seen_image` も新から画像を引く。求めがもう別の世代なら O の差し替えだけ行う
        （並びは次の求めのものになっている）。VLM が空・失敗なら何もしない（即席の文が残る）。
        """
        agent = self._agent
        try:
            entities = await extract_entities(text, agent._scene_backend, image_b64=image_b64)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("event-loop 見たものの意味づけに失敗: %s", e)
            return
        labels = [str(d.get("label", "")).strip() for d in entities if d.get("label")]
        if not labels:
            logger.info("event-loop 見たが、意味づけは何も返さなかった（即席の印を残す）")
            return
        logger.info("event-loop 意味づけが返った %d 件：%.60s", len(labels), "、".join(labels))
        new_id = await agent._oif.write(
            MI(
                id="",
                content=f"{prefix}見えたもの：{'、'.join(labels)}"[
                    : agent.config.completion_content_max
                ],
                timestamp=None,
                direction="観察",
                parent_id=self._req.request_id if gen == self._request_generation else None,
                image_path=image_path,
            ),
            **agent._observation_perspective(),
        )
        if not new_id:
            return
        self._revise(old_id, new_id)
        if gen != self._request_generation:
            return
        if (
            self._req.seen_texts.pop(old_id, None) is not None
        ):  # 最終版の経緯も書き直した中身に（記-t）
            self._req.seen_texts[new_id] = f"{prefix}見えたもの：{'、'.join(labels)}"
        self._req.turn_records = [
            (new_id if (i == old_id and r == "見た") else i, r) for i, r in self._req.turn_records
        ]

    def _revise(self, old_id: str, new_id: str) -> None:
        """記録を改めた記録で置き換える（`改訂`）。見えたものの意味づけと、言いたかったことを畳むのが使う。"""
        self._agent._oif.supersede(old_id, new_id, kind=KIND_REVISION)

    async def _fold_told(self, items: "list[tuple[str, str]]") -> None:
        """伝えた言いたかったことを畳み、根づきを普通（0）に戻す（出-as 段 7・本人の決定）。

        「伝えた：…」の記録を書いて元の記録を置き換える。同じことが何度も上がってこないように。
        """
        import re

        for obs_id, content in items:
            body = re.sub(r"^\[[^\]]*\]\s*", "", str(content))  # 想起の頭「[そばに居た] 」を外す
            with contextlib.suppress(Exception):
                new_id = await self._write_origin("独白", f"{unsaid.TOLD}{body}")
                if new_id:
                    self._revise(obs_id, new_id)
                self._agent._oif.set_groundedness(obs_id, 0, lower=True)
                logger.info("event-loop 言いたかったことを伝えたので畳んだ：%.40s", body)

    def _addressee(self) -> "tuple[str | None, str]":
        """言いたかった相手（面を立てる id と、呼び名）。分からなければ (None, "")。"""
        agent = self._agent
        with contextlib.suppress(Exception):
            if agent.speaker_known() and agent._pmm.current_speaker_id:
                return str(agent._pmm.current_speaker_id), self._current_speaker_name()
        return None, ""

    def _version_content(self, *, aborted: bool = False) -> str:
        """いまの求めの状態を、1つの版の content として組み立てる（組み方は `workspace.version_text`・出-bw）。"""
        return workspace.version_text(self._req, aborted=aborted)

    @property
    def _in_flight_count(self) -> int:
        """まだ結果が届いていない調べものの数。**手で数えず、器の列から導く。**"""
        return sum(1 for lk in self._req.lookups if lk.in_flight)

    @property
    def _thinking_capped(self) -> bool:
        """**考えた回数が上限に達したか**（暴走の歯止め・2026-09-12 実機で露見）。

        反復の上限（`event_max_iterations`）は、主LLM の返りで反復が 0 へ戻るので、輪が
        閉じたときには効かない。実機ではカメラが映像を返さず `see` が空で完了し、主LLM が
        「目の前を見る」を57回出し続けた——同じ語は二度と投げず**完了として積む**ので、その
        完了が次の反復を起こし、主LLM が同じ要求を出す。止めたのは人の手だった。

        「考えた回数を材料として渡し、主LLM に判断させる」（環-h）は、この実測で否定された。
        `考え=57回目` と渡していても止まらなかった。見えるはずのものが見えていない状況で
        もう一度見ようとするのは自然な判断で、指示では止まらない。**機械の歯止めを置く。**
        """
        return self._thinking_round >= max(1, self._agent.config.max_thinking_rounds)

    @property
    def _thinking_round(self) -> int:
        """この求めで、主LLM を呼ぶのが何回目か（これから呼ぶ回を含む）。

        主LLM を投げっぱなしにしてからは、返りで反復を 0 へ戻すので `[反復] N/M` は
        常に小さいままになり、**何回目かの手がかりが消えた**。器（`self._req.lookups`）は求めの
        終わりに空になるので、そこの `主LLM` を数えれば求めごとの回数になる。

        機械の歯止めは置かない。**回数を材料として渡し、切り上げるかは判断に任せる。**
        """
        return sum(1 for lk in self._req.lookups if lk.action == "主LLM") + 1

    @property
    def _ledger(self) -> IntentLedger:
        """意図の台帳（出-cc）。殻だけの器（`__new__` で組んだ試験の装置）でも、最初に使うときに作る。"""
        led = self.__dict__.get("_ledger_obj")
        if led is None:
            led = self.__dict__["_ledger_obj"] = IntentLedger()
        return led

    @_ledger.setter
    def _ledger(self, value: IntentLedger) -> None:
        self.__dict__["_ledger_obj"] = value

    def _open_lookup(self, action: str, query: str) -> Lookup:
        """意図の器を作り、求めに足し、台帳に載せる（出-cc）。器を作る口はここだけ。"""
        lk = Lookup(
            index=self._next_lookup_index(),
            action=action,
            query=query,
            generation=self._request_generation,
        )
        self._req.lookups.append(lk)
        self._ledger.open(
            lk, request_id=str(self._req.request_id or ""), origin=self._req.trigger_kind
        )
        return lk

    def _lookup_for(self, c: "Trigger") -> "Lookup | None":
        """完了の器。uuid を持てば台帳で引き（出-cc）、持たなければ語で引く（試験の組み立て）。"""
        if c.intent_id:
            return self._ledger.get(c.intent_id)
        return self._lookup_of(c.query)

    def _drop_orphan_completions(self) -> bool:
        """意図の無い完了を捨てる（出-cc・2026-10-13・本人の決定）。捨てて何も残らなければ真（反復を起こさない）。

        10/13 20:00:33 に、閉じた求めの `look` の完了が取込に数えられ、想起と調停が回ってタイマー中に話した。
        """
        kept = [c for c in self._drained_completions if self._lookup_for(c) is not None]
        for c in self._drained_completions:
            if c not in kept:
                logger.info("event-loop 意図の無い完了を捨てる：%.40s", c.query)
                self._ledger.late(c.intent_id)
        dropped = len(kept) < len(self._drained_completions)
        self._drained_completions[:] = kept
        return dropped and not kept

    def _lookup_of(self, query: str) -> "Lookup | None":
        """語で1件を引く。同じ語は二度投げないので、引き当ては一意になる。"""
        return next((lk for lk in self._req.lookups if lk.query == query), None)

    def _next_lookup_index(self) -> int:
        """この求めの中での通し番号。**器の数から決まる**（別の変数で数えない）。"""
        return len(self._req.lookups) + 1

    def _dispatch_lookup(
        self, action: str, tool_input: dict, query: str, intent_id: str | None
    ) -> None:
        """RH：調べる動作を非同期に実行し、結果を QC へ積む（投げっぱなし・待たない）。"""
        # **この求めで一度調べた語は、二度と調べない。** `deferred` は自前で同じ意図を
        # 止めるが、`recall`・`see`・`look` は素通りで、実機では同じ `recall` を4反復
        # 続けて投げた（語は MD5 まで一致）。4回とも同じ記憶を取ってきて無駄になった。
        # `recall` は DB を引くだけなので、引き直しても結果は変わらず取り直しの意味がない。
        # 止めた調査は完了として積む。投げずに黙って帰ると、完了も時間切れも来ないまま
        # 飛行中の数だけが残り、駆動体が待ち続ける（`deferred` が投げられなかったときと
        # 同じ形にする）。
        seen = self._lookup_of(query)
        if seen is not None:
            logger.info("event-loop すでに調べた語なので投げない：%.40s", query)
            # **器は増やさない。** 同じ語の器が2つできると、語で引いたときどちらが返るか
            # 決まらない。飛行中の数は器から導くので、以前のように数だけ増やして釣り合いを
            # 取る必要もない（環-g・段は で挙動が変わったところ）。完了だけを積む——投げずに
            # 黙って帰ると、完了も時間切れも来ないまま駆動体が待ち続ける。
            #
            # **前に返ったものをそのまま返す**（出-ag-ろ 穴 3）。以前は「結果は W にある」と
            # 返していたが、タイマーの返りの反復は想起をしないので W に無かった。材料の無い W を
            # 見て、調停は掛かっていないタイマーを「セットしました」と言った（実機 2026-09-21 17:34）。
            from ..core.completion_kind import ALREADY_TRIED

            earlier = (
                f"前の結果：{seen.result}" if seen.result is not None else "（まだ返っていない）"
            )
            self._ledger.reopen(
                seen
            )  # 前の意図は閉じているので、台帳に戻して完了を引けるようにする（出-cc）
            self._triggers.put_nowait(
                Trigger(
                    kind="完了",
                    query=query,
                    result=f"「{query}」は{ALREADY_TRIED}。{earlier}",  # 完了の表が見る印
                    intent_id=seen.intent_id,
                    index=seen.index,
                    # 前の失敗の印を引き継ぐ（出-be）。無いと届いた側が「失敗していない」で上書きし、
                    # 「別の曲が鳴っている」の結果を完了の表が「かけた → 黙る」と読んだ（実機 11:05:16）。
                    failed=bool(seen.failed),
                )
            )
            return

        lk = self._open_lookup(action, query)
        # 完了はこの器の uuid を持って戻る（出-cc）。呼び手の `intent_id` はいつも None だった。
        task = asyncio.create_task(
            self._run_lookup(action, tool_input, query, lk.intent_id, lk.index)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        self._ensure_wait_watch()

    def _main_llm_in_flight(self) -> bool:
        """主LLM が考えている最中か（返りがまだ来ていない）。"""
        return any(lk.in_flight and lk.action == "主LLM" for lk in self._req.lookups)

    def _sounding(self) -> bool:
        """合図の声が、いま鳴っているか（鳴っているあいだ待ちの電子音は休む・出-bu）。"""
        cue = getattr(self, "_cue_sound", None)
        return cue is not None and not cue.done()

    def _ensure_wait_watch(self) -> None:
        """待たせている時間の見張りを、**求めごとに 1 本**立てる（出-au 段 2）。立っていれば何もしない。"""
        if self._req.trigger_kind != "発話":
            return  # つなぎは会話の求めだけ（出-aq 段 6）。情動と機器は待たせる相手が居ない
        watch = self.__dict__.get("_wait_watch")
        if watch is not None and not watch.done():
            return
        with contextlib.suppress(RuntimeError):  # 走っている loop が無ければ立てない
            self._wait_watch: "asyncio.Task[None] | None" = asyncio.get_running_loop().create_task(
                self._watch_waiting(self._request_generation)
            )

    def _start_beeps(self) -> None:
        """待ちの電子音の見張りを、求めごとに 1 本立てる（出-bu）。立っていれば何もしない。"""
        beeps = self.__dict__.get("_beeps")
        if beeps is not None and not beeps.done():
            return
        self._beep_count = 0
        with contextlib.suppress(RuntimeError):  # 走っている loop が無ければ立てない
            self._beeps: "asyncio.Task[None] | None" = asyncio.get_running_loop().create_task(
                self._beep_while_waiting(self._request_generation)
            )

    def _stop_beeps(self) -> None:
        """答えの声を出し始めた・求めが閉じた。電子音を止め、鳴らした回数を 1 行残す。"""
        beeps = self.__dict__.get("_beeps")
        if beeps is None:
            return
        self._beeps = None
        if not beeps.done():
            beeps.cancel()
        logger.info("event-loop 待ちの電子音 %d 回", getattr(self, "_beep_count", 0))

    async def _beep_while_waiting(self, gen: int) -> None:
        """合図が鳴り終わってから答えの声が始まるまで、機械音 A を間隔をあけて 1 回ずつ鳴らす（出-bu）。

        10/12 22:15 に、合図「はい」から最初のつなぎまで 13 秒無音だった。音楽が鳴っているあいだも鳴らす（出-cl・2026-10-15・
        本人の決定カ・つなぎの声をやめたので、鳴らさないと待ちのあいだ何も鳴らない）。音楽は絞らない（`dif.beep`）。
        合図の声が鳴っている・準備中のあいだは休む。鳴らすのにかかった時間を引いて、次まで待つ。
        """
        interval = float(getattr(self._agent.config, "wait_beep_interval_seconds", 1.0))
        loop = asyncio.get_running_loop()
        with contextlib.suppress(asyncio.CancelledError):
            while gen == self._request_generation:
                started = loop.time()
                if not self._sounding():
                    with contextlib.suppress(Exception):
                        await self._dif.beep()
                    self._beep_count += 1
                await asyncio.sleep(max(0.0, interval - (loop.time() - started)))

    def _stop_wait_watch(self) -> None:
        """求めが閉じた・打ち切られた。見張りを止める。"""
        self._stop_beeps()  # 待ちの電子音も求めと一緒に止める（出-bu）
        watch = self.__dict__.get("_wait_watch")
        if watch is not None and not watch.done():
            watch.cancel()
        self._wait_watch = None

    def _waiting_on(self) -> "list[Lookup]":
        """待たせているもの（調べものと主LLM）。見張りは会話の求めにだけ立つ（出-aq 段 6）。"""
        return [lk for lk in self._req.lookups if lk.in_flight]

    async def _watch_waiting(self, gen: int) -> None:
        """会話の求めで待たせているあいだ、窓を延ばし続ける（2026-10-07 本人の決定）。

        窓は 10 秒なので、調べものや主LLM を待つあいだに切れると、出来上がった答えが独り言になる。見張りは
        `WAIT_WATCH_TICK_SEC`（1 秒）ごとに起き、待っているものがあれば窓を延ばす（開いているときだけ）。待ちが終わって
        見張りが閉じたら、最後に延ばしたところから 10 秒で閉じる。**会話の求めだけ**（出-aq 段 6）。

        以前は待ちが 5 秒を超えると「まだかかっている」を積み、軽量LLM が書いたつなぎを言っていた。考えているあいだは合図の
        声と待ちの電子音で分かるので、つなぎはやめた（出-cl 段 2・2026-10-15・本人の決定ア）。
        """
        with contextlib.suppress(asyncio.CancelledError):
            while True:
                await asyncio.sleep(WAIT_WATCH_TICK_SEC)
                if gen != self._request_generation:
                    return  # 打ち切られた求めの見張り
                if not self._waiting_on():
                    return  # もう結果が来ている（次に飛ばすときに立て直す）
                self._wake_window().extend(time.monotonic())

    def _dispatch_main_llm(
        self,
        *,
        messages: list,
        system,
        effort: "str | None",
        capped: bool,
        memories: "list[Recalled]",
        w_id_map: dict[str, str],
        verdict_view: str,
        recent_frame: str,
        max_tokens: int,
        retried: bool = False,
        original_text: str = "",
        say_asked: bool = False,
    ) -> None:
        """主LLM を投げる（**待たない**・環-h・段は）。

        調べものと同じ扱いにする——求めの台帳（`self._req.lookups`）へ1件積んで飛行中に数え、版に載せ、世代で
        打ち切れるようにする。**重複の判定は通さない**（同じ求めで何度も呼ぶ）。

        発話前の検査の差し戻し（言い直し）も**この口から投げる**。同じ口を通るので、
        言い直しも `action="主LLM"` として積まれ、**考えた回数に数えられる**
        （2026-09-09 の決定）。
        """
        index = self._next_lookup_index()
        lk = self._open_lookup("主LLM", f"主LLM{index}")
        task = asyncio.create_task(
            self._run_main_llm(
                index=index,
                intent_id=lk.intent_id,
                messages=messages,
                system=system,
                effort=effort,
                capped=capped,
                memories=memories,
                w_id_map=w_id_map,
                verdict_view=verdict_view,
                recent_frame=recent_frame,
                max_tokens=max_tokens,
                retried=retried,
                original_text=original_text,
                say_asked=say_asked,
            )
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        self._ensure_wait_watch()  # 主LLM の待ちも見張る（会話の求めだけ・出-au 段 2）

    async def _run_main_llm(
        self,
        *,
        index: int,
        messages: list,
        system,
        effort: "str | None",
        capped: bool,
        memories: "list[Recalled]",
        w_id_map: dict[str, str],
        verdict_view: str,
        recent_frame: str,
        max_tokens: int,
        retried: bool,
        original_text: str = "",
        intent_id: "str | None" = None,
        say_asked: bool = False,
    ) -> None:
        """RH：主LLM を呼び、返りを QC へ積む（投げっぱなしの担い手）。"""
        started = time.monotonic()
        # 発話だけに絞る理由は2つあり、**別のことである**（1つの名前へまとめない）。
        # `capped`＝連鎖上限なので、調べさせずに必ず閉じる。
        # `retried`＝言い直しなので、答え直すだけでよい（調べ直すためではない）。
        narrow = ("say",)
        say_only = capped or retried or say_asked
        # プロンプトキャッシュを使うか（出-bs 段 2）。層 3 が直近 7 日の続き方で入り切りする。使わないなら固定部分と
        # 道具の印を付けない（組を 1 本の文にすれば、どの backend も印を付けない）。
        cache_on = main_llm_cache_on()
        # 窓の会話の中では、say だけの回も道具の一覧を変えない（環-ar 段 D）。一覧を変えると、道具から後ろ（安定部と
        # 窓の会話）のキャッシュが全部外れる。say だけであることは毎回変わる部分（最後）で伝え、ほかの呼び出しは捨てる。
        # キャッシュを使わない日は守るものが無いので、いまどおり say だけを渡す。
        say_only_in_window = (
            say_only
            and cache_on
            and _caches_window(self._agent.backend)
            and self._talk_of(self._req.window_session) is not None
        )
        if say_only_in_window:
            system = _with_say_only_note(system)
        if not cache_on and isinstance(system, tuple):
            system = system_text(system, sep="\n\n")
        # say で言い直すよう差し戻した呼び出しも、言い直しと同じく say だけを渡す（出-ck）。
        tools = self._tools(
            actions=narrow if say_only and not say_only_in_window else _FULL_ACTIONS,
            cache_tools=cache_on,
        )
        result = await self._main_llm_turn(system, messages, tools, max_tokens, effort)
        if say_only_in_window and any(tc.name != "say" for tc in result.tool_calls):
            # 道具は渡してあるが、この回は say だけ（環-ar 段 D）。ほかの呼び出しは投げずに捨てる。
            kept = [tc for tc in result.tool_calls if tc.name == "say"]
            logger.info(
                "event-loop 主LLM の say だけの回なので道具の呼び出しを捨てた：%s",
                "/".join(tc.name for tc in result.tool_calls if tc.name != "say"),
            )
            result = replace(result, tool_calls=kept)
            if not kept:
                # 捨てたら何も言えない。黙って終わらないよう、say だけを渡してもう一度呼ぶ（キャッシュはこの 1 回だけ外れる）。
                logger.info("event-loop say が無かったので、say だけを渡して呼び直す")
                # この並びは二度と使わないので、道具にキャッシュの印は付けない（書き込みの割増を払わない）。
                tools = self._tools(actions=narrow, cache_tools=False)
                result = await self._main_llm_turn(system, messages, tools, max_tokens, effort)
                # 文で書いた呼び出しを拾い直すと say 以外も来うるので、もう一度ふるう。
                result = replace(
                    result, tool_calls=[tc for tc in result.tool_calls if tc.name == "say"]
                )
        # **生成の秒数は必ず残す**（出-k-い）。何が返ったか（道具・字数・写真の有無）も添える。
        # 所要時間を出していたのは調停だけで、主LLM はログの時刻差から手で引くしかなかった。
        say = next((tc for tc in result.tool_calls if tc.name == "say"), None)
        chars = len(str(say.input.get("text", ""))) if say else len(result.text or "")
        if result.stop_reason == "max_tokens":
            # 予算で切れた。返事は捨てない（`say` が壊れていれば素テキストへ倒れる既存経路）。
            logger.info("event-loop 主LLM の出力が max_tokens=%d で切れた", max_tokens)
        logger.info(
            "event-loop 主LLM %.2f 秒（effort=%s 道具=%s %d 字 写真=%s）",
            time.monotonic() - started,
            effort or "-",
            "/".join(tc.name for tc in result.tool_calls) or "なし",
            chars,
            "あり" if _has_image(messages) else "なし",
        )
        self._triggers.put_nowait(
            Trigger(
                kind="決定",
                query=f"主LLM{index}",
                index=index,
                intent_id=intent_id,
                decision=Decision(
                    result=result,
                    memories=memories,
                    w_id_map=w_id_map,
                    verdict_view=verdict_view,
                    recent_frame=recent_frame,
                    system=system,
                    effort=effort,
                    capped=capped,
                    max_tokens=max_tokens,
                    retried=retried,
                    original_text=original_text,
                    say_asked=say_asked,
                ),
            )
        )

    async def _main_llm_turn(
        self, system, messages: list, tools: list[dict], max_tokens: int, effort
    ):
        """主LLM を 1 回呼ぶ。失敗は空の返りにし、文で書いた道具の呼び出しは拾い直す。"""
        agent = self._agent
        # 当たりは層 3 が時刻と道具の数で数える（使わない週も同じ物差し・出-bs 段 2）。
        measure_record("主LLM", 道具=len(tools))
        try:
            result, _raw = await agent.backend.stream_turn(
                system=system,
                messages=messages,
                tools=tools,
                # 返事の予算（出-k-ろ）。`config.max_tokens`（4096 固定）は使わない。
                max_tokens=max_tokens,
                on_text=None,
                effort=effort,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("event-loop 主LLM の呼び出しに失敗: %s", e)
            from ..backends.types import TurnResult as _TR

            result = _TR(stop_reason="end_turn", text="")
        if not result.tool_calls and _looks_like_text_tool_call(result.text or ""):
            # 道具呼び出しを**文で**書いてきた（Sonnet 5・`<invoke>` 約 50 回に 1 回、関数呼び出し風
            # `say(text="…")` も・2026-09-15）。読めるなら呼び出しに直す。素テキストのままだと
            # 画面と O にそのまま出るか、沈黙になる。
            recovered = tool_calls_from_text(result.text)
            if recovered:
                logger.info(
                    "event-loop 主LLM が道具呼び出しを文で書いたので拾い直す：%s",
                    "/".join(tc.name for tc in recovered),
                )
                result = replace(result, stop_reason="tool_use", text="", tool_calls=recovered)
        return result

    async def _run_lookup(
        self, action: str, tool_input: dict, query: str, intent_id: str | None, index: int = 0
    ) -> None:
        """調べものを実行し、**かかった秒数を必ず残す**（出-k-い）。中身は `_run_lookup_body`。"""
        started = time.monotonic()
        try:
            await self._run_lookup_body(action, tool_input, query, intent_id, index)
        finally:
            # 入力も残す（`days=2`・`query=…`）。無いと「何で呼んだか」が後から分からない。
            logger.info(
                "event-loop 調べもの %s %.2f 秒：%.40s（%s）",
                action,
                time.monotonic() - started,
                query,
                " ".join(f"{k}={str(v)[:40]}" for k, v in tool_input.items()) or "入力なし",
            )

    async def _run_lookup_body(
        self, action: str, tool_input: dict, query: str, intent_id: str | None, index: int = 0
    ) -> None:
        """`recall` は同期で結果が返る。deferred は投げるだけで、完了は自身が QC へ積む。"""
        # **確認待ちのあいだ、掛け直しは機械が落とす**（出-ag-ろ・2026-09-21）。
        # 言葉で頼んでも守られなかった——1 回の「3 分測って」で `set_timer` が 3 回投げられ、
        # 確認待ちを作り直したうえで「セットしました」と言った（実機 17:34・掛かっていない）。
        if self._confirm_blocks(action):
            pc = getattr(self._agent, "_pending_confirm", None)
            what = getattr(pc, "what", "") or "掛けること"
            self._triggers.put_nowait(
                Trigger(
                    kind="完了",
                    query=query,
                    result=f"いま「{what}」でいいか確かめている。先に答えてもらう",
                    intent_id=intent_id,
                    index=index,
                    failed=True,
                )
            )
            return
        if action in ("see", "look"):
            # 飛行中の数は減らさない。`recall` と同じく取込が1件につき1つ減らす。
            out = await self._run_camera(action, tool_input)
            self._triggers.put_nowait(
                Trigger(kind="完了", query=query, result=out, intent_id=intent_id, index=index)
            )
            return
        if (
            action in _TIMER_ACTIONS
            or action in _ALARM_ACTIONS
            or action in _STOPWATCH_ACTIONS
            or action in _CONFIRM_ACTIONS
        ):
            # 呼ぶ前の預かり。**新しい預かりが置かれたか**を、呼んだ後に見比べる。
            before_confirm = getattr(self._agent, "_pending_confirm", None)
            # タイマー（知-n）／アラーム（知-q）／ストップウォッチ（知-u）の道具。その場で返る。落ちても求めは閉じる。
            if action in _STOPWATCH_ACTIONS:
                tool = getattr(self._agent, "_stopwatch_tool", None)
                if tool is None:
                    out, failed = "ストップウォッチの道具が無い", True
                else:
                    out, ok = await tool.call(action, tool_input, now=self._req.began_at)
                    failed = not ok
            elif action in _CONFIRM_ACTIONS:
                # 確認待ちへの答え（出-y）。預かった入力で機械が掛ける／捨てる。起点は「いい」と言った瞬間。
                out, ok = await self._agent.resolve_confirm(
                    action == "confirm", now=self._req.began_at
                )
                failed = not ok
            elif action in _ALARM_ACTIONS:
                tool = getattr(self._agent, "_alarm_tool", None)
                if tool is None:
                    out, failed = "アラームの道具が無い", True
                else:
                    out, ok = await tool.call(action, tool_input)
                    failed = not ok
            else:
                tool = getattr(self._agent, "_timer_tool", None)
                if tool is None:
                    out, failed = "タイマーの道具が無い", True
                else:
                    # 起点は**人が言った瞬間**（調停や主LLM を通る 3 秒は数えない）。
                    out, ok = await tool.call(action, tool_input, now=self._req.began_at)
                    failed = not ok
            # 道具が確認待ちを置いたら、**その場で問いを出す**（出-ag-ろ・2026-09-21）。
            # LLM に任せると、掛けていないのに「掛けた」と言う（実機 17:34）。
            await self._ask_confirm_if_fresh(before_confirm)
            self._triggers.put_nowait(
                Trigger(
                    kind="完了",
                    query=query,
                    result=str(out),
                    intent_id=intent_id,
                    index=index,
                    failed=failed,
                )
            )
            return
        if action in _MCP_LOOKUPS or _personal_action(action):
            # MCP の同期の道具。結果はその場で返るので、完了として積む（`recall` と同じ）。
            # `vault`・`journal` は話者の英字で道具名が決まる（知-g-い・知-az）。主LLM が道具名で呼んだときはそのまま。
            tool_name = (
                action
                if _personal_action(action)
                else (
                    self._personal_tool_name(action)
                    if action in _PERSONAL_PREFIXES.values()
                    else _MCP_LOOKUPS[action][0]
                )
            )
            failed = False
            try:
                if not tool_name:
                    raise RuntimeError("いま話している人の記録の道具が無い")
                out, ok = await self._dif.call_tool(tool_name, tool_input)
                if not ok:
                    # 生の文（`TypeError …`）は対処できる情報ではない。ログにだけ残し、
                    # 人へは「道具が使えなかった」だけを運ぶ（出-o）。
                    logger.warning("event-loop %s が使えなかった：%.200s", tool_name, out)
                    out, failed = f"「{query}」は道具が使えず失敗した", True
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("event-loop %s の実行に失敗: %s", tool_name, e)
                out, failed = f"「{query}」は道具が使えず失敗した", True
            self._triggers.put_nowait(
                Trigger(
                    kind="完了",
                    query=query,
                    result=str(out),
                    intent_id=intent_id,
                    index=index,
                    failed=failed,
                )
            )
            return
        if action in _MUSIC_ACTIONS:
            tool = getattr(self._agent, "_music_tool", None)
            if tool is None:
                out, failed = "音楽の道具が無い", True
            else:
                out, ok = await tool.call(action, tool_input)
                failed = not ok
            self._triggers.put_nowait(
                Trigger(
                    kind="完了",
                    query=query,
                    result=str(out),
                    intent_id=intent_id,
                    index=index,
                    failed=failed,
                )
            )
            return
        if action in _WIDEN_ACTIONS:
            out = await self._run_widen(action, tool_input)
            self._triggers.put_nowait(
                Trigger(kind="完了", query=query, result=out, intent_id=intent_id, index=index)
            )
            return
        if action != "recall":
            try:
                # 投げっぱなしの道具には意図の uuid を渡し、完了と一緒に返してもらう（出-cc）。
                carried = {**tool_input, "_intent_id": intent_id} if intent_id else tool_input
                text, dispatched = await self._dif.lookup(action, carried)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("event-loop %s の実行に失敗: %s", action, e)
                self._triggers.put_nowait(
                    Trigger(
                        kind="完了",
                        query=query,
                        result=f"「{query}」は道具が使えず失敗した",
                        intent_id=intent_id,
                        index=index,
                        failed=True,
                    )
                )
                return
            if not dispatched:
                # 投げられなかった（クエリが空・同時実行の上限・同じ意図が進行中）。背景
                # タスクが無いので完了も時間切れも来ない。ここで閉じないと飛行中の数が
                # 戻らず、駆動体が完了キューだけを待ち続けて何も処理しなくなる。
                logger.info("event-loop %s は投げられなかった：%.60s", action, text)
                self._triggers.put_nowait(
                    Trigger(kind="完了", query=query, result=text, intent_id=intent_id, index=index)
                )
                return
            # 投げられた。**ここで飛行中の数を減らさない。** 減らすと、結果が返る前に
            # 「調査中ではない」ことになり、情動や機器で別の反復が起きて同じ調べものを
            # 投げ直す（実機で1つの求めに検索が4本走り、つなぎを4回喋った）。減らすのは
            # 完了の取込1点に揃える。
            return
        # 自分が出した検索が、いま書いた版そのもの（語を丸ごと含む）を拾わないよう、版が書き終わるのを待って、
        # いまの版で除く（出-bz-ろ）。版は列に並んで書くので、待たないと検索が走る時点で版の id が決まっていない。
        tail = self.__dict__.get("_version_tail")
        if tail is not None:
            with contextlib.suppress(Exception):
                await asyncio.shield(tail)
        exclude = self._req.live_version_id or self._recall_exclude_id
        try:
            out, _ = await self._agent._memory_tool.call(
                "recall",
                tool_input,
                exclude_ids=[exclude] if exclude else None,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("event-loop recall の実行に失敗: %s", e)
            out = f"（recall を実行できなかった：{e}）"
        self._triggers.put_nowait(
            Trigger(kind="完了", query=query, result=str(out), intent_id=intent_id, index=index)
        )
        logger.debug(
            "event-loop RH 完了をQCへ（id=%s qsize=%d 意図=%.8s）",
            id(self),
            self._triggers.qsize(),
            intent_id or "-",
        )

    def _recall_tree(self, tool_input: dict) -> str:
        """記憶の木の節を指して、その節の要約を読む（記-m 段 2）。機械が O から引くだけで、LLM は使わない。

        節ごとに見出し（【2025-08】）を付けて返す。書かれていない節は「まだ要約が無い」と返し、主LLM は
        日を指し直せる。人を指せばその人の面から（家族に無い名前は断る）、省けば家族全体。
        """
        agent = self._agent
        nodes = tool_input.get("nodes") or []
        if isinstance(nodes, str):
            nodes = [nodes]
        name = str(tool_input.get("person") or "").strip()
        pid = None
        if name:
            pid = agent._pmm.find_person_id_by_name(name)
            if pid is None:
                return f"「{name}」は家族に見つからないので、その人の木はたどれない"
        whose = f"（{name}との要約）" if name else "（家族全体の要約）"
        parts: list[str] = []
        for node in [str(n).strip() for n in nodes if str(n).strip()]:
            rows = agent._oif.tree_summaries(node, person_id=pid)
            if rows is None:
                parts.append(f"【{node}】読めない（年 2025・月 2025-08・日 2025-08-15 の形で指す）")
            elif not rows:
                parts.append(f"【{node}】まだ要約が無い")
            else:
                parts.append(f"【{node}】\n" + "\n".join(f"- {r.get('content', '')}" for r in rows))
        if not parts:
            return "節が指されていない（年 2025・月 2025-08・日 2025-08-15 の形で指す）"
        return whose + "\n" + "\n".join(parts)

    def _recall_thread(self, tool_input: dict) -> str:
        """記録の前後の話を継起でたどる（記-q・2026-10-15）。機械が O から引くだけで、LLM は使わない。

        件数は主LLM が決め、前後それぞれ `THREAD_MAX_STEPS`（10 件〔仮〕）で止める。止めたら返りの先頭にそう書く。
        """
        oif = self._agent._oif
        short = str(tool_input.get("id") or "").strip()
        full = oif.id_from_short(short) if short else None
        if full is None:
            return f"id「{short}」の記録は見つからない（作業状態に載った 12 桁の id を渡す）"

        def _steps(key: str) -> int:
            try:
                return max(0, int(tool_input.get(key) or 0))
            except (TypeError, ValueError):
                return 0

        before, after = _steps("before"), _steps("after")
        notes = []
        if before > THREAD_MAX_STEPS:
            notes.append(f"前は {THREAD_MAX_STEPS} 件までにした")
        if after > THREAD_MAX_STEPS:
            notes.append(f"後は {THREAD_MAX_STEPS} 件までにした")
        before, after = min(before, THREAD_MAX_STEPS), min(after, THREAD_MAX_STEPS)
        rows = oif.thread(full, before=before, after=after)
        logger.info("event-loop recall_thread：前 %d・後 %d 件", before, after)
        head = f"（{THREAD_MAX_STEPS} 件まで：" + "・".join(notes) + "）\n" if notes else ""
        if len(rows) <= 1:
            return head + "前後につながる記録が無い"

        def _label(off: int) -> str:
            return "【これ】" if off == 0 else (f"【前 {-off}】" if off < 0 else f"【後 {off}】")

        n_before = sum(1 for r in rows if int(r.get("offset", 0)) < 0)
        n_after = sum(1 for r in rows if int(r.get("offset", 0)) > 0)
        lines = [f"{_label(int(r.get('offset', 0)))}{r.get('content', '')}" for r in rows]
        return head + f"（id {short} の前 {n_before} 件・後 {n_after} 件）\n" + "\n".join(lines)

    async def _run_widen(self, action: str, tool_input: dict) -> str:
        """思い出し方を変えて引き直す（出-ah・2026-09-21）。

        **引くのは既存の口**（`workspace.recall`／`Workspace.recent_text`）で、新しい引き方は
        作らない。動かすのは軸だけ——面・件数と重み・時期と幅・直近の窓。**効き目はこの 1 回だけ**
        で、求めには何も残さない（次の反復の W は元の条件で組む）。上限を超えた指定は丸める。
        """
        from ..config import MemoryConfig
        from ..core import recall_options as ro

        if action == "recall_tree":
            return self._recall_tree(tool_input)
        if action == "recall_thread":
            return self._recall_thread(tool_input)
        agent = self._agent
        cfg = MemoryConfig()
        query = str(tool_input.get("query") or self._req.cue or "")
        viewpoint = agent._pmm.current_speaker_id or AGENT_SELF_ID
        head = ""  # 何を広げたか。**返りだけを見て分かるようにする**（出-ah）
        weights = cfg.recall_weights(self._req.trigger_kind)
        k = cfg.recall_k
        time_ref: "float | None" = None
        span: "float | None" = None

        if action == "recall_as":
            name = str(tool_input.get("person") or "").strip()
            pid = agent._pmm.find_person_id_by_name(name) if name else None
            if pid is None:
                return f"「{name}」は家族に見つからないので、その人の面では思い出せない"
            viewpoint = pid
            head = f"（{agent._pmm.get_person_name(pid) or name}の面で思い出した）\n"
        elif action == "recall_deeper":
            k = ro.clamp_k(tool_input.get("n"))
            way = str(tool_input.get("way") or "").strip()
            weights = ro.weights_for(way, weights)
            head = f"（{k} 件まで" + (f"・{way}" if way in ro.WAYS else "") + "思い出した）\n"
        elif action == "recall_when":
            got = ro.parse_when(str(tool_input.get("date") or ""), tool_input.get("span_days"))
            if got is None:
                return "日付が読めない（2026-08-15 の形で渡す）"
            time_ref, span = got
            head = f"（{str(tool_input.get('date')).strip()} のころ・前後 {int(span)} 日で思い出した）\n"
        else:  # recall_recent——直近のやりとりを広げて読み直す
            minutes, turns = ro.clamp_recent(tool_input.get("minutes"), tool_input.get("turns"))
            ws = await workspace.recall(
                agent._oif,
                query,
                viewpoint=viewpoint,
                weights=weights,
                req=self._req,
            )
            ws.max_age_sec = minutes * 60
            body = ws.recent_text(turns)
            head = f"（直近 {minutes} 分・{turns} 往復まで広げた）\n"
            return head + (body or "そのあいだには、まだ何も無い")

        ws = await workspace.recall(
            agent._oif,
            query,
            viewpoint=viewpoint,
            weights=weights,
            req=self._req,
            time_ref=time_ref,
            time_span_days=span,
        )
        # 母数を添える（出-ah）。取りこぼしがあるかどうかを、返りだけで判断できるようにする。
        tail = f"\n（候補 {cfg.recall_primary_n} 件から {min(k, len(ws.memories))} 件・載せられるのは {k} 件まで）"
        body = _widen_lines(ws, k)
        return f"{head}{body}{tail}"

    async def _run_camera(self, action: str, tool_input: dict) -> str:
        """目と首を動かし、**見えたものを言葉にして**返す。

        `see` が返すテキストは「撮って保存した」と言うだけで、何が写っているかは画像の
        ほうにある。完了キューはテキストしか運ばないので、`知覚在席` §3-2 が定める
        意味づけ（I 側・必要時・VLM）を通す。`look` も向いた先を撮って帰るので同じ経路（2026-09-16）。

        カメラも VLM も落ちる前提の機器なので、例外はここで畳む。見た事実まで失うと
        求めが閉じないまま残る。
        """
        agent = self._agent
        try:
            text, image_b64 = await agent._camera.call(action, tool_input)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("event-loop %s の実行に失敗: %s", action, e)
            return f"（{action} を実行できなかった：{e}）"
        if action not in _CAMERA_ACTIONS or not image_b64:
            return str(text)
        # どの定点を見たかを記録に残す。`ユースケース③` の「見た定点の印」で、これが
        # 根づきで薄れ、W 構築で薄れた順に上がることで巡回が創発する。
        where = await self._current_pose_name()
        prefix = f"{where}を見た。" if where else ""
        image_path = getattr(agent._camera, "last_capture_path", None)
        # **VLM を待たない。** 意味づけ（Gemini へ画像を送る）は実測 2.3 秒で、`see` の帰りの
        # ほぼ全部だった。まずローカルの人検出（YOLO・COCO 80 種・数十ミリ秒）で即席の印を
        # 書いて完了を積み、主LLM は画像そのものを見て話す（`_seen_image`）。VLM は背景で
        # 投げ、返ったら印を差し替える（`_refine_seen_mark`）——記憶の文はそちらの細かさで残る。
        if self._req.see_by == "調停":
            return await self._read_photo_into_state(
                str(text), image_b64=image_b64, image_path=image_path, prefix=prefix
            )
        labels = await self._quick_labels(image_path)
        logger.info(
            "event-loop %s見えたもの（即席）%d 件：%.60s",
            f"{where}で" if where else "",
            len(labels),
            "、".join(labels),
        )
        # 印は**見たことだけ**にする。`see` が返す "You see the current view
        # (saved to …)" は撮ったことを LLM へ伝える文で、見た内容ではない。想起は
        # 印の文でベクトルを作るので、毎回同じ英語の定型句とファイルパスが入ると
        # ノイズになる（実機で観測）。
        mark = f"{prefix}見えたもの：{'、'.join(labels)}" if labels else (prefix or "見た。")
        # **書くのはここである。** 実際にカメラを回した経路だけを通る。取込の側で
        # `action == "see"` を見て書くと、重複抑止で弾かれた完了（「すでに調べた。結果は
        # W にある」）まで印になり、見ていないのに見た印が立つ（実機で観測）。
        obs_id = await self._write_seen_mark(mark, image_path=image_path)
        if obs_id:
            task = asyncio.create_task(
                self._refine_seen_mark(
                    obs_id,
                    text=str(text),
                    image_b64=image_b64,
                    image_path=image_path,
                    prefix=prefix,
                    gen=self._request_generation,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        return f"{prefix}{text} 見えたもの：" + "、".join(labels)

    async def _read_photo_into_state(
        self, text: str, *, image_b64: str, image_path: str | None, prefix: str
    ) -> str:
        """調停が見ると決めた帰り：写真を読むのを**待ち**、結果をシステムの状態として残す（出-au 段 5-7a）。

        Jev は写真を見られないので、調停に写真を渡す形（v0.46）はやめた。見えたものは O の記録と完了の文に（W に載る）、
        写っている人の見立ては顔ぶれに（話者は付けない・知-ai）。読めなければ即席のラベル（YOLO）で書く。
        """
        agent = self._agent
        labels, people = await read_photo(
            image_b64, agent._scene_backend, family_md=_family_and_people(agent)
        )
        if not labels:
            labels = await self._quick_labels(image_path)
        who = "、".join((p["name"] or "誰か") + f"（{p['confidence']:.1f}）" for p in people)
        logger.info("event-loop 写真を読んだ：もの %d 件・人 %d 人", len(labels), len(people))
        mark = f"{prefix}見えたもの：{'、'.join(labels)}" + (
            f"。写っている人：{who}" if who else ""
        )
        await self._write_seen_mark(mark, image_path=image_path)
        if people:
            await self._apply_seen_people(people)
        return f"{prefix}{text} {mark[len(prefix) :]}"

    async def _quick_labels(self, image_path: str | None) -> list[str]:
        """即席の意味づけ（ローカルの人検出モデルで、写っているものの名前）。無ければ空。"""
        detector = getattr(self._agent, "_person_detector", None)
        if detector is None or not image_path:
            return []
        try:
            return list(await detector.labels(image_path))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("event-loop 即席の意味づけに失敗: %s", e)
            return []

    async def _current_pose_name(self) -> str:
        """いま向いている定点の名前。どの定点でもない（移動中）なら空。

        どの定点でもない向きの映像に定点名を付けると、その定点の記録が別の場所の景色で
        汚れる（`知覚在席` §3-3 の振動中ゲートと同じ理由）。
        """
        agent = self._agent
        try:
            camera = getattr(agent, "_camera", None)
            if camera is None:
                return ""
            poses = await agent.poses()
            if not poses:
                return ""
            position = await camera.position()
            if position is None:
                return ""
            pose = nearest_pose(poses, position[0], position[1], agent.config.camera.pose_tolerance)
            return pose.name if pose else ""
        except Exception:  # noqa: BLE001
            logger.debug("いまどの定点を向いているか分からなかった")
            return ""

    async def _intake(self) -> "tuple[int, Decision | None]":
        """取込：駆動体が受けた完了（と QC の残り）を O に書き、open 意図を解決する。

        返りは (取り込んだ件数, 主LLM の決定 or None)。**決定があれば「出す反復」になる**
        （環-h・段は）。決定も器へ入れて飛行中から外すが、**版には中身を載せない**——
        `see` と同じで、中身は `_finish` の「自分が答えた」が持つ。
        """
        # `_drained_completions` は作り直さず中身だけ移す。駆動体は `self._drained_completions.append(await get())` の
        # append を await の前に束縛するので、ここで差し替えると駆動体が捨てられた古い
        # リストへ積み、完了が黙って失われる（実機で観測）。
        # **列には触らない。** 列から取り出す口は `_take_trigger` だけ（2026-09-16）。以前は
        # 3 本のキューだったころの「QC を drain」がそのまま残り、列にいた会話入力を完了として
        # 取ってしまった（Future は誰にも解決されず、GUI が 88 秒待った）。
        items = list(self._drained_completions)
        self._drained_completions.clear()
        logger.debug(
            "event-loop 取込（id=%s items=%d inflight=%d qsize=%d）",
            id(self),
            len(items),
            self._in_flight_count,
            self._triggers.qsize(),
        )

        decided: "Decision | None" = None
        taken = 0
        for c in items:
            if c.kind in _NEW_REQUEST_KINDS:
                # 起きないはずのこと（列の口は 1 つ）。静かに戻さず、大きく失敗させる——
                # 待ち手が居れば例外を返して解放し、無ければ捨てる（次の出入り・発火で再び来る）。
                logger.error("event-loop 取込に %s が紛れた（query=%.40s）", c.kind, c.query)
                if c.future is not None and not c.future.done():
                    c.future.set_exception(RuntimeError(f"取込に{c.kind}が紛れた"))
                continue
            # 届いた結果を器へ入れる。**これで飛行中でなくなる**（数は導出）。
            query, result_text = c.query, c.result
            if c.kind == "決定":
                decided = c.decision
                # 版には返りの中身を載せない（`see` と同じ）。求めの状態としては
                # 「何番が返ったか」だけあればよい。
                result_text = "（返りを実行した）"
            lk = self._lookup_for(c)
            if lk is None:
                # 対応する開いた意図が無い完了（打ち切り後・閉じた後に遅れて届いた等）。駆動体が先に捨てるので
                # ここへは来ないはずだが、来ても数えずに捨てる（出-cc：数えると反復が回り、閉じた求めで話した）。
                logger.warning(
                    "event-loop 取込：対応する意図が無い完了を捨てる（query=%.40s）", query
                )
                self._ledger.late(c.intent_id)
                continue
            taken += 1
            action = lk.action
            if action in _CAMERA_ACTIONS:
                self._see_returned = True  # `look` も向いた先を撮って帰る（`see` と同じ扱い）
                # 版には結果を載せない。見たことは `_run_camera` が鎖の外へ独立した
                # 記録として書いており（会話の「自分が答えた」と同じ位置）、版にも
                # 載せると同じ出来事が2件になって、想起でどちらも上がり W の枠を食う。
                # 求めの状態としては「何番が届いたか」だけあればよい。
                # **W のどこにあるかを言う。**「観察に記録した」だと、主LLM は別の場所を
                # 見ろと読み、W の最下位にある `見えたもの` より重く見て `see` を繰り返した
                # （2026-09-12 実機・5回）。
                result_text = "（見えたものは、作業状態の『わたしが見た』の行にある）"
            if lk is not None:
                lk.result = result_text
                lk.failed = bool(getattr(c, "failed", False))
                if (
                    self._req.trigger_kind == "発話"
                    and not lk.failed
                    and _action_family(action) in _LOOKUP_ACTIONS
                    and action not in _CAMERA_ACTIONS
                ):
                    # 調べた結果は、窓の会話に短い 1 行で足す（環-ar 段 A・全文は W の作業状態）。
                    self._talk_line("わたし", _talk_result(query, result_text))
                self._ledger.close(
                    lk.intent_id, state=_intents.RETURNED, failed=lk.failed, result=c.result
                )
                if lk.index not in self._req.just_returned:
                    self._req.just_returned.append(lk.index)  # W の最上部に載せる（出-x）
                if lk.failed:
                    # その求めのあいだ、この道具を候補から外す（構造で呼び直しを起こさない）。
                    self._req.failed_actions.add(_action_family(lk.action))
        if taken:
            # 求めの新しい版を書き、直前の版を畳む（1本の鎖）。主LLM の答えだけなら、書き終わりを待つものが無いので
            # 裏で書く（答えの声を待たせない・出-bz-ろ）。調べものの結果は直後の想起で使うので待つ（本人）。
            if all(c.kind == "決定" for c in items):
                self._write_version_later()
            else:
                await self._write_version()
        return taken, decided

    # この反復で使える動作の表。値＝その動作のツール定義を取り出す関数で、引数はループ。
    # 身体を1つ繋ぐたびにここへ1行足すだけで済むようにしてある（see・look・net など）。
    # **口が持っている機器の定義は口が答える**（`_dif`）。まだ口を通していないものだけが
    # `ip._agent` を見る（カメラは 段3・記憶は OIF の担当）。
    _ACTIONS: dict = {
        "say": lambda ip: ip._dif.speak_defs(),
        "recall": lambda ip: [
            d for d in ip._agent._memory_tool.get_tool_definitions() if d.get("name") == "recall"
        ],
        # 思い出し方を変えて引き直す（出-ah）。説明文に**使いどころ**を書く——道具があっても、
        # いつ使うかが伝わらなければ使われない。
        "recall_as": lambda ip: [_widen_def("recall_as")],
        "recall_deeper": lambda ip: [_widen_def("recall_deeper")],
        "recall_when": lambda ip: [_widen_def("recall_when")],
        "recall_recent": lambda ip: [_widen_def("recall_recent")],
        "recall_tree": lambda ip: [_widen_def("recall_tree")],
        "recall_thread": lambda ip: [_widen_def("recall_thread")],
        # net（投げっぱなしの外部呼び出し）。結果は完了キュー経由で後の反復に届く。
        "search_deferred": lambda ip: ip._dif.lookup_defs("search_deferred"),
        "fetch_deferred": lambda ip: ip._dif.lookup_defs("fetch_deferred"),
        # 身体。カメラが無ければ空を返し、繋がっていない身体は渡さない。
        "see": lambda ip: _camera_tool_def(ip._agent, "see"),
        "look": lambda ip: _camera_tool_def(ip._agent, "look"),
        # 家の決まり（`obsidian-memo`）。**家族ティアだけ**を載せる——個人ティア
        # （`ask_vault_yusuke`）は話者ゲートができるまで載せない。
        "house_rules": lambda ip: ip._dif.tool_defs("get_house_rules"),
        # 家族の予定（`family-calendar`・知-j）。家族ティアなのでゲート無し。
        "family_schedule": lambda ip: ip._dif.tool_defs("get_family_schedule"),
        # Notion の目次と日次記録（`notion-memo`・知-k）。中身は基本すべて家族ティア。
        "notion_search": lambda ip: ip._dif.tool_defs("search_notion"),
        # 日次記録は 1 人の記録（2026-10-11・知-az）。持ち主があれば `get_journal_<latin>` なので頭で引き、
        # 話者ゲートが本人以外を落とす。持ち主の設定が無い `get_journal` は家族の誰とでも通る。
        "journal": lambda ip: ip._dif.tool_defs_with_prefix("get_journal"),
        # 個人ティアの記録（知-g-い）。`ask_vault_*` を全部出し、話者ゲート（`_gated`）が本人以外を落とす。
        "vault": lambda ip: ip._dif.tool_defs_with_prefix("ask_vault_"),
        # タイマー（知-n）。器（`agent._timer_tool`）が無ければ渡さない。
        "set_timer": lambda ip: _timer_def(ip._agent, "set_timer"),
        "cancel_timer": lambda ip: _timer_def(ip._agent, "cancel_timer"),
        "pause_timer": lambda ip: _timer_def(ip._agent, "pause_timer"),
        "resume_timer": lambda ip: _timer_def(ip._agent, "resume_timer"),
        # アラーム（知-q）。器（`agent._alarm_tool`）が無ければ渡さない。
        "set_alarm": lambda ip: _alarm_def(ip._agent, "set_alarm"),
        "cancel_alarm": lambda ip: _alarm_def(ip._agent, "cancel_alarm"),
        # ストップウォッチ（知-u）。器（`agent._stopwatch_tool`）が無ければ渡さない。
        # 音楽（知-aa）。器が無ければ空（`spotifyd` が居ない機体では渡さない）。
        "play_music": lambda ip: _music_def(ip._agent, "play_music"),
        "stop_music": lambda ip: _music_def(ip._agent, "stop_music"),
        "next_track": lambda ip: _music_def(ip._agent, "next_track"),
        "music_volume": lambda ip: _music_def(ip._agent, "music_volume"),
        "music_suggestion_reply": lambda ip: _suggestion_reply_def(ip._agent),
        "start_stopwatch": lambda ip: _stopwatch_def(ip._agent, "start_stopwatch"),
        "stop_stopwatch": lambda ip: _stopwatch_def(ip._agent, "stop_stopwatch"),
        # 確認待ちへの答え（出-y）。預かりが生きているあいだだけ。
        "confirm": lambda ip: _confirm_defs(ip._agent, "confirm"),
        "decline": lambda ip: _confirm_defs(ip._agent, "decline"),
    }

    def _vault_tool_name(self) -> str:
        """話者の英字から `ask_vault_<latin>` を解く。本人の道具が無ければ空（呼べない）。"""
        return self._personal_tool_name("vault")

    def _personal_tool_name(self, action: str) -> str:
        """話者ゲートを通った道具の名前（`vault` → `ask_vault_<latin>`・`journal` → `get_journal_<latin>`）。

        本人の道具が無ければ空（呼べない）。
        """
        names = {str(d.get("name", "")) for d in self._gated(self._ACTIONS[action](self))}
        return next(iter(sorted(names)), "")

    def _gated(self, defs: list[dict]) -> list[dict]:
        """話者ゲート（知-f）。主LLM の道具と調停の候補は**同じ出口**を通る。"""
        return gate_personal_tools(
            defs,
            speaker=self._current_speaker_name(),
            members=parsing.parse_family_md(
                md if isinstance(md := getattr(self._agent, "_family_md", ""), str) else ""
            ),
        )

    def _extra_actions(self, exclude: frozenset[str] = frozenset()) -> tuple[str, ...]:
        """調停に載せる同期の道具（MCP とタイマー）。繋がっているものだけ、かつこの求めで失敗していないもの。

        `exclude`＝この反復で返ってきた道具（`Workspace.returned_actions`・出-x）。返りが「確かめて」でも
        調停が同じ道具を選び直さないよう候補から外す（掛け直しは「いい」の側で機械が行う）。

        **タイマーが鳴った知らせの求めでは、掛ける・測り始めるを載せない**（2026-09-16 実機 17:00）。
        鳴った知らせに応えてタイマーを掛けることはなく、掛け直しは見出しが毎回違うので同語
        二度投げの守りも効かない（3 本掛け直して 3 回「かしこまりました」）。止めるのは残す。
        """
        timer_ringing = self._req.trigger_kind == "機器" and str(self._req.request_text).startswith(
            "[タイマー]"
        )
        return tuple(
            a
            for a in (
                "house_rules",
                "family_schedule",
                "notion_search",
                "journal",
                "vault",
                *_TIMER_ACTIONS,
                *_ALARM_ACTIONS,
                *_STOPWATCH_ACTIONS,
                *_CONFIRM_ACTIONS,
                # 音楽（知-ak 段 6）。すすめた曲への返事は、返事を待っているときだけ道具の定義が出る（出-ay 段 4-4f）。
                *_MUSIC_ACTIONS,
            )
            if a not in self._req.failed_actions
            and a not in exclude
            and not (timer_ringing and a in ("set_timer", "start_stopwatch"))
            and self._gated(self._ACTIONS[a](self))
        )

    async def _recall_at(self, decision, ws, *, cue, viewpoint, weights):
        """調停が時期を指していたら、その時期を基準に引き直す。指していなければそのまま。

        失敗は飲む（引き直せなくても、いまの基準で引いた W が手元にある）。移したことは
        ログに残す——後から「どの基準で引いた W か」を辿れないと、想起の順を再構成できない。
        """
        if not decision.time_ref:
            return ws
        moved = None
        with contextlib.suppress(Exception):
            moved = await workspace.recall(
                self._agent._oif,
                cue,
                viewpoint=viewpoint,
                weights=weights,
                req=self._req,
                time_ref=datetime.fromisoformat(decision.time_ref).timestamp(),
                time_span_days=decision.time_span_days or None,
            )
            logger.info(
                "event-loop 想起の基準を移す：%s（幅 %s 日）",
                decision.time_ref,
                decision.time_span_days or "既定",
            )
        if moved is None:
            return ws
        # 引き直した W も、同じ窓の会話を持つ（環-ar 段 D：主LLM へ渡す W と system が窓の会話を失わない）。
        try:
            moved.set_window(self._talk_of(self._req.window_session))
        except Exception:  # noqa: BLE001
            logger.warning("event-loop 引き直した W に窓の会話を入れられなかった", exc_info=True)
        return moved

    def _tone_note(self) -> str:
        """いまの相手向けの口調を 1 行で（出-ak・2026-09-23）。分からなければ空。

        **口調の正本は `ME.md`。** ここは「どちらの行を当てるか」だけを決め、行はそのまま
        渡す。写しを持つと、`ME.md` を書き換えても機械の行が古いまま残る。

        規則の側（`personality-from-me`）や顔ぶれの行だけでは効かなかった（実測 0/12）。
        `[返事]` の行に置き、W の過去の発話に相手を添え、顔ぶれに大人と書いて、揃って 11/12。
        """
        from ..core.tone import is_adult, tone_line

        agent = self._agent
        try:
            rows = agent._pmm.presence_status()
        except Exception:  # noqa: BLE001
            return ""
        speaker = next((r for r in rows if r.get("is_speaker")), None)
        if not speaker:
            return ""
        name = str(speaker.get("name") or "")
        adult = is_adult(name, str(getattr(agent, "_family_md", "") or ""))
        if adult is None:
            return ""  # 関係が書かれていない人。決めつけない
        line = tone_line(str(getattr(agent, "_me_md", "") or ""), adult=adult)
        if not line:
            return ""
        who = "大人" if adult else "子ども"
        return f"いま向き合っている相手は{who}。{line}"

    def _missing_tools(self) -> "list[str]":
        """いま使えない道具の呼び名（出-al・2026-09-22）。

        能力の一覧が「道具の名前で有効になる」と書いているもののうち、いま取れないもの。
        **繋がっていない道具は候補から黙って消える**ので、ここで名前を残さないと、
        無いという事実がどこにも書かれない。数えられなければ空（黙る）。
        """
        from ..capability_state import live_tool_names, load_capabilities, missing_tools

        try:
            manifest = load_capabilities()
            if not manifest:
                import pathlib

                manifest = (
                    pathlib.Path(__file__).resolve().parents[3] / "capabilities.yaml"
                ).read_text(encoding="utf-8")
            live = set(live_tool_names(self._agent))
            return [name for _, name in missing_tools(manifest, live)]
        except Exception:  # noqa: BLE001
            logger.debug("使えない道具を数えられなかった（黙る）", exc_info=True)
            return []

    def _tools(
        self, *, actions: tuple[str, ...] = ("say", "recall"), cache_tools: bool = True
    ) -> list[dict]:
        """この反復で使える動作のツール定義を返す。

        表に無い名前は黙って落とす。まだ繋いでいない身体を渡そうとしても壊れないように
        しておく（段階3 の次で see・look・search_deferred を載せる）。

        **道具の定義もキャッシュに載せる**（出-i）。安定部だけ（3,342トークン）では
        Haiku 系の最小長 4,096 に届かず**1回も効かない**。道具（2,329トークン）を載せると
        跨いで**安定部ごと全部が乗り**、1000ターン1反復で 738円 → 366円 になる。

        印は**最後の1つ**に付ける。`cache_control` は「ここまで」を意味する境目なので、
        道具の並びの末尾に付ければ**道具全体とその前の安定部**が範囲に入る。複数付けると
        区切りが増え、書き込みが増える。

        `cache_tools=False` は、**安定部だけで効くモデル**のためにある（`sonnet-5` は
        最小長が低く、道具を載せると読み出し料が増えて 581円 → 635円 と高くなる）。
        """
        defs: list[dict] = []
        # この求めで道具が使えなかった動作（出-o）。求めの器が無い呼び方（土台だけの試験）でも
        # 落ちないよう、無ければ空とみなす。
        failed = getattr(getattr(self, "_req", None), "failed_actions", ())
        for name in actions:
            build = self._ACTIONS.get(name)
            if build is None:
                logger.debug("event-loop 未接続の動作を要求された（無視する）: %s", name)
                continue
            if name in failed:
                # この求めで道具が使えなかった動作は渡さない（出-o・呼び直しを構造で止める）。
                continue
            with contextlib.suppress(Exception):
                defs.extend(build(self))
        # 話者ゲート（知-f）。個人ティアの道具はその人のターン以外では存在しない。
        # `_ACTIONS` にどう足されても、ここを通る。
        defs = self._gated(defs)
        if cache_tools and defs:
            # **共有されている定義を書き換えない。** `get_tool_definitions()` は同じ辞書を
            # 返すことがあり、そこへ印を付けると次に取ったときも残る（`cache_tools=False`
            # が効かなくなる）。最後の1つだけを写して印を付ける。
            defs[-1] = {**defs[-1], "cache_control": {"type": "ephemeral"}}
        return defs

    async def _begin_request(self, *, kind: str, text: str, utterance: str = "") -> None:
        """求めを始める。**3つの入口（会話入力・情動・機器）はここを通る。**

        やることは同じである——求めをリセットし、来た事実を O へ書き、求めの id を置き、
        起点を控え、手がかりを置く。入口ごとに違うのは、起点の種別・文面・`utterance`
        の3つだけなので、それだけを受け取る（`モジュール分割設計` 環-e-に）。

        **鎖は進めない。** 求めの中は版チェーン（`_write_version` の `改訂`）が担い、
        求めをまたいで畳む理由はない（環-g・段に）。
        """
        self._req.utterance = utterance
        # この求めを起こした発話の声（知-ae）。名乗りが話者に付いたら、この声を貯める（段 5）。`_pending_heard` と
        # 同じく脇から受けて空にする（求めの始め方に渡すものは種別・文面・発話の 3 つだけ・環-e-に）。
        self._req.voice = getattr(self, "_pending_voice", None)
        self._pending_voice = None
        self._req.source = getattr(self, "_pending_source", "") if kind == "発話" else ""
        self._pending_source = ""
        self._req.named = bool(getattr(self, "_pending_named", False)) if kind == "発話" else False
        self._pending_named = False
        self._req.cue_rang = False
        self._req.cue_played = False
        self._req.skills = []  # 窓の外で選んだスキルは、その求めのあいだだけ（環-au）
        self._req.trigger_kind = kind
        # 窓の会話の番号は、人の言葉なら門が決めた番号（環-ar 段 B）。始めるまでに窓が切れても、窓で受けた言葉の求めにする。
        # 門を通らない求め（情動・機器）と、門が番号を付けていない言葉は、いま窓が開いているかで決める。
        wake = self._wake_window()
        session = getattr(self, "_pending_window_session", 0) if kind == "発話" else 0
        self._pending_window_session = 0
        self._req.window_session = session or (
            wake.session if wake.is_open(time.monotonic()) else 0
        )
        self._req.began_at = datetime.now(timezone.utc)  # 人が言った瞬間（タイマーの起点・知-n）
        self._req.request_text = text[:500]
        self._req.live_version_id = None
        self._ledger.close_request(self._req.lookups, state=_intents.CLOSED)
        self._req.lookups.clear()
        self._req.iterations = 0
        self._req.iterations_capped = False
        self._req.added = []  # 言い足したことは求めごと（出-au 段 3）
        self._req.seen_texts = {}  # 見たもの・止められた理由も求めごと（記-t の最終版の材料）
        self._req.block_reason = ""
        self._req.draft = ""
        # 明けた瞬間の求めにだけ載る（情-h）。`__new__` で組んだ装置（テスト）でも落ちないよう既定を持つ。
        self._req.heard_while_silent = getattr(self, "_pending_heard", [])
        self._pending_heard = []
        # 確認待ち（出-y）。この求めの W の最上部に載せる。預かりが無い・寿命切れなら空。
        self._req.confirm_frame = self._confirm_frame()
        # **人の言葉は、その人がやったことである。** `actor` の面（`situated_memories`）は
        # 話者に立てる。想起は `_active_memory()`＝話者の面を引くので、`__self__` の面に
        # しか立てないと、**その人の面にはその人が言ったことが1件も無くなる**。
        #
        # 情動と機器はパジュ自身のことなので `__self__` でよい。3つの入口で違うのはここと、
        # 起点の種別・文面・`utterance` である。
        obs_id = await self._write_origin(kind, text)
        self._req.request_id = obs_id
        self._notify_request_state(True)
        # このターンを起こした記録を控え、前のターンとつなぐ。控えないと、問いだけが
        # やりとりの関係にも拡散想起の母集合にも入らない。
        self._note_origin(obs_id)
        self._req.cue = text[:500]

    async def push_utterance(
        self,
        utterance: str,
        on_text=None,
        source: str = "keyboard",
        arrived: "float | None" = None,
        voice: Any = None,
        no_speech: "float | None" = None,
        logprob: "float | None" = None,
    ) -> str:
        """人の言葉を待ち行列へ積み、**その反復の出力**を返す（環-f-い-2）。

        4つのきっかけ（会話入力・知覚イベント・情動発火・完了）が同じ列に並ぶ。積む口が
        `push_completion`・`push_affect`・`push_device` と揃い、**順序づけは
        `_take_trigger()` の1箇所**にある。

        **返すのは求めの終わりではなく、最初の反復の出力である。** 1反復＝1出力なので、
        ツールを投げた反復は発話を持たず空文字を返す。続きは、完了が列へ届いて駆動体が
        起こす次の反復が担う。呼び手（GUI・CUI）が待っているのもここまでで、これは
        `begin_request` だったころと同じ意味である。

        **門はここで、届いた時刻で判定する**（出-au 段 1-2・`設計方針_判定の段` §2.1）。声は門を通ってから、
        初めて何かを起こす。以前は門より前に打ち切りと時刻の印を付けていたので、窓の外で捨てる声（テレビ・
        家族の話）でも、調べもの・考えている途中の主LLM・情動やタイマーの求めが取り消され、「ひとりの回数」が
        0 に戻った。駆動体の中で判定すると、反復を回している間（調停は最大 5 秒）は判定も打ち切りも遅れる。

        **打ち切りは名前がある入力でだけ。** 名前の無い入力は、飛行中の調べものがある間は待たせる
        （`_take_trigger`）。

        `on_text` は出力先（駆動体が起こす反復も使う）。`source` は声（`voice`）かキーボード
        （`keyboard`）か（出-as 段 2）。`arrived` は届いた時刻（`time.monotonic()`・出-au 段 1-1）で、
        無ければいま。
        """
        agent = self._agent
        self._on_text = on_text or self._on_text
        self._ensure_driver()
        # 名前が区切りの直後にあれば、そこから後ろだけを受ける（出-bh・10/11 08:18 に前の会話がくっついた呼びかけを捨てた）。
        from ..core.wake_window import from_name

        cut = from_name(utterance, list(getattr(agent.config, "agent_names", None) or []))
        if cut != utterance:
            logger.info("event-loop 名前より前を落とした（%d 字）", len(utterance) - len(cut))
            utterance = cut
        fut: "asyncio.Future[str]" = asyncio.get_running_loop().create_future()
        trigger = Trigger(
            kind="会話入力",
            query=utterance,
            future=fut,
            source=source,
            arrived=time.monotonic() if arrived is None else arrived,
            voice=voice,
            no_speech=no_speech,
            logprob=logprob,
        )
        if await self._swallow_if_unheard(trigger):
            self._notify_heard(utterance, trigger.quiet_open)
            self._log_swallowed_voice(voice)  # 点数だけ残す（知-as 段 1）
            return ""  # 捨てた・黙っていて控えた。打ち切りも時刻の印も付けない
        shown = utterance
        if trigger.named or trigger.near_name:
            # 窓を開けた聞き違いの名前を直して先へ渡す（窓の中の 1 字違いも・出-db 段 D）。画面には聞こえたままの形に名前を添える（知-ap）。
            from ..core.wake_window import fix_name

            fixed, shown = fix_name(
                utterance, list(getattr(agent.config, "agent_names", None) or [])
            )
            if fixed != utterance:
                logger.info("event-loop 名前の聞き違いを直した：%s", shown[: shown.find("）") + 1])
                trigger.query = fixed
        # 門を通った声だけ、誰が話しているかを照らす（知-ae）。捨てた声は聞いていないのと同じなので照らさない。
        # 吹き出しの名前は知らせた時点の話者を読むので、先に声で話者を決める（10/13 18:56 にこうきの名前で出て、返事はパパ宛てだった）。
        await self._match_voice(voice)
        self._notify_heard(shown, True)
        # 人が話しかけた時刻の印。**在席の証拠には使わない**（2026-09-17：マイクはテレビ・物音・
        # 聞き違いを拾う）。使い道は「ひとりの回数」のリセット（情-d・`step_drives`）と記録。
        agent._last_human_at = time.time()
        if trigger.named:
            # 名前で呼ばれたら、**調べかけを打ち切る**（言い直し・別の用事・「パジュ、止めて」）。結果は
            # 捨てるが、**何を打ち切ったかは記録に残す**。名前の無い入力（「うん」）では止めない。
            await self._abort_lookups()
        self._triggers.put_nowait(trigger)
        return await fut

    async def _abort_lookups(self) -> None:
        """飛行中の調査を打ち切る（人に話しかけられたとき）。

        飛行中のツール呼び出しを止め、まだ取り込んでいない完了を捨て、**何を打ち切ったかを
        O に残す**。打ち切りも版のひとつなので、直前の版を `改訂` で畳む——鎖は「打ち切った」
        1件へ収束する（親と子をまとめて閉じる操作は要らない・`_write_version`）。

        結果を捨てるのは、残すと次の求めの W に無関係な完了が載るためである。ただし
        **打ち切った事実は残す**（あとで「あのとき何を調べていたか」を辿れる）。
        """
        in_flight = [lk for lk in self._req.lookups if lk.in_flight]
        if not self._background_tasks and not in_flight and self._req.request_id is None:
            return
        dropped = [f"{lk.index}番：{lk.action}「{lk.query}」" for lk in in_flight]
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()
        drained = 0
        while not self._triggers.empty():
            item = self._triggers.get_nowait()
            # **捨てるのは完了だけ。** 情動と機器は別のきっかけで、言い直しで無かったことに
            # はならない（キューが3本だったころも、打ち切りは完了キューしか触らなかった）。
            if item.kind in _NEW_REQUEST_KINDS:
                self._held.append(item)
            else:
                drained += 1
        drained += len(self._drained_completions)
        self._drained_completions.clear()

        self._request_generation += 1
        self._stop_wait_watch()
        if dropped or drained:
            logger.info(
                "event-loop 調べかけを打ち切る（%s／取り込まなかった完了 %d件）",
                "・".join(dropped) or "投げた先なし",
                drained,
            )
        if self._req.request_id:
            # 打ち切りも版のひとつ。何を打ち切ったかは版の content が持つので、
            # **飛行中の一覧を消す前**に、かつ親を捨てる前に書く。
            # 打ち切りの版は、最終版の形で書く（記-t 段 1）。
            pending_final = None
            with contextlib.suppress(Exception):
                pending_final = await self._close_final("打ち切り", "", aborted=True)
            # ここまでが一つのやりとりである。打ち切りの版の親は、この求めの起点だから
            # である。閉じないと並びが次のターンへ持ち越され、一つのやりとりに起点が2つ
            # 入る。答えも要約も無いやりとりになるが、それが起きた事実そのものである。
            _aborted = self._close_exchange()
            _xid = None
            if _aborted:
                with contextlib.suppress(Exception):
                    _xid = self._agent._oif.link(
                        KIND_EXCHANGE, [(i, r, n) for n, (i, r) in enumerate(_aborted)]
                    )
            if pending_final is not None:
                self._spawn_final(pending_final, _xid)
        self._req.request_id = None
        self._req.live_version_id = None
        self._ledger.close_request(self._req.lookups, state=_intents.ABORTED)
        self._req.lookups.clear()
        self._req.cue = ""
        self._req.told_unsaid = []
        self._req.added = []
        self._req.draft = ""
        self._req.seen_image_path = None
        self._req.failed_actions.clear()
        self._req.fired_axis = ""

    def _emit(self, text: str) -> None:
        """発話を表示先へ渡す。素テキストと say 動作の両方で知らせる。"""
        if not text:
            return
        if self._on_text is None and self._on_action is None:
            # 出口が無いまま話した。起動直後の自発の求め（メモ・タイマー）で起きた（2026-09-15
            # 21:50）——アプリが `set_output` を起動時に渡していなかった。黙って消えると追えない。
            logger.warning("event-loop 表示先が無いまま話した（%d 字）：%.40s", len(text), text)
        if self._on_text is not None:
            self._on_text(text)
        if self._on_action is not None:
            # 表示先で落ちても発話は止めない。ただし**黙らない**——握りつぶすと
            # 「表示関数に渡したのに画面に無い」を追えない（2026-09-12 実機で露見）。
            try:
                self._on_action("say", {"text": text})
            except Exception:  # noqa: BLE001
                logger.exception("event-loop 表示先が受け取れなかった：%.40s", text)

    def set_request_state_listener(self, listener) -> None:
        """求めが開いた／閉じたを知らせる先を登録する（環-j）。

        GUI の停止ボタンはこれに従う。会話入力の `Future` は最初の反復で解決するので、
        調べものを投げた時点で GUI のターンは終わり、続きの反復（駆動体の上）には待ち手が
        居ない。求めが開いているあいだ停止を有効にしておき、押されたら `abort_current`。
        """
        self._on_request_state = listener

    def set_heard_listener(self, listener) -> None:
        """会話入力を受けた／捨てたを知らせる先を登録する（出-au 段 1-4）。

        画面は、会話ログに出すのを受けた入力だけにし、捨てた入力は状態の行に「聞いていない」と出す。
        以前は書き起こしが確定した時点・打った時点でログに出しており、窓の外の声も「聞いた」ように見えた。
        """
        self._on_heard = listener

    def _notify_heard(self, text: str, heard: bool) -> None:
        listener = getattr(self, "_on_heard", None)  # 殻だけの器（テスト）でも落ちない
        if listener is None:
            return
        try:
            listener(text, heard)
        except Exception:  # noqa: BLE001
            logger.exception("event-loop 受けた／捨てたを知らせられなかった")

    def _notify_request_state(self, open_: bool) -> None:
        listener = getattr(self, "_on_request_state", None)  # 殻だけの器（テスト）でも落ちない
        if listener is None:
            return
        try:
            listener(open_)
        except Exception:  # noqa: BLE001
            logger.exception("event-loop 求めの状態を知らせられなかった")

    async def abort_current(self, *, reason: str = "停止") -> None:
        """いま開いている求めを丸ごと打ち切る（停止ボタン・環-j）。

        話しかけ直しの打ち切り（`_abort_lookups`）と同じ口を使う——飛行中の調べものと
        主LLM の背景タスクを cancel し、未取込の完了を捨て、世代を進め、打ち切った版を
        O に残す。世代が進むので、飛行中だった返りは出す前に畳まれる。**すでに鳴り始めた
        声は止められない**（合成済みの再生は別課題）。
        """
        if self._req.request_id is None and not self._background_tasks:
            return
        logger.info("event-loop %s により求めを打ち切る", reason)
        await self._abort_lookups()
        self._notify_request_state(False)

    def set_output(self, on_text, on_action=None) -> None:
        """発話の表示先を登録する。人の発話を待たずに出口が定まる（起動時にアプリが渡す）。

        `on_action`：GUI の表示経路。ログ表示・ひとりごと判定・音声タグの除去がそちらに
        集まっているので、同じ約束（`("say", {"text": …})`）で通知すればそのまま効く。
        """
        self._on_text = on_text
        self._on_action = on_action or self._on_action

    def start(self) -> None:
        """駆動体だけを起こす。以後はキュー到来で反復が回る。"""
        self._ensure_driver()

    def push_completion(
        self,
        query: str,
        result: str,
        index: int = 0,
        *,
        failed: bool = False,
        intent_id: "str | None" = None,
    ) -> None:
        """RH（資源ハンドラ）が deferred の完了を QC へ積む。

        投げっぱなしの外部呼び出し（検索・取得）の結果は、完了キュー→O 経由で次の反復の
        入力になる（正本③）。スレッドから呼ばれても届くよう、ループへ委譲する。
        """
        # 打ち切った求めの完了は捨てる。外部呼び出しは投げた時点で飛んでおり、止められない。
        # deferred の完了は必ず器（`_dispatch_lookup`）を作ってから飛ぶので、**器の無い完了は
        # 打ち切られたものしか無い**（打ち切りは器を空にする）。世代だけ見ていると器が
        # 消えたあとの完了が素通りし、停止ボタンで打ち切った求めが反復 2/5 として続いた
        # （2026-09-13 実機・M）。
        # uuid を持つ完了は台帳で引く（出-cc）。台帳に無ければ閉じた後に届いたものとして記録して捨てる。
        _lk = self._ledger.get(intent_id) if intent_id else self._lookup_of(query)
        if _lk is None or _lk.generation != self._request_generation:
            logger.info("event-loop 打ち切った求めの完了なので捨てる：%.40s", query)
            self._ledger.late(intent_id)
            return
        loop = getattr(self, "_asyncio_loop", None)
        item = Trigger(
            kind="完了",
            query=query,
            result=str(result),
            index=index or (_lk.index if _lk is not None else 0),
            failed=failed,
            intent_id=intent_id,
        )
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._triggers.put_nowait, item)
        else:
            self._triggers.put_nowait(item)

    def push_affect(self, drive_name: str, prompt: str) -> None:
        """T が drive 発火を待ち行列へ積む（AIF 経由・I は時計を見ない）。"""
        self._triggers.put_nowait(Trigger(kind="情動", query=drive_name, result=prompt))

    async def record_device(self, kind: str, content: str) -> None:
        """機器の出来事を、求めを立てずに記憶へ記録だけする（出-as §2.7・人の出入り・メモ）。

        書き方は知らせの求めが始まるときと同じ（`[種別] 中身`・`direction="機器"`・パジュ自身の面）。
        """
        with contextlib.suppress(Exception):
            await self._write_origin("機器", f"[{kind}] {content[:_DEVICE_MAX]}", max_chars=None)

    async def _write_origin(
        self, kind: str, text: str, *, max_chars: "int | None" = _ORIGIN_MAX
    ) -> str:
        """起点の記録を O へ書く（求めの始まり・`record_device` の 2 つの口が共有する・同じ形で書く）。

        会話は話者の面、情動と機器はパジュ自身の面（`__self__`）。
        """
        agent = self._agent
        perspective = (
            agent._conversation_perspective()
            if kind == "発話"
            else agent._observation_perspective()
        )
        return await agent._oif.write(
            MI(
                id="",
                content=text if max_chars is None else text[:max_chars],
                timestamp=None,
                direction=kind,
            ),
            **perspective,
        )

    def _suggestion_frame(self) -> str:
        """`[音楽のおすすめ]`：話しかけてよいとき（bond・esteem）に勧める材料、勧めた日の会話では返事の受け方。"""
        if getattr(self._agent, "_music_tool", None) is None:
            return ""
        from ..core import music_suggestion as ms

        try:
            return ms.frame(
                ms.stored(),
                today=ms.today(),
                talking=self._talking(),
                conversation=self._req.trigger_kind == "発話",
            )
        except Exception:  # noqa: BLE001
            return ""

    def _note_suggestion_offered(self, said: str) -> None:
        """声にした返事に、すすめたい曲の名前が入っていたら「勧めた」と印をつける（知-aa 段 4）。"""
        if not self._talking() or getattr(self._agent, "_music_tool", None) is None:
            return
        from ..core import music_suggestion as ms

        with contextlib.suppress(Exception):
            s = ms.stored()
            if ms.mark_offered(s, said, today=ms.today()):
                ms.store(s)
                logger.info(
                    "event-loop 音楽をすすめた：%s", s.candidate.title if s.candidate else ""
                )

    async def _judge_companion(self, utterance: str) -> str:
        """相手の言葉から相手の気分を見立てる（出-av）。会話の求めだけで、情動と機器は見立てない。

        判定は Jev（`Evaluator.infer_companion_mood`・出-au 段 5-6）。落ちても主LLM は止めない——
        分からない（`absent`）として、何も置かない。
        """
        if self._req.trigger_kind != "発話" or not utterance:
            return "absent"
        try:
            mood = await self._agent._evaluator.infer_companion_mood(utterance)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("event-loop 相手の気分を見立てられなかった", exc_info=True)
            return "absent"
        return mood if isinstance(mood, str) else "absent"

    async def _music_now(self) -> str:
        """鳴っているあいだの様子を読んで曲送りを記録し（知-aa 段 2）、`[音楽]` の枠を返す。鳴っていなければ「いまは何も鳴っていない」、読めなければ空。

        読むのは T の見張りと同じ口（`music_watch.observe`）。主LLM が「いま何の曲？」に答えられるように。
        """
        tool = getattr(self._agent, "_music_tool", None)
        if tool is None or getattr(self._agent, "_music_state", None) is None:
            return ""
        from ..tools.music import NOT_PLAYING

        status = await self._music_status()
        if status == {}:
            return NOT_PLAYING  # パジュはかけていない（読まない・知-ak 段 5）
        try:
            return await tool.frame(status) if status else ""
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("event-loop 音楽の枠を組めなかった", exc_info=True)
            return ""

    async def _music_status(self) -> "dict | None":
        """いまの音楽の様子（鳴っているか・曲名・アーティスト・音量）。パジュがかけていなければ空の辞書、読めなければ None。

        主LLM の `[音楽]` の枠（`_music_now`）と、調停が「いま何の曲？」に答える文（出-bf・`core/music_now`）が同じ
        読み方を使う。読むのは T の見張りと同じ口（`music_watch.observe`）で、曲送りもここで記録する（知-aa 段 2）。
        """
        agent = self._agent
        tool = getattr(agent, "_music_tool", None)
        state = getattr(agent, "_music_state", None)
        if tool is None or state is None:
            return None
        if not getattr(state, "playing", False):
            return {}
        from .music_watch import observe

        try:
            status = await observe(
                io=tool._io,
                bus=tool._bus(),
                state=state,
                record=lambda text: self.note_device("音楽", text),
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("event-loop 音楽の様子を読めなかった", exc_info=True)
            return None
        return status or None

    def note_device(self, kind: str, content: str) -> None:
        """T から（DIF 経由）：機器の出来事を記録だけする。書くのは非同期なので、タスクとして立てる。"""
        with contextlib.suppress(
            RuntimeError
        ):  # 走っている loop が無ければ記録しない（落とさない）
            task = asyncio.get_running_loop().create_task(self.record_device(kind, content))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    def push_device(self, kind: str, content: str, *, passes_gate: bool = False) -> None:
        """T が人の出入りを待ち行列へ積む（DIF 経由・I は時計を見ない）。`passes_gate` はタイマー（知-n）。"""
        self._triggers.put_nowait(
            Trigger(
                kind="機器",
                query=kind,
                result=content,
                passes_gate=passes_gate,
            )
        )

    def _ensure_driver(self) -> None:
        """駆動体：待ち行列の到来で次の反復を起こす（イベント駆動・時計は見ない）。

        **装置が event loop より長生きすることがある。** `asyncio.run` を2度呼ぶ形（テスト）や、
        GUI が回す loop を作り直す形である。`asyncio.Queue` は**最初に使った loop に縛られる**
        ので、そのままだと次の loop で `is bound to a different event loop` が出続ける。
        loop が変わったら中身を移して作り直す。
        """
        loop = asyncio.get_running_loop()
        if self._asyncio_loop is not None and self._asyncio_loop is not loop:
            logger.debug("event-loop 別の event loop になったので待ち行列を作り直す")
            old, self._triggers = self._triggers, asyncio.Queue()
            while not old.empty():
                self._triggers.put_nowait(old.get_nowait())
            if self._driver is not None:
                self._driver.cancel()
                self._driver = None
        self._asyncio_loop = loop
        if self._driver is None or self._driver.done():
            self._driver = asyncio.create_task(self._drive())

    async def _take_trigger(self) -> "Trigger | None":
        """次に処理するきっかけを1つ返す。`None` は「届いた完了を取り込んで反復する」。

        **順序づけはここ1箇所にある**（環-f-い-1）。以前は3つに散っていた——キューの分け方
        （調査中は完了キューだけを待つ）・取り出したあとの分岐（機器 ＞ 情動 ＞ 完了）・
        受け箱への溜め方である。入口を1つ足すたびに3箇所を触ることになっていた。

        規則は3つ。

        1. **調査中は、新しい求めを始めるきっかけを待たせる**（保留箱へ）。飛行中の調査が
           あるあいだに情動や人の出入りで別の連鎖を始めると、1つの求めの途中に別の話が
           割り込む。聞いている側には、軽量LLM と主LLM が交互に喋る＝別々の人格が居るように
           聞こえる（実機で観測）。**取りこぼしではなく待たせるだけ**である。
        2. **完了は受け箱へまとめて溜める。** 1反復でまとめて取り込む。
        3. **新しい求めを始めるきっかけは1つだけ選ぶ**（会話入力 ＞ 機器 ＞ 情動）。
           **選ばれなかったものは保留箱へ回す。** 3本のキューだったころは、機器と情動が
           同時に届くと情動を取り出したまま黙って捨てていた
           （`if device is not None: … elif affect …`）。

        **優先順位は「同時に届いた中から1つ選ぶとき」の規則である。** 保留箱で待っている
        ものは**待った順に**片づく（待たせたのだから、待った順で出す）。

        **名前の無い会話入力は、飛行中の調べもの（主LLM を含む）がある間は待たせる**（出-au 段 1-3）。
        新しい求めを始めると `_begin_request` が前の求めの状態を空にし、届く結果の行き場が無くなる——打ち切りと
        同じになる。打ち切ってよいのは名前がある入力だけである。前の求めの答えが出てから、別の求めとして扱う
        （段 3 で「前の求めに添える」を入れるまでの既定）。飛行中の印は取り込んだ時点で外れるので、届いた
        結果を取り込む反復は必ず先に回る。
        """
        carry: list[Trigger] = []
        while True:
            # **列から取り出す口はここだけ。** 列を空にして仕分ける（完了は完了箱へ、新しい
            # 求めを始めるものは手元へ）。取込（`_intake`）は列に触らない（2026-09-16）。
            batch = carry + []
            carry = []
            while not self._triggers.empty():
                batch.append(self._triggers.get_nowait())
            fresh = [t for t in batch if t.kind in _NEW_REQUEST_KINDS]
            self._drained_completions.extend(t for t in batch if t.kind not in _NEW_REQUEST_KINDS)
            # ① 会話入力は保留箱より常に先（調査中でも）。保留箱を列より先に返していた近道が、
            #    列で待つ人の言葉より保留箱の入室を先に走らせ、会話入力が別の求めの取込に
            #    横取りされる隙を作った（実機 12:40・88 秒返らず）。
            busy = bool(self._in_flight_count)
            # 主LLM が考えている最中の名前の無い入力は待たせる（返りの時点で考え直すかを決める・段 3）。
            # 調べものだけが飛んでいるなら、取り出して前の求めに添える（`_attach_to_request`）。
            thinking = self._main_llm_in_flight()
            utterance = next(
                (
                    t
                    for t in self._held + fresh
                    if t.kind in _NEVER_HELD_KINDS and (t.named or not thinking)
                ),
                None,
            )
            if utterance is not None:
                if utterance in self._held:
                    self._held.remove(utterance)
                self._held.extend(t for t in fresh if t is not utterance)
                return utterance
            if busy:
                # 調査中は新しい求めを待たせる（名前の無い会話入力も）。取りこぼしではなく待たせるだけ。
                self._held.extend(fresh)
            elif self._held:
                # ② 待たせていたぶんを、待った順に。いま届いたものはその後ろへ。
                self._held.extend(fresh)
                return self._held.pop(0)
            elif fresh:
                # ③ 同時に届いた中から 1 つ選ぶ（機器 ＞ 情動）。選ばれなかったものは保留箱へ。
                chosen = max(fresh, key=lambda t: _TRIGGER_PRIORITY[t.kind])
                self._held.extend(t for t in fresh if t is not chosen)
                return chosen
            if self._drained_completions:
                return None
            # 何も無い（または調査中に情動・機器だけが届いた）。列を待ち、届いたら同じ仕分けへ。
            carry = [await self._triggers.get()]

    async def _drive(self) -> None:
        """待ち行列の到来で次の反復を起こす（イベント駆動・時計は見ない・正本③）。

        待つのは受ける側だけで、時計を持つのは T（自律機構）である。上限は設けない：
        終了は `close()` の cancel が待ちの最中でも即座に効くので、定期的に目を覚ます
        必要がない（目を覚ますこと自体が「時計を見る」動作になる）。

        **待ち行列は1本である**（IIF・環-f-い-1）。会話入力・知覚イベント・情動発火・完了の
        **4つのきっかけが同じ列に並ぶ**（会話入力・知覚イベント・情動発火・完了）。
        """
        while True:
            try:
                trigger = await self._take_trigger()
                # 会話入力は `push_utterance` が届いた時刻で門に通してある（出-au 段 1-2）。
                if (
                    trigger is not None
                    and trigger.kind != "会話入力"
                    and await self._swallow_if_unheard(trigger)
                ):
                    continue
                if self._skip_while_confirming(trigger):
                    continue
                if trigger is None and self._drop_orphan_completions():
                    continue  # 意図の無い完了しか無かった（出-cc）
                if trigger is None:
                    logger.debug(
                        "event-loop 駆動体が完了を受領（id=%s inbox=%d）",
                        id(self),
                        len(self._drained_completions),
                    )
                    await self._iterate()
                elif (
                    trigger.kind == "会話入力"
                    and not trigger.named
                    and self._in_flight_count
                    and not self._confirming()  # 確認待ちのあいだは添えず、3 択の求めにする（知-o）
                ):
                    # 調べもの中の名前の無い入力は、新しい求めを始めず前の求めに添える（出-au 段 3）。
                    await self._attach_to_request(trigger)
                elif trigger.kind == "会話入力":
                    logger.debug("event-loop 駆動体が会話入力を受領")
                    await self._begin_utterance(trigger)
                elif trigger.kind == "機器":
                    logger.debug("event-loop 駆動体が機器を受領（%s）", trigger.query)
                    await self._begin_device(
                        trigger.query,
                        trigger.result,
                        passes_gate=getattr(trigger, "passes_gate", False),
                    )
                else:
                    logger.debug("event-loop 駆動体が情動を受領（%s）", trigger.query)
                    await self._begin_affect(trigger.query, trigger.result)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("event-loop 駆動体で例外: %s", e)
                # **必ず一度譲る。** 待つ前に投げる例外（作り直し前の待ち行列など）だと、
                # ここで譲らなければ await を挟まない密な繰り返しになり、event loop ごと
                # 止まる（実機では固まって見える）。
                await asyncio.sleep(0)

    def _waiting_words(self) -> "list[Trigger]":
        """主LLM が考えているあいだに名前無しで言い足され、保留箱で待っている言葉。"""
        held = self.__dict__.get("_held", [])  # 殻だけの器（テスト）でも落ちない
        return [t for t in held if t.kind == "会話入力" and not t.named]

    async def _rethink_or_speak(self, draft: str) -> str:
        """主LLM の返りが来たとき、考えているあいだに名前無しで言い足された言葉があれば、**考え直す**か
        **そのまま出す**かを Jev に決めさせる（出-au 段 5-1・`設計方針_判定の段` §2.2.3・§2.4）。

        待っている言葉が無ければ聞かない。Jev が使えない・確信度が低いときは倒し先の「そのまま出す」——返りを声にし、
        待っている言葉は答えの後に別の新しい求めになる（`_take_trigger` の保留箱）。
        """
        waiting = self._waiting_words()
        if not waiting:
            return jev_judges.AS_IS
        agent = self._agent
        return await jev_judges.judge_rethink(
            getattr(agent, "_jev", None),
            question=self._req.utterance or self._req.cue,
            draft=draft,
            added=[t.query for t in waiting],
            min_conf=float(getattr(agent.config, "jev_confidence_min", 0.6)),
        )

    async def _rethink(self, draft: str) -> str:
        """考え直す（出-au 段 5-1）。返事は声にせず、待っていた言葉を前の求めに添え、下書きを W に載せて決め直す。"""
        waiting = self._waiting_words()
        for t in waiting:
            self._held.remove(t)
            await self._attach_to_request(t)
        self._req.draft = draft
        logger.info("event-loop 言い足されたことで考え直す（%d 件）", len(waiting))
        return await self._iterate()

    async def _attach_to_request(self, trigger: "Trigger") -> None:
        """名前の無い入力を、調べもの中の求めに添える（出-au 段 3・`設計方針_判定の段` §2.4）。

        新しい求めを始めると、`_begin_request` が前の求めの状態を空にし、届く結果の行き場が無くなる。打ち切って
        よいのは名前で呼ばれたときだけである。O へは人の言葉として書き（話者の面）、やりとりの役割は「添え」
        （起点にすると 1 つのやりとりに起点が 2 つになり、境目が崩れる）。結果が届いた後の決める反復で、W の
        作業状態の枠に「言い足したこと」として載る。待っている呼び手（画面）にはすぐ返す。
        """
        text = trigger.query
        obs_id = ""
        with contextlib.suppress(Exception):
            obs_id = await self._write_origin("発話", text) or ""
        self._note_record(obs_id, "添え")
        self._req.added.append((obs_id, text))
        self._talk_line(
            self._current_speaker_name() or "相手", text, obs_id
        )  # 窓の会話（環-ar 段 A）
        logger.info("event-loop 調べているあいだの言葉を求めに添える：%.40s", text)
        if trigger.future is not None and not trigger.future.done():
            trigger.future.set_result("")

    async def _begin_utterance(self, trigger: "Trigger") -> None:
        """会話入力で新しい連鎖を始め、**その反復の出力を待ち手へ返す**（環-f-い-2）。

        **待ち手が中断したら反復も止める。** GUI の停止ボタンは呼び手のタスクを cancel
        する形で、以前は反復がそのタスクの上で回っていたので cancel がそのまま効いた。
        列へ移すと反復は駆動体の上で回るので、待ち手の cancel を受けてこちらから止める。
        止めたことは駆動体の外へ出さない——駆動体まで畳むと、次のきっかけで起きなくなる。
        """
        fut = trigger.future
        if fut is None or fut.done():
            # 待ち手がもう居ない（中断された・呼び手が消えた）。反復を始める理由がない。
            return
        stopped = False
        self._pending_source = trigger.source  # 声かキーボードか（出-bg：合図は声だけ）
        # 名前で呼ばれたか（出-cx：返事の話し終わりから窓を開ける）
        self._pending_named = trigger.named
        self._pending_window_session = (
            trigger.window_session
        )  # 門が決めた窓の会話の番号（環-ar 段 B）
        task = asyncio.create_task(self._utterance_iteration(trigger.query, trigger.voice))

        def _stop(f: "asyncio.Future[str]") -> None:
            nonlocal stopped
            if f.cancelled() and not task.done():
                stopped = True
                task.cancel()

        fut.add_done_callback(_stop)
        try:
            spoken = await task
        except asyncio.CancelledError:
            if stopped:
                logger.info("event-loop 待ち手が中断したので反復を止めた")
                return
            raise
        except Exception as e:  # noqa: BLE001
            if not fut.done():
                fut.set_exception(e)
            return
        if not fut.done():
            fut.set_result(spoken)
            # **待っている呼び手を、次のきっかけより先に起こす。** 待ち行列が空でなければ
            # `Queue.get()` は譲らずに返るので、譲らなければ駆動体がそのまま次の反復まで
            # 進み、呼び手の `await` はそのあとで再開する。`begin_request` だったころは
            # 反復が呼び手のタスクの上で回っていて、戻った時点が求めの続きより前だった。
            await asyncio.sleep(0)

    async def _utterance_iteration(self, utterance: str, voice: Any = None) -> str:
        """人の言葉を O へ書き、1反復回す。**打ち切りは `push_utterance` が済ませている。**"""
        self._pending_voice = voice
        await self._begin_request(kind="発話", text=utterance, utterance=utterance)
        # 窓の会話（環-ar 段 A）。求めの窓の番号が決まってから足す（前の求めの番号で、閉じた会話に足さない）。
        self._talk_line(
            self._current_speaker_name() or "相手", utterance, str(self._req.request_id or "")
        )
        return await self._iterate()

    async def _begin_affect(self, drive_name: str, prompt: str) -> None:
        """情動で新しい連鎖を始める。取込＝来た事実（情動）を O に書き、鎖の起点にする。

        情動は中身を持たないので、取り込み時に想起で状況づける（正本③ 手順1・2）。
        """
        await self._begin_request(kind="情動", text=f"[内的な促し:{drive_name}] {prompt}")
        self._req.fired_axis = str(drive_name or "").lower()  # 内部状態の言葉で明示する（情-f）
        if self._req.fired_axis == "bond":
            # BOND は想起の手がかりに家族の予定とメモの新しい行を並べ、W を家族へ寄せる（情-q）。O の起点は変えない。
            from ..core.bond_cue import cue_for
            from . import notes_watch, schedule_watch

            with contextlib.suppress(Exception):
                self._req.cue = cue_for(
                    self._req.cue, schedule_watch.stored() or "", notes_watch.last_added()
                )
        await self._iterate()

    async def _begin_device(self, kind: str, content: str, *, passes_gate: bool = False) -> None:
        """機器（タイマー・アラームなど）で新しい連鎖を始める。取込＝来た事実を O に書き、鎖の起点にする。

        `passes_gate` は確かめて掛けたタイマー（知-n）。
        """
        await self._begin_request(kind="機器", text=f"[{kind}] {content}")
        self._req.passes_gate = bool(passes_gate)
        await self._iterate()

    async def close(self) -> None:
        # **待たせたままの呼び手を残さない。** 列と保留箱に会話入力が残っていると、
        # `push_utterance` の `await` が永久に返らない（環-f-い-2）。
        for item in list(self._held) + list(self._triggers._queue):  # type: ignore[attr-defined]
            if item.future is not None and not item.future.done():
                item.future.cancel()
        if self._driver is not None:
            self._driver.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._driver
            self._driver = None
        self._stop_wait_watch()

    async def _iterate(self) -> str:
        """1反復：取込 → W 構築 → 生成 → 出力（発話 or ツール投げ）で終わる。

        本体は `loop/iteration.Iteration`（メソッドオブジェクト・環-ab E）。1 反復の状態と段はそちらが持つ。
        """
        from .iteration import Iteration

        return await Iteration(self).run()

    async def _decide(
        self,
        *,
        utterance: str,
        ws: "workspace.Workspace",
        present_ctx: str,
        capped: bool,
        round_: int,
    ) -> "ArbiterDecision":
        """この反復の分岐を決める。**see の帰りは、出した側が判断する。**

        `ws`＝この反復で組んだ W。調停へ渡す写し・返ったもの・断られたときに縮める口を、ここで取り出す
        （出-bm・反復の本体を短く保つため W ごと受け取る）。

        `ws.returned_actions`＝この反復で道具から返ったもの（出-x）。タイマー・
        アラームの帰りなら、調停の先導文を「返りを受ける」型にし、返った道具を候補から外す。

        主LLM が見ると決めたなら、その続きは主LLM（調停を飛ばす）。画像を受け取るのも主LLM
        だけである（`_user_content`）。ここで軽量LLM に判定し直させると、`light` を選んで
        **画像を見ずに**即席のラベル文だけで答えたり、`recall` へ逸れたりした（2026-09-12
        実機）。判定し直す理由が無いうえ、調停の約 1 秒も消える。思考の深さは see を出した
        ときの値を引き継ぐ。

        調停が見ると決めたなら（v0.44）、帰りも調停が判断する。**そのときは調停にも写真を
        渡す**（v0.46）。即席のラベル（YOLO）はこの部屋で `bench` 1 語になり、材料不足で毎回
        full へ倒れた（実機）。写真があれば `light` で「机と椅子が見えます」と答えられる。
        """
        # 窓が開いていれば、窓の会話の全部を渡す。Jev は W の直近の枠の代わりに、書き手は起点の言葉のすぐ前に、主LLM は
        # system の安定部の後に置き、書き手と主LLM の W からは直近の枠を外す（環-ar 段 B・C・D）。主LLM へ戻る see の帰りも
        # 同じ W を使うので、早く戻る道より前で入れる。
        ws.set_window(self._talk_of(self._req.window_session))
        if ws.window_talk is not None:
            logger.debug(
                "event-loop 窓の会話 %d 行（%d 字）:\n%s",
                ws.window_talk.count("\n"),
                len(ws.window_talk),
                ws.window_talk,
            )
        if self._see_returned:
            self._see_returned = False
            if self._req.see_by == "主LLM":
                logger.info("event-loop 主LLM が出した see の帰りなので調停を飛ばして主LLM へ戻す")
                # 見えたものを語るだけなので low（課題5 G 章「ループ側で決まる effort」）。
                return ArbiterDecision(branch="full", effort="low")
            # 写真は渡さない（出-au 段 5-7a）。読み取りは状態として W に載っている（`_read_photo_into_state`）。
            logger.info("event-loop 調停が出した see の帰りなので調停が判断する（読み取りは W に）")
        # W の全文は DEBUG。調停が何を見て選んだかは、これが無いと後から追えない（Jev に聞いた回の文は arbiter_records にある）。
        writer_w = ws.for_arbiter
        logger.debug("event-loop 調停へ渡す W:\n%s", writer_w)
        # 判定は Jev、文章は要るときだけ軽量LLM（出-au 段 5-7・`Arbiter`）。
        # 声で話しかけた求めの最初の判断では、聞こえた合図（機械音 A）を鳴らし続け、決まったら反応の合図に替える（出-bg）。
        cue = self._start_reaction_cue(ws.returned_lookups)
        heard = self._take_heard_check(ws)
        try:
            decision = await self._arbiter().decide(
                self._arbiter_input(
                    utterance=utterance,
                    workspace_ctx=writer_w,  # 狭い窓（記-h）。窓の中では [直近のやりとり] を外す
                    # 書き手の文で起点の言葉のすぐ前に置く（環-ar 段 C）
                    window_ctx=ws.window_talk or "",
                    present_ctx=present_ctx,
                    capped=capped,
                    round_=round_,
                    returned=ws.returned_actions,
                    returned_lookups=ws.returned_lookups,
                    # Jev へ送る W と [いま湧いたこと] は、重なりを除いて測る（出-bw）。書き手は Jev の字数の上限で記憶を削らない W を見る。
                    jev_workspace=getattr(ws, "for_jev_arbiter", None),
                    jev_utterance=(
                        ws.jev_cue()
                        if utterance and utterance == self._req.cue and hasattr(ws, "jev_cue")
                        else ""
                    ),
                    heard=heard,
                    # パジュ用のスキル（環-au）。判断のたびに読み直す（直したらすぐ効く）。読むのはループの外で。
                    # 問いを聞くのは人の言葉の求めの最初の判断（道具の返りではない回）だけ。載っているものがあれば、書き手の枠の
                    # ためにも読む。
                    skills=await self._load_skills()
                    if (self._req.trigger_kind == "発話" and not ws.returned_lookups)
                    or self._active_skills()
                    else (),
                    active_skills=self._active_skills(),
                ),
                on_decided=cue.decided if cue is not None else None,
            )
            self._keep_skills(decision.skills)
        finally:
            if cue is not None:
                cue.stop()  # 調停が落ちても・時間切れでも、機械音 A は必ず止める
        self._note_confirm_prompt(decision)
        self._keep_heard_check(decision)
        # 何を選んだかは INFO（出-k-い の材料。DEBUG では実機で見えなかった）。
        logger.info(
            "event-loop 調停=%s effort=%s action=%s",
            decision.branch,
            decision.effort,
            decision.action if decision.branch == "action" else "-",
        )
        return decision

    def _start_reaction_cue(self, returned_lookups) -> "_ReactionCue | None":
        """声で話しかけた求めの最初の判断なら、機械音 A を鳴らし始める（出-bg・2026-10-10 本人の決定）。

        キーボード・情動・機器・道具の返りの反復では鳴らさない。求めに 1 回だけ。
        """
        req = self._req
        if req.trigger_kind != "発話" or req.source != "voice" or returned_lookups or req.cue_rang:
            return None
        req.cue_rang = True
        try:
            thinking = self._dif.start_thinking()
        except Exception:  # noqa: BLE001
            logger.debug("event-loop 機械音を鳴らせなかった", exc_info=True)
            return None
        return _ReactionCue(self, thinking)

    def _arbiter(self) -> Arbiter:
        agent = self._agent
        return Arbiter(jev=getattr(agent, "_jev", None), writer=agent._utility_backend)

    def _arbiter_input(
        self,
        *,
        utterance: str,
        workspace_ctx: str,
        present_ctx: str,
        capped: bool = False,
        round_: int = 1,
        returned: frozenset[str] = frozenset(),
        returned_lookups: "tuple[tuple[str, bool, str], ...]" = (),
        jev_workspace: "Callable[[int], str | None] | None" = None,
        jev_utterance: str = "",
        heard: "dict | None" = None,
        window_ctx: str = "",
        skills: "tuple[skills_mod.Skill, ...]" = (),
        active_skills: "tuple[str, ...]" = (),
    ) -> ArbiterInput:
        """調停へ渡す材料。判定（`_decide`）が見る。"""
        from ..capability_state import load_summary

        agent = self._agent
        silence_note = self._silence_note()
        return ArbiterInput(
            utterance=utterance,
            workspace_ctx=workspace_ctx,
            # 文脈に合わない言葉の読み直し（出-db 段 B）。キーボードで打った言葉は話しかけなので、名前が無くても
            # 名前つきと同じに読む（本人の決定ア）。窓の開け閉めには使わない。
            named=self._req.named or self._req.source == "keyboard",
            # 軽く返すか黙るかの判定に渡す（出-db 段 E）。人の言葉の求めのときだけ読む。
            agent_names=tuple(getattr(agent.config, "agent_names", None) or ()),
            recent_ctx=self._recent_for_judges()
            if self._req.trigger_kind == "発話" and not returned_lookups
            else "",
            typed=self._req.source == "keyboard",
            self_understanding=load_summary() or getattr(agent, "_me_md", ""),
            family_md=getattr(agent, "_family_md", ""),
            people_md=str(getattr(agent, "_people_md", "") or ""),
            family_now=family_now.current_text(),
            self_image=_self_image_text(),
            season_env=_season_env_text(),
            present_ctx=present_ctx,
            now_ctx=f'(now :datetime "{clock.now_local_str()}")'
            + silence_note
            + (self._patrol_note() if self._req.trigger_kind == "情動" else ""),
            capped=capped,
            thinking_round=round_,
            can_see=getattr(agent, "_camera", None) is not None,
            origin=self._req.trigger_kind,
            extra_actions=self._extra_actions(exclude=returned),
            tool_return=bool(returned & workspace.RETURN_WITHOUT_RECALL),
            # 完了を機械で分ける（出-ay 段 4-2）。主LLM は外部の道具を道具の名前（`get_family_schedule`）で呼ぶので、
            # 完了の表が引く動作の名前（`family_schedule`）に揃える（段 5b）。揃えないと表に当たらなかった。
            returned=tuple((_action_family(a), failed, r) for a, failed, r in returned_lookups),
            fired_axis=str(
                getattr(self._req, "fired_axis", "") or ""
            ),  # 情動を軸で決める（出-ay 段 4-3）
            timer_active=self._timer_active(),  # 操作の言葉の守り（出-aa）
            current_speaker=self._current_speaker_name(),
            silenced=bool(silence_note),
            talking=self._talking(),
            suggestion_ctx=self._suggestion_frame() if self._talking() else "",  # 知-aq
            music_status=self._music_status,  # 「いま何の曲？」に答えるときだけ読む（出-bf）
            jev_workspace=jev_workspace,
            jev_utterance=jev_utterance,
            window_ctx=window_ctx,
            skills=skills,
            active_skills=active_skills,
            jev_max_chars=int(getattr(agent.config, "jev_state_max_chars", 28_000)),
            **self._confirm_inputs(),
            **_heard_inputs(heard),
        )

    def _take_heard_check(self, ws) -> "dict | None":
        """会話の求めの最初の判断なら、聞き取りの確認の預かりを取り出して消す（出-cm）。切れていれば渡さない。

        答えに使うのは 1 回だけ（答えがどれでも消す）。道具の返りの反復・情動・機器では取り出さない。
        """
        held = self._heard_check
        if held is None or self._req.trigger_kind != "発話":
            return None
        if ws.returned_actions or ws.returned_lookups:
            return None
        self._heard_check = None
        if time.monotonic() - float(held.get("at", 0.0)) > HEARD_CHECK_TTL_SEC:
            logger.info("event-loop 聞き取りの確認は切れていたので使わない：%s", held.get("action"))
            return None
        return held

    def _keep_heard_check(self, decision) -> None:
        """見立てつきの聞き返しなら、聞き取りの確認として預ける（出-cm）。本文はログに出さない。"""
        check, text = getattr(decision, "heard_check", None), getattr(decision, "text", "")
        if not isinstance(check, dict) or not check or not isinstance(text, str) or not text:
            return  # 辞書でない返り（土台だけの調停の差し替え）は預けない
        self._heard_check = {**check, "question": text, "at": time.monotonic()}
        logger.info("event-loop 聞き取りの確認を預けた：%s", check.get("action"))

    def _confirming(self) -> bool:
        """確認待ちが生きているか（知-o）。器が無い・真偽でない返り（土台だけの器の差し替え）なら偽。"""
        alive = getattr(self._agent, "confirm_alive", None)
        return callable(alive) and alive() is True

    def _confirm_inputs(self) -> dict:
        """確認待ちのあいだ調停へ渡す、問いの文・決まった促し・促す間隔が空いたか（知-o・2026-10-13）。"""
        if not self._confirming():
            return {}
        from ..core.confirm_state import prompt_due, prompt_text

        pc = self._agent._pending_confirm
        last = getattr(self._agent, "_confirm_prompted_at", None)
        return {
            "confirm_question": str(getattr(pc, "text", "") or "確認の問い"),
            "confirm_prompt": prompt_text(pc),
            "confirm_prompt_due": prompt_due(last, now=time.time()),
        }

    def _note_confirm_prompt(self, decision) -> None:
        """調停が決まった促しを選んだら、その時刻を残す（次の促しまで 20 秒〔仮〕空ける・知-o）。"""
        prompt = self._confirm_inputs().get("confirm_prompt")
        if prompt and getattr(decision, "branch", "") == "light" and decision.text == prompt:
            self._agent._confirm_prompted_at = time.time()

    def _skip_while_confirming(self, trigger: "Trigger | None") -> bool:
        """確認待ちのあいだ、排他で止めるきっかけか（知-o・2026-10-13・本人の決定）。

        情動の求めは立てない（10/13 20:00:25 に、確認待ちのあいだ見回りを始めた）。意図の無い完了は、確認待ちに関係なく
        駆動体が捨てる（`_drop_orphan_completions`・出-cc）。機器と会話入力はここでは止めない。
        """
        if not self._confirming():
            return False
        if trigger is not None and trigger.kind == "情動":
            logger.info("event-loop 確認待ちなので情動の求めを立てない：%s", trigger.query)
            return True
        return False

    def _talking(self) -> bool:
        """情動が話しかける軸（bond・esteem）で起きた求めか（出-at）。"""
        return self._req.trigger_kind == "情動" and self._req.fired_axis in TALKING_AXES

    async def _act_on_decision(self, decision: Decision, *, utterance: str, gen: int) -> str:
        """主LLM の決定を実行する（環-h・段ろ）。

        出口は4つある——打ち切られた求め／道具投げ／発話／素テキスト。**閉じるかどうかは
        決定で決まる**：道具を投げた反復は閉じず、続きは完了で起きる。

        環-h では主LLM が投げっぱなしになり、**返りは別の反復（出す反復）で実行される**。
        いま切り出しておけば、h-は で**呼び元が変わるだけ**になる。

        **器のまま受け取る**（に-5-い）。`Decision` は返りと「投げたときに見ていたもの」を
        束ねた器なので、呼び口でばらさない。ばらせば、欄が増えるたびに呼び口も伸びる。
        引数で足すのは、器が持っていないものだけ——`utterance`（誰の言葉で始まった求めか）
        と `gen`（この反復が属する世代）である。

        差し戻しも投げっぱなしなので（段は-2）、**出す反復は1つの求めで2回起きうる**。
        1回目は検査して違反なら投げ返し、2回目（`decision.retried`）は検査せずそのまま出す。
        """
        agent = self._agent
        say_tc = next((tc for tc in decision.result.tool_calls if tc.name == "say"), None)
        # 上限の反復では調べる動作を渡していないので、返ってきても投げない（連鎖を必ず閉じる）。
        lookup_tc = (
            None
            if decision.capped
            else next((tc for tc in decision.result.tool_calls if tc.name in _LOOKUP_ACTIONS), None)
        )

        # 発話と動作が一緒に来たら、発話はつなぎとして出し、その反復の出力は動作とする。
        # 以前は say を見つけた時点で閉じており、同じ応答に入っていた検索を捨てていた。
        if gen != self._request_generation:
            logger.info("event-loop 打ち切られた求めの反復なので畳む（生成後）")
            return ""

        # 返しようがないと主LLM が決めた（出-db 段 F）。say の中に書いても、文だけでも、考え直しや写真の見立てより先に見て、
        # Jev に聞かず、差し戻さず、声にも画面にも出さずに閉じる。言い直しの返りには指示を渡していないので見ない。
        mark_text = (
            str(say_tc.input.get("text", "")) if say_tc is not None else decision.result.text
        )
        if (
            lookup_tc is None
            and not decision.retried
            and not decision.say_asked
            and _is_silence_mark(mark_text or "")
        ):
            logger.info("event-loop 主LLM が返しようがないと決めたので黙る")
            self._play_silent_cue("主LLM の印")  # 合図（えっと など）の後で黙った（出-dc）
            await self._finish("", decision.memories, "沈黙", gen=gen)
            return ""

        # 主LLM が考えているあいだに名前無しで言い足されたことがあれば、考え直すかを Jev が決める（出-au 段 5-1）。
        # 返事（`say`）を出す返りだけが対象——道具を投げる返りは声にしないので、考え直す理由が無い。
        if say_tc is not None and lookup_tc is None:
            draft = str(say_tc.input.get("text", "")).strip()
            if draft and await self._rethink_or_speak(draft) == jev_judges.RETHINK:
                return await self._rethink(draft)

        if lookup_tc is not None:
            # 調べものと一緒の `say` は声にしない（出-aq 段 7）。待たせる間は合図の声と電子音が受ける。
            if lookup_tc.name in _CAMERA_ACTIONS:
                self._req.see_by = "主LLM"
            self._start_lookup(
                utterance or self._req.cue, dict(lookup_tc.input), action=lookup_tc.name
            )
            logger.info("event-loop 出力=%s（続きは完了で起きる）", lookup_tc.name)
            return ""

        if say_tc is not None:
            # 写真からの見立て（出-an）。**口に出すなら機械にも渡す**——調停と同じ道を通す。
            await self._apply_seen_people(say_tc.input.get("seen_people"))
            text = str(say_tc.input.get("text", "")).strip()
            # **1回だけ**言い直させる。言い直した応答は検査しない（際限なく往復させない）。
            # 記憶の申告は Jev だけが出す（記-r・2026-10-14・本人の決定）。主LLM が `say` に申告を書いてきても使わない。
            # 検査と同じ呼び出しで聞く（`verdicts`）。検査しなかった返事は `None` で、話したあとに背景で聞く。
            violation, verdicts = (
                (None, None)
                if decision.retried
                else await self._speech_check_violation(
                    text,
                    decision.recent_frame,
                    decision.memories,
                    w_id_map=decision.w_id_map,
                )
            )
            if violation:
                # 差し戻しは新しい1通で投げる。say の tool_use を含む往復をそのまま組むと、
                # 結果を返さないまま次を送ることになり backend が受け付けない。
                #
                # **ここで閉じない。** 話してしまえば、この求めに答えが2件書かれる。言い直しは
                # 完了キューを通って**次の出す反復**が出す（段は-2）。同期で待っていたころは、
                # そのあいだ打ち切りが効かなかった。
                logger.info("event-loop 発話前の検査が違反を捕まえた：%s", violation)
                # 言い直しに根拠を添える（出-n）。違反の一文だけだと、検査しない 2 度目に
                # 別の捏造が出た（2026-09-13 16:32「東京は晴れ 30℃」）。
                # 差し戻した返事は声にならないので、申告は当てない（言い直しの返事を背景で聞く）。
                facts = self._checker_facts(decision.memories, w_id_map=decision.w_id_map)
                self._dispatch_main_llm(
                    messages=[
                        agent.backend.make_user_message(
                            self._user_content(
                                f"{utterance or self._req.cue}\n\n"
                                f"[SELF-CHECK] いま言おうとした「{text}」には問題がある："
                                f"{violation}\n使ってよい事実は次のとおり。事実に無いことは"
                                f"言わない（分からないなら分からないと言う）。\n"
                                f"{facts}\n"
                                f"これを直して、もう一度 say() で答える。",
                                decision.memories,
                            )
                        )
                    ],
                    system=_without_silence_note(decision.system),
                    effort=decision.effort,
                    capped=decision.capped,
                    memories=decision.memories,
                    w_id_map=dict(decision.w_id_map),
                    verdict_view=decision.verdict_view,
                    recent_frame=decision.recent_frame,
                    max_tokens=decision.max_tokens,
                    retried=True,
                    original_text=text,
                )
                self._write_version_later()  # 投げ直したあとの版は裏で書く（出-bz-ろ）
                return ""
            if verdicts is not None:
                workspace.apply_memory_verdicts(
                    self._agent._oif,
                    decision.verdict_view,
                    verdicts,
                    decision.w_id_map,
                    origin=self._req.request_id or "",
                )
                # 使った言いたかったことを控える。声に出せたら `_finish` が畳む（出-as 段 7）。
                self._req.told_unsaid = unsaid.told(verdicts, decision.w_id_map, decision.memories)
            spoken, outcome = await self._speak(text)
            if verdicts is None:
                self._declare_light_memory_use(
                    utterance=utterance or self._req.cue,
                    reply=spoken or text,
                    workspace_ctx="",
                    w_id_map=decision.w_id_map,
                    verdict_view=decision.verdict_view,
                    memories=decision.memories,
                    spoken=outcome == "発話",
                    rows=past_rows(decision.memories, decision.w_id_map),
                )
            await self._finish(spoken, decision.memories, outcome, gen=gen)
            return spoken

        if decision.retried and decision.original_text:
            # 言い直しが say を返さなかった。**元の応答で出す**（黙るよりはよい）。
            logger.info("event-loop 言い直しが say を返さなかったので元の応答で出す")
            spoken, outcome = await self._speak(decision.original_text)
            await self._finish(spoken, decision.memories, outcome, gen=gen)
            return spoken

        # どちらも無ければ素テキストへフォールバック（表示はここで1回）。
        # **声にはならない**（音になるのは say() だけ）。`_finish` が `独白` として残す。
        logger.debug("event-loop 決定=none（素テキスト）")
        text = (decision.result.text or "").strip()
        if text and await self._send_back_for_say(decision, text, utterance=utterance):
            return ""
        if text:
            self._emit(text)
        await self._finish(text, decision.memories, "沈黙", gen=gen)
        return text

    async def _send_back_for_say(self, decision: Decision, text: str, *, utterance: str) -> bool:
        """say の無い文が返事なら、主LLM へ 1 回だけ「say で言って」と差し戻す（出-ck・2026-10-15・本人の決定チ）。

        文そのものは機械が声にしない——独り言として書いた文を外に出すことになる（本人）。聞くのは声にしてよい求め
        （会話・情動の bond・esteem）で、言い直し（`retried`）でも差し戻しの返り（`say_asked`）でもないときだけ。
        差し戻したら真（呼び手は閉じずに言い直しを待つ）。言い直しは発話前の検査を通す。
        """
        req = self._req
        if decision.retried or decision.say_asked:
            return False
        if req.trigger_kind != "発話" and not self._talking():
            return False
        agent = self._agent
        reply = await jev_judges.judge_bare_reply(
            getattr(agent, "_jev", None),
            heard=utterance or req.cue,
            text=text,
            min_conf=float(getattr(agent.config, "jev_confidence_min", 0.6)),
        )
        logger.info(
            "event-loop 主LLM が say を呼ばずに文で返した（起点=%s・%d 字）→ Jev：%s",
            req.trigger_kind,
            len(text),
            "返事" if reply else "独り言",
        )
        if not reply:
            return False
        logger.info("event-loop say で言い直すよう主LLM へ差し戻す")
        self._dispatch_main_llm(
            messages=[
                agent.backend.make_user_message(
                    self._user_content(
                        f"{utterance or req.cue}\n\n[SELF-CHECK] いま返した「{text}」は say() を呼んで"
                        "いないので、相手に聞こえていない。相手への返事なら、say() で言う。",
                        decision.memories,
                    )
                )
            ],
            system=_without_silence_note(decision.system),
            effort=decision.effort,
            capped=decision.capped,
            memories=decision.memories,
            w_id_map=dict(decision.w_id_map),
            verdict_view=decision.verdict_view,
            recent_frame=decision.recent_frame,
            max_tokens=decision.max_tokens,
            say_asked=True,
        )
        self._write_version_later()  # 投げ直したあとの版は裏で書く（出-bz-ろ）
        return True

    async def _speech_check_violation(
        self,
        text: str,
        recent: str,
        memories: "list[Recalled]",
        *,
        w_id_map: "dict[str, str] | None" = None,
    ) -> "tuple[str | None, list | None]":
        """発話の前に規則違反を見て、同じ呼び出しで記憶の申告も聞く（出-f・記-r）。`(違反の説明 | None, 申告)` を返す。

        申告は Jev の答え（`[{"id", "verdict"}]`）。検査を切っている・返事が空なら聞いていないので `None`
        （呼び手が背景で聞く）。Jev が使えない・失敗なら空の並び（軽量LLM の道と同じ倒し先：申告しない）。

        **応答の文字列を機械で削らない。** 機械が出すのは、見たか・記憶が載ったかという
        推測の要らない事実だけで、規則に反するかどうかの判断は Jev がする（出-au 段 5-3）。照らす決まりは
        文と事実だけで判断できるものに絞った版（`rules_for_checker`・出-n）。Jev が使えない・確信度が低いときは
        倒し先の「破っていない」。
        """
        from .prompt import rule_list, rules_for_checker

        agent = self._agent
        if not agent.config.speech_check or not text:
            return None, None
        started = time.monotonic()
        tts = getattr(agent, "_tts", None)
        # 記憶の中身は事実の欄と申告の問いの 2 か所に載る。上限（出-bw）を超えるなら、記憶 1 件の字数だけを縮める。
        rows = past_rows(memories, w_id_map or {})
        bare = self._checker_facts(memories, w_id_map=w_id_map, cap=0)
        cap = fitting_cap(
            len(bare) + len(recent) + len(text),
            rows,
            int(getattr(agent.config, "jev_state_max_chars", 28_000)),
        )
        if cap < PAST_CHARS:
            logger.info("event-loop 発話前の検査：長いので記憶 1 件を %d 字に縮める", cap)
            rows = past_rows(memories, w_id_map or {}, cap)
        asked: list = []
        violation, verdicts = await jev_judges.judge_speech_and_verdicts(
            getattr(agent, "_jev", None),
            response=text,
            recent=recent,
            facts=self._checker_facts(memories, w_id_map=w_id_map, cap=cap),
            rows=rows,
            rules=rule_list(
                rules_for_checker(
                    allow_tts_tags=bool(getattr(tts, "understands_tags", False)),
                    confirming=getattr(agent, "_pending_confirm", None) is not None,
                )
            ),
            min_conf=float(getattr(agent.config, "jev_confidence_min", 0.6)),
            on_asked=lambda *got: asked.append(got),
        )
        if asked:
            self._record_speech_check(*asked[0], violation=violation)
        logger.info(
            "event-loop 発話前の検査 %.2f 秒（違反=%s）",
            time.monotonic() - started,
            "あり" if violation else "なし",
        )
        return violation, verdicts

    def _record_speech_check(
        self, state: str, questions: dict, answer: Any, *, violation: "str | None"
    ) -> None:
        """発話前の検査で Jev に聞いた回を、調停の記録に道「発話前の検査」で残す（知-bn・本人の決定ア-1）。

        差し戻した返事は声にならず、どこにも残らなかった。検査が正しいのか厳しすぎるのかを比べるため、通った回も残す。
        書くのは別スレッドで、中身はログに出さない（`arbiter_records.record`）。
        """
        from ..store import arbiter_records

        arbiter_records.record(
            origin=self._req.trigger_kind,
            utterance=self._req.request_text or self._req.cue,
            state=state,
            questions=questions,
            answer=dict(getattr(answer, "answers", None) or {}),
            outcome=violation or "破っていない",
            path="発話前の検査",
            final="差し戻した" if violation else "話す",
        )

    def _checker_facts(
        self,
        memories: "list[Recalled]",
        *,
        w_id_map: "dict[str, str] | None" = None,
        cap: int = PAST_CHARS,
    ) -> str:
        """発話前の検査へ渡す事実（出-n）。言い直しにも同じものを添える。

        届いた結果（この求めの open な記録＝版・完了 O）は、主LLM が申告しなくても根拠に
        なる。届いたことはループが知っているので、申告の有無に頼らない。
        """
        saw = any(role == "見た" for _, role in self._req.turn_records)
        found = self._seen_image(memories)
        seen_mark = None
        if found is not None:
            rec = next((r for r in memories if getattr(r.mi, "image_path", None) == found[1]), None)
            seen_mark = rec.mi.content if rec is not None else None
        open_set = set(workspace.open_ids(self._req))
        arrived = [r.mi.content for r in memories if r.mi.obs_id in open_set and r.mi.content]
        return facts_ctx(
            saw=saw,
            memories=memories,
            picture=found is not None,
            seen=seen_mark,
            past=list(past_rows(memories, w_id_map or {}, cap).values()) if cap > 0 else [],
            arrived=arrived,
        )

    def _declare_light_memory_use(
        self,
        *,
        utterance: str,
        reply: str,
        workspace_ctx: str,
        w_id_map: dict[str, str],
        verdict_view: str,
        memories: "list | None" = None,
        spoken: bool = False,
        rows: "dict[str, str] | None" = None,
    ) -> None:
        """発話前の検査で聞かなかった返事の申告を、背景で聞いて当てる（出-h-ろ・判定は Jev・出-au 段 5-5・記-r）。

        軽量LLM が答えた反復と、主LLM の言い直し（検査しない）・検査を切った反復がここを通る。記憶の行の文は、
        軽量LLM の道は W の文（`workspace_ctx`）から、主LLM の道は記憶の中身（`rows`）から取る。

        **待たない。** 申告は答えを出したあとの後片付けで、発話を待たせる理由がない
        （実測 1.03 秒）。

        **投げるときに写して閉じ込める。** 走っているあいだに次の反復が来れば、ループの
        `w_id_map` も申告の面（`verdict_view`）も作り直されている。12桁が偶然当たれば黙って別の記憶へ当たり、
        面が変われば別の人の記憶が育つ。だから引数で受け、ここで写す（環-h ②・出-h-ろ ③）。
        """
        if not w_id_map:
            return
        w_id_map = dict(w_id_map)
        kept = list(memories or [])  # 走っているあいだに作り直されないよう写す
        # 共起の起点も投げた時点で写す（記-s）。届くまでに次の求めが始まれば `request_id` は次のものになる。
        origin = self._req.request_id or ""
        agent = self._agent

        async def _run() -> None:
            try:
                # 判定は Jev（出-au 段 5-5）。使えない・確信度が低い記憶は申告しない（倒し先）。
                raw = await jev_judges.judge_verdicts(
                    getattr(agent, "_jev", None),
                    utterance=utterance,
                    reply=reply,
                    workspace_ctx=workspace_ctx,
                    ids=list(w_id_map),
                    min_conf=float(getattr(agent.config, "jev_confidence_min", 0.6)),
                    rows=rows,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # 握りつぶさない。申告が出ていないことに気づけないと、記憶が育たない
                # 理由が分からなくなる。
                logger.warning("event-loop 申告を聞けなかった: %s", e)
                return
            workspace.apply_memory_verdicts(agent._oif, verdict_view, raw, w_id_map, origin=origin)
            # 声に出した返事で使った言いたかったことを畳む（出-as 段 7）。
            if spoken:
                told = unsaid.told(raw, w_id_map, kept)
                if told:
                    await self._fold_told(told)

        self._verdict_tasks.add(task := asyncio.create_task(_run()))
        task.add_done_callback(self._verdict_tasks.discard)

    async def _speak(self, text: str, *, branch: str = "full") -> tuple[str, str]:
        """声に出す。返りは **(実際に出した文, 結末)**。**反復は閉じない。**

        身体を持つ以上、発話は相手が居て初めて意味を持つ。出せないとき（情動の軸・窓・出口の門）は
        本文を返して結末を `独白`／`沈黙` にし、`_finish` が「考えたが言わなかった」として O に書く
        （出-as §2.6：保留して後で配ることはしない）。

        **閉じるのは呼び手（`_iterate`）である**（環-e-に・段4）。以前はここが `_finish` を
        3通りに呼び分けており、話す動作が求めの寿命の終わりまで持っていた。核が殻を呼び返す
        形で、結末を決めるのも `memories` を受け取るのも、閉じるためだけだった。
        """
        if not text:
            return "", "沈黙"
        if self._req.trigger_kind == "情動" and self._req.fired_axis not in TALKING_AXES:
            # **会話をしようとするのは BOND と ESTEEM だけ**（出-as §2.1・本人の決定）。見回る・探すなどの
            # 発火は行動だけで、返事の文は声に出さない（声にしなかっただけの独り言として残る）。
            logger.info("event-loop %s の発火なので話しかけない：%.40s", self._req.fired_axis, text)
            return text, "沈黙"
        # **頼まれたことへの返事は窓を見ない**（出-bj・2026-10-11・本人の決定イ）。窓で独り言にしていたが、7.5 秒話し・
        # 書き起こし 2.58 秒・調停 1.5 秒で 10 秒を超え、受けた入力への返事を独り言にした。窓は開け直さない（延ばすのは
        # 開いているときだけ）。
        blocked = self._delivery_block_reason()
        if blocked:
            # **止められた発話は、すべて独り言にする**（出-as §2.6・2026-09-26）。思ったこと自体は残す——
            # 本文を返して `_finish` が「考えたが言わなかった」（役割 `独白`）で O に書く。
            logger.info("event-loop %s ので独り言は言わずに残す（積まない）", blocked)
            self._req.block_reason = blocked  # 最終版の「言えなかった（理由）」（記-t）
            return text, "独白"
        self._stop_beeps()  # 答えの声が始まる。待ちの電子音を止める（出-bu）
        await self._say_aloud(text, careful=self._careful_voice(branch))
        self._note_suggestion_offered(text)
        self._stamp_said()
        self._extend_window_for_conversation()  # 返事から窓の長さぶん（出-as 段 4）
        self._emit(text)
        return text, "発話"

    def _delivery_block_reason(self) -> str:
        """配信ゲート。発話を出せない理由を返す（出せるなら空文字）。

        正本③ の「配信ゲート（結果有り＋在席）」。見るのは在席だけで、会話の求めには掛けない（出-as §2.3）。

        **確かめたうえで掛けたタイマーが鳴った求めは通り抜ける**（知-n・`passes_gate`）。頼んだ本人が
        「静かな時間でも鳴らしていい」と言ったものなので、在席にも掛けない。
        """
        agent = self._agent
        if getattr(self._req, "passes_gate", False):
            return ""
        # 「黙っていて」の依頼は**入口**で見る（`_swallow_if_unheard`・情-h・2026-09-16）。以前は
        # ここ（出口）で止めていたので、発話ごとに求めが立って調停・主LLM が回ってから止まり、
        # 他人への返事を通す例外（案イ）も入り込んだ。黙っているあいだに求めが立つのは解く
        # きっかけ（タイマーが鳴る・本人の「話していい」・止める頼み）だけで、それは通す。
        # **会話の求めは在席で止めない**（出-as §2.3・2026-09-26）。名前で呼ばれた・打たれた窓の中なので、
        # カメラに映っていなくても相手は居る。在席で止めるのは、情動が自分から話しかけるときだけ（§2.1）。
        # 静穏時間の門は外した——夜に話さないことは情動の溜まり方（静穏時間の倍率）が持つ。
        if self._req.trigger_kind == "発話":
            return ""
        if not agent._someone_here():  # 顔ぶれか在席（知-ai）
            return "聞く相手が居ない"
        return ""

    async def _dispatch_arbiter_action(self, decision, *, utterance: str) -> None:
        """調停が選んだ調べもの（see・look・recall・検索・道具）を投げる。

        投げる前につなぎは言わない（出-aq 段 7）。答えまで 5 秒かかったときだけ、待ちの知らせが言う。
        `see`・`look` は帰りの判断も調停がするので、誰が出したかを控える（`_decide`）。
        """
        tool_input = _tool_input_of(decision)
        if decision.action == "look":
            tool_input = self._fill_look_target(tool_input)
        if decision.action in _CAMERA_ACTIONS:
            self._req.see_by = "調停"
        self._start_lookup(utterance, tool_input, action=decision.action)

    # ── 黙っているあいだ（情-h）────────────────────────────────────────────

    def _load_silence(self):
        from ..silence_state import load_silence

        return load_silence()

    async def _swallow_if_unheard(self, trigger: "Trigger") -> bool:
        """入口の門：聞けないものは求めを立てない。

        - **知らない声**（知-ar）：声を登録した誰にも基準まで届かない声は、いちばん先で捨てる（基準が未設定なら効かない）。
        - **黙っているよう頼まれている**（情-h）：会話入力・機器・情動のどれも O に残して器
          （`_muted`・`silence_hold`）に控えるだけにする。通すのは解くきっかけだけ（`lifts`）。控えたものは
          聞けるようになった最初の求めへ渡す（`_pending_heard` → `Request.heard_while_silent`）。
        - **窓の外の入力**（出-as 段 3・出-au 段 1-2）：声もキーボードも捨てる。誰も見えないのに声がしたのは
          見に行く理由なので、SEEKING を押し上げる（案ア）。タイマーの操作の言葉にも名前が要る。

        会話入力はここを `push_utterance` から**届いた時刻で**通る（駆動体は会話入力をもう一度は見ない）。
        名前で呼ばれたか（`Trigger.named`）もここで決める。
        """
        from ..core.silence_hold import lifts
        from ..core.wake_strictness import is_name_only, is_name_repeat, near_name_only
        from ..core.wake_window import heard_name
        from ..silence_state import is_silenced

        now = time.time()
        names = list(getattr(self._agent.config, "agent_names", None) or [])
        if trigger.kind == "会話入力" and trigger.source == "voice":
            if self._voice_stranger(trigger.voice):
                # **知らない声は無条件に捨てる**（知-ar）。名前・黙っていての依頼・窓より前。気配として SAFETY だけ上げる。
                await self._nudge_by_voice(trigger, stranger=True)
                return self._swallowed(trigger)
        if trigger.kind == "会話入力":
            if trigger.source == "voice" and is_name_repeat(trigger.query, names):
                # 名前だけの繰り返しは想定外（2026-10-07 本人の決定）。窓が開いていても、値に関わらず捨てる。
                logger.info(
                    "event-loop 名前の繰り返しなので捨てた（%d 字）", len(trigger.query or "")
                )
                await (
                    self._nudge_by_name()
                )  # 窓は開けないが、情動は動かす（出-ch 段 3・本人の決定タ）
                return self._swallowed(trigger)
            trigger.named = heard_name(trigger.query, names)
            near = (
                trigger.named and trigger.source == "voice" and near_name_only(trigger.query, names)
            )
            window_open = self._wake_window().is_open(self._arrival(trigger))
            if near and window_open:
                # 窓の中の 1 字違いは、名前の証拠にしない。名前の無い言葉と同じに宛先の判定を通す（出-db 段 D・
                # 2026-10-25・本人の決定 3-c）。聞き違えた名前の直し（知-ap）だけは行う（`near_name`）。
                trigger.named = False
                trigger.near_name = True
            elif trigger.named and trigger.source == "voice" and not window_open:
                near_ok = False
                if near:
                    # 1 字違いでだけ当たった名前は、宛先を Jev に確かめてから窓を開ける（出-db 段 D・本人の決定 3-c）。
                    # 以前は言葉の意味を聞いていた（出-ch サ）が、言葉だけでは本物と崩れた声が分からなかった。
                    # Jev が答えなければ捨てる（崩れた音で窓を開けない）。名前がそのまま当たるときは待たない。
                    started = time.monotonic()
                    near_ok = await self._near_name_is_for_me(trigger)
                    logger.info(
                        "event-loop 1 字違いの名前の宛先を聞いた（%.2f 秒）",
                        time.monotonic() - started,
                    )
                trigger.named = self._wake_admits(trigger, names, near_ok=near_ok)
        req = None
        with contextlib.suppress(Exception):
            req = self._load_silence()
        if is_silenced(req, now=now):
            assert req is not None
            if lifts(
                trigger.kind,
                trigger.query,
                names=names,
                reason=getattr(req, "reason", "") or "",
            ):
                if trigger.kind == "会話入力":
                    # 通した言葉（名前つきの「話していいよ」・名前つきのタイマーの操作の言葉）への返事を声に
                    # するため、窓を開ける（出-as 段 6）。開けないと、返事が「窓が切れた後」になって独り言になる。
                    self._wake_window().open(self._arrival(trigger))
                    trigger.window_session = self._wake_window().session  # 窓で受けた（環-ar 段 B）
                    self._pending_heard = self._take_muted()  # 解く言葉の求めがまとめの求めになる
                return False
            await self._note_muted(trigger, now)
            return self._swallowed(trigger)
        if (
            trigger.kind == "会話入力"
            and trigger.source == "voice"
            and is_name_only(trigger.query, names)
        ):
            # 声で名前だけなら返事をしない。値が足りれば黙って窓を開け、続けて言う用件を名前なしで受ける。ふつうは
            # 名前のあとに用件が続く（出-ch 段 3・2026-10-16・本人の決定ア・カ）。呼ばれたことで情動はいつも動かす。
            if trigger.named:
                self._wake_window().open(self._arrival(trigger))
                trigger.quiet_open = True
            logger.info(
                "event-loop 名前だけなので黙って窓を%s（%d 字）",
                "開けた" if trigger.named else "開けなかった",
                len(trigger.query or ""),
            )
            await self._nudge_by_name()
            return self._swallowed(trigger)
        if trigger.kind == "会話入力":
            # タイマーの操作の言葉（止めて・一時停止・再開）にも**名前が要る**（出-au 段 1-2・`設計方針_判定の段`
            # §2.1）。以前は名前が無くても通していたが（出-ab）、打ち切りには名前が要る決まりにそろえた。
            if self._music_word_while_playing(trigger):
                # 鳴っているあいだの音楽の言葉は、名前が無くても受ける（出-cp・2026-10-15・本人の決定ア）。マイクの関門は
                # 通しているのに、ここで捨てて止まらなかった（10/14 23:21）。**窓は開けない**——開けるとテレビや家族の話も
                # 10 秒受ける。家族どうしかも聞かない（「ストップ」は短く、家族どうしと読まれやすい）。返事は頼まれたことへの
                # 返事なので、窓を見ずに話す（出-bj）。
                logger.info(
                    "event-loop 音楽の言葉なので名前なしで受けた（%d 字）", len(trigger.query or "")
                )
                if self._muted:
                    self._pending_heard = self._take_muted()
                return False
            if not self._wake_window_admits(trigger):
                # **窓の外の入力は捨てる**（出-as 段 3）。記録もしない——呼ばれていない話は聞いて
                # いないのと同じ。家族どうしの話や食事のあいさつにまで返事をしていた（実機 09-26）。
                logger.info("event-loop 窓の外の入力なので捨てる：%.40s", trigger.query)
                await self._nudge_by_voice(trigger)
                return self._swallowed(trigger)
            trigger.window_session = self._wake_window().session  # 窓で受けた（環-ar 段 B）
            if not trigger.named:
                if trigger.source == "voice" and await self._between_family(trigger):
                    # **パジュ宛てでない話は受けない**（家族どうし・どこにも向いていない・出-au 段 5-2・出-db 段 C）。窓の外の入力と
                    # 同じく、何も起こさない。
                    logger.info(
                        "event-loop パジュ宛てでないので受けない（%d 字）", len(trigger.query or "")
                    )
                    return self._swallowed(trigger)
                self._wake_window().extend(self._arrival(trigger))
        if self._muted:
            self._pending_heard = self._take_muted()
        return False

    def _confirm_frame(self) -> str:
        """`[確認待ち]` の枠（`agent.confirm_frame`）。器が無ければ空（`__new__` で組んだ装置でも落ちない）。"""
        fn = getattr(self._agent, "confirm_frame", None)
        try:
            return str(fn() or "") if callable(fn) else ""
        except Exception:  # noqa: BLE001
            return ""

    def _timer_active(self) -> bool:
        """タイマーが動いている／一時停止中（`[タイマー]` の枠がある）か、音が鳴っているか。"""
        if _timer_frame(self._agent):
            return True
        return bool(getattr(getattr(self._agent, "_dif", None), "ringing", False))

    def _wake_window(self) -> WakeWindow:
        """窓の器。`__new__` で組んだ装置（試験）でも落ちないよう、無ければ閉じた窓を置く。"""
        wake = self.__dict__.get("_wake")
        if wake is None:
            wake = self.__dict__["_wake"] = WakeWindow()
        return wake

    def _window_talk(self) -> WindowTalk:
        """いまの窓の会話（環-ar 段 A）。窓の番号が変わったら、新しい空の会話にする。

        開いた時刻は、窓の器が持つ時刻（`time.monotonic()`）を、いまの時計に引き直して見出しに使う。
        """
        wake = self._wake_window()
        talk = self.__dict__.get("_talk")
        if talk is None or talk.session != wake.session:
            ago = max(0.0, time.monotonic() - wake.started) if wake.session else 0.0
            opened = datetime.now(clock.local_tz()) - timedelta(seconds=ago)
            talk = self.__dict__["_talk"] = WindowTalk(session=wake.session, opened_at=opened)
        return talk

    def _play_silent_cue(self, why: str) -> None:
        """合図を鳴らした後で黙ると決まったら、黙る音を 1 回鳴らす（出-dc・2026-10-26・本人の決定イ）。

        鳴らすのは、決まった動作の合図が実際に鳴った求め（`cue_played`）で、まだ道具を使っていない（頼まれたことをやり終えて
        黙るのではない）ときだけ。待ちの電子音は、黙ると決まったので止める（鳴り続けると、まだ考えているように聞こえる）。
        """
        self._stop_beeps()
        # 主LLM の呼び出しも `lookups` に「主LLM」として積まれる。数えるのは道具だけ（`_main_llm_in_flight` と同じ見分け）。
        used_tool = any(lk.action != "主LLM" for lk in self._req.lookups)
        if not self._req.cue_played or used_tool:
            return
        from ..core import reaction_cue as rc

        logger.debug("event-loop 黙ったので黙る音を鳴らす（%s）", why)
        try:
            task = self._dif.play_cue(rc.SILENT)
        except Exception:  # noqa: BLE001
            logger.warning("event-loop 黙る音を鳴らせなかった（%s）", why, exc_info=True)
            return
        if task is not None:
            self._cue_sound = task  # 鳴っているあいだ、待ちの電子音は休む（`_sounding`）

    def _skill_store(self) -> "list[str]":
        """選んだスキルの貯め場所（環-au）。窓の会話の中なら窓の会話（窓が閉じるまで）、窓の外ならいまの求め。"""
        talk = self._talk_of(self._req.window_session)
        return talk.skills if talk is not None else self._req.skills

    def _active_skills(self) -> "tuple[str, ...]":
        """いま載せるスキルの名前。"""
        return tuple(self._skill_store())

    def _keep_skills(self, names: "tuple[str, ...]") -> None:
        """調停の Jev が選んだスキルを貯める（足すだけで、外さない・本人の決定ア）。"""
        kept = self._skill_store()
        kept[:] = skills_mod.merged(tuple(kept), names)

    async def _load_skills(self) -> "tuple[skills_mod.Skill, ...]":
        """スキルを読み直す（判断のたびに・直したらすぐ効く）。読むのはループの外。主LLM の枠も、この読んだものを使う。"""
        self._skills = tuple(await asyncio.to_thread(skills_mod.load_skills))
        return self._skills

    async def _skills_text(self) -> str:
        """主LLM へ渡す [やり方] の枠（書き手と同じ文・環-au）。載せるものが無ければ空。"""
        names = self._active_skills()
        if not names:
            return ""
        loaded = self.__dict__.get("_skills")
        skills = loaded if loaded is not None else await self._load_skills()
        logger.debug("event-loop 主LLM にスキルを載せる：%s", "・".join(names))
        return skills_mod.skills_frame(skills, names)

    def _talk_line(self, who: str, text: str, obs_id: str = "") -> None:
        """窓の会話に 1 行足す（環-ar 段 A）。窓が開いているときか、いまの求めがその窓の中で始まったときだけ。

        窓が閉じた後に届いた返事（出-bj）は、その求めが窓の中で始まっていれば同じ会話の行にする。窓の外で始まった
        求めの声や結果は、閉じた会話に足さない。
        """
        if self._in_window():
            self._window_talk().add(datetime.now(clock.local_tz()), who, text, obs_id)

    def _in_window(self) -> bool:
        """いまの窓の会話の中か（環-ar 段 A）。窓が開いているか、いまの求めがその窓の中で始まったとき。"""
        wake = self._wake_window()
        if wake.session == 0:
            return False
        return wake.is_open(time.monotonic()) or self._req.window_session == wake.session

    def _window_text(self) -> str:
        """窓の会話の中なら、窓の会話の全部（1 つの文）。外なら空（いまどおりの直近を使う）。"""
        return self._window_talk().render() if self._in_window() else ""

    def _talk_of(self, session: int) -> "WindowTalk | None":
        """窓の会話の番号 `session` の会話がいまの会話なら、その会話。違えば None（環-ar 段 B）。

        判定（宛先・返すか黙るか）と Jev へ送る W は、これで窓の会話か直近かを決める。前の窓の番号を持つ声や求めは、
        新しい窓の会話を見ない。
        """
        if not session or session != self._wake_window().session:
            return None
        return self._window_talk()

    def _extend_window_for_conversation(self) -> None:
        """会話の求めで返事を出したら、そこから窓の長さ（10 秒）へ延ばす（開いているときだけ）。"""
        if self._req.trigger_kind == "発話":
            self._wake_window().extend(time.monotonic())

    @staticmethod
    def _arrival(trigger: "Trigger") -> float:
        """会話入力が届いた時刻（`time.monotonic()`）。付いていなければいま。"""
        return trigger.arrived or time.monotonic()

    def _wake_admits(
        self, trigger: "Trigger", names: "list[str]", *, near_ok: bool = False
    ) -> bool:
        """窓が閉じているときの声で、名前で窓を開けてよいか（2026-10-05 実機の聞き違い「パジュー」・本人の決定）。

        基準は 2 段（`core/wake_strictness`）。**厳しい**のは、誰の声か分からないときだけ。居ない（顔ぶれも在席も無い）だけでは
        厳しくしない（知-bg ア・2026-10-19 本人の決定）——カメラに写らなくなって 1〜2 分のうちに家族の声で呼んだのを、誰も
        居ない部屋の音と同じ基準で捨てていた（10/17 17:22・10/18 07:56）。居るかどうかはログにだけ残す。捨てたら段と値を
        ログに残す（本文は出さない・値は実機で貯めて見直す）。最後に人を見た定点と秒も残す（知-bg 段 2・2026-10-17・
        本人の決定ウ）。呼んだ時点で在席がどれだけ切れていたかを、この行だけで数えるため。
        """
        from ..config import STTConfig
        from ..core.wake_strictness import admits

        cfg = getattr(self._agent.config, "stt", None)
        if not isinstance(cfg, STTConfig):
            cfg = STTConfig()
        alone = True
        with contextlib.suppress(Exception):
            alone = not self._agent._someone_here()
        unknown = self._voice_unknown(trigger.voice)
        strict = unknown  # 居ないだけでは厳しくしない（知-bg ア）
        ok, why = admits(
            trigger.query,
            names,
            strict=strict,
            no_speech=trigger.no_speech,
            logprob=trigger.logprob,
            cfg=cfg,
            near_ok=near_ok,
        )
        # 通したときも値を残す（出-ch）。捨てた値だけでは、上限を上げて物音で起きたかを見直せない。本文は出さない。
        logger.info(
            "event-loop %s：%s（無音らしさ=%s・確かさ=%s・%d 字・%s・声%s）%s",
            "名前で起こした" if ok else "名前で起こさなかった",
            ("厳しい" if strict else "ふだん") if ok else why,
            "-" if trigger.no_speech is None else f"{trigger.no_speech:.3f}",
            "-" if trigger.logprob is None else f"{trigger.logprob:.3f}",
            len(trigger.query or ""),
            "居ない" if alone else "居る",
            "が分からない" if unknown else "が当たる",
            self._last_seen_note(),
        )
        return ok

    def _last_seen_note(self) -> str:
        """名前で起こす判定のログの末尾：最後に人を見た定点と秒（知-bg 段 2）。読めなければ何も足さない。"""
        sensor = getattr(self._agent, "_occupancy_sensor", None)
        if sensor is None:
            return "・在席の見張りなし"
        try:
            seen = sensor.last_person()
        except Exception:  # noqa: BLE001
            return ""
        if seen is None:
            return "・起動してから人を見ていない"
        if not (isinstance(seen, tuple) and len(seen) == 2 and isinstance(seen[1], (int, float))):
            return ""
        return f"・最後に人を見たのは {seen[0]}・{seen[1]:.0f} 秒前"

    def _voice_unknown(self, voice: Any) -> bool:
        """誰の声か分からないか：声の特徴が無い、または、どの家族にも付け替えの閾値（0.35）以上で当たらない。"""
        if voice is None:
            return True
        try:
            cfg = self._agent.config.recognition
            scores = self._voice_scores(voice)
            return not any(v >= cfg.voice_switch_threshold for v in scores.values())
        except Exception:  # noqa: BLE001
            return True

    def _voice_stranger(self, voice: Any) -> bool:
        """知らない声か（知-ar）：声紋があり、声を登録した誰にも `voice_stranger_threshold` まで届かない。

        `_voice_unknown`（声紋が無いと真・付け替えの閾値で見る）とは別物。判断できないもの（基準が未設定・声紋が無い・
        登録した声が 1 つも無い・点数を読めない）は知らない声にしない。
        """
        threshold = getattr(self._agent.config.recognition, "voice_stranger_threshold", None)
        if threshold is None or voice is None:
            return False
        try:
            scores = self._voice_scores(voice)
        except Exception:  # noqa: BLE001
            logger.warning("知らない声か確かめられなかった（通す）", exc_info=True)
            return False
        if not scores:
            return False
        best = max(scores.values())
        if best >= threshold:
            return False
        logger.info(
            "event-loop 知らない声なので捨てた（最も近い %.2f・基準 %.2f）", best, threshold
        )
        return True

    def _voice_scores(self, voice: Any) -> "dict[str, float]":
        """人ごとの似かた（登録の重心と今日の重心のうち近いほう・`core/voice_speaker.scores`）。読めなければ例外。

        門を通った声（`_match_voice`）も、窓の外で捨てた声（`_log_swallowed_voice`）も、ここで測る。捨てた声は測って
        記録するだけで、話者・今日の声には触らない（知-as 段 1）。
        """
        from datetime import datetime

        from ..core import voice_speaker
        from ..store.clock import local_tz

        store = self._voice_store()
        today = datetime.now(local_tz()).date()
        return voice_speaker.scores(
            voice,
            store.centroids("voice", "registered"),
            store.centroids("voice", "today", day=today),
        )

    def _log_voice_scores(self, heard: bool, scores: "dict[str, float]", verdict: Any) -> None:
        """声の似かたを 1 行（知-as 段 1）。全員の点数・いまの話者・在席と付け替えの基準・判定。本文は出さない。"""
        cfg = self._agent.config.recognition
        occupied = self._occupied()
        strict = cfg.voice_switch_threshold if occupied else cfg.voice_alone_threshold
        current = self._current_speaker_pid()
        who = self._call_name_for(verdict.person_id) if verdict.person_id else None
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        logger.info(
            "声の似かた：%s・いま %s・在席%s（付け替え %.2f）→ %s%s %s %.2f｜%s",
            "通した" if heard else "捨てた",
            (self._call_name_for(current) or current) if current else "既定",
            "あり" if occupied else "なし",
            strict,
            "" if heard else "通していたら ",
            verdict.action,
            who or verdict.person_id or "-",
            verdict.score,
            "・".join(f"{self._call_name_for(pid) or pid} {v:.2f}" for pid, v in ranked)
            or "基準なし",
        )

    def _log_swallowed_voice(self, voice: Any) -> None:
        """窓の外で捨てた声も、点数と「通していたら」の答えだけ残す（知-as 段 1）。話者・今日の声には触らない。"""
        if voice is None:
            return
        from ..core import voice_speaker

        cfg = self._agent.config.recognition
        try:
            scores = self._voice_scores(voice)
            strict = cfg.voice_switch_threshold if self._occupied() else cfg.voice_alone_threshold
            verdict = voice_speaker.decide(
                self._current_speaker_pid(), scores, loose=cfg.voice_threshold, strict=strict
            )
            self._log_voice_scores(False, scores, verdict)
        except Exception:  # noqa: BLE001
            logger.warning("捨てた声の似かたを測れなかった", exc_info=True)

    def _music_word_while_playing(self, trigger: "Trigger") -> bool:
        """声で、名前が無く、窓が閉じていて、音楽が鳴っていて、音楽の言葉か（出-cp）。"""
        if trigger.source != "voice" or trigger.named:
            return False
        if self._wake_window().is_open(self._arrival(trigger)):
            return False
        state = getattr(self._agent, "_music_state", None)
        if getattr(state, "playing", False) is not True:
            return False
        from ..core.music_rules import passes_unnamed

        table: tuple = ()
        with contextlib.suppress(Exception):
            got = self._agent._music_table()
            table = got if isinstance(got, tuple) else ()
        return passes_unnamed(trigger.query or "", table)

    def _wake_window_admits(self, trigger: "Trigger") -> bool:
        """会話入力を窓で受けるか（出-as 段 3・出-au 段 1-2・`設計方針_判定の段` §2.1）。受けたら窓を開ける／延ばす。

        **声もキーボードも**、名前があれば開けて受け、窓が開いていれば延ばして受ける。判定は**届いた時刻**で
        する。**カメラに誰も映っていなくても**窓の中なら受ける（呼ばれた・打たれたことが居る証拠）。
        """
        now = self._arrival(trigger)
        if trigger.named:
            self._wake_window().open(now)
            return True
        # 名前の無い入力は、窓の中なら受ける。**延ばすのは宛先を決めた後**（出-au 段 5-2・`_swallow_if_unheard`）——
        # 家族どうしの話で窓を延ばさないため。
        return self._wake_window().is_open(now)

    def _recent_for_judges(self, session: "int | None" = None) -> str:
        """判定に渡す直前のやりとり。窓が開いていれば窓の会話の全部、閉じていれば直近 3 往復（W と同じ時間の上限）。

        窓が開いたばかりで行がまだ無ければ空（環-ar 段 B）。前の窓の話を、いまの会話の直前として載せない。
        時間の上限を付けないと、再起動をまたいだ前の晩のやりとりが「直前」に載る（出-ae(2) と同じ）。読めなければ空。
        """
        talk = self._talk_of(self._req.window_session if session is None else session)
        if talk is None:
            return self._recent_from_memory(talk=True)
        # 窓の会話はやりとりの代わり。タイマー・入室などの [そのあいだに起きたこと] は Jev の W と同じく添える。
        return "\n\n".join(p for p in (talk.render(), self._recent_from_memory(talk=False)) if p)

    def _recent_from_memory(self, *, talk: bool) -> str:
        """記録から引いた直近 3 往復（W と同じ時間の上限）。`talk` が偽なら [そのあいだに起きたこと] だけ。読めなければ空。"""
        with contextlib.suppress(Exception):
            agent = self._agent
            _rows, recent, _ids = workspace.render_recent(
                agent._oif,
                workspace.recent_chains(agent._oif, 3),
                3,
                max_age_sec=int(agent.config.memory.recent_exchanges_max_sec),
                talk=talk,
            )
            return str(recent or "")
        return ""

    async def _between_family(self, trigger: "Trigger") -> bool:
        """窓の中の名前の無い声がパジュ宛てでないか（家族どうし・どこにも向いていない）を Jev に聞く（出-au 段 5-2・出-db 段 C・
        `設計方針_判定の段` §2.2.3）。

        送るのはその言葉・直前のやりとり（窓の会話の全部・`_recent_for_judges`）・顔ぶれ・ME.md の呼び名・声の似かた。ロボット宛ての確率が
        `jev_judges.ADDRESSEE_ROBOT_MIN` 未満なら宛てでない（真）。Jev が使えないときは受ける（偽）。
        """
        # 窓の会話かは、門で受けたときの番号で見る。判定するまでに窓が切れても、窓の会話で判定する。
        verdict = await self._ask_addressee(
            trigger, recent=self._recent_for_judges(trigger.window_session)
        )
        return verdict == jev_judges.NOT_TO_PAJU

    async def _near_name_is_for_me(self, trigger: "Trigger") -> bool:
        """窓が閉じているときの 1 字違いの名前が、パジュ宛てか（出-db 段 D・本人の決定 3-c）。Jev が答えなければ偽（捨てる）。

        直前のやりとりは送らない。窓が閉じたなら会話は終わっていて、終わった話の続きと読まれると崩れた音でも
        ロボット宛てが上がる（以前の意味の判定も送っていなかった。線 0.3 を測った実験でも、窓の閉じた言葉は 0 行だった）。
        """
        verdict = await self._ask_addressee(trigger, recent="", drop_unanswered=True)
        return verdict == jev_judges.TO_PAJU

    async def _ask_addressee(
        self, trigger: "Trigger", *, recent: str, drop_unanswered: bool = False
    ) -> str:
        """宛先の判定（`jev_judges.judge_addressee`）に、顔ぶれ・呼び名・声の似かたを添えて聞く。"""
        agent = self._agent
        present = ""
        with contextlib.suppress(Exception):
            present = _present_ctx(agent)
        return await jev_judges.judge_addressee(
            getattr(agent, "_jev", None),
            text=trigger.query,
            recent=recent,
            present=str(present or ""),
            names=list(getattr(agent.config, "agent_names", None) or []),
            voice=self._voice_line(trigger.voice),
            drop_unanswered=drop_unanswered,
        )

    def _voice_line(self, voice: Any) -> str:
        """宛先の判定に渡す声の似かた（呼びかけ名と点数・近い順）。読めなければ空（出-db 段 C・本人の決定 3-d）。

        声だけで止める決まりにはしない（声の登録が弱い人やお客さんの本当の頼みまで黙る）。判定の材料にするだけ。
        """
        if voice is None:
            return ""
        try:
            scores = self._voice_scores(voice)
            best: dict[str, float] = {}
            for pid, v in scores.items():
                name = self._call_name_for(pid)
                if name and v > best.get(name, -2.0):
                    best[name] = v  # 同じ呼びかけ名に人物が 2 つあれば、近いほうだけ
        except Exception:  # noqa: BLE001
            logger.warning("宛先の判定に声の似かたを渡せなかった", exc_info=True)
            return ""
        ranked = sorted(best.items(), key=lambda nv: nv[1], reverse=True)
        return "・".join(f"{n} {v:.2f}" for n, v in ranked)

    async def _nudge_by_voice(self, trigger: "Trigger", *, stranger: bool = False) -> None:
        """窓の外で捨てた声で、欲求を押し上げる（情-p・2026-10-10・本人の決定）。

        カメラに誰か居るとき（顔ぶれか在席）だけ。知っている声（家族に 0.35 以上・`_voice_unknown` が偽）なら BOND を
        `voice_bond_nudge`（0.05）、知らない声（当たらない・声の特徴が無い）なら SAFETY を `voice_safety_nudge`（0.10）。
        誰も居なければ何もしない。声で届いた入力だけ（キーボードは押し上げない）。`stranger` は門の先で捨てた知らない声
        （知-ar）で、いつも SAFETY 側。以前は誰も見えないとき SEEKING へ
        0.50 足し、家族の会話のあいだ声 2 つごとに検索した（10/10 の SEEKING からの検索 68 回）。
        """
        if trigger.source != "voice" or self._nobody_here():
            return
        from ..config import DriveConfig

        cfg = DriveConfig()
        with contextlib.suppress(Exception):
            if stranger or self._voice_unknown(trigger.voice):
                await self._agent._nudge_drive(
                    "safety", cfg.voice_safety_nudge, "知らない声がした（誰か居る）"
                )
            else:
                await self._agent._nudge_drive(
                    "bond", cfg.voice_bond_nudge, "知っている声がした（誰か居る）"
                )

    async def _nudge_by_name(self) -> None:
        """声で名前だけを聞いたら BOND を `voice_name_bond_nudge`（0.05）押し上げる（出-ch 段 3・本人の決定カ・サ）。

        `_nudge_by_voice` と違い、居るか・誰の声かは見ない。呼ばれたことが、誰かが居る証拠になる。
        """
        from ..config import DriveConfig

        with contextlib.suppress(Exception):
            await self._agent._nudge_drive(
                "bond", DriveConfig().voice_name_bond_nudge, "名前を呼ばれた"
            )

    def _nobody_here(self) -> bool:
        """出口の門と同じ「居る」（`agent._someone_here`・顔ぶれか在席・知-ai）が無いか。読めなければ居る扱い。"""
        try:
            return not self._agent._someone_here()
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _swallowed(trigger: "Trigger") -> bool:
        if trigger.future is not None and not trigger.future.done():
            trigger.future.set_result("")  # 呼び手（GUI）は吹き出しだけ出して待たない
        return True

    async def _note_muted(self, trigger: "Trigger", now: float) -> None:
        """黙っていたあいだに届いたものを O に残し、器に控える。"""
        agent = self._agent
        if trigger.kind == "会話入力":
            who = self._current_speaker_name()
            text = trigger.query
            content = f"（黙っていたあいだに聞いた）{trigger.query}"
            perspective = agent._conversation_perspective()
        elif trigger.kind == "機器":
            who = ""
            text = f"{trigger.query}（{trigger.result}）"
            content, perspective = (
                f"（黙っていたあいだに起きた）[{trigger.query}] {trigger.result}",
                agent._observation_perspective(),
            )
        else:
            who = ""
            text = trigger.query
            content, perspective = (
                f"（黙っていたあいだに湧いた）[内的な促し:{trigger.query}]",
                agent._observation_perspective(),
            )
        obs_id = ""
        with contextlib.suppress(Exception):
            obs_id = (
                await agent._oif.write(
                    MI(id="", content=content[:500], timestamp=None, direction=trigger.kind),
                    **perspective,
                )
                or ""
            )
        if self._muted_since is None:
            self._muted_since = now
        self._muted.append(Heard(kind=trigger.kind, text=text, who=who, obs_id=obs_id, at=now))
        logger.info("黙っているので聞くだけ：%s %.40s", trigger.kind, text)

    def _take_muted(self) -> "list[Heard]":
        items, self._muted = self._muted, []
        self._muted_since = None
        return items

    def check_silence_lifted(self) -> None:
        """T が tick ごとに呼ぶ：明示の依頼が期限で明けたら記録を消す。求めは立てない（出-as §2.7）。"""
        with contextlib.suppress(Exception):
            from ..silence_state import clear_silence, is_silenced

            req = self._load_silence()
            lifted = not is_silenced(req, now=time.time())
            if lifted and req is not None and not req.reason.startswith("timer:"):
                # 明示の依頼が期限で解けたら**記録も消す**（情-l-ろ・実機 2026-09-18 20:46）。残すと、
                # タイマーの沈黙（`hush_for_timer`）が「人の依頼が生きている」と負ける。
                clear_silence()
                logger.info("黙る期限が来たので沈黙を消した：%s", req.person or "不明")
            # **明けても何もしない**（出-as §2.7・2026-09-26）。以前は「沈黙が明けた」の求めを立てて
            # 話しかけていた。黙っていたあいだに聞いたことは、次の会話の求めに乗る（入口の `_take_muted`）。

    def _confirm_blocks(self, action: str) -> bool:
        """確認待ちのあいだ落とす動作か（出-ag-ろ・`core/confirm_state.blocks`）。"""
        from ..core.confirm_state import blocks

        pc = getattr(self._agent, "_pending_confirm", None)
        ttl = float(getattr(getattr(self._agent, "config", None), "confirm_ttl_sec", 300.0))
        return blocks(action, pc, now=time.time(), ttl=ttl)

    async def _ask_confirm_if_fresh(self, before) -> None:
        """道具の呼び出しで**新しく置かれた**預かりだけ、その場で問いを出す（出-ag-ろ）。"""
        pc = getattr(self._agent, "_pending_confirm", None)
        if pc is None or pc is before:
            return
        await self._ask_confirm_aloud(pc)

    async def _ask_confirm_aloud(self, pc) -> None:
        """確認の問いを**機械が出す**（出-ag-ろ・本人の決定）。

        枠に用意した文をそのまま読む。LLM に作らせると、掛けていないのに「掛けた」と言う
        （実機 17:34）。配信の門（在席・静穏・黙っていて）はふだんの発話と同じく効く。
        """
        text = getattr(pc, "text", "") or ""
        if not text or self._delivery_block_reason():
            return
        await self._say_aloud(text)
        # 問いの話し終わりに窓が開き、答えはその窓で受ける。答える相手が読めるよう、問いを窓の会話の行にする（環-ar 段 B）。
        # 人の言葉で始まった求めでなければ、宛先は書かない（いまの話者の名前は、前に話した人のことがある）。
        to = self._current_speaker_name() if self._req.trigger_kind == "発話" else ""
        self._talk_line(f"わたし（{to}へ）" if to else "わたし", text)
        self._stamp_said()
        self._emit(text)
        logger.info("確認の問いを出した：%.40s", text)

    def _careful_voice(self, branch: str) -> bool:
        """じっくり読む声で読むか（環-u・2026-09-21・`core/voice_rules`）。

        主LLM の発話で、かつ**外へ問い合わせた返り**から起きた反復のときだけ真。合成に 3.4 秒
        かかるが、その場面は相手がすでに数秒待っている。
        """
        from ..core.voice_rules import careful_voice

        return careful_voice(branch, getattr(self, "_returned_now", frozenset()))

    async def _say_aloud(
        self, text: str, *, careful: bool = False, gain: "float | None" = None
    ) -> None:
        """声を出す口（本応答・つなぎ・確認の問い）。**`dif.speak` を呼ぶのはここだけ**。

        倍率はここで必ず付ける（`gain` を渡さなければ `_voice_gain()`）。口が分かれていたころ、つなぎの声だけ
        倍率を渡し忘れ、普段の 4 倍の声で鳴った（2026-10-07 実機 23:18:39「承知いたしました。少々お待ちください。」）。
        """
        # 会話の求めで窓が開いていれば、話しているあいだは開けておき、話し終わりから 10 秒にする（本人
        # 「話し終わってから１０秒」）。閉じていれば触らない——声で窓を開け直さない（確認待ちと名前で受けた話しかけは
        # 話し終わりに開ける・下の finally）。
        wake = self._wake_window()
        hold = self._req.trigger_kind == "発話" and wake.is_open(time.monotonic())
        if hold:
            wake.hold()
        try:
            await self._dif.speak(
                text, gain=self._voice_gain() if gain is None else gain, careful=careful
            )
        finally:
            if hold:
                wake.release(time.monotonic())
            if self._confirming() or (self._req.trigger_kind == "発話" and self._req.named):
                # 確認待ちのあいだの声（問い・聞き返し）は、窓が閉じていても話し終わりから開ける（出-cw・本人の決定イ）。
                # 10/18 07:55 に 12.9 秒の頼みで窓が問いの前に閉じ、名前の無い「はい」を窓の外として捨てた。
                # 名前で受けた話しかけの返事も同じ（出-cx・2026-10-21 本人の決定）。名前つきで 10 秒を超えて話すと窓は
                # 話し始めから 10 秒で切れていて、返事のあとの名前なしの声を捨てていた。考えている最中には開けない。
                # 開け直しても、同じ窓の会話の続き（環-ar 段 A）。
                wake.open(time.monotonic(), resume=self._req.window_session == wake.session)

    def _voice_gain(self) -> float:
        """この求めの声の倍率。タイマーが鳴った知らせ（`[タイマー]`）だけ `TIMER_VOICE_GAIN`（既定 1.0）。

        機器の音量には触らず、その 1 回の再生データにだけ掛ける。鳴り終わった次の発話は元の音量。
        """
        text = str(self._req.request_text)
        if self._req.trigger_kind == "機器" and (
            text.startswith("[タイマー]") or text.startswith("[アラーム]")
        ):
            with contextlib.suppress(Exception):
                return float(getattr(self._agent.config, "timer_voice_gain", 1.0) or 1.0)
        # 普段の声は `TTS_GAIN`（環-q-ろ）。ミキサーは音を基準に決めたので声は下げる。
        with contextlib.suppress(Exception):
            raw = getattr(self._agent.config, "tts_gain", 1.0)
            if isinstance(raw, (int, float)) and raw > 0:
                return float(raw)
        return 1.0

    def _stamp_said(self) -> None:
        """自分が声を出した時刻を打つ（話者の寿命・`speaker_hold_sec`・2026-09-17）。

        起点が人の発話で、相手が分かっているなら、話者の指定も延びる（`_speaker_confirmed_at`・
        知-t）——会話中は切れない。
        """
        agent = getattr(self, "_agent", None)
        if agent is None:
            return
        now = time.time()
        req = getattr(self, "_req", None)
        if getattr(req, "trigger_kind", "") == "発話":
            with contextlib.suppress(Exception):
                if agent.speaker_known():
                    agent._speaker_confirmed_at = now

    def _patrol_note(self) -> str:
        """`[いま]` に添える「見ていない順」の 1 行（知-c・2026-09-17）。センサが無ければ空。

        見た印は W に浮かないことがある（印が無い定点は想起に掛からない・08-01）ので、在/不在の層が
        持つ「最後に見た時刻」から出す。渡すのは情動が起点の求めだけ（呼び手が決める）。
        """
        sensor = getattr(self._agent, "_occupancy_sensor", None)
        if sensor is None:
            return ""
        try:
            order = sensor.stale_order()
        except Exception:  # noqa: BLE001
            return ""
        if not order:
            return ""
        parts = [f"{p} 未" if s is None else f"{p} {int(s // 60)} 分" for p, s in order]
        return "\n見ていない順：" + "・".join(parts)

    def _fill_look_target(self, tool_input: dict) -> dict:
        """情動が起点で調停が `look` の行き先を書かなかったら、最も長く見ていない定点を機械で入れる。

        人に頼まれた `look`（起点が発話）は LLM が読んだ行き先をそのまま使う。
        """
        if tool_input.get("pose") or tool_input.get("direction"):
            return tool_input
        if getattr(self._req, "trigger_kind", "") != "情動":
            return tool_input
        sensor = getattr(self._agent, "_occupancy_sensor", None)
        if sensor is None:
            return tool_input
        try:
            pose = sensor.stalest_pose()
        except Exception:  # noqa: BLE001
            return tool_input
        if not pose:
            return tool_input
        logger.info("event-loop look の行き先を機械で決めた：%s（最も長く見ていない定点）", pose)
        return {**tool_input, "pose": pose}

    def _silence_note(self) -> str:
        """調停へ渡す「いま黙っている」の一行（無ければ空）。黙っている前提が無いと「解かれた」と読めない。"""
        with contextlib.suppress(Exception):
            from ..core.silence_rules import silence_note
            from ..silence_state import load_silence

            note = silence_note(load_silence(), now=time.time())
            return f"\n{note}" if note else ""
        return ""

    async def _apply_requests(self, decision, *, utterance: str = "") -> None:
        """調停が読んだ人の求め（沈黙の依頼・名乗り）を掛ける（反復本体の呼び口は 1 つ・番人 231 行）。"""
        self._apply_silence(decision, utterance=utterance)
        # **否定が先**（出-am）。「ちがう、ママだよ」で パパ を外してから ママ を付ける。
        # 逆だと、付けたばかりの人を否定で消すことになる。
        await self._apply_not_person(getattr(decision, "not_person", ""))
        await self._apply_speaker_claim(decision)

    async def _apply_seen_people(self, people: "list | None") -> None:
        """調停が写真から見立てた人を、顔ぶれへ入れる（出-ae-は・2026-09-22）。

        **推し量ること自体は禁じない。推し量ったなら顔ぶれにも使う**（本人の決定）。15:57 は、
        写真を見て「パパ、おかえりなさい！」と言いながら機械は `unconfirmed` のままだった。

        - **写った人を顔ぶれに入れるか、持ち時間を数え直すだけ**（知-ai・2026-10-05・`set_guessed_present`）。
          写っていない人は消さない——写真は部屋の一部しか写さない。人ごとの持ち時間（10 秒・知-bg）で切れる。
          以前（知-ag）は見立てで入った人をこの並びに置き換えていたので、見回りのたびに顔ぶれが入れ替わった。
        - **話者は付けない**（知-ai）。話者は声・名乗り・`/speaker`・`[名前]` が決める。
        - 家族に当たらない名前と名前の無い人は、**名前の分からない顔ぶれ**として数だけ残す。
          `participants` には入らないので、誰にも対応しない記憶空間は作らない。
        - 見立てが空の反復では顔ぶれを触らない。写真を見ていない反復で顔ぶれを消さないため。

        呼び口は**並び**で受ける。写真の読み取り（`scene.read_photo`・出-au 段 5-7a）と、主LLM の申告
        （`say()` の `seen_people`・出-an）が同じ道を通る——どちらも写真からの推し量りで、
        証拠の種類は同じなので、確信度の上限も揃える。
        """
        from ..core.seen_people import SEEN_CONFIDENCE_MAX, parse_seen_people

        people = list(people or [])
        if not people:
            return
        agent = self._agent
        known, unknown = parse_seen_people(people, str(getattr(agent, "_family_md", "") or ""))
        pmm = getattr(agent, "_pmm", None)
        if pmm is None:
            return
        seen: list[tuple[str, float]] = []
        for name, conf in known:
            pid = pmm.find_person_id_by_name(name)
            if pid is None:
                unknown += 1  # 家族の記述にはあるが人物表に無い。居たことは残す
                continue
            seen.append((pid, conf))
        await pmm.set_guessed_present(seen)
        pmm.note_unknown_present(unknown, confidence=SEEN_CONFIDENCE_MAX)
        # 見立てで話者は付けない（知-ai・2026-10-05・本人の決定ア）。顔ぶれは誰が居るか、話者は誰が話しているかで、
        # 話者は声・名乗り・`/speaker`・`[名前]` が決める。以前（出-as 段 9a）は 1 人なら話者にしていたので、
        # SEEKING の見回りの写真と声が話者を交互に書き換えた。

    async def _apply_not_person(self, claim: str) -> None:
        """身元を否定されたら、その人を顔ぶれから外し、話者を戻す（出-am・2026-09-22）。

        顔ぶれへ**入る**口は 3 つある（顔・声の名乗り・写真からの見立て）のに、**出る**口は
        時間切れと顔の消失しか無かった。誤った見立ては寿命が来るまで残る。

        **由来は問わない**（本人の決定・2026-09-22）。顔で入った人でも、目の前の人が
        「パパじゃない」と言えば外す。名前を言わずに否定されたら、いま話者としている人を
        外す——他に指すものが無く、否定は目の前のやりとりについて言われる。その名前は Jev が選択肢
        「いま話者としている人」から選んで書いてくる（`loop/arbiter`）。

        **空は「打ち消しなし」**（出-ax・2026-10-07）。以前は空ならいまの話者を外していたが、この関数は
        調停のたびに無条件で呼ばれるので、声で付けた話者が毎回 0.4〜2.3 秒で外れた（実機 12:49〜13:28 の 6 回）。
        """
        from ..core.speaker_claim import resolve_claim

        agent = self._agent
        family = str(getattr(agent, "_family_md", "") or "")
        raw = (claim or "").strip()
        if not raw:
            return
        name = resolve_claim(raw, family)
        if name is None:
            logger.info("否定された「%s」は家族に無いので何もしない", raw)
            return
        pid = agent._pmm.find_person_id_by_name(name)
        if pid is None:
            logger.info("否定された「%s」は人物表に無いので何もしない", name)
            return
        await agent._pmm.person_left(pid)  # 顔ぶれから外す（話者だったら話者も外れる）
        agent._persons.reset_to_default()
        logger.info("「%s ではない」と言われたので顔ぶれから外した", name)

    async def _apply_speaker_claim(self, decision) -> None:
        """調停が読んだ名乗り（`speaker_claim`）を、**その発話の声が名乗った本人に当たるとき**だけ話者に付ける。

        知-ai（2026-10-05・本人の決定・`設計方針_在席と顔ぶれ` v0.1）で改めた。名乗りは言葉だけでは顔ぶれに入れず、
        名乗った本人に対する声の閾値を下げる印にする（`voice_claim_threshold`・0.30）。在席（カメラ）は見ない。
        当たれば `/speaker` と同じ効きで話者に付け（顔ぶれにも入る）、その声を登録の声と今日の声に足す。
        本人の声の基準がまだ無い・声の特徴が無い・届かないときは何もしない（最初の登録は `/voice`）。

        以前（知-w・知-w-ろ）は在席があれば名乗りだけで付け、無ければ 30 秒預かっていた。テレビの「パパだよ」でも
        付きえたうえ、在席を顔ぶれの門に使う上下が逆の形だった。
        """
        from ..core.speaker_claim import resolve_claim

        claim = str(getattr(decision, "speaker_claim", "") or "")
        if not claim:
            return
        agent = self._agent
        name = resolve_claim(claim, str(getattr(agent, "_family_md", "") or ""))
        if name is None:
            logger.info("名乗り「%s」は家族に無いので話者にしない", claim)
            return
        voice = getattr(self._req, "voice", None)
        pid = agent._pmm.find_person_id_by_name(name)
        if voice is None or not pid:
            logger.info("名乗り「%s」に声の特徴が無いので話者にしない", name)
            return
        score = self._claim_score(pid, voice)
        threshold = agent.config.recognition.voice_claim_threshold
        if score is None or score < threshold:
            logger.info(
                "名乗り「%s」の声が本人に当たらない（%s・閾値 %.2f）",
                name,
                "基準なし" if score is None else f"{score:.2f}",
                threshold,
            )
            return
        await self._set_speaker(name, f"名乗り＋声 {score:.2f}")
        self._learn_voice(name, voice)

    def _claim_score(self, pid: str, voice: Any) -> "float | None":
        """名乗った本人の登録の声・今日の声に対する似かた（近いほう）。本人の基準が無ければ None。"""
        from datetime import datetime

        from ..core import voice_speaker
        from ..store.clock import local_tz

        try:
            store = self._voice_store()
            today = datetime.now(local_tz()).date()
            registered = {
                k: v for k, v in store.centroids("voice", "registered").items() if k == pid
            }
            todays = {
                k: v for k, v in store.centroids("voice", "today", day=today).items() if k == pid
            }
        except Exception:  # noqa: BLE001
            logger.warning("名乗りの声を照らせなかった", exc_info=True)
            return None
        return voice_speaker.scores(voice, registered, todays).get(pid)

    def _learn_voice(self, name: str, voice: Any) -> None:
        """名乗りが本人の声に当たった発話の声を、その人の登録の声と今日の声に足す（知-ae 段 5・知-ai）。

        登録の声（日をまたいで残る）を育てるのは、名乗りが当たったときと `/voice` だけにする——付け替えの閾値の
        当たり（`_match_voice`）は今日の声にだけ足す。すでに同じ話者でも足す。失敗しても話者の付け方は変えない。
        """
        if voice is None:
            return
        pid = self._agent._pmm.find_person_id_by_name(name)
        if not pid:
            return
        from datetime import datetime

        from ..store.clock import local_tz

        cfg = self._agent.config.recognition
        try:
            store = self._voice_store()
            today = datetime.now(local_tz()).date()
            store.add(pid, "voice", "registered", voice, cap=cfg.registered_max)
            store.add(pid, "voice", "today", voice, cap=cfg.today_max, day=today)
            logger.info("名乗りの声を %s の基準に足した", name)
        except Exception:  # noqa: BLE001
            logger.warning("名乗りの声を基準に足せなかった（話者は付けた）", exc_info=True)

    def _voice_store(self):
        """声の特徴の器（共有接続・短く使う・知-ae）。"""
        from ..db import get_db
        from ..store.recognition_embeddings import RecognitionEmbeddingStore

        return RecognitionEmbeddingStore(get_db().conn())

    def _current_speaker_pid(self) -> "str | None":
        """いまの話者の人物 id。既定の人（話者が明示されていない）なら None。"""
        persons = self._agent._persons
        if not getattr(persons, "active_is_explicit", False):
            return None
        return self._agent._pmm.find_person_id_by_name(str(persons.active_name or ""))

    def _call_name_for(self, person_id: str) -> "str | None":
        """人物 id から呼びかけ名（話者の帳面の書き方）。`FAMILY.md` に無ければ人物表の名前。"""
        from ..core import parsing
        from ..core.speaker_claim import call_name_of

        row = next(
            (p for p in self._agent._pmm.list_persons() if str(p.get("id")) == person_id), None
        )
        if row is None:
            return None
        name = str(row.get("name") or "")
        family = str(getattr(self._agent, "_family_md", "") or "")
        member = next((m for m in parsing.parse_family_md(family) if m.get("name") == name), None)
        return call_name_of(member) if member else (name or None)

    async def _match_voice(self, voice: Any) -> None:
        """門を通った声で、誰が話しているかを決める（知-ae 段 4・知-ai 段 4・`設計方針_在席と顔ぶれ` v0.1）。

        **在席（カメラ）を待たない**（知-ai）。声で誰か分かるなら、それは顔ぶれの証拠になる。規則は
        `core/voice_speaker.decide`：いまの話者は緩い閾値（0.25）で続け、別の人への付け替えは状況で変える——在席あり
        `voice_switch_threshold`（0.35）、無し `voice_alone_threshold`（0.45）。どちらでもなければ既定の人に戻す。
        続けたときも、その人の顔ぶれの持ち時間を数え直す。付け替えの閾値で当たった声は今日の声に足す。
        失敗しても会話は止めない。
        """
        if voice is None:
            return
        from datetime import datetime

        from ..core import voice_speaker
        from ..store.clock import local_tz

        cfg = self._agent.config.recognition
        try:
            store = self._voice_store()
            today = datetime.now(local_tz()).date()
            store.drop_old_today(today)
            scores = self._voice_scores(voice)
            strict = cfg.voice_switch_threshold if self._occupied() else cfg.voice_alone_threshold
            verdict = voice_speaker.decide(
                self._current_speaker_pid(),
                scores,
                loose=cfg.voice_threshold,
                strict=strict,
            )
            if verdict.sure and verdict.person_id:
                store.add(verdict.person_id, "voice", "today", voice, cap=cfg.today_max, day=today)
        except Exception:  # noqa: BLE001
            logger.warning("声で話者を照らせなかった（会話は続ける）", exc_info=True)
            return
        with contextlib.suppress(Exception):
            self._log_voice_scores(True, scores, verdict)
        if verdict.action == "switch" and verdict.person_id:
            name = self._call_name_for(verdict.person_id)
            if name:
                await self._set_speaker(name, f"声 {verdict.score:.2f}")
        elif verdict.action == "unknown" and self._current_speaker_pid() is not None:
            self._agent._persons.reset_to_default()
            with contextlib.suppress(Exception):
                self._agent._pmm.clear_speaker()
            logger.info(
                "声が誰にも当たらないので、話者を既定の人に戻した（最も近い %.2f）", verdict.score
            )
        elif verdict.action == "keep" and verdict.person_id:
            with contextlib.suppress(Exception):
                self._agent._pmm.refresh_signal(verdict.person_id)  # 顔ぶれの持ち時間を数え直す

    def _occupied(self) -> bool:
        try:
            return float(self._agent._occupancy() or 0.0) > 0.0
        except Exception:  # noqa: BLE001
            return False

    async def _set_speaker(self, name: str, why: str) -> None:
        """話者を付ける（`/speaker` と同じ効き：`set_active`・`_speaker_set_at`・PMM 同期）。名乗りと写真の見立てが使う。"""
        agent = self._agent
        if str(getattr(agent._persons, "active_name", "") or "") == name and getattr(
            agent._persons, "active_is_explicit", False
        ):
            # 同じ話者のままでも、その人だと分かる印なので顔ぶれの持ち時間は数え直す（知-ai）。
            with contextlib.suppress(Exception):
                pid = agent._pmm.find_person_id_by_name(name)
                if pid:
                    agent._pmm.refresh_signal(pid)
            return
        agent._persons.set_active(name)
        agent._speaker_set_at = time.time()
        with contextlib.suppress(Exception):
            await agent._sync_pmm_speaker(name)
        logger.info("話者を付けた：%s（%s）", name, why)

    def _apply_silence(self, decision, *, utterance: str = "") -> None:
        """調停が読んだ沈黙の依頼を掛ける／解く（反復本体の呼び口は 1 つ）。

        **名前の関門は窓にまとめた**（出-as §2.5・2026-09-26）。ここへ来る会話入力は、入口で窓を通った
        もの（名前で呼ばれてから 1 分・またはキーボード）だけなので、ここでは名前を見ない。解くのは、
        調停が読めなくても「話していい」と言えば解く（2026-09-16 実機「話していいよ」が効かなかった）。
        """
        from ..core.silence_rules import is_release

        if decision.silence_minutes:
            self._accept_silence(decision.silence_minutes)
        elif decision.lift_silence or is_release(utterance):
            self._release_silence()

    def _release_silence(self) -> None:
        """「もう話していいよ」と解かれたら、依頼を消す（2026-09-16 実機）。

        「待てぃ」を調停が「黙っていて」と読み 60 分の依頼になったあと、解く口が無かった
        （解ける条件が退室と期限だけで、`clear_silence()` に呼び手が無かった）。話しかけられた
        だけでは解かない（黙っていてほしい人が用事だけ言うことはある）。解くのは、頼んだ本人が
        話していいと言ったとき。
        """
        with contextlib.suppress(Exception):
            from ..silence_state import clear_silence, load_silence

            req = load_silence()
            if req is None:
                return
            if req.reason.startswith("timer:"):
                # タイマー由来の沈黙は鳴る・止めるまで（タイマーは従来どおり・出-as §2.4）。
                logger.info("話していいと言われたが、タイマーの沈黙なので解かない")
                return
            # **誰が言っても解く**（出-as §2.5）。話者が分からない家では本人かを決められなかった。
            clear_silence()
            logger.info(
                "話していいと言われたので黙るのを解く（頼んだのは %s）", req.person or "不明"
            )

    def _accept_silence(self, asked_minutes: int) -> None:
        """黙っている依頼を受ける。宛先は、いま話している相手。

        `asked_minutes` は調停が読み取った分数で、`-1` は「頼まれたが長さの指定なし」。
        既定と上限は Config が持つ。
        """
        agent = self._agent
        with contextlib.suppress(Exception):
            from ..silence_state import SilenceRequest, save_silence

            # **話者が分からなくても受ける**（出-as §2.5）。頼んだ人は分かれば記録し、分からなければ空。
            who = self._current_speaker_name()
            from ..silence_state import resolve_minutes

            minutes = resolve_minutes(
                asked_minutes,
                default=max(1, int(getattr(agent.config, "silence_minutes", 15))),
                maximum=max(1, int(getattr(agent.config, "silence_max_minutes", 60))),
            )
            if minutes <= 0:
                return
            save_silence(SilenceRequest(person=who, until=time.time() + minutes * 60))

    def _current_speaker_name(self) -> str:
        """いま話している相手の名前。分からなければ空文字。

        顔ぶれ表の `is_speaker` を先に見て、無ければ `/speaker` で明示された名前を使う。
        黙っている依頼の宛先を決めるのと、その依頼を誰の返事に掛けるかを決めるのは、
        同じ「誰が話しているか」なので 1 箇所で引く。
        """
        agent = self._agent
        with contextlib.suppress(Exception):
            for row in agent._pmm.presence_status():
                if row.get("is_speaker"):
                    return str(row.get("name") or "")
        if getattr(agent._persons, "active_is_explicit", False):
            known = True
            with contextlib.suppress(Exception):
                known = bool(agent.speaker_known())  # 切れていれば「分からない」（知-t）
            if known:
                return str(agent._persons.active_name or "")
        return ""

    def _start_lookup(self, utterance: str, tool_input: dict, *, action: str = "recall") -> None:
        """open 意図を O に残し、RH へ投げる（待たない）。意図は常に高々1件に保つ。"""
        self._pending_lookup = (utterance, tool_input, action)
        self._background_tasks.add(t := asyncio.create_task(self._dispatch_and_write_version()))
        t.add_done_callback(self._background_tasks.discard)

    async def _dispatch_and_write_version(self) -> None:
        _utterance, tool_input, action = self._pending_lookup
        query = _query_label(action, tool_input)
        # **先に投げる。** 版の content は飛行中の一覧を含むので、投げてから書かないと
        # 「1番：… を起動中」が入らない。投げられなかった場合（重複など）は完了が積まれ、
        # 次の取込で版が書かれる。
        self._dispatch_lookup(action, tool_input, query, None)
        version_id = await self._write_version()
        # 自分が出した検索が、いま書いたばかりの版そのものを拾わないように狭く除外する。
        # 版の content は語を丸ごと含むので、その語で探せば必ず上位に来る。
        self._recall_exclude_id = version_id
        # 発話の記録は鎖の外だが、検索を始めたことはそこからも辿れたほうがよい。
        # 一度だけ足す（何を調べているかは版の側が持つ）。埋め込みの作り直しを含むので
        # 別スレッドへ逃がす（同期呼び出しでイベントループを止めない）。
        if self._req.request_id:
            with contextlib.suppress(Exception):
                await self._agent._oif.append(
                    self._req.request_id, self._agent._memory.LOOKUP_STARTED_NOTE
                )

    async def _finish(
        self, text: str, memories: "list[Recalled]", outcome: str, *, gen: "int | None" = None
    ) -> None:
        """求めが閉じた反復の後始末：総括ログと永続化。

        閉じ方は `outcome` が持つ——`発話`（声になった）・`沈黙`（地の文だけで声にならず
        `独白` として残る）・`保留`（配信ゲートに止められた）・`独白`（相手が居ない独り言。
        積まずに、思ったことだけ残す・情-c）。**発話だけではない。**

        **ここでは何も畳まない。** 求めの版チェーンは `_write_version` が `改訂` で畳み、
        自分が答えた記録は鎖の外にある。要約と内省は背景で遅れて来て、その記録を supersede
        する。
        """
        await self._settle_versions()  # 裏で書いている版を待ってから閉じる（出-bz-ろ）
        agent = self._agent
        logger.info(
            # **数えるのは考えた回数**。反復は主LLM の返りで 0 へ戻るので、閉じた時点では
            # 常に 0 で、この求めに何回かかったかを伝えない（環-h）。
            "event-loop 終了: 考えた回数=%d 結末=%s 上限到達=%s text_len=%d",
            self._thinking_round - 1,
            outcome,
            "はい" if self._req.iterations_capped else "いいえ",
            len(text),
        )
        # 自分が言ったことを、**発話の時点で同期に** O へ書く。背景の永続化（要約・内省）を
        # 待つと2秒遅れ、そのあいだに次の反復が起きると「さっき何と言ったか」を拾えない
        # （実機で「それだけ？」に聞き返した）。要約は後から来て、この記録を supersede する。
        #
        # **声になったかで書き分ける。** 主LLM が `say` を呼ばず地の文だけを返した反復は、
        # 画面には出るが**声にはなっていない**（規則 `voice-only-from-say`）。それを
        # 「自分が答えた」と書くと、相手が聞いていない文が答えとして残り、直近のやりとりの枠が
        # 「わたし」として読み返す。**残す価値はある**ので、区別して残す——役割 `独白` は
        # やりとりの項にならないが（`recent_exchanges` が引く役割に無い）、拡散想起の
        # 母集合には入る（`HIDDEN_ROLES` に入れない）。
        # **打ち切られた求めの後始末は、次の求めに触らない**（出-au 段 1-5）。求めの状態は 1 つを共有しており、
        # 返事を声にしている最中に名前で呼ばれて打ち切られると、状態はもう次の求めのものになっている。声になった
        # 事実だけを書き（次の求めの id を親にしない）、状態・やりとり・背景の後始末には触らない。
        aborted = gen is not None and gen != self._request_generation
        if aborted and outcome != "発話":
            return
        if aborted:
            logger.info("event-loop 打ち切られた求めの返事なので、言った事実だけ残す")
        spoken = outcome == "発話"
        # **話そうとして止められた発話は「〇〇に言いたかったこと」**（出-as 段 7・§2.6）。相手の面に立て、
        # 根づきを 2 にする——情動が発火して相手が映ったとき、想起で上がってくるように。相手が分からなければ
        # パジュ自身の面に「誰かに」。声にしなかっただけの独り言（結末「沈黙」）はいままでどおり。
        wanted = outcome == "独白"
        who_id, who_name = self._addressee() if wanted else (None, "")
        if spoken:
            content = f"自分が答えた：{text}"
        elif wanted:
            content = unsaid.content(who_name, text)
        else:
            content = f"考えたが言わなかった：{text}"
        perspective = (
            dict(writer_id=AGENT_SELF_ID, participants=[who_id])
            if who_id and not aborted
            else agent._observation_perspective()
        )
        answer_id = None
        if text:
            with contextlib.suppress(Exception):
                answer_id = await agent._oif.write(
                    MI(
                        id="",
                        content=content[:500],
                        timestamp=None,
                        direction="発話" if spoken else "独白",
                        parent_id=None if aborted else self._req.request_id,
                    ),
                    **perspective,
                )
            if wanted and answer_id:
                with contextlib.suppress(Exception):
                    agent._oif.set_groundedness(answer_id, unsaid.GROUNDEDNESS)
        if aborted:
            return
        if spoken and self._req.told_unsaid:
            await self._fold_told(list(self._req.told_unsaid))
        if spoken and text and (self._req.trigger_kind == "発話" or self._talking()):
            # 声になった返事だけを、窓の会話に足す（環-ar 段 A）。声にしなかった文は相手に届いていない。人の言葉への
            # 返事と、パジュから話しかけた言葉だけで、タイマーなどの知らせは会話の行にしない（確認の問いは
            # `_ask_confirm_aloud` が足す）。
            to = self._current_speaker_name()
            self._talk_line(f"わたし（{to}へ）" if to else "わたし", text, str(answer_id or ""))
        self._note_record(answer_id, "答え" if spoken else "独白")
        # **自分が答えた記録は鎖の外**。何も畳まない。求めの版チェーンは、最後の版
        # （結果が届いた状態）のまま残る。まとめ知識の MI を作る場合は、それが最後の版を
        # 畳む（未実装・`設計方針_求めの版チェーン`）。
        # **最終版を書く**（記-t 段 1）。求めの状態を消す前に組む。収まれば閉じたその場で、超えれば下で背景に出す。
        pending_final = await self._close_final(outcome, text)
        self._stop_wait_watch()  # 待たせている時間の見張りは求めと一緒に閉じる（出-au 段 2）
        self._req.request_id = None
        self._req.live_version_id = None
        self._ledger.close_request(self._req.lookups, state=_intents.CLOSED)
        self._req.lookups.clear()
        self._req.told_unsaid = []
        self._req.added = []
        self._req.draft = ""
        self._req.seen_image_path = None
        self._req.failed_actions.clear()
        self._req.fired_axis = ""
        self._req.iterations = 0
        self._req.iterations_capped = False
        self._notify_request_state(False)
        # やりとりへ渡す区間を取り出してから捨てる（渡す前に消すと空で渡る）。共起はターンの記録を使わない
        # （起点と答えに使った記憶だけ・記-s）。
        noted = self._close_exchange() or []
        self._req.turn_records, self._req.exchange_start = [], 0
        # **やりとりの関係はここで同期で書く。** 背景（要約待ち）で書くと、閉じた 0.1 秒後に
        # 起きた反復（待たされていた入室）が直近のやりとりを空で引く（2026-09-13 実機）。
        # 要約は背景でできたときに、この関係の末尾へ足される。
        exchange_id: int | None = None
        if noted:
            with contextlib.suppress(Exception):
                exchange_id = agent._oif.link(
                    KIND_EXCHANGE, [(i, r, n) for n, (i, r) in enumerate(noted)]
                )
        if pending_final is not None:
            self._spawn_final(pending_final, exchange_id)
        try:
            origin = self._req.utterance or self._req.cue
            arousal = await agent._turn_arousal(origin, text)
            agent._spawn_background_task(
                agent._run_post_response_pipeline(
                    user_input=origin,
                    final_text=text,
                    camera_used=False,
                    arousal=arousal,
                    memories=memories,
                    # 会話要約は背景で作られるので、向こうでこの関係の末尾に足す。
                    exchange_id=exchange_id,
                ),
                name="event-post-response",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("event-loop persistence spawn failed: %s", e)


class _ReactionCue:
    """話しかけたときの合図の 1 回分（出-bg）。Jev が決めたら機械音 A を止め、決めた動作に合わせた合図を流す。"""

    def __init__(self, ip: "InformationProcessing", thinking: Any) -> None:
        self._ip = ip
        self._thinking = thinking
        #: 合図を始めた求めの世代。決まるまでに打ち切られて次の求めになっていたら、合図の印を次の求めに立てない（出-dc）。
        self._gen = getattr(ip, "_request_generation", 0)

    def decided(self, final: str) -> None:
        from ..core import reaction_cue as rc

        self.stop()
        cue = rc.cue_for(final)
        if cue == rc.NONE or self._gen != getattr(self._ip, "_request_generation", 0):
            # 決まるまでに打ち切られて次の求めになっていたら、古い求めの合図も待ちの電子音も鳴らさない（出-dc）。
            return
        task = self._ip._dif.play_cue(cue)
        if task is None:
            return
        self._ip._req.cue_played = True  # 黙ったら黙る音を鳴らす目印（出-dc）
        self._ip._start_beeps()  # 合図が鳴り終わったら、答えまで待ちの電子音（出-bu）。合図の音は `_sounding` で休む
        self._ip._cue_sound = task  # 合図が鳴っているあいだ、待ちの電子音は休む（`_sounding`）

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self._thinking.stop()
