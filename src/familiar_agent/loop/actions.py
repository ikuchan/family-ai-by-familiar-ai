"""動作の表と、道具の定義（2026-11-23 に `event_loop.py` から移した）。

主LLM が選べる動作の名前、調べものの見出し、道具の定義（タイマー・音楽・印刷など）の組み立てを持つ。あわせて、
道具の定義や判定に添える小さなもの（起点の記録の字数の上限 `_ORIGIN_MAX`・`_DEVICE_MAX`、画像の有無、文の中の道具呼びの
見分け、季節と自己像と家族の顔ぶれの文、聞こえた声の入力）も置く。どれも状態を持たない関数と表で、ループ
（`event_loop.InformationProcessing`）が呼ぶ。ここではループを回さない。
"""

from __future__ import annotations

import contextlib
import logging
import re
from datetime import datetime

from ..core.completion_kind import is_spent
from .arbiter import CONFIRM_QUERY

logger = logging.getLogger(__name__)

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
    "print_document",  # 印刷（知-bo 段 2）。主LLM だけに渡す（調停の候補には載せない）
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
# 印刷（知-bo 段 2・2026-11-22）。主LLM だけが呼ぶ（`agent._print_tool`）。刷る前に 1 回確かめる。
_PRINT_ACTIONS = ("print_document",)
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
    """持ち主の英字が付いた道具名なら動作名（`ask_vault_ichiro` → `vault`）。違えば空。"""
    return next((a for p, a in _PERSONAL_PREFIXES.items() if name.startswith(p)), "")


# 調べる動作＝結果が後の反復に届くもの。投げたらその反復は終わる。
# `see`・`look` も含める。結果はその場で返るが、それを見て何を言うかは次の反復が決める
# （`recall` と同じ）。ここに入れないと 1反復1出力 が崩れる。
_LOOKUP_ACTIONS = (
    "recall",
    "search_deferred",
    "fetch_deferred",
    "deep_research",  # 別モデルに深く調べさせる（出-dh）
    "see",
    "look",
    # MCP の同期の道具は即座に返るが、それを見て何を言うかは次の反復が決める（`recall` と
    # 同じ）。**主LLM は道具名で呼ぶ**ので、動作名だけでなく道具名も並べる（2026-09-13 まで
    # 動作名 `house_rules` しか無く、主LLM が `get_house_rules` を呼んでも捨てられていた）。
    *_MCP_LOOKUPS.keys(),
    *_TIMER_ACTIONS,
    *_ALARM_ACTIONS,
    *_PRINT_ACTIONS,
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
    if action in _PRINT_ACTIONS:
        what = str(tool_input.get("title") or "").strip()
        return "印刷する" + (f"「{what}」" if what else "")
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


def _searches(req, *, arrived: bool) -> int:
    """この求めの調べものの数（失敗は数えない）。`arrived` なら結果が届いたものだけ、でなければ飛んでいるものも数える。"""
    return sum(
        1
        for lk in req.lookups
        if _action_family(lk.action) == "search_deferred"
        and not lk.failed
        and (lk.result is not None or not arrived)
    )


def _search_in_flight(req) -> bool:
    """この求めに、結果のまだ届いていない調べもの（search_deferred・deep_research）があるか（出-dl）。"""
    return any(
        _action_family(lk.action) in ("search_deferred", "deep_research") and lk.in_flight
        for lk in req.lookups
    )


def _search_arrived(req) -> bool:
    """この求めに、結果の届いた調べものがあるか（出-dh）。失敗と、使い切って投げなかった器は数えない。"""
    return any(
        _action_family(lk.action) == "search_deferred"
        and bool(lk.result)  # 0 件（空の結果）は、答えに使える結果ではない
        and not lk.failed
        and not is_spent(lk.result)
        for lk in req.lookups
    )


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
        logger.debug("季節とまわりを読めなかった（無しで続ける）: %s", e, exc_info=True)
        return ""


def _self_image_text() -> str:
    """自己像（層 2）の枠。DB の現在値（無ければ種）を文にする。読めなければ空（degrade）。"""
    try:
        from ..core import self_image

        return self_image.render(self_image.load())
    except Exception as e:  # noqa: BLE001
        logger.debug("自己像を読めなかった（無しで続ける）: %s", e, exc_info=True)
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


def _widen_lines(ws, k: int, now: datetime) -> str:
    """引き直した記憶を、`recall` と同じ読み方の行にする（出-ah）。載せるのは `k` 件まで。

    W と同じく、`now` からの距離を添える（出-ds）。記憶（O）に書く文では外す（`workspace.strip_ago`）。
    """
    from .workspace import _lines, about_names, me_of, named

    picked = sorted(ws.memories, key=lambda r: r.fit, reverse=True)[:k]
    if not picked:
        return "その引き方では、思い出せるものが無かった"
    names: dict = {}
    with contextlib.suppress(Exception):
        names = dict(ws.oif.actors([r.mi.obs_id for r in picked]))
    return _lines(picked, named(names, me_of(ws.oif)), about=about_names(ws.oif, picked), now=now)


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
        logger.warning("ストップウォッチの枠を組めなかった: %s", e, exc_info=True)
        return ""


def _alarm_def(agent, name: str) -> list[dict]:
    """アラームの道具定義から 1 つだけ。器が無ければ空（知-q）。"""
    tool = getattr(agent, "_alarm_tool", None)
    if tool is None:
        return []
    return [d for d in tool.get_tool_definitions() if d.get("name") == name]


def _print_def(agent) -> list[dict]:
    """印刷の道具定義。器が無い、PDF を作れない（Qt のアプリが無い・画面の無い起動）なら空（知-bo）。

    作れないのに渡すと、家族に「刷るね。いい？」と聞いてから刷れないと言うことになる。
    """
    tool = getattr(agent, "_print_tool", None)
    if tool is None or not tool.can_render():
        return []
    return tool.get_tool_definitions()


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
        logger.warning("アラームの枠を組めなかった: %s", e, exc_info=True)
        return ""


def _timer_frame(agent) -> str:
    """`[タイマー]` の枠（動いているもの・直前に鳴ったもの）。器が無い・読めなければ空（degrade）。"""
    tool = getattr(agent, "_timer_tool", None)
    if tool is None:
        return ""
    try:
        return str(tool.frame() or "")
    except Exception as e:  # noqa: BLE001
        logger.warning("タイマーの枠を組めなかった: %s", e, exc_info=True)
        return ""


def _looks_like_text_tool_call(text: str) -> bool:
    """文で書かれた道具呼び出しらしいか（`<invoke>` か、先頭が `名前(` で末尾が `)`）。"""
    t = text.strip()
    return "<invoke name=" in t or bool(
        re.match(r"^[A-Za-z_][A-Za-z0-9_]*\(", t) and t.endswith(")")
    )
