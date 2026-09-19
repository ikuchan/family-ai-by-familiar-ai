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
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..backends.types import TurnResult

from ..poses import nearest_pose
from ..scene import extract_entities
from ..store import clock
from .arbiter import Decision as ArbiterDecision, arbitrate
from ..store.relations import KIND_EXCHANGE, KIND_RESOLVE, KIND_REVISION
from ..io.dif import DIF
from ..core import measure, parsing
from ..core.silence_hold import Heard
from ..core.tool_gate import gate_personal_tools
from ..core.tool_text import tool_calls_from_text
from ..io.oif import MI, Recalled
from ..person_memory_manager import AGENT_SELF_ID
from .coherence import facts_ctx, used_lines
from .generator import _iter_ctx, _pi_ctx, _present_ctx
from . import reply_budget, workspace
from .request import Lookup, Request
from .prompt import build_event_system_prompt

logger = logging.getLogger(__name__)

# 連鎖が続けられる反復で渡す動作。上限に達した反復では say だけにして必ず閉じる。
_FULL_ACTIONS = (
    "say",
    "recall",
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
    "start_stopwatch",  # ストップウォッチ（知-u・別物）
    "stop_stopwatch",
)
_TIMER_ACTIONS = ("set_timer", "cancel_timer", "pause_timer", "resume_timer")
# ストップウォッチ（知-u・2026-09-18）。タイマーとは別物・別の道具（`agent._stopwatch_tool`）。
_STOPWATCH_ACTIONS = ("start_stopwatch", "stop_stopwatch")
# アラーム（知-q・2026-09-18）。タイマーとは別物・別の道具（`agent._alarm_tool`）。
_ALARM_ACTIONS = ("set_alarm", "cancel_alarm")
# 確認待ちへの答え（出-y・2026-09-18）。預かりが生きているあいだだけ候補と道具に載る（`agent.confirm_alive`）。
# 機械が預かった入力で掛ける／捨てる。返りは道具の帰りと同じ道（想起なし・`workspace.RETURN_WITHOUT_RECALL`）。
_CONFIRM_ACTIONS = ("confirm", "decline")
#: 在席が無い名乗りを預かる秒数（知-w-ろ・2026-09-19・本人が 30 と決めた）。テレビの名乗りの直後に人が入る誤りは短さで受ける。
PENDING_CLAIM_SEC = 30.0
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
    "journal": ("get_journal", "日次記録を見る"),
    "get_journal": ("get_journal", "日次記録を見る"),
    # 個人ティアの記録（`obsidian-memo`・知-g-い）。道具名は話者の英字で決まる（`ask_vault_<latin>`）
    # ので、表には動作名だけを置き、道具名は `_vault_tool_name()` が解く。数十秒かかるが、
    # 調べものは投げっぱなしなので別経路は要らない（MCP の時間切れだけ長い・`call_timeout_for`）。
    "vault": ("ask_vault_", "記録に「{query}」を聞く"),
}
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
)


async def _result_or_failure(task) -> "tuple[str | None, bool]":
    """待ち合わせて (結果, 落ちたか) を返す。落ちたら (None, True)（繋がない側へ倒す）。

    判定が来ないターンは新しい話の始まりとして扱う。誤って繋ぐと、関係のない会話が
    文脈に混ざる（`設計方針_MI間の関係`）。落ちたことは計測ログに残す（記-i）。
    """
    try:
        return await task, False
    except Exception as e:  # noqa: BLE001
        logger.debug("続き先の判定を受け取れなかった（続行する）: %s", e)
        return None, True


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
        return {"confirm": "「いい」と言われて掛ける", "decline": "確かめたものをやめる"}[action]
    if action == "look":
        return f"{tool_input.get('pose') or tool_input.get('direction') or ''}を見に行く"
    if action.startswith("ask_vault_"):
        action = "vault"
    if action in _MCP_LOOKUPS:
        return _MCP_LOOKUPS[action][1].format(
            query=str(tool_input.get("query") or tool_input.get("question") or "").strip(),
            days=_days_of(tool_input),
        )
    return str(tool_input.get("query") or tool_input.get("url", "")).strip()


SCHEDULE_DAYS_MAX = 14  # カレンダー MCP の `days` の上限（`mcp/calendar_mcp/server.py` と同じ）


def _days_of(tool_input: dict) -> int:
    """入力の `days`（無ければ道具の既定 1）。見出しの期間に使う。"""
    try:
        return max(1, min(SCHEDULE_DAYS_MAX, int(tool_input.get("days", 1))))
    except (TypeError, ValueError):
        return 1


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
    if action.startswith("ask_vault_"):
        return "vault"
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


def _timer_def(agent, name: str) -> list[dict]:
    """タイマーの道具定義から 1 つだけ。器が無ければ空（知-n）。"""
    tool = getattr(agent, "_timer_tool", None)
    if tool is None:
        return []
    return [d for d in tool.get_tool_definitions() if d.get("name") == name]


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


def _elapsed_label(created_at, now_epoch: float) -> str:
    """いつのことかを「経過時間（時刻）」で書く。片方だけでは足りない。"""
    with contextlib.suppress(Exception):
        stamp = created_at.timestamp()
        hours = (now_epoch - stamp) / 3600.0
        ago = f"{int(hours * 60)}分前" if hours < 1 else f"約{int(hours)}時間前"
        return f"{ago}（{created_at.astimezone().strftime('%m/%d %H:%M')}）"
    return "いつか"


def _log_recall_weights(trigger, base, used, memories) -> None:
    """採用した5軸重みと、その重みで出た上位のスコアを残す（INFO）。

    重みは反復ごとに揺らぐので、後から「どの重みでどう並んだか」を紐づけられないと、
    値を実挙動から選べない。**記憶の内容は出さない**（INFO 以上に会話・記憶内容を出さない
    方針。中身は DEBUG の `recall score` の内訳にある）。
    """

    def _fmt(w):
        return "(%.2f,%.2f,%.2f,%.2f,%.2f)" % (w.w_r, w.w_t, w.w_e, w.w_g, w.w_p)

    top = "/".join("%.3f" % r.fit for r in memories[:3])
    logger.info(
        "event-loop 想起 trigger=%s w=%s 基底=%s 上位=%s %d件",
        trigger,
        _fmt(used),
        _fmt(base),
        top or "なし",
        len(memories),
    )


@dataclass
class Decision:
    """主LLM の返りと、**投げたときに見ていたもの**（環-h・段は）。

    主LLM は投げっぱなしになり、返りは別の反復（出す反復）で実行される。そのあいだに別の
    完了が届けば、W も対応表も次の反復のもので作り直される。だから**返りと一緒に運ぶ**。
    に-5-に-2 で対応表は属性でなくなったので、いまは**ここが唯一の持ち場**である。

    - `memories`：共起は「その反復で一緒に活性した記録」なので、**主LLM が見た W** でなければ
      意味がない
    - `w_id_map`：申告（`memory_verdicts`）は W に印字された12桁で返るので、その W を作った
      ときの対応表でないと引けない
    - `mem`：**申告を当てる面**。`situated_memories` は人ごとで、想起は話者の面を通る
      （`_active_memory()`）。基底の記憶へ書くと視点が `__self__` へ寄り、話者が同定
      されている場面で申告が0行に当たる（出-h-ろ ③）
    - `system`・`effort`：整合チェックの差し戻しで**主LLM をもう一度呼ぶ**のに要る
    - `capped`：上限の反復では調べる動作を渡していないので、返ってきても投げない
    - `retried`：これは言い直しの返りか。真なら**もう検査しない**（1回だけ）
    - `original_text`：言い直しが `say` を返さなかったときの戻り先。差し戻しが同期
      だったころは同じ関数の変数だったので運ぶ必要がなかったが、投げっぱなしにすると
      失われる（段は-2）
    """

    result: "TurnResult"
    memories: "list[Recalled]"
    w_id_map: dict[str, str]
    mem: object
    recent_frame: str  # W の直近の枠（整合チェックが「さっき何を言ったか」として見る）
    system: object
    effort: "str | None"
    capped: bool
    max_tokens: int = 0  # 返事の予算（出-k-ろ）。言い直しも同じ値で呼ぶ
    retried: bool = False
    original_text: str = ""


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


@dataclass
class Trigger:
    """**きっかけ**：W 構築を起動する出来事。IIF の待ち行列に並ぶ1件（環-f-い-1）。

    用語一覧の「きっかけ（trigger）」そのものである——**会話入力・知覚イベント・情動発火・
    完了の4つ**。`Completion`（資源から返ってきたもの）という名前だったが、情動と機器も同じ
    列に並ぶようになったので、中身に合う語へ改めた（2026-09-11）。

    以前は `(語, 結果, 意図id, 種別, 番号)` の5つ組で、**位置で意味が決まっていた**。種別に
    よって埋まる欄が違う（`進捗` は結果も番号も持たない）のに、位置で運んでいた。

    | 種別 | いつ | 埋まる欄 |
    |---|---|---|
    | `完了` | 調べものが終わった | `query`・`result`・`index` |
    | `進捗` | 調べものが遅い（`_watch_slow_lookup`） | `query` |
    | `決定` | 主LLM が返った | `decision`（`Decision`） |
    | `情動` | drive が発火した（AIF 経由） | `query`（drive の名）・`result`（促しの文） |
    | `機器` | 人が出入りした（DIF 経由） | `query`（種別）・`result`（中身）・`release_pending` |
    | `会話入力` | 人が話しかけた | `query`（人の言葉）・`future`（呼び手が待っている） |

    **`会話入力` だけが返事を持って帰る。** 呼び手（GUI・CUI）はその反復の出力を待っている
    ので、`future` に入れて返す。待っているのは呼び手だけで、**反復は駆動体の上で回る**
    （環-f-い-2）。

    **種別に `発話` を使わない。** その語はコードの中で3つの別物を指している——人が話しかけた
    こと・パジュがつないだ一言・パジュが答えたこと（`direction="発話"` として DB にも入って
    いる）。用語一覧が定めるきっかけの語は `会話入力` である。

    **`決定` は投げたときの W を一緒に運ぶ。** 共起は「その反復で一緒に活性した記録」なので
    主LLM が実際に見た W でなければ意味がなく、申告（`memory_verdicts`）は W に印字された
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
    # `機器` だけが使う。在席がゼロから立ち上がった瞬間に真で、保留した発話を先に配る。
    release_pending: bool = False
    # 確かめて掛けたタイマーが鳴った（知-n）。配信ゲートを通り抜ける。
    passes_gate: bool = False
    # `会話入力` だけが使う。呼び手がここで返事を待っている。
    future: "asyncio.Future[str] | None" = None


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
        # 要素＝(何を探したか, 結果, 起点の open 意図 id, 種別)。種別＝完了｜進捗。
        # 「進捗」は結果ではないので、飛行中の数も一覧も触らず、意図も supersede しない。
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

        # 直前に書いた版の id。`recall` ツールが自分自身を拾わないための除外に使う。
        self._recall_exclude_id: str | None = None
        # 求めの世代。打ち切るたびに1つ進める。**走っている反復と、飛んでいる調査の完了**を
        # 古い世代として捨てるのに使う。打ち切りの時点で外部呼び出しは既に飛んでおり、
        # 反復もフルLLM の返りを待っている最中なので、止めるには番号で見分けるしかない。
        self._request_generation = 0
        # 「まだかかっている」を受けたか。次の反復でつなぎだけ出して閉じない。
        self._slow_notice_received = False
        # see の帰りで起きた反復か。調停を飛ばして主LLM へ戻す（`_decide`）。
        self._see_returned = False
        self._background_tasks: set[asyncio.Task] = set()
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
        # 駆動体（キュー到来で次の反復を起こす）と、そこへ渡す取込待ちの完了。
        self._driver: asyncio.Task | None = None
        self._asyncio_loop: asyncio.AbstractEventLoop | None = None
        self._drained_completions: list[Trigger] = []
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
        # 在席が無くて使えなかった名乗り（名前・時刻）。30 秒預かる（知-w-ろ）。
        self._pending_claim: "tuple[str, float] | None" = None

    def _note_origin(self, obs_id: str | None) -> None:
        """このターンの起点を控える。

        **何に続くかはここで決めない。** 続き先は、そのターンを作るのに使った W の中に
        しかない（`workspace.link_follows`）。段 3 では「直前の起点へ無条件に繋ぐ」形に
        していたが、
        それは鎖の種類を機構の側で数え上げることになり、並行して走る本数に上限が生まれた。
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
        agent = self._agent
        content = self._version_content(aborted=aborted)
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
        if version_id:
            self._note_record(version_id, "版")
            if self._req.live_version_id and self._req.live_version_id != version_id:
                agent._oif.supersede(self._req.live_version_id, version_id, kind=KIND_REVISION)
            self._req.live_version_id = version_id
            # 手がかり（次の反復の想起クエリ）は、いまの版そのものにする。
            self._req.cue = content
        return version_id

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
        if image_path:
            self._req.seen_image_path = image_path
        return obs_id

    def _build_system(
        self, *, present_ctx: str, workspace_ctx: str, iter_ctx: str
    ) -> "tuple[str, str]":
        """主LLM の system 文（安定部・可変部）を組む。材料の出所はここに集める。"""
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
        return build_event_system_prompt(
            self_understanding=load_summary() or getattr(agent, "_me_md", ""),
            family_md=getattr(agent, "_family_md", ""),
            self_image=_self_image_text(),
            present_ctx=present_ctx,
            pi_ctx=_pi_ctx(self._req),
            iter_ctx=iter_ctx,
            workspace_ctx=workspace_ctx,
            # 角括弧タグを許すかは合成の担い手が決める（`根拠台帳` §9）。
            allow_tts_tags=self._dif.understands_tags,
            origin=self._req.trigger_kind,
            # 見回りの行き先の材料（知-c）。自発の求めだけ。
            now_note=self._patrol_note() if self._req.trigger_kind == "情動" else "",
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

        画像を受け取るのは主LLM だけである（調停・整合チェック・申告の軽量LLM は文字だけ）。
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
        agent._oif.supersede(old_id, new_id, kind=KIND_REVISION)
        if gen != self._request_generation:
            return
        self._req.turn_records = [
            (new_id if (i == old_id and r == "見た") else i, r) for i, r in self._req.turn_records
        ]

    def _version_content(self, *, aborted: bool = False) -> str:
        """いまの求めの状態を、1つの版の content として組み立てる。

        求めは1本の版チェーンとして進み、各版が状態を表す。**どの版にも求めそのものを
        入れる**（前の版は畳まれて想起の候補から外れるので、辿る道が残らない）。

        並行する調査は通し番号で列挙し、鎖は分岐させない。状態は「求め」の状態であって
        個々の調査の状態ではないので、3件飛んでいても求めの状態はひとつである。

        結果はここでは切らない。切るなら書き込みの上限で切る。ここで切ると、どこで短く
        なったのかが追えなくなる。
        """
        parts: list[str] = []
        for lk in sorted(self._req.lookups, key=lambda x: x.index):
            if lk.action == "主LLM":
                # 主LLM には探す語が無く、**返りの中身も版には載せない**（`see` と同じで、
                # 中身は `_finish` の「自分が答えた」が持つ）。求めの状態としては
                # 「何番が返ったか」だけあればよい。
                if aborted:
                    parts.append(f"{lk.index}番：考えるのを打ち切った")
                elif lk.in_flight:
                    parts.append(f"{lk.index}番：考えている")
                else:
                    parts.append(f"{lk.index}番：考えて答えた")
                continue
            if lk.in_flight:
                verb = "を打ち切った" if aborted else "を起動中"
                parts.append(f"{lk.index}番：{lk.action}「{lk.query}」{verb}")
            elif lk.failed:
                # 結果ではない。生のエラー文は載せない（対処できる情報ではない・出-o）。
                parts.append(f"{lk.index}番：{lk.action}「{lk.query}」は道具が使えず失敗した")
            else:
                parts.append(f"{lk.index}番：{lk.action}「{lk.query}」の結果が届いた：{lk.result}")

        head = f"「{self._req.request_text}」と聞かれた"
        if not parts:
            return head + ("（打ち切った）" if aborted else "")
        return f"「{self._req.request_text}」と聞かれ、" + "／".join(parts)

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
            self._triggers.put_nowait(
                Trigger(
                    kind="完了",
                    query=query,
                    result=f"「{query}」はこの求めですでに調べた。結果は W にある。",
                    intent_id=intent_id,
                    index=seen.index,
                )
            )
            return

        index = self._next_lookup_index()
        self._req.lookups.append(
            Lookup(index=index, action=action, query=query, generation=self._request_generation)
        )
        task = asyncio.create_task(self._run_lookup(action, tool_input, query, intent_id, index))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        watch = asyncio.create_task(self._watch_slow_lookup(query, self._request_generation))
        self._background_tasks.add(watch)
        watch.add_done_callback(self._background_tasks.discard)

    async def _watch_slow_lookup(self, query: str, gen: int) -> None:
        """調べものが遅いとき、**1回だけ**「まだかかっている」を積む（案G-3・案イ）。

        時計で定期的に起こすのではなく、**遅いという事実**が起点になる。繰り返すと結局
        「一定時間ごとに言う」になるので、1回で終える。結果が先に来たら何もしない
        （その時点で飛行中の一覧から消えている）。
        """
        seconds = float(getattr(self._agent.config, "lookup_slow_seconds", 5.0))
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(seconds)
            if gen != self._request_generation:
                return  # 打ち切られた求めの見張り
            lk = self._lookup_of(query)
            if lk is None or not lk.in_flight:
                return  # もう結果が来ている
            logger.info("event-loop 調べものが %.0f 秒を超えた：%.40s", seconds, query)
            self._triggers.put_nowait(Trigger(kind="進捗", query=query))

    def _dispatch_main_llm(
        self,
        *,
        messages: list,
        system,
        effort: "str | None",
        capped: bool,
        memories: "list[Recalled]",
        w_id_map: dict[str, str],
        mem: object,
        recent_frame: str,
        max_tokens: int,
        retried: bool = False,
        original_text: str = "",
    ) -> None:
        """主LLM を投げる（**待たない**・環-h・段は）。

        調べものと同じ扱いにする——求めの台帳（`self._req.lookups`）へ1件積んで飛行中に数え、版に載せ、世代で
        打ち切れるようにする。**重複の判定は通さない**（同じ求めで何度も呼ぶ）。

        整合チェックの差し戻し（言い直し）も**この口から投げる**。同じ口を通るので、
        言い直しも `action="主LLM"` として積まれ、**考えた回数に数えられる**
        （2026-09-09 の決定）。
        """
        index = self._next_lookup_index()
        self._req.lookups.append(
            Lookup(
                index=index,
                action="主LLM",
                query=f"主LLM{index}",
                generation=self._request_generation,
            )
        )
        task = asyncio.create_task(
            self._run_main_llm(
                index=index,
                messages=messages,
                system=system,
                effort=effort,
                capped=capped,
                memories=memories,
                w_id_map=w_id_map,
                mem=mem,
                recent_frame=recent_frame,
                max_tokens=max_tokens,
                retried=retried,
                original_text=original_text,
            )
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

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
        mem: object,
        recent_frame: str,
        max_tokens: int,
        retried: bool,
        original_text: str = "",
    ) -> None:
        """RH：主LLM を呼び、返りを QC へ積む（投げっぱなしの担い手）。"""
        agent = self._agent
        started = time.monotonic()
        try:
            result, _raw = await agent.backend.stream_turn(
                system=system,
                messages=messages,
                # 発話だけに絞る理由は2つあり、**別のことである**（1つの名前へまとめない）。
                # `capped`＝連鎖上限なので、調べさせずに必ず閉じる。
                # `retried`＝言い直しなので、答え直すだけでよい（調べ直すためではない）。
                tools=self._tools(actions=("say",) if capped or retried else _FULL_ACTIONS),
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
                decision=Decision(
                    result=result,
                    memories=memories,
                    w_id_map=w_id_map,
                    mem=mem,
                    recent_frame=recent_frame,
                    system=system,
                    effort=effort,
                    capped=capped,
                    max_tokens=max_tokens,
                    retried=retried,
                    original_text=original_text,
                ),
            )
        )

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
        if action in _MCP_LOOKUPS or action.startswith("ask_vault_"):
            # MCP の同期の道具。結果はその場で返るので、完了として積む（`recall` と同じ）。
            # `vault` は話者の英字で道具名が決まる（知-g-い）。主LLM が道具名で呼んだときはそのまま。
            tool_name = (
                action
                if action.startswith("ask_vault_")
                else (self._vault_tool_name() if action == "vault" else _MCP_LOOKUPS[action][0])
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
        if action != "recall":
            try:
                text, dispatched = await self._dif.lookup(action, tool_input)
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
        try:
            out, _ = await self._agent._memory_tool.call(
                "recall",
                tool_input,
                exclude_ids=[self._recall_exclude_id] if self._recall_exclude_id else None,
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

        progress = [c for c in items if c.kind == "進捗"]
        items = [c for c in items if c.kind != "進捗"]
        if progress:
            # 「まだかかっている」は結果ではない。飛行中の数も一覧も触らず、意図も
            # supersede しない。次の反復で、調停に短い一言を書かせるためだけに起こす。
            self._slow_notice_received = True
        decided: "Decision | None" = None
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
            lk = self._lookup_of(query)
            if lk is None:
                # 対応する開いた意図が無い完了（打ち切り後に遅れて届いた等）。無言で落とさない。
                logger.warning(
                    "event-loop 取込：対応する意図が無い完了を捨てる（query=%.40s）", query
                )
            action = lk.action if lk is not None else "recall"
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
                if lk.index not in self._req.just_returned:
                    self._req.just_returned.append(lk.index)  # W の最上部に載せる（出-x）
                if lk.failed:
                    # その求めのあいだ、この道具を候補から外す（構造で呼び直しを起こさない）。
                    self._req.failed_actions.add(_action_family(lk.action))
        if items:
            # 求めの新しい版を書き、直前の版を畳む（1本の鎖）。
            await self._write_version()
        return len(items), decided

    # この反復で使える動作の表。値＝その動作のツール定義を取り出す関数で、引数はループ。
    # 身体を1つ繋ぐたびにここへ1行足すだけで済むようにしてある（see・look・net など）。
    # **口が持っている機器の定義は口が答える**（`_dif`）。まだ口を通していないものだけが
    # `ip._agent` を見る（カメラは 段3・記憶は OIF の担当）。
    _ACTIONS: dict = {
        "say": lambda ip: ip._dif.speak_defs(),
        "recall": lambda ip: [
            d for d in ip._agent._memory_tool.get_tool_definitions() if d.get("name") == "recall"
        ],
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
        "journal": lambda ip: ip._dif.tool_defs("get_journal"),
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
        "start_stopwatch": lambda ip: _stopwatch_def(ip._agent, "start_stopwatch"),
        "stop_stopwatch": lambda ip: _stopwatch_def(ip._agent, "stop_stopwatch"),
        # 確認待ちへの答え（出-y）。預かりが生きているあいだだけ。
        "confirm": lambda ip: _confirm_defs(ip._agent, "confirm"),
        "decline": lambda ip: _confirm_defs(ip._agent, "decline"),
    }

    def _vault_tool_name(self) -> str:
        """話者の英字から `ask_vault_<latin>` を解く。本人の道具が無ければ空（呼べない）。"""
        names = {str(d.get("name", "")) for d in self._gated(self._ACTIONS["vault"](self))}
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
            )
            if a not in self._req.failed_actions
            and a not in exclude
            and not (timer_ringing and a in ("set_timer", "start_stopwatch"))
            and self._gated(self._ACTIONS[a](self))
        )

    def _action_of_query(self, query: str) -> str:
        """その語をどの動作で投げたか。分からなければ recall とみなす。"""
        lk = self._lookup_of(query)
        return lk.action if lk is not None else "recall"

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
        agent = self._agent
        self._req.utterance = utterance
        self._req.trigger_kind = kind
        self._req.began_at = datetime.now(timezone.utc)  # 人が言った瞬間（タイマーの起点・知-n）
        self._req.request_text = text[:500]
        self._req.live_version_id = None
        self._req.lookups.clear()
        self._req.iterations = 0
        self._req.iterations_capped = False
        # 明けた瞬間の求めにだけ載る（情-h）。`__new__` で組んだ装置（テスト）でも落ちないよう既定を持つ。
        self._req.heard_while_silent = getattr(self, "_pending_heard", [])
        self._pending_heard = []
        # 確認待ち（出-y）。この求めの W の最上部に載せる。預かりが無い・寿命切れなら空。
        self._req.confirm_frame = self._confirm_frame()
        await self._apply_pending_claim()  # 預かった名乗り（知-w-ろ）
        # **人の言葉は、その人がやったことである。** `actor` の面（`situated_memories`）は
        # 話者に立てる。想起は `_active_memory()`＝話者の面を引くので、`__self__` の面に
        # しか立てないと、**その人の面にはその人が言ったことが1件も無くなる**。
        #
        # 情動と機器はパジュ自身のことなので `__self__` でよい。3つの入口で違うのはここと、
        # 起点の種別・文面・`utterance` である。
        perspective = (
            agent._conversation_perspective()
            if kind == "発話"
            else agent._observation_perspective()
        )
        obs_id = await agent._oif.write(
            MI(id="", content=text[:500], timestamp=None, direction=kind),
            **perspective,
        )
        self._req.request_id = obs_id
        self._notify_request_state(True)
        # このターンを起こした記録を控え、前のターンとつなぐ。控えないと、問いだけが
        # やりとりの関係にも拡散想起の母集合にも入らない。
        self._note_origin(obs_id)
        self._req.cue = text[:500]

    async def push_utterance(self, utterance: str, on_text=None) -> str:
        """人の言葉を待ち行列へ積み、**その反復の出力**を返す（環-f-い-2）。

        4つのきっかけ（会話入力・知覚イベント・情動発火・完了）が同じ列に並ぶ。積む口が
        `push_completion`・`push_affect`・`push_device` と揃い、**順序づけは
        `_take_trigger()` の1箇所**にある。

        **返すのは求めの終わりではなく、最初の反復の出力である。** 1反復＝1出力なので、
        ツールを投げた反復は発話を持たず空文字を返す。続きは、完了が列へ届いて駆動体が
        起こす次の反復が担う。呼び手（GUI・CUI）が待っているのもここまでで、これは
        `begin_request` だったころと同じ意味である。

        **打ち切りはここでやる。** 積む前に前の調査を止める。駆動体側へ移すと、走っている
        反復が終わるまで打ち切りが遅れる（調停の時間切れなら最大5秒）。打ち切りは「人が
        言い直した」瞬間の判断であって、順序づけではない。

        `on_text` は出力先（駆動体が起こす反復も使う）。
        """
        agent = self._agent
        # 人が話しかけた時刻の印。**在席の証拠には使わない**（2026-09-17：マイクはテレビ・物音・
        # 聞き違いを拾う）。使い道は「ひとりの回数」のリセット（情-d・`step_drives`）と記録。
        agent._last_human_at = time.time()
        self._on_text = on_text or self._on_text
        self._ensure_driver()

        # 調べかけの途中に話しかけられたら、**その調査を打ち切る**。人が言い直したとき、
        # 前の調査を続ける意味はない（実機で「これはどこの地方の天気？」に答えられず、
        # 言い直されたあとも同じ検索を繰り返した）。結果は捨てるが、**何を打ち切ったかは
        # 記録に残す**。
        await self._abort_lookups()

        fut: "asyncio.Future[str]" = asyncio.get_running_loop().create_future()
        self._triggers.put_nowait(Trigger(kind="会話入力", query=utterance, future=fut))
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
        if dropped or drained:
            logger.info(
                "event-loop 調べかけを打ち切る（%s／取り込まなかった完了 %d件）",
                "・".join(dropped) or "投げた先なし",
                drained,
            )
        if self._req.request_id:
            # 打ち切りも版のひとつ。何を打ち切ったかは版の content が持つので、
            # **飛行中の一覧を消す前**に、かつ親を捨てる前に書く。
            with contextlib.suppress(Exception):
                await self._write_version(aborted=True)
            # ここまでが一つのやりとりである。打ち切りの版の親は、この求めの起点だから
            # である。閉じないと並びが次のターンへ持ち越され、一つのやりとりに起点が2つ
            # 入る。答えも要約も無いやりとりになるが、それが起きた事実そのものである。
            _aborted = self._close_exchange()
            if _aborted:
                with contextlib.suppress(Exception):
                    self._agent._oif.link(
                        KIND_EXCHANGE, [(i, r, n) for n, (i, r) in enumerate(_aborted)]
                    )
        self._req.request_id = None
        self._req.live_version_id = None
        self._req.lookups.clear()
        self._req.cue = ""
        self._req.said_fillers.clear()
        self._req.speech_to_deliver.clear()
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
        self, query: str, result: str, index: int = 0, *, failed: bool = False
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
        _lk = self._lookup_of(query)
        if _lk is None or _lk.generation != self._request_generation:
            logger.info("event-loop 打ち切った求めの完了なので捨てる：%.40s", query)
            return
        loop = getattr(self, "_asyncio_loop", None)
        item = Trigger(
            kind="完了",
            query=query,
            result=str(result),
            index=index or (_lk.index if _lk is not None else 0),
            failed=failed,
        )
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._triggers.put_nowait, item)
        else:
            self._triggers.put_nowait(item)

    def push_affect(self, drive_name: str, prompt: str) -> None:
        """T が drive 発火を待ち行列へ積む（AIF 経由・I は時計を見ない）。"""
        self._triggers.put_nowait(Trigger(kind="情動", query=drive_name, result=prompt))

    def push_device(
        self, kind: str, content: str, *, release_pending: bool = False, passes_gate: bool = False
    ) -> None:
        """T が人の出入りを待ち行列へ積む（DIF 経由・I は時計を見ない）。`passes_gate` はタイマー（知-n）。"""
        self._triggers.put_nowait(
            Trigger(
                kind="機器",
                query=kind,
                result=content,
                release_pending=release_pending,
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
            utterance = next((t for t in self._held + fresh if t.kind in _NEVER_HELD_KINDS), None)
            if utterance is not None:
                if utterance in self._held:
                    self._held.remove(utterance)
                self._held.extend(t for t in fresh if t is not utterance)
                return utterance
            if self._in_flight_count:
                # 調査中は新しい求めを待たせる。取りこぼしではなく待たせるだけ。
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
                if trigger is not None and await self._swallow_if_unheard(trigger):
                    continue
                if trigger is None:
                    logger.debug(
                        "event-loop 駆動体が完了を受領（id=%s inbox=%d）",
                        id(self),
                        len(self._drained_completions),
                    )
                    await self._iterate()
                elif trigger.kind == "会話入力":
                    logger.debug("event-loop 駆動体が会話入力を受領")
                    await self._begin_utterance(trigger)
                elif trigger.kind == "機器":
                    logger.debug("event-loop 駆動体が機器を受領（%s）", trigger.query)
                    await self._begin_device(
                        trigger.query,
                        trigger.result,
                        trigger.release_pending,
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
        task = asyncio.create_task(self._utterance_iteration(trigger.query))

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

    async def _utterance_iteration(self, utterance: str) -> str:
        """人の言葉を O へ書き、1反復回す。**打ち切りは `push_utterance` が済ませている。**"""
        await self._begin_request(kind="発話", text=utterance, utterance=utterance)
        return await self._iterate()

    async def _begin_affect(self, drive_name: str, prompt: str) -> None:
        """情動で新しい連鎖を始める。取込＝来た事実（情動）を O に書き、鎖の起点にする。

        情動は中身を持たないので、取り込み時に想起で状況づける（正本③ 手順1・2）。
        """
        await self._begin_request(kind="情動", text=f"[内的な促し:{drive_name}] {prompt}")
        self._req.fired_axis = str(drive_name or "").lower()  # 内部状態の言葉で明示する（情-f）
        await self._iterate()

    async def _begin_device(
        self, kind: str, content: str, release_pending: bool, *, passes_gate: bool = False
    ) -> None:
        """機器（人の出入り）で新しい連鎖を始める。取込＝来た事実を O に書き、鎖の起点にする。

        `release_pending` が真なら、聞く相手が居らず保留していた発話を先に配る。在席が
        ゼロから立ち上がった瞬間だけ真になる（寿命は `pending_speech` 側が持つので、
        新しいキューは作らない）。`passes_gate` は確かめて掛けたタイマー（知-n）。
        """
        await self._begin_request(kind="機器", text=f"[{kind}] {content}")
        self._req.passes_gate = bool(passes_gate)
        if release_pending:
            await self._release_pending_speech()
        await self._iterate()

    async def _release_pending_speech(self) -> None:
        """保留していた発話を取り出し、**W へ流す分として持つ**（鮮度切れは捨てる）。

        MI の content へ差し込まない。保留の記録（`direction="保留"`）は観測なので、想起でも
        W に上がってくる（実機のログで、入室の反復の想起上位4件が保留 O だった）。content に
        も差し込むと同じ話が二重に載る。

        **いつ言いたかったか**を添える。経過時間だけだと「23時台に言いたかった」という文脈が
        落ち、時刻だけだと日付をまたいだとき「昨夜」か「今朝」か決まらない。両方あれば、
        言葉を組み立てる側が自然な言い方を選べる。

        配った分は `pending_speech` から消し、元の O も supersede する（消さないと、想起で
        W に上がり続けて何度も蒸し返す）。
        """
        store = getattr(self._agent, "_pending_store", None)
        if store is None:
            return
        try:
            from ..config import PendingSpeechConfig

            cfg = PendingSpeechConfig()
            now_epoch = time.time()
            released: list[str] = []
            for row in store.list_active():
                score = store.freshness_score(row, now_epoch, cfg)
                if store.is_expired(row, score, cfg):
                    store.delete(row["id"])
                    continue
                content = str(row.get("content", "")).strip()
                if content:
                    released.append(
                        f"- {_elapsed_label(row.get('created_at'), now_epoch)}：{content}"
                    )
                store.delete(row["id"])
                with contextlib.suppress(Exception):
                    self._agent._oif.supersede(
                        row["observation_id"], self._req.request_id, kind=KIND_RESOLVE
                    )
            self._req.speech_to_deliver = released
            if released:
                # 何件を W へ流したかを残す。system プロンプトの全文は出していないので、
                # これが無いと「載ったが触れられなかった」のか「そもそも載っていない」のか
                # を区別できない（実機で、配られたのに発話が触れなかった）。
                logger.info("event-loop 保留を配る：%d件", len(released))
        except Exception as e:  # noqa: BLE001
            logger.exception("保留していた発話を取り出せなかった: %s", e)

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

    async def _iterate(self) -> str:
        """1反復：取込 → W 構築 → 生成 → 出力（発話 or ツール投げ）で終わる。"""
        from ..config import MemoryConfig

        agent = self._agent
        utterance = self._req.utterance
        # この反復が属する世代。打ち切られたら（世代が進んだら）、フルLLM の返りを待って
        # いる最中でも、出力せずに畳む。実機で、打ち切った直後に走っていた反復が
        # fetch_deferred を投げ、返事も1つ余計に出た。
        gen = self._request_generation
        max_chain = max(1, agent.config.event_max_iterations)
        # 1. 取込：駆動体が受けた完了を O に書き、open 意図を解決する。
        drained, decided = await self._intake()
        if decided is not None:
            # **出す反復は数えない。** 数えるのは軽量LLM が司る反復だけである。さらに
            # **主LLM の返りで 0 へ戻す**——主LLM が調査結果を見て「足りない」と判断した
            # なら、それは新しい一巡である。上限は暴走防止の安全弁であって、材料を見た
            # うえで再度調べることを止めるためのものではない
            # （`設計方針_主LLMを投げっぱなしにする` ⑤）。
            self._req.iterations = 0
            self._req.iterations_capped = False
            # **出す反復。** 想起も調停も回さない——回すと軽量LLM が主LLM の決定を覆せて
            # しまい、「そのまま出す」と矛盾する（`設計方針_主LLMを投げっぱなしにする`）。
            return await self._act_on_decision(decided, utterance=utterance, gen=gen)
        # ここから先は**決める反復**である（決定は上で捌いて返っている）。数えるのはここだけ。
        self._req.iterations += 1
        chain = self._req.iterations
        if drained:
            logger.debug("event-loop iter=%d/%d QC取込=%d件", chain, max_chain, drained)

        # 2. REC（想起）：O（＋現入力）→ W。W は派生なので反復末に捨てる。
        # 一律の規則：取込で書いた記録（＝鎖の先頭）は検索から外し、W へは決定的に加える。
        # 素通しだと問いと同一文の記録が必ず上位に来て、限られた枠から本物の記憶を押し出す。
        # 手がかりは「取り込んだもの」＝鎖の先頭（反復1なら人の発話、反復2以降なら完了 O）。
        # 最初の発話で探し続けると、いま届いた完了とは無関係な検索になる（④ の想起クエリ）。
        mem = agent._active_memory()
        # **誰の面から引くか。** `_active_memory()` と同じ選び方である（話者が居なければ
        # パジュ自身）。想起は口を通すので、面は `View.viewpoint` で言う（環-e-い）。
        # 申告（`apply_memory_verdicts`）はまだ記憶そのものを受け取るので `mem` も持つ。
        viewpoint = agent._pmm.current_speaker_id or AGENT_SELF_ID
        # **取込 O を候補から外さない。** 手がかりは取込の content そのものなので、候補に
        # 入れば必ず上位に来る。以前はこれを「枠を食う」と嫌って外していたが、いま届いた
        # 結果を全文で見せる必要がある以上、1位に来るのが正しい順位である。手組みで W へ
        # 足すのをやめ、候補集合の一員として同じ採点を通す（正本 [D-想起起動] の1本の流れ）。
        cue = self._req.cue or utterance
        _mcfg = MemoryConfig()
        # 5軸の重みは trigger 種別で決める（`課題5_パラメータ仮案` §280）。選ぶ基準は
        # 「この求めを何が始めたか」ではなく **「この反復を何を手がかりに動くか」**である。
        # 反復1の手がかりは人の言葉だが、完了が届いて起きた反復の手がかりは結果の本文で、
        # 性質が違う。`self._req.trigger_kind` を書き換えないのは、そちらが静穏時間のゲート
        # （`_delivery_block_reason`）に使われており、人に話しかけられて始まった求めを夜間に
        # 保留させてしまうためである。
        trigger = "完了" if drained else self._req.trigger_kind
        w_base = _mcfg.recall_weights(trigger)
        weights = _mcfg.jitter_weights(w_base)
        ws = await workspace.recall(
            agent._oif, cue, viewpoint=viewpoint, weights=weights, req=self._req
        )
        _log_recall_weights(trigger, w_base, weights, ws.memories)
        self._req.just_returned.clear()  # 「いま道具から返った」はこの反復の W にだけ載せる
        # 続き先の判定を投げる。**待たずに先へ進む。** 調停と並行して走らせれば、
        # 実測 0.72 秒（`根拠台帳` §29）はほぼ隠れる。受け取るのはシステム文を組む
        # 直前で、そこは待つ（続きでなければ直近のやりとりを載せてはいけない）。
        follows_task = asyncio.ensure_future(
            agent._evaluator.judge_follows(ws.for_main, utterance or "")
        )

        # 誰と話していると思って喋ったかを残す。これが無いと、口調がおかしいときに
        # 「話者が渡っていない」のか「渡ったが口調が従っていない」のかを切り分けられない。
        present_ctx = _present_ctx(agent)
        # **この求めで何回目に考えるか。** 主LLM の返りで反復は 0 へ戻るので、`iter=N/M`
        # だけではどの一巡か分からない（ログでも、渡す文脈でも同じ）。ここで1度だけ数え、
        # ログ・調停・主LLM の三箇所へ同じ値を渡す（別々に数えると食い違う）。
        round_ = self._thinking_round
        logger.debug(
            "event-loop iter=%d/%d 考え=%d回目 在席=%s", chain, max_chain, round_, present_ctx
        )

        # **上限は2つ。** 反復（1回の一巡の長さ）と、考えた回数（求め全体で主LLM を呼んだ数）。
        # 反復は主LLM の返りで 0 へ戻るので、輪が閉じたときは考えた回数だけが効く。
        capped = chain >= max_chain or self._thinking_capped
        if capped:
            # 上限で打ち切ったことは、後からログだけで判別できる必要がある（DEBUG の
            # iter=N/M からは「たまたま N 回で終わった」のか「打ち切った」のか分からない）。
            if self._thinking_capped:
                logger.info(
                    "event-loop 考えた回数 %d/%d 上限に達したため探索を打ち切る",
                    round_,
                    self._agent.config.max_thinking_rounds,
                )
            else:
                logger.info(
                    "event-loop 反復 %d/%d 上限に達したため探索を打ち切る", chain, max_chain
                )
            self._req.iterations_capped = True
        decision = await self._decide(
            utterance=utterance or self._req.cue,
            workspace_ctx=ws.for_arbiter,  # 同じ W・狭い窓（記-h）
            present_ctx=present_ctx,
            capped=capped,
            round_=round_,
            memories=ws.memories,
            returned=ws.returned_actions,
        )
        # 「いまは話しかけないで」と読めたら、その人が居るあいだ黙る。この反復の受け答えは
        # 出したうえで（頼みに無言で応じるのは不自然）、次の反復から止める。解くのも同じ口。
        await self._apply_requests(decision, utterance=utterance or self._req.cue)
        # 調停が時期を指した（「去年の夏の話」）なら、その基準で想起し直して W を組み直す。
        # 想起は調停より前に走るので、この反復に効かせるには引き直すしかない。実測 17〜50ms
        # で、指定があったときだけ走る。
        if decision.time_ref:
            with contextlib.suppress(Exception):
                ws = await workspace.recall(
                    agent._oif,
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
        memories, workspace_ctx, w_id_map = ws.memories, ws.for_main, ws.id_map

        if gen != self._request_generation:
            logger.info("event-loop 打ち切られた求めの反復なので畳む（調停後）")
            return ""

        # 「まだかかっている」で起きた反復は、**つなぎだけ出して閉じない**。求めは調査待ちの
        # まま続く。ここで light を選ばせると別の答えを出して終わってしまい、あとから届く
        # 結果に行き場が無くなる（案ハ）。
        if self._slow_notice_received:
            self._slow_notice_received = False
            await self._say_filler(decision.text)
            logger.info("event-loop 反復 %d/%d 出力=つなぎ（調べもの待ち）", chain, max_chain)
            return ""

        # (a') 情動の求めで調停が「黙る」（light・text 空）と決めた：主LLM を起こさず沈黙で閉じる（出-w）。
        if decision.branch == "light" and not decision.text and self._req.trigger_kind == "情動":
            logger.info("event-loop 調停が黙ると決めたので沈黙で閉じる（情動）")
            await self._finish("", memories, "沈黙")
            return ""
        # (a) 軽量で閉じる：フルLLM を起こさず、軽量LLM の応答で反復を終える。
        if decision.branch == "light" and decision.text:
            spoken, outcome = await self._speak(decision.text)
            # **記憶が育つ経路は申告1本しかない。** 主LLM を起こさない反復もそこを通す
            # （出-h-ろ）。聞くのは背景で、閉じるのは待たない。
            self._declare_light_memory_use(
                utterance=utterance or self._req.cue,
                reply=spoken or decision.text,
                workspace_ctx=workspace_ctx,
                w_id_map=ws.verdict_map,
                mem=mem,
            )
            await self._finish(spoken, memories, outcome)
            return spoken

        # (c) 定型：探すと決まっている反復も、フルLLM を起こさず投げて閉じる。
        if decision.branch == "action" and decision.query and not capped:
            await self._dispatch_arbiter_action(decision, utterance=utterance or self._req.cue)
            logger.info(
                "event-loop 反復 %d/%d 出力=%s（調停・続きは完了で起きる）",
                chain,
                max_chain,
                decision.action,
            )
            return ""

        # (b) 軽量つなぎ→フル（正本③ 段5 の内部二段）。フル生成は effort=high で10秒近く
        # かかり、そのあいだ無音になる。つなぎで体感の待ち時間を埋める。**1つの work の
        # 内部二段**であって別の出力ではない（1反復1出力は保たれる）。
        # effort=low は実測 0.8〜3.6 秒で返るので挟まない（かえってテンポが悪くなる）。
        # **材料が届いた反復でも挟まない**（`drained`）。待つものがもう無いのに「待って」と
        # 言う理由がない。実機では、検索結果が届いた1秒後に「うん、任せてね！」が出て、
        # 一言目（ですます）と本応答（ですます）のあいだでそこだけ口調が割れた。
        if decision.branch == "full" and decision.text and decision.effort != "low" and not drained:
            await self._say_filler(decision.text)

        # 判定は辺を書くだけで W は変えない（記-h）。結末は計測ログへ（記-i）。
        await self._note_follows(ws.for_main, utterance or "", w_id_map, follows_task)
        # 返事の予算（出-k-ろ）：長さは数字で渡し、`max_tokens` はそこから固定する。
        budget = reply_budget.decide(
            effort=decision.effort,
            researched=self._researched(),
            w_count=len(memories),
            origin=self._req.trigger_kind,
        )
        system = self._build_system(
            present_ctx=present_ctx,
            workspace_ctx=workspace_ctx,
            iter_ctx=_iter_ctx(
                chain=chain,
                max_chain=max_chain,
                thinking_round=round_,
                capped=capped,
                budget=budget,
            ),
        )
        # 生成中はストリームしない：ツールを選ぶ反復で出る前置きの地の文が表示され重複するため。
        # 起点が人の発話ならそのまま、情動・機器なら内的な出来事として渡す。空文字を送ると
        # 何がこの反復を起こしたのか分からなくなる（API も空メッセージを受け付けない）。
        user_msg = agent.backend.make_user_message(
            self._user_content(utterance or self._req.cue, memories)
        )
        # **投げて終わる。** 返りは QC を通り、次の反復（出す反復）が実行する。
        self._dispatch_main_llm(
            messages=[user_msg],
            system=system,
            effort=decision.effort,
            capped=capped,
            memories=memories,
            w_id_map=dict(ws.verdict_map),  # 申告の母数は過去の列だけ（出-n 4）
            mem=mem,
            recent_frame=ws.recent_text(ws.n_main),
            max_tokens=budget.max_tokens,
        )
        await self._write_version()
        logger.info(
            "event-loop 反復 %d/%d 考え=%d回目 出力=主LLM（続きは返りで起きる）",
            chain,
            max_chain,
            round_,
        )
        return ""

    async def _decide(
        self,
        *,
        utterance: str,
        workspace_ctx: str,
        present_ctx: str,
        capped: bool,
        round_: int,
        memories: "list[Recalled] | None" = None,
        returned: frozenset[str] = frozenset(),
    ) -> "ArbiterDecision":
        """この反復の分岐を決める。**see の帰りは、出した側が判断する。**

        `returned`＝この反復で道具から返ったもの（`Workspace.returned_actions`・出-x）。タイマー・
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
        from ..capability_state import load_summary

        agent = self._agent
        image_b64: "str | None" = None
        if self._see_returned:
            self._see_returned = False
            if self._req.see_by == "主LLM":
                logger.info("event-loop 主LLM が出した see の帰りなので調停を飛ばして主LLM へ戻す")
                # 見えたものを語るだけなので low（課題5 G 章「ループ側で決まる effort」）。
                return ArbiterDecision(branch="full", effort="low")
            found = self._seen_image(memories)
            image_b64 = found[0] if found else None
            logger.info(
                "event-loop 調停が出した see の帰りなので調停が判断する（写真%s）",
                "つき" if image_b64 else "なし",
            )
        # W の全文は DEBUG。調停が何を見て選んだかは、これが無いと後から追えない。
        logger.debug("event-loop 調停へ渡す W:\n%s", workspace_ctx)
        decision = await arbitrate(
            agent._utility_backend,
            utterance=utterance,
            workspace_ctx=workspace_ctx,
            self_understanding=load_summary() or getattr(agent, "_me_md", ""),
            family_md=getattr(agent, "_family_md", ""),
            self_image=_self_image_text(),
            present_ctx=present_ctx,
            now_ctx=f'(now :datetime "{clock.now_local_str()}")'
            + self._silence_note()
            + (self._patrol_note() if self._req.trigger_kind == "情動" else ""),
            capped=capped,
            thinking_round=round_,
            can_see=getattr(agent, "_camera", None) is not None,
            image_b64=image_b64,
            origin=self._req.trigger_kind,
            extra_actions=self._extra_actions(exclude=returned),
            tool_return=bool(returned & workspace.RETURN_WITHOUT_RECALL),
            timer_active=self._timer_active(),  # 操作の言葉の守り（出-aa）
        )
        # 何を選んだかは INFO（出-k-い の材料。DEBUG では実機で見えなかった）。
        logger.info(
            "event-loop 調停=%s effort=%s action=%s",
            decision.branch,
            decision.effort,
            decision.action if decision.branch == "action" else "-",
        )
        return decision

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

        if lookup_tc is not None:
            if say_tc is not None:
                await self._say_filler(str(say_tc.input.get("text", "")).strip())
            if lookup_tc.name in _CAMERA_ACTIONS:
                self._req.see_by = "主LLM"
            self._start_lookup(
                utterance or self._req.cue, dict(lookup_tc.input), action=lookup_tc.name
            )
            logger.info("event-loop 出力=%s（続きは完了で起きる）", lookup_tc.name)
            return ""

        if say_tc is not None:
            workspace.apply_memory_verdicts(
                decision.mem, say_tc.input.get("memory_verdicts"), decision.w_id_map
            )
            text = str(say_tc.input.get("text", "")).strip()
            # **1回だけ**言い直させる。言い直した応答は検査しない（際限なく往復させない）。
            violation = (
                None
                if decision.retried
                else await self._coherence_violation(
                    text,
                    decision.recent_frame,
                    decision.memories,
                    verdicts=say_tc.input.get("memory_verdicts"),
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
                logger.info("event-loop 整合チェックが違反を捕まえた：%s", violation)
                # 言い直しに根拠を添える（出-n）。違反の一文だけだと、検査しない 2 度目に
                # 別の捏造が出た（2026-09-13 16:32「東京は晴れ 30℃」）。
                facts = self._checker_facts(
                    decision.memories,
                    verdicts=say_tc.input.get("memory_verdicts"),
                    w_id_map=decision.w_id_map,
                )
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
                    system=decision.system,
                    effort=decision.effort,
                    capped=decision.capped,
                    memories=decision.memories,
                    w_id_map=dict(decision.w_id_map),
                    mem=decision.mem,
                    recent_frame=decision.recent_frame,
                    max_tokens=decision.max_tokens,
                    retried=True,
                    original_text=text,
                )
                await self._write_version()
                return ""
            spoken, outcome = await self._speak(text)
            await self._finish(spoken, decision.memories, outcome)
            return spoken

        if decision.retried and decision.original_text:
            # 言い直しが say を返さなかった。**元の応答で出す**（黙るよりはよい）。
            logger.info("event-loop 言い直しが say を返さなかったので元の応答で出す")
            spoken, outcome = await self._speak(decision.original_text)
            await self._finish(spoken, decision.memories, outcome)
            return spoken

        # どちらも無ければ素テキストへフォールバック（表示はここで1回）。
        # **声にはならない**（音になるのは say() だけ）。`_finish` が `独白` として残す。
        logger.debug("event-loop 決定=none（素テキスト）")
        text = (decision.result.text or "").strip()
        if text:
            self._emit(text)
        await self._finish(text, decision.memories, "沈黙")
        return text

    async def _coherence_violation(
        self,
        text: str,
        recent: str,
        memories: "list[Recalled]",
        *,
        verdicts=None,
        w_id_map: "dict[str, str] | None" = None,
    ) -> "str | None":
        """発話の前に規則違反を見る（出-f）。違反の説明を返す。無ければ None。

        **応答の文字列を機械で削らない。** 機械が出すのは、見たか・記憶が載ったかという
        推測の要らない事実だけで、規則に反するかどうかの判断は軽量LLM がする。
        """
        agent = self._agent
        if not agent.config.coherence_check or not text:
            return None
        started = time.monotonic()
        violation = await agent._evaluator.check_response_coherence(
            text,
            recent=recent,
            facts=self._checker_facts(memories, verdicts=verdicts, w_id_map=w_id_map),
        )
        logger.info(
            "event-loop 整合チェック %.2f 秒（違反=%s）",
            time.monotonic() - started,
            "あり" if violation else "なし",
        )
        return violation

    async def _note_follows(
        self, w_text: str, utterance: str, w_id_map: "dict[str, str]", follows_task
    ) -> None:
        """続き先の判定を受け取り、辺を書き、結末を計測ログへ 1 行書く（記-i）。

        結末は 5 通り——続き（辺を書いた）／途切れ（none）／未判定（言葉が無い・W が空で
        判定を呼んでいない）／落ちた（例外・時間切れ）／不一致（返った id が W に無い・自分の
        起点）。以前は「続き」しか形に残らず、継起が 0 本のとき「判定が壊れている」のか
        「続きの場面が無かった」のかを切り分けられなかった。REST 内省が読む（記-a-に）。
        """
        follows, failed = await _result_or_failure(follows_task)
        linked = workspace.link_follows(self._agent, self._req, w_id_map, follows)
        if not w_text.strip() or not utterance.strip():
            outcome = "未判定"
        elif failed:
            outcome = "落ちた"
        elif follows is None:
            outcome = "途切れ"
        elif linked:
            outcome = "続き"
        else:
            outcome = "不一致"
        measure.record(
            "続き先",
            結末=outcome,
            相手=(follows or "-")[:12] if outcome == "続き" else "-",
            起点=(self._req.request_id or "-").replace("-", "")[:12],
        )

    def _checker_facts(
        self,
        memories: "list[Recalled]",
        *,
        verdicts=None,
        w_id_map: "dict[str, str] | None" = None,
    ) -> str:
        """整合チェックへ渡す事実（出-n）。言い直しにも同じものを添える。

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
            used=used_lines(verdicts, w_id_map or {}, memories),
            arrived=arrived,
        )

    def _declare_light_memory_use(
        self,
        *,
        utterance: str,
        reply: str,
        workspace_ctx: str,
        w_id_map: dict[str, str],
        mem: object,
    ) -> None:
        """**軽量LLM** が答えて閉じた反復の申告を、背景で聞いて当てる（出-h-ろ）。

        **待たない。** 申告は答えを出したあとの後片付けで、発話を待たせる理由がない
        （実測 1.03 秒）。

        **投げるときに写して閉じ込める。** 走っているあいだに次の反復が来れば、ループの
        `w_id_map` も `mem` も作り直されている。12桁が偶然当たれば黙って別の記憶へ当たり、
        面が変われば別の人の記憶が育つ。だから引数で受け、ここで写す（環-h ②・出-h-ろ ③）。
        """
        if not w_id_map:
            return
        w_id_map = dict(w_id_map)
        backend = self._agent._utility_backend

        async def _run() -> None:
            try:
                raw = await workspace.ask_verdicts(
                    backend, utterance=utterance, reply=reply, workspace_ctx=workspace_ctx
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # 握りつぶさない。申告が出ていないことに気づけないと、記憶が育たない
                # 理由が分からなくなる。
                logger.warning("event-loop 軽量LLM の申告を聞けなかった: %s", e)
                return
            workspace.apply_memory_verdicts(mem, raw, w_id_map)

        self._verdict_tasks.add(task := asyncio.create_task(_run()))
        task.add_done_callback(self._verdict_tasks.discard)

    async def _speak(self, text: str) -> tuple[str, str]:
        """声に出す。返りは **(実際に出した文, 結末)**。**反復は閉じない。**

        身体を持つ以上、発話は相手が居て初めて意味を持つ（正本③ の配信ゲート＝結果有り＋在席）。
        居ないときは「話したかったができなかった」を O に残して `pending_speech` へ積み、
        次に人が現れたときに気づけるようにする。溜めたものの寿命（鮮度切れ・参照先 supersede で
        失効）は `pending_speech` 側が持つ。

        **閉じるのは呼び手（`_iterate`）である**（環-e-に・段4）。以前はここが `_finish` を
        3通りに呼び分けており、話す動作が求めの寿命の終わりまで持っていた。核が殻を呼び返す
        形で、結末を決めるのも `memories` を受け取るのも、閉じるためだけだった。
        """
        if not text:
            return "", "沈黙"
        if self._req.trigger_kind == "機器" and str(self._req.request_text).startswith("[退室]"):
            # **退室の知らせに返事する相手は居ない**（知-r・2026-09-18 12:36 実機：最後の検出から
            # 滞留窓 180 秒の内なので出口が通り、「出ていかれたんですね」を声に出した）。
            # 思ったことは独白として O に残る。入室・タイマー・メモは従来どおり。
            logger.info("event-loop 退室の知らせなので独り言として残す（相手は居ない）")
            return text, "独白"
        blocked = self._delivery_block_reason()
        if blocked and self._req.said_fillers and blocked != "黙っているよう頼まれている":
            # **つなぎを出したなら本応答も出す**（環-i）。つなぎと本応答は別々にゲートを引く
            # ので、あいだで在席の証拠（顔の検出・5 分の窓）が切れると「見てみますね」だけ
            # 出て本文が保留になった（実機）。つなぎを聞いた相手が居た事実を優先する。
            # 名前で呼ばれた「黙っていて」だけは、つなぎの後でも守る。
            logger.info("event-loop %s が、つなぎを出した相手へ本応答を出す", blocked)
            blocked = ""
        if blocked:
            if self._req.trigger_kind == "情動":
                # **独り言は相手が居なければ言わない、し、あとでも言わない**（情-c）。
                # その場に居なければ無かったことになる。ただし思ったこと自体は残す——本文を
                # 返して `_finish` が「考えたが言わなかった」（役割 `独白`）で O に書く。
                # 理由（居ない／静穏時間／黙っていて）に依らず同じ。
                logger.info("event-loop %s ので独り言は言わずに残す（積まない）", blocked)
                return text, "独白"
            await self._hold_speech(text, blocked)
            logger.info("event-loop %s ので発話を保留し pending_speech へ積む", blocked)
            # 不在の会話入力は入口で止まる（`_swallow_if_unheard`）ので、ここに来る保留は
            # 機器の知らせ（本文が機器側で決まっている）だけ。SEEKING の押し上げも入口で行う。
            return "", "保留"
        await self._dif.speak(text, gain=self._voice_gain())
        self._stamp_said()
        self._emit(text)
        return text, "発話"

    async def _say_filler(self, text: str) -> None:
        """つなぎの一言を出す（内容にコミットしない前置き）。配信ゲートは同じく効かせる。

        本応答ではないので、これで反復を閉じない。溜める（`pending_speech`）のも本応答の
        役目なので、出せない場面では黙って落とす。
        """
        if not text or self._delivery_block_reason():
            return
        if text.rstrip().endswith(("？", "?")):
            # **つなぎは疑問文にしない。** 調停が検索を投げながら「何を調べましょうか？」と
            # 聞き返した（2026-09-13 実機）。相手は答えるべきか待つべきか分からない。規則は
            # 文で頼んでいたが守られなかったので、機械で落とす。
            logger.info("event-loop つなぎが疑問文なので出さない：%.40s", text)
            return
        agent = self._agent
        await self._dif.speak(text)
        self._stamp_said()  # つなぎも自分の発話
        self._emit(text)
        # 言ったことを覚えておく。覚えないと、調停は「もう一言伝えた」ことを知らないまま
        # 同じことをまた言う（実機で1秒差に同じ文が2回出た）。抑止で黙らせるのではなく、
        # 判断できる材料を渡して解く。
        #
        self._req.said_fillers.append(text)
        # **O へ書く**（段 4）。054 で外したのは、想起の候補を食うからだった。役割が
        # 「想起に出さない」を担う形になったので、項として持ちながら想起から外せる。
        # 書かないと、相手が聞いた会話とパジュが読み返す会話が食い違う。
        obs_id = await agent._oif.write(
            MI(
                id="",
                content=f"つなぎに言った：{text}"[:500],
                timestamp=None,
                direction="発話",
                parent_id=self._req.request_id,
            ),
            **agent._observation_perspective(),
        )
        self._note_record(obs_id, "つなぎ")

    def _delivery_block_reason(self) -> str:
        """配信ゲート。発話を出せない理由を返す（出せるなら空文字）。

        正本③ の「配信ゲート（結果有り＋在席）」に静穏時間を併せる。以前は静穏時間を
        deferred の配信側だけが見ており、自発発話は素通りしていた。判定をここへ集める。

        **確かめたうえで掛けたタイマーが鳴った求めは通り抜ける**（知-n・`passes_gate`）。頼んだ本人が
        「静かな時間でも鳴らしていい」と言ったものなので、在席・静穏・沈黙の依頼のどれにも掛けない。
        """
        agent = self._agent
        if getattr(self._req, "passes_gate", False):
            return ""
        # 「黙っていて」の依頼は**入口**で見る（`_swallow_if_unheard`・情-h・2026-09-16）。以前は
        # ここ（出口）で止めていたので、発話ごとに求めが立って調停・主LLM が回ってから止まり、
        # 他人への返事を通す例外（案イ）も入り込んだ。黙っているあいだに求めが立つのは解く
        # きっかけ（タイマーが鳴る・本人の「話していい」・止める頼み）だけで、それは通す。
        if agent._social_presence_permission() == 0.0:
            return "聞く相手が居ない"
        # 静穏時間は「**自分から**話しかけない時間」で、話しかけられたのに黙るための
        # ものではない。起点を区別せず掛けていたため、夜に話しかけても返事が出ず、
        # 保留されて翌朝に届く動きになっていた（実機で観測）。在席と「黙っていて」の
        # 依頼は起点によらず掛かるので、ここだけを分ける。
        if self._req.trigger_kind != "発話":
            with contextlib.suppress(Exception):
                if agent._in_quiet_hours():
                    return "静穏時間である"
        return ""

    async def _dispatch_arbiter_action(self, decision, *, utterance: str) -> None:
        """調停が選んだ調べもの（see・look・recall・検索・道具）を投げる。

        つなぎの一言はここで即出す（フルLLM を経由しないぶん速い・正本③ 段5 の内部二段）。
        `see`・`look` は帰りの判断も調停がするので、誰が出したかを控える（`_decide`）。
        """
        await self._say_filler(decision.text)
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
        """聞けないあいだは、届いたものを O に残すだけで求めを立てない（入口の門）。

        聞けない理由は 2 つ。**黙っているよう頼まれている**（情-h・2026-09-16：会話入力・機器・
        情動のどれも止め、通すのは解くきっかけだけ）と、**誰も見えない**（2026-09-17：会話入力
        だけ止める。機器の知らせは出口の `pending_speech` で本文ごと溜め、情動は出口で独白に
        なる）。どちらも同じ器（`_muted`・`silence_hold`）に溜め、聞けるようになった最初の求めへ
        渡す（`_pending_heard` → `Request.heard_while_silent`）。以前の不在は出口で止めており、
        返事の本文を作ってから溜めていた（テレビの声への返事が、人が映った瞬間に出る）。
        誰も見えないのに声がしたのは見に行く理由なので、ここで SEEKING を押し上げる（案ア）。
        """
        from ..core.silence_hold import lifts
        from ..core.timer_rules import is_control_word
        from ..silence_state import is_silenced

        now = time.time()
        req = None
        with contextlib.suppress(Exception):
            req = self._load_silence()
        if is_silenced(req, now=now, nobody_since=self._nobody_since()):
            assert req is not None
            speaker = self._current_speaker_name() if trigger.kind == "会話入力" else ""
            if lifts(
                trigger.kind,
                trigger.query,
                speaker=speaker,
                asker=req.person,
                reason=getattr(req, "reason", "") or "",
            ):
                if trigger.kind == "会話入力":
                    self._pending_heard = self._take_muted()  # 解く言葉の求めがまとめの求めになる
                return False
            await self._note_muted(trigger, now)
            return self._swallowed(trigger)
        if trigger.kind == "会話入力" and self._nobody_visible():
            if self._timer_active() and is_control_word(trigger.query):
                # タイマーが動いている／鳴っているときの操作の言葉（止めて・一時停止・再開）は、誰も
                # 見えなくても通す（出-ab・実機 2026-09-18 22:13：鳴っている最中の「止めて」が飲まれた）。
                # 沈黙の門（情-m）と同じ規則。テレビの短い「止めて」で止まる副作用は許容済み。
                return False
            await self._note_muted(trigger, now, why="誰も見えなかった")
            with contextlib.suppress(Exception):
                await self._agent._nudge_seeking()
            return self._swallowed(trigger)
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

    def _nobody_since(self) -> "float | None":
        """センサが「誰も居ない」を見始めた時刻（`agent.nobody_since`）。読めなければ None（解けない側）。"""
        try:
            since = self._agent.nobody_since()
        except Exception:  # noqa: BLE001
            return None
        return float(since) if isinstance(since, (int, float)) else None

    def _timer_active(self) -> bool:
        """タイマーが動いている／一時停止中（`[タイマー]` の枠がある）か、音が鳴っているか。"""
        if _timer_frame(self._agent):
            return True
        return bool(getattr(getattr(self._agent, "_dif", None), "ringing", False))

    def _nobody_visible(self) -> bool:
        """配信ゲートと同じ「居るか」（`agent._social_presence_permission`）。読めなければ居る扱い。"""
        try:
            return self._agent._social_presence_permission() == 0.0
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _swallowed(trigger: "Trigger") -> bool:
        if trigger.future is not None and not trigger.future.done():
            trigger.future.set_result("")  # 呼び手（GUI）は吹き出しだけ出して待たない
        return True

    async def _note_muted(self, trigger: "Trigger", now: float, *, why: str = "黙っていた") -> None:
        """聞けなかったあいだに届いたものを O に残し、器に控える（理由は `why`）。"""
        agent = self._agent
        if trigger.kind == "会話入力":
            who = self._current_speaker_name()
            text = trigger.query
            content = (
                f"（誰も見えないあいだに聞いた）{trigger.query}"
                if why == "誰も見えなかった"
                else f"（黙っていたあいだに聞いた）{trigger.query}"
            )
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
        self._muted.append(
            Heard(kind=trigger.kind, text=text, who=who, obs_id=obs_id, at=now, why=why)
        )
        logger.info(
            "%sので聞くだけ：%s %.40s",
            "黙っている" if why == "黙っていた" else "誰も見えない",
            trigger.kind,
            text,
        )

    def _take_muted(self) -> "list[Heard]":
        items, self._muted = self._muted, []
        self._muted_since = None
        return items

    def check_silence_lifted(self) -> None:
        """T が tick ごとに呼ぶ：期限切れで明けたのに何も届かないとき、まとめの求めを起こす。

        見るのは器の中で**理由が「黙っていた」のもの**だけ（情-k・2026-09-18 12:38 実機）。器は
        不在で聞いたもの（理由「誰も見えなかった」）とも共有しており、そちらを「沈黙が明けた」と
        読むと、誰も見えないのに求めが立って LLM が回った。不在の分は人が映った求めに載る。
        """
        with contextlib.suppress(Exception):
            from ..silence_state import clear_silence, is_silenced

            req = self._load_silence()
            lifted = not is_silenced(req, now=time.time(), nobody_since=self._nobody_since())
            if lifted and req is not None and not req.reason.startswith("timer:"):
                # 明示の依頼が退室（誰も居ないを 60 秒）で解けたら**記録も消す**（情-l-ろ・実機 2026-09-18
                # 20:46）。残すと、タイマーの沈黙（`hush_for_timer`）が「人の依頼が生きている」と負け、
                # 人が映ればまた黙る。タイマー由来は鳴る・止めるまで（`unhush_timer`）。
                clear_silence()
                logger.info("頼んだ人が居なくなったので沈黙を消した：%s", req.person)
            if not any(getattr(h, "why", "黙っていた") == "黙っていた" for h in self._muted):
                return
            if lifted:
                self.push_device(
                    "沈黙が明けた", f"黙っていたあいだに {len(self._muted)} 件届いていた"
                )

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
        """自分が声を出した時刻を打つ（在席の証拠・`presence_said_sec`・2026-09-17）。

        起点が人の発話で、相手が分かっているなら、話者の指定も延びる（`_speaker_confirmed_at`・
        知-t）——会話中は切れない。
        """
        agent = getattr(self, "_agent", None)
        if agent is None:
            return
        now = time.time()
        agent._last_said_at = now
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
        sensor = getattr(self._agent, "_presence_sensor", None)
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
        sensor = getattr(self._agent, "_presence_sensor", None)
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
        await self._apply_speaker_claim(decision)

    async def _apply_speaker_claim(self, decision) -> None:
        """調停が読んだ名乗り（`speaker_claim`）を、在席があるときだけ話者に付ける（知-w・2026-09-19）。

        在席（居るか）はカメラだけで決める（マイクは証拠にしない）。カメラが人を見ていれば、声の名乗りは
        `/speaker` と同じ効き（`set_active`・`_speaker_set_at`・PMM 同期）。家族に無い名前は付けない。
        入口で飲まれた発話はここまで来ないので、在席なしの名乗りは自然に使われない。
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
        if not self._someone_present():
            # 在席が無い名乗りは 30 秒預かる（知-w-ろ・実機 13:27：見回りの帰りに名乗りを読んだ 6 秒後に
            # センサが人を見た）。人を見た最初の求め（`_begin_request`）で生きていれば付ける。
            self._pending_claim = (name, time.time())
            logger.info("名乗り「%s」があるが在席が無いので %.0f 秒預かる", name, PENDING_CLAIM_SEC)
            return
        await self._set_speaker(name, "在席あり")

    def _someone_present(self) -> bool:
        try:
            return float(self._agent._social_presence_permission() or 0.0) > 0.0
        except Exception:  # noqa: BLE001
            return False

    async def _set_speaker(self, name: str, why: str) -> None:
        """名乗りで話者を付ける（`/speaker` と同じ効き：`set_active`・`_speaker_set_at`・PMM 同期）。"""
        agent = self._agent
        if str(getattr(agent._persons, "active_name", "") or "") == name and getattr(
            agent._persons, "active_is_explicit", False
        ):
            return
        agent._persons.set_active(name)
        agent._speaker_set_at = time.time()
        with contextlib.suppress(Exception):
            await agent._sync_pmm_speaker(name)
        logger.info("名乗りで話者を付けた：%s（%s）", name, why)

    async def _apply_pending_claim(self) -> None:
        """預かった名乗りを、人を見た最初の求めで付ける（30 秒以内・知-w-ろ）。過ぎていれば捨てる。"""
        kept = getattr(self, "_pending_claim", None)
        if kept is None:
            return
        name, at = kept
        if time.time() - at > PENDING_CLAIM_SEC:
            self._pending_claim = None
            logger.info("預かった名乗り「%s」は %.0f 秒過ぎたので捨てた", name, PENDING_CLAIM_SEC)
            return
        if self._someone_present():
            self._pending_claim = None
            await self._set_speaker(name, "預かり・在席が付いた")

    def _apply_silence(self, decision, *, utterance: str = "") -> None:
        """調停が読んだ沈黙の依頼を掛ける／解く（反復本体の呼び口は 1 つ）。

        読むのは調停だが、機械の守りを重ねる（`core.silence_rules`・2026-09-16 実機）：
        名前で呼ばれていなければ掛けない（「待てぃ」「しなよ」で 60 分黙った）、本人が
        「話していい」と言えば調停が読めなくても解く（「話していいよ」が効かなかった）。
        """
        from ..core.silence_rules import is_release, names_me

        if decision.silence_minutes:
            names = list(getattr(self._agent.config, "agent_names", None) or [])
            if names and not names_me(utterance, names):
                logger.info(
                    "黙っているよう読めたが、名前で呼ばれていないので受けない：%.40s", utterance
                )
                return
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
            who = self._current_speaker_name()
            if who and who == req.person:
                clear_silence()
                logger.info("黙っていてと頼んだ本人（%s）が話していいと言ったので解く", who)
            else:
                logger.info(
                    "話していいと言われたが、頼んだのは %s なので解かない（言ったのは %s）",
                    req.person,
                    who or "不明",
                )

    def _accept_silence(self, asked_minutes: int) -> None:
        """黙っている依頼を受ける。宛先は、いま話している相手。

        `asked_minutes` は調停が読み取った分数で、`-1` は「頼まれたが長さの指定なし」。
        既定と上限は Config が持つ。
        """
        agent = self._agent
        with contextlib.suppress(Exception):
            from ..silence_state import SilenceRequest, save_silence

            who = self._current_speaker_name()
            if not who:
                logger.info("黙っているよう頼まれたが、誰からか分からないので受けない")
                return
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

        在席表の `is_speaker` を先に見て、無ければ `/speaker` で明示された名前を使う。
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

    # 配信ゲートが返す理由を、記録に書く形（過去形）へ言い換える。**表に無い理由は
    # そのまま書く**——増えたときに黙って誤った文へ倒さないためである。
    _HELD_REASON_PAST = {
        "黙っているよう頼まれている": "黙っているよう頼まれていた",
        "聞く相手が居ない": "聞く相手が居なかった",
        "静穏時間である": "静穏時間だった",
    }

    async def _hold_speech(self, text: str, reason: str) -> None:
        """話せなかった内容を O に残し、`pending_speech` へ積む（想起系は汚さない）。

        **止められた理由も書く。** 以前は理由に関わらず「聞く相手が居なかった」と固定で
        書いており、「黙っていてと言われたのでやめた」が「誰も居なかった」として残って
        いた。保留は後で配られるので（`_release_pending_speech`）、パジュはその文面を
        読んで話し始める。理由が違えば、話し出し方も違う。
        """
        agent = self._agent
        why = self._HELD_REASON_PAST.get(reason, reason)
        obs_id = await agent._oif.write(
            MI(
                id="",
                content=f"話したかったが、{why}：{text}"[:500],
                timestamp=None,
                direction="保留",
            ),
            **agent._observation_perspective(),
        )
        if obs_id:
            with contextlib.suppress(Exception):
                agent._pending_store.add(obs_id, None)

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

    async def _finish(self, text: str, memories: "list[Recalled]", outcome: str) -> None:
        """求めが閉じた反復の後始末：総括ログと永続化。

        閉じ方は `outcome` が持つ——`発話`（声になった）・`沈黙`（地の文だけで声にならず
        `独白` として残る）・`保留`（配信ゲートに止められた）・`独白`（相手が居ない独り言。
        積まずに、思ったことだけ残す・情-c）。**発話だけではない。**

        **ここでは何も畳まない。** 求めの版チェーンは `_write_version` が `改訂` で畳み、
        自分が答えた記録は鎖の外にある。要約と内省は背景で遅れて来て、その記録を supersede
        する。
        """
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
        spoken = outcome == "発話"
        answer_id = None
        if text:
            with contextlib.suppress(Exception):
                answer_id = await agent._oif.write(
                    MI(
                        id="",
                        content=(
                            f"自分が答えた：{text}" if spoken else f"考えたが言わなかった：{text}"
                        )[:500],
                        timestamp=None,
                        direction="発話" if spoken else "独白",
                        parent_id=self._req.request_id,
                    ),
                    **agent._observation_perspective(),
                )
        self._note_record(answer_id, "答え" if spoken else "独白")
        # **自分が答えた記録は鎖の外**。何も畳まない。求めの版チェーンは、最後の版
        # （結果が届いた状態）のまま残る。まとめ知識の MI を作る場合は、それが最後の版を
        # 畳む（未実装・`設計方針_求めの版チェーン`）。
        self._req.request_id = None
        self._req.live_version_id = None
        self._req.lookups.clear()
        self._req.said_fillers.clear()
        self._req.speech_to_deliver.clear()
        self._req.seen_image_path = None
        self._req.failed_actions.clear()
        self._req.fired_axis = ""
        self._req.iterations = 0
        self._req.iterations_capped = False
        self._notify_request_state(False)
        # 母集合とやりとりへ渡す分を取り出してから捨てる（渡す前に消すと空で渡る）。
        # やりとりは**区間**、母集合は**全部**である（打ち切りの分も次へ持ち越している）。
        noted = self._close_exchange() or []
        turn_records = [i for i, _ in self._req.turn_records]
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
        try:
            origin = self._req.utterance or self._req.cue
            arousal = await agent._turn_arousal(origin, text)
            agent._spawn_background_task(
                agent._run_post_response_pipeline(
                    user_input=origin,
                    final_text=text,
                    camera_used=False,
                    camera_image=None,
                    observation_action_name=None,
                    observation_action_input=None,
                    companion_mood="engaged",
                    arousal=arousal,
                    memories=memories,
                    # 会話要約は背景で作られるので、向こうでこの関係の末尾に足す。
                    exchange_id=exchange_id,
                    # ループが作った記録も拡散想起の母集合へ。載せないと、閉じた逐語へ
                    # 辿り着く辺が WR に無い（実機で、逐語の WR 掲載数が0だった）。
                    extra_cooccurring_ids=turn_records,
                ),
                name="event-post-response",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("event-loop persistence spawn failed: %s", e)
