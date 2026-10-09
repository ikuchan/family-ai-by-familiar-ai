"""完了（自分の動作の結果が届いた）を、機械が「何が起きたか」に分ける（出-ay 段 4-2・2026-10-09・`設計方針_判定の段` §2.2.5）。

調停の組み替えで、完了の 1 回目（何が起きたか）は機械が決め、2 回目（どうするか）は選択肢が 2 つ以上のときだけ Jev に聞く
と決めた（本人）。ここは道具の名前・失敗の印・結果の文・求めの起点から、種類と並べる動作を返すだけで、何も呼ばない。

動作の鍵：`silent`（黙る）・`tell_light`（軽く伝える）・`reply_light`（軽く返す）・`talk_light`（軽く話しかける）・
`reply_full`（考えて返す＝主LLM）・道具の名前（そのまま投げる）。

失敗で特別に扱うのは 2 つだけ。「別の曲が鳴っている」（知-al）はかけ直し、検索の結果が空なら記憶か調べ直すかを Jev に聞く。
ほかの失敗（使えない・見つからない・時間切れ・知らない文）は、どれも軽く伝える。道具ごとに言い回しが違うので、言い回しで
細かく分けない。
"""

from __future__ import annotations

#: 調べものの道具（頼まれたら答えを返す・自分からなら覚えておく）。`ask_vault_` で始まる本人の記録も含む。
_RESEARCH = frozenset(
    {
        "search_deferred",
        "fetch_deferred",
        "recall",
        "recall_as",
        "recall_deeper",
        "recall_when",
        "recall_recent",
        "recall_tree",
        "house_rules",
        "family_schedule",
        "notion_search",
        "journal",
        "vault",
    }
)
_LOOK = frozenset({"look", "see"})

#: 成功した操作 → 並べる動作（本人の表・2026-10-09）。
_DONE: "dict[str, tuple[str, tuple[str, ...]]]" = {
    "play_music": ("音楽をかけた", ("silent",)),
    "stop_music": ("音楽を止めた", ("tell_light",)),
    "next_track": ("次の曲・音量を変えた", ("silent",)),
    "music_volume": ("次の曲・音量を変えた", ("silent",)),
    "set_timer": ("タイマー・アラームを掛けた", ("tell_light",)),
    "set_alarm": ("タイマー・アラームを掛けた", ("tell_light",)),
    "cancel_timer": ("タイマー・アラームを止めた・一時停止・再開した", ("tell_light",)),
    "pause_timer": ("タイマー・アラームを止めた・一時停止・再開した", ("tell_light",)),
    "resume_timer": ("タイマー・アラームを止めた・一時停止・再開した", ("tell_light",)),
    "cancel_alarm": ("タイマー・アラームを止めた・一時停止・再開した", ("tell_light",)),
    "start_stopwatch": ("ストップウォッチで測り始めた", ("tell_light",)),
    "stop_stopwatch": ("ストップウォッチを止めた", ("tell_light",)),
    "confirm": ("確認待ちに答えた", ("tell_light",)),
    "decline": ("確認待ちに答えた", ("tell_light",)),
}


def _research(action: str) -> bool:
    return action in _RESEARCH or action.startswith("ask_vault_")


def kind_of(
    action: str, *, failed: bool, result: "str | None", origin: str
) -> "tuple[str, tuple[str, ...]] | None":
    """(何が起きたか, 2 回目に並べる動作)。表に無い道具は None（いままでの判定に任せる）。"""
    text = (result or "").strip()
    if failed:
        if "別の曲が鳴っている" in text:
            return "頼んだものと違うものになった", ("play_music",)
        return "うまくいかなかった", ("tell_light",)
    if action == "search_deferred" and not text:
        return "検索の結果が 0 件", ("recall", "search_deferred")
    self_started = origin == "情動"
    if _research(action):
        if self_started:
            return "自分から調べに行った結果が届いた", ("silent", "talk_light", "search_deferred")
        return "頼まれた調べものの答えが届いた", ("reply_full", "reply_light", "search_deferred")
    if action in _LOOK:
        if self_started:
            return "自分から見に行った結果が届いた", ("look", "talk_light", "silent")
        return "頼まれて見た結果が届いた", ("reply_light", "look")
    return _DONE.get(action)
