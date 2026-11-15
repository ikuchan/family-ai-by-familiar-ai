"""ゲームのおすすめ（環-ba・2026-11-15・本人の決定）。

親愛や承認が溜まって話しかける発火（bond・esteem）のときに、その日まだ誘っていなければ、`skills/` にある遊び
（しりとり、マジカルバナナ）に「〇〇しない？」と誘ってよいことにする。誘うかは書く側（主LLM・軽量LLM）が決める。
数え方は音楽のおすすめ（`music_suggestion`）と同じで、声にした文に遊びの名前が入っていた日だけ「誘った」と数える
（1 日 1 回まで）。誘いに乗った返事でやり方を選べるよう、各スキルの「いつ要る」に「パジュが誘って」を書いてある。

DB（`agent_state` の鍵 `game_suggestion`）に持つのは、最後に誘った日だけ。日の区切りは音楽のおすすめと同じ口
（`music_suggestion.today`）で決める。
"""

from __future__ import annotations

from . import state_json

STATE_KEY = "game_suggestion"
HEADING = "[遊びのおすすめ]"


def last_offered_on() -> str:
    """最後に誘った日（YYYY-MM-DD）。無い・読めなければ空。"""
    raw = state_json.read(STATE_KEY)
    return str(raw.get("last_offered_on") or "") if isinstance(raw, dict) else ""


def store_offered_on(day: str) -> bool:
    return state_json.write(STATE_KEY, {"last_offered_on": day})


def frame(names: "list[str]", *, today: str, last: str) -> str:
    """主LLM と軽量LLM の文に載せる枠。その日まだ誘っていなければ載せる。載せなければ空。

    話しかける発火かは、呼び手（`event_loop._game_frame`）が見る。音楽が鳴っているあいだは、情動が発火しても何もしない
    （T の発火の口と `event_loop._begin_affect` が
    `music_state.playing` で見る）ので、ここへは来ない。
    """
    if not names or last == today:
        return ""
    return (
        f"{HEADING}\nいっしょにできる遊び：{'、'.join(names)}。"
        "話しかけるなら、「〇〇しない？」と誘ってよい（誘わなくてもよい）。"
        "ほかのおすすめと同じ声で続けて誘わない（誘うのは 1 回の声で 1 つまで）。"
    )


def offered(said: str, names: "list[str]") -> bool:
    """声にした文に遊びの名前が入っているか。"""
    return any(n and n in said for n in names)
