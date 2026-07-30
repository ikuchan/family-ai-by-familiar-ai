"""音声の確定文がキューへ1件だけ積まれる（実機で「2回答える」が起きた）。

2026-07-30 の実機で、1つの「おはよう。」に2回答えた。GUI のキューに2件入っていた。

真因は、確定した書き起こしを**二人が積んでいた**ことである。`realtime_stt_session` の
`_committed_relay` は、確定文を表示用の差し込み口へ渡した直後に、入力キューへも積む。

    if self.on_committed: self.on_committed(text)   # 表示用の差し込み口
    await self._committed_queue.put(text)           # 入力の本線

そして `gui.py` は `_input_queue` **そのもの**を `committed_queue` として渡していた。
差し込み口に配線された `_on_realtime_stt_committed` もキューへ積んでいたため、必ず2件に
なった。当時入れた GUI の重複除去が効かなかったのは、片方がその窓を通らないからである。

口は3つあり、役割は分かれている。

| 口 | 行き先 | 役割 |
|---|---|---|
| `on_partial` | `set_status("🎤 聞いています")` | 一瞬出して消す |
| `on_committed` | `append_line("[話者] 発言")` | 会話ログに残す |
| `_committed_queue` | `_process_queue` | **入力** |

**表示側で重複除去をしてはいけない。** そこで落とすと、画面に出ないのにエージェントが
答える食い違いが起きる（入力は別の口から入るため）。書き起こしどうしの重複は、両方の口
より前にあるセッション側（3 秒の窓・`VoiceLoopGuard`）が済ませている。
"""

from __future__ import annotations


def _stub_window():
    """`_on_realtime_stt_committed` が触る分だけ持つ最小の窓。"""
    import asyncio

    from familiar_agent.gui import FamiliarWindow

    class _Log:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def append_line(self, text: str) -> None:
            self.lines.append(text)

    class _Stream:
        def __init__(self) -> None:
            self.cleared = 0

        def clear_status(self) -> None:
            self.cleared += 1

    win = FamiliarWindow.__new__(FamiliarWindow)
    win._closing = False
    win._log = _Log()
    win._stream = _Stream()
    win._input_queue = asyncio.Queue()
    win._companion_display_name = "話者"
    win._pmm = None
    return win


def test_a_committed_transcript_is_not_enqueued_by_the_display_hook():
    """**入力はセッションがキューへ入れる。** 表示の口では積まない。

    ここで積むと、セッション側の `_committed_queue.put()` と合わせて必ず2件になる。
    """
    from familiar_agent.gui import FamiliarWindow

    win = _stub_window()
    FamiliarWindow._on_realtime_stt_committed(win, "おはよう。")

    assert win._input_queue.qsize() == 0, "表示の口で積んでいる（2回答える原因）"


def test_a_committed_transcript_is_shown_in_the_conversation_log():
    """積まないが、表示はする（会話ログに残す役割はこの口が持つ）。"""
    from familiar_agent.gui import FamiliarWindow

    win = _stub_window()
    FamiliarWindow._on_realtime_stt_committed(win, "おはよう。")

    assert any("おはよう。" in line for line in win._log.lines)
    assert win._stream.cleared == 1      # 「聞いています」を消す


def test_the_display_hook_does_not_drop_repeats():
    """**表示側で落とさない。** 落とすと、画面に出ないのに答える食い違いになる。

    入力は別の口（セッションのキュー）から入るので、表示を止めても入力は止まらない。
    """
    from familiar_agent.gui import FamiliarWindow

    win = _stub_window()
    FamiliarWindow._on_realtime_stt_committed(win, "うん")
    FamiliarWindow._on_realtime_stt_committed(win, "うん")

    assert len(win._log.lines) == 2, "2回言ったのに1回しか出ていない"


def test_both_entrances_report_the_same_queue_size_for_one_input(caplog):
    """入口が違っても、1件目のログはどちらも `queue=1` になる。

    数える時点がずれている。キーボードは積んでから出すが、音声は積む前に出す（積むのは
    セッション側で、この口を呼んだ直後）。`pending` でその差を埋めていなければ、同じ1件が
    音声では `queue=0`、キーボードでは `queue=1` と出て、ログを並べたときに読み違える。
    """
    import logging

    from familiar_agent.gui import FamiliarWindow

    win = _stub_window()
    with caplog.at_level(logging.INFO, logger="familiar_agent.gui"):
        FamiliarWindow._on_realtime_stt_committed(win, "おはよう。")
        stt_line = caplog.messages[-1]

        # キーボード側は、積んでから同じログを出す。
        win._input_queue.put_nowait("おはよう。")
        FamiliarWindow._log_input_queued(win, "keyboard")
        keyboard_line = caplog.messages[-1]

    assert "queue=1" in stt_line, f"音声側が積む前の数を出している: {stt_line}"
    assert "queue=1" in keyboard_line
    assert "stt" in stt_line and "keyboard" in keyboard_line


# ── 起動メッセージは実物の担い手を出す ─────────────────────────────────────────
#
# `🎤 Realtime STT ON (ElevenLabs)` の固定文字列だったので、既定のローカル
# （faster-whisper）で動いていても ElevenLabs と表示していた。どちらで書き起こして
# いるかは実機の切り分けで最初に見る情報である。

def test_the_startup_line_names_the_engine_actually_in_use():
    """既定（ローカル）なら faster-whisper と出す。ElevenLabs とは出さない。"""
    import asyncio

    from familiar_agent.gui import FamiliarWindow

    win = _stub_window()
    win._realtime_stt = _StubController("faster-whisper")
    win._set_last_error = lambda _err: None

    asyncio.run(FamiliarWindow._start_realtime_stt(win))

    line = win._log.lines[-1]
    assert "faster-whisper" in line, f"担い手が出ていない: {line}"
    assert "ElevenLabs" not in line


def test_the_startup_line_names_elevenlabs_when_switched_back():
    import asyncio

    from familiar_agent.gui import FamiliarWindow

    win = _stub_window()
    win._realtime_stt = _StubController("ElevenLabs")
    win._set_last_error = lambda _err: None

    asyncio.run(FamiliarWindow._start_realtime_stt(win))

    assert "ElevenLabs" in win._log.lines[-1]


class _StubController:
    """`_start_realtime_stt` が触る分だけ持つ包み。"""

    def __init__(self, engine_label: str) -> None:
        self.engine_label = engine_label
        self.on_partial = None
        self.on_committed = None
        self.on_restart = None

    async def start(self, loop, queue) -> None:
        return None
