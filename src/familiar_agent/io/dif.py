"""外部機器接続 DIF：I と外の機械をつなぐ唯一の口（`設計図` ③-2）。

設計は口を4つ（IIF・DIF・AIF・OIF）と定め、この4つ以外にコンポーネントどうしが直接
つながる線を置かない。ところが**外部の機械へは口が無く**、`loop/event_loop.py` が
スピーカーと調べものの道具を直接掴んでいた。

**挙動は変えない。** 返り値の形も、例外の畳み方も、順序もそのままで、口は転送し、
通ったものを debug に残すだけである（AIF と同じ作り）。

**主LLM は入らない**（出-c でモデルは資源とした）。記憶は OIF の担当である。

段1で通すのは**声と調べもの**だけである。知覚（`see`・`look`）は 知-c が実機で挙動を
追っている最中なので、そこが済むまで触らない——同じ箇所を切り出しと機能変更の両方で
触ると、どちらで挙動が変わったのかを切り分けられなくなる。
"""

from __future__ import annotations

import contextlib
import logging

logger = logging.getLogger(__name__)

# ログに載せる発話の長さ。記憶の内容と同じ扱いで、debug でも先頭だけにする。
_TRAIL_CHARS = 24


class DIF:
    """I と外部の機械の唯一の出入り口。

    掴んでいる相手を持ち替えず、`agent` 越しに読む。道具は起動の途中で差し替わる
    ことがあり、生成時に写しを取ると古い相手を掴んだままになる。
    """

    def __init__(self, agent) -> None:
        self._agent = agent

    # ── 声 ────────────────────────────────────────────────────────────────

    @property
    def understands_tags(self) -> bool:
        """合成の担い手が角括弧タグを指示として解するか（`根拠台帳` §9）。

        整え方・`say` の説明・規則 `no-tts-tags` の3箇所がこの値を見る。声の担い手を
        知っているのは DIF なので、聞き先をここへ寄せる。
        """
        tts = self._agent._tts
        return bool(tts and tts.understands_tags)

    async def speak(self, text: str) -> None:
        """声に出す。

        **例外は飲む。** 機器は落ちる前提のもので、声が出せなかったことでターンごと
        壊すわけにはいかない（移す前と同じ扱い）。
        """
        tts = self._agent._tts
        if tts is None:
            return
        logger.debug("DIF speak → %s", text[:_TRAIL_CHARS])
        with contextlib.suppress(Exception):
            await tts.call("say", {"text": text})

    # ── 調べもの ──────────────────────────────────────────────────────────

    async def lookup(self, kind: str, params: dict) -> tuple[str, bool]:
        """調べものを投げる。返りは (文面, 背景タスクを作ったか)。

        **ここでは畳まない。** 投げられなかったときに開いた意図を閉じるのは呼び手の
        仕事で、口が例外を飲むと飛行中の数が合わなくなる。
        """
        agent = self._agent
        tool = agent._deferred_search if kind == "search_deferred" else agent._deferred_fetch
        logger.debug("DIF lookup → %s", kind)
        return await tool.dispatch(params)


__all__ = ["DIF"]
