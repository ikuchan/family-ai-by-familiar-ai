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
import time
import logging

logger = logging.getLogger(__name__)

# ログに載せる発話の長さ。記憶の内容と同じ扱いで、debug でも先頭だけにする。
_TRAIL_CHARS = 24


class DIF:
    """I と外部の機械の唯一の出入り口。

    **要るものだけを受け取る。** `agent` を丸ごと持てば、口はその 89 個の属性すべてに
    手が届く。届く必要のないものへ届く形は、口を1枚挟んだ意味を消す。

    `tts` は声の担い手（無い機体では `None`）、`search` と `fetch` は調べものの道具、
    `mcp` は MCP の道具を持つ側。どれも `agent` の `__init__` で一度作られたきりで
    差し替わらないので、写しを持つ。`ip` は機器の出来事を受ける側の I（`InformationProcessing`）で、
    外から中へ入る向きだけが使う。

    **持つ側ごとに1つ作る。** I は外へ出る向き（`tts`・`search`・`fetch`・`mcp`）を、
    T は中へ入る向き（`ip`）を持つ。AIF が `tonic.py` と `agent.py` に1つずつ在るのと
    同じで、口は薄い転がしなので、要る向きだけを持てばよい。
    """

    def __init__(self, *, tts=None, search=None, fetch=None, mcp=None, ip=None) -> None:
        self._tts = tts
        self._search = search
        self._fetch = fetch
        self._mcp = mcp
        self._ip = ip

    # ── 声 ────────────────────────────────────────────────────────────────

    @property
    def understands_tags(self) -> bool:
        """合成の担い手が角括弧タグを指示として解するか（`根拠台帳` §9）。

        整え方・`say` の説明・規則 `no-tts-tags` の3箇所がこの値を見る。声の担い手を
        知っているのは DIF なので、聞き先をここへ寄せる。
        """
        return bool(self._tts and self._tts.understands_tags)

    def speak_defs(self) -> list[dict]:
        """声の道具の定義。無い機体では空。"""
        return self._tts.get_tool_definitions() if self._tts else []

    async def speak(self, text: str) -> None:
        """声に出す。

        **例外は飲む。** 機器は落ちる前提のもので、声が出せなかったことでターンごと
        壊すわけにはいかない（移す前と同じ扱い）。
        """
        if self._tts is None:
            return
        logger.debug("DIF speak → %s", text[:_TRAIL_CHARS])
        started = time.monotonic()
        with contextlib.suppress(Exception):
            await self._tts.call("say", {"text": text})
        # 合成＋再生の秒数は必ず残す（出-k-い）。返事が出るまでの体感にそのまま乗る。
        logger.info("DIF 声 %.2f 秒（%d 字）", time.monotonic() - started, len(text))

    # ── 調べもの ──────────────────────────────────────────────────────────

    def lookup_defs(self, kind: str) -> list[dict]:
        """調べものの道具の定義。"""
        tool = self._search if kind == "search_deferred" else self._fetch
        return tool.get_tool_definitions()

    async def lookup(self, kind: str, params: dict) -> tuple[str, bool]:
        """調べものを投げる。返りは (文面, 背景タスクを作ったか)。

        **ここでは畳まない。** 投げられなかったときに開いた意図を閉じるのは呼び手の
        仕事で、口が例外を飲むと飛行中の数が合わなくなる。
        """
        tool = self._search if kind == "search_deferred" else self._fetch
        logger.debug("DIF lookup → %s", kind)
        return await tool.dispatch(params)

    # ── MCP の道具 ────────────────────────────────────────────────────────

    def tool_defs(self, name: str) -> list[dict]:
        """MCP の道具を**名前で1本だけ**取り出す。

        **話者ゲートはこちら側の責任である。** サーバー側からは誰が話しているか見えない
        ので、個人ティアの道具名には人が入っている（`ask_vault_yusuke`）。**名前に人が
        入っている道具は、その人のターン以外では出さない**——`description` でお願いする
        のではなく、定義リストから落とす。存在しない道具は呼べない。

        MCP のサーバーは落ちる前提のもので、引けなければ空で返す。
        """
        if self._mcp is None:
            return []
        with contextlib.suppress(Exception):
            return [d for d in self._mcp.get_tool_definitions() if d.get("name") == name]
        return []

    # ── 機器の出来事（外 → I） ─────────────────────────────────────────────

    def device(self, kind: str, content: str, *, release_pending: bool = False) -> None:
        """人の出入りなど、機器が出した出来事を I の待ち行列へ入れる。

        カメラが出す事実なので DIF の担当である（T が渡すのは、時計を持つのが T
        だからにすぎない）。
        """
        logger.debug("DIF device → %s／%s", kind, content[:_TRAIL_CHARS])
        self._ip.push_device(kind, content, release_pending=release_pending)


__all__ = ["DIF"]
