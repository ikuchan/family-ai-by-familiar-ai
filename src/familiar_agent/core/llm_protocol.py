"""バックエンドが守る約束（出-d-は）。

**6つのバックエンドは、既に同じ5つのメソッドを同じ署名で持っていた。** だが抽象基底も
`Protocol` も無く、**暗黙の約束で並んでいた**——新しいモデルを足すとき、何を実装すれば
足りるのかがコードから読めなかった。ここがその答えである。

**揃っていないものは約束に含めない。** `complete_with_image` は3つ（Anthropic・
OpenAICompatible・Gemini）、`make_system_message` は2つ（Kimi・GLM）しか持たない。
呼ぶ側は `hasattr` で見る（`scene.py` が画像の経路でそうしている）。**持たないことが
普通である面を必須にすると、持たないバックエンドが約束を破ることになる。**

**`runtime_checkable` にしてあるが、実行時の `issubclass` は名前しか見ない。** 引数まで
揃っているかは `tests/test_backend_protocol.py` が確かめる。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ..backends import ToolCall, TurnResult


@runtime_checkable
class LLMBackend(Protocol):
    """主LLM・軽量LLM のどちらにもなれるものの面。

    **重みを持たない。** API のクライアントか、別プロセスのサーバか、CLI の呼び出しである
    （モデル資源 MR の型枠は重みを持つものだけが従う・出-c）。
    """

    def make_user_message(self, content: str | list) -> dict:
        """人の側の1発言を、そのモデルが受け取る形にする。"""
        ...

    def make_assistant_message(self, result: TurnResult, raw_content: Any) -> dict:
        """モデルの応答を、次の往復へ持ち越せる形にする。"""
        ...

    def make_tool_results(
        self, tool_calls: list[ToolCall], results: list[tuple[str, str | None]]
    ) -> list[dict]:
        """道具の結果を、そのモデルが受け取る形にする。"""
        ...

    async def stream_turn(
        self,
        system: str | tuple[str, str],
        messages: list,
        tools: list[dict],
        max_tokens: int,
        on_text: Callable[[str], None] | None,
        effort: str | None = None,
    ) -> tuple[TurnResult, Any]:
        """1往復を流す。`on_text` へ届いた端から渡し、まとめた結果を返す。"""
        ...

    async def complete(
        self, prompt: str, max_tokens: int, *, system: str | None = None
    ) -> str:
        """1問1答。**返るのは文字列だけ**である。

        `system` は立ち位置と文脈（出-e）。**native な口で渡す**——プロンプトの先頭へ
        足すのとは別物で、モデルはシステム文と利用者の文を違う重みで扱う。native な口を
        持たない `cli` だけが前置きで代替する。既定 `None` なら、渡さないのと同じである。

        形のある答え（数値・選択・はい／いいえ・JSON）が要るなら
        `core.structured_ask` の口を通す——ここで直に受けて呼び出し側が解釈すると、
        モデルを替えたときに壊れる場所が散らばる（出-d の出発点）。
        """
        ...
