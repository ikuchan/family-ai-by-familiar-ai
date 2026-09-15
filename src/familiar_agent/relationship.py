"""話者の名前の器（`PersonRegistry`）。

`/speaker` や `[名前]` で明示された話者と、既定名（パジュが誰と話しているか決め打ちしないための
印）を持つ。以前ここにあった関係の追跡（trust／intimacy の表）は 2026-09-15（環-d）に撤去した——毎ターン書いていたが、読むのは呼び手の無い宛先選びだけだった。
関係は O の関係の面（`situated_memories`）と `人物` のまとめ（REST 内省の層 1 ①）が担う。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PersonRegistry:
    """話者の名前と、いま誰と話しているか。"""

    def __init__(self, default_name: str):
        self._default_name = default_name
        self._names: list[str] = [default_name]
        self._active_name: str = default_name
        # 話者が明示されたか（`/speaker` や `[名前]`）。既定名のままを「その人と話している」
        # と受け取ると、家族の誰かを決め打ちしてその人向けの口調で話し始めることになる。
        self._active_explicit: bool = False

    def register(self, name: str) -> None:
        """名前を知っている人に加える（FAMILY.md の呼び方・`/speaker`）。"""
        if name and name not in self._names:
            self._names.append(name)

    @property
    def active_name(self) -> str:
        return self._active_name

    @property
    def default_name(self) -> str:
        return self._default_name

    @property
    def active_is_explicit(self) -> bool:
        """話者が明示されたか。既定名のままなら偽。"""
        return self._active_explicit

    def set_active(self, name: str) -> None:
        self.register(name)
        self._active_name = name
        self._active_explicit = True

    def reset_to_default(self) -> None:
        self._active_name = self._default_name
        self._active_explicit = False

    def known_names(self) -> list[str]:
        """知っている名前（既定名を含む・登録順）。"""
        return list(self._names)

    def close(self) -> None:
        """持ち物は無い（以前は関係の追跡を閉じていた）。呼び手のために残す。"""
