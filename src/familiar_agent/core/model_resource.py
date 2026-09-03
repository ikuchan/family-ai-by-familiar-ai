"""モデル資源（MR）。重みを持つものが従う型枠（出-c）。

**引き受けるのは推論そのものではなく、モデルを持つことに伴う関心事である。** 遅延読み込み
（二度目は試さない）・失敗の記憶・並行制御（読み込みは1回だけ）・載せる先の解決・準備の
問い合わせ。推論は各実装が `_load()` で作ったものを使って自分で行う。

**同じ形が3回、少しずつ違って書かれていた**（`モジュール分割設計` §出-c）。失敗フラグの
名前も、読み込み口の形も、載せる先の決め方も揃わず、並行制御は3つのうち1つにしか無かった。
「二度目以降は試さない」という同じ文言のコメントが2箇所にあった。

**失敗の約束は宣言で分ける。** `fatal=True` なら例外を投げ、そうでなければ縮退する
（`ensure()` が `None` を返し、呼び出し側が「見えなかった」として続ける）。3通りに割れて
いたのは誤りではなく、出-b が「埋め込みは致命・他は縮退」と決めた結果である。**どちらを
選んだかがコードの一行に出る**形にした。

**モデルは口ではなく資源である**（`設計図` v0.85）。置き場は「何に密着しているか」で決まる
——言語の符号化器は OIF の内側、見え・顔・検出・音声は DIF の内側、LLM はどの口にも
属さない。この型枠はそのどれにも共通で効く。
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class ModelResource(ABC):
    """重みを持つものの共通の型枠。

    `_load()` だけを実装すればよい。読む回数・失敗の扱い・載せる先は型枠が持つ。
    """

    def __init__(
        self,
        *,
        name: str,
        fatal: bool = False,
        device_env: str | None = None,
    ) -> None:
        """
        `name` はログに出る呼び名（「人検出」「見えのエンコーダ」など）。
        `fatal` は「これが無ければ続けられない」の宣言。既定は縮退（False）。
        `device_env` は載せる先を指定する環境変数の名前（無ければ自動判定だけ）。
        """
        self._mr_name = name
        self._mr_fatal = fatal
        self._mr_device_env = device_env
        self._mr_model: Any = None
        self._mr_failed = False
        self._mr_lock = threading.Lock()

    @abstractmethod
    def _load(self) -> Any:
        """重みを読み、使える形にして返す。**失敗したら例外を投げる**（型枠が扱う）。"""

    def ensure(self) -> Any:
        """読み込み済みのモデルを返す。まだなら1回だけ読む。

        **二度目以降は試さない。** 毎回試すと、重みの無い環境で呼ぶたび重くなる。
        **並行して呼ばれても読むのは1回だけ**。在席の常駐と想起が同時に触りうる。
        """
        if self._mr_model is not None or self._mr_failed:
            return self._mr_model
        with self._mr_lock:
            # 待っているあいだに別のスレッドが読み終えている場合がある。
            if self._mr_model is not None or self._mr_failed:
                return self._mr_model
            try:
                self._mr_model = self._load()
                logger.info("%sのモデルを読み込んだ（%s）", self._mr_name, self.device)
            except Exception as e:  # noqa: BLE001
                self._mr_failed = True
                if self._mr_fatal:
                    logger.exception("%sのモデルを読み込めない（続けられない）", self._mr_name)
                    raise
                logger.exception("%sのモデルを読み込めない（縮退して続ける）: %s",
                                 self._mr_name, e)
            return self._mr_model

    @property
    def ready(self) -> bool:
        """読み込み済みか。**読みにはいかない**（問い合わせが副作用を持たない）。"""
        return self._mr_model is not None

    @property
    def failed(self) -> bool:
        """読み込みに失敗したか（二度目を試さない印）。"""
        return self._mr_failed

    @property
    def device(self) -> str:
        """載せる先。**環境変数 → 自動判定 → CPU** の順で決める。

        環境変数を最優先にするのは、テストが並列に走るためである（ワーカーごとにモデルを
        載せると GPU の VRAM を使い切ってプロセスごと落ちる）。
        """
        if self._mr_device_env:
            chosen = os.environ.get(self._mr_device_env, "").strip()
            if chosen:
                return chosen
        return "cuda" if self._cuda_available() else "cpu"

    def _cuda_available(self) -> bool:
        """GPU が使えるか。`torch` が無い構成でも落ちない。"""
        try:
            import torch
        except Exception:  # noqa: BLE001
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False
