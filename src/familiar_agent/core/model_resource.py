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

**永続的な失敗と一時的な失敗を分ける。** カメラやマイクが無い構成では、ライブラリごと
入っていないことがある（`ImportError`）し、重みが取れないこともある（`FileNotFoundError`）。
**これらは何度試しても同じ**なので、`retries` の指定に関わらず即座に記憶して二度と試さない。
再試行すると呼ぶたびに数秒を失うだけになる（`load_whisper_model` は書き起こしのたびに
呼ばれる）。一時的な失敗（VRAM の不足・重みの取得中のネットワーク断）だけを `retries` の
回数まで試し、**使い切ったら記憶する。永久に試し続けない。**

**先読みは起動時に、別スレッドで行う。** `pre_warm()` は起動を止めない。読み終わる前に
最初の呼び出しが来たら、そこで待つ。

**モデルは口ではなく資源である**（`設計図` v0.85）。置き場は「何に密着しているか」で決まる
——言語の符号化器は OIF の内側、見え・顔・検出・音声は DIF の内側、LLM はどの口にも
属さない。この型枠はそのどれにも共通で効く。
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: 何度試しても結果が変わらない失敗。ライブラリが入っていない（`ImportError`）、
#: 重みが取れない（`FileNotFoundError`／`OSError`）。カメラやマイクの無い構成では
#: これが普通に起きるので、再試行の対象にしない。
_PERMANENT = (ImportError, FileNotFoundError)


def local_place(name: str) -> str:
    """Hugging Face の名前を、手元の控えの場所に引き直す（環-ax）。手元に無ければ名前のまま返す（外から取る）。

    名前のまま渡すと、手元に控えがあっても `transformers` が外へ問い合わせる（字句分けの `is_base_mistral` が
    `model_info` を呼ぶ。`local_files_only` でも止まらない）。2026-11-08 21:56、パソコンを起動し直した直後で
    ネットがつながる前に、埋め込みを読めず起動をやめた。場所を渡せば問い合わせない。引くのは通信しない。
    手元の版に留まるので、埋め込みの版が勝手に変わって記録の点とずれることもない。
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return str(snapshot_download(name, local_files_only=True))
    except LocalEntryNotFoundError:
        logger.info("モデル %s は手元に無いので、外から取る", name)
    except Exception:  # noqa: BLE001
        logger.warning("モデル %s の手元の控えを引けないので、外から取る", name, exc_info=True)
    return name


#: Hugging Face のモデルを 1 つずつ読むための錠（出-dr）。`transformers` は読み込みのあいだ torch の既定の型を
#: 変えて戻す（`torch.set_default_dtype`）。既定の型はプロセスに 1 つなので、別のスレッドで同時に読むと出入りが重なり、
#: 既定が bf16 のまま残ることがある。2026-11-14 21:26 の起動では、Ruri（bf16）と並んで読んだあと、見えのエンコーダが
#: 「expected scalar type Float but found BFloat16」で 17 分落ち続けた。
_HF_LOAD_LOCK = threading.Lock()


def load_local_first(name: str, load: "Callable[[str], T]") -> T:
    """手元の控えの場所から読み（`local_place`）、読めなければ名前で読み直す（控えが欠けていれば外から足す）。

    読むのは 1 つずつ（`_HF_LOAD_LOCK`）。読み終えて既定の型が float32 でなければ、WARNING を残して float32 に戻す。
    """
    with _HF_LOAD_LOCK:
        try:
            place = local_place(name)
            if place == name:
                return load(name)
            try:
                return load(place)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "モデル %s を手元の控えから読めないので、名前で読み直す", name, exc_info=True
                )
                return load(name)
        finally:
            _restore_default_dtype(name)


def _restore_default_dtype(name: str) -> None:
    """読み終えて torch の既定の型が float32 でなければ、戻して WARNING に残す（出-dr）。"""
    try:
        import torch
    except ImportError:
        return
    if torch.get_default_dtype() != torch.float32:
        logger.warning(
            "モデル %s を読んだあと torch の既定の型が %s のままだったので、float32 に戻した",
            name,
            torch.get_default_dtype(),
        )
        torch.set_default_dtype(torch.float32)


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
        retries: int = 0,
    ) -> None:
        """
        `name` はログに出る呼び名（「人検出」「見えのエンコーダ」など）。
        `fatal` は「これが無ければ続けられない」の宣言。既定は縮退（False）。
        `device_env` は載せる先を指定する環境変数の名前（無ければ自動判定だけ）。
        `retries` は**一時的な失敗**を試し直す回数。既定は 0（一度きり）。
        """
        self._mr_name = name
        self._mr_fatal = fatal
        self._mr_device_env = device_env
        self._mr_retries_left = max(0, retries)
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
                logger.info(
                    "%sのモデルを読み込んだ（%s）",
                    self._mr_name,
                    self.device or "載せる先はライブラリ任せ",
                )
            except _PERMANENT as e:
                # 無いものは何度試しても無い。回数に関わらず記憶する。
                self._mr_failed = True
                if self._mr_fatal:
                    logger.exception("%sのモデルが無い（続けられない）", self._mr_name)
                    raise
                logger.warning("%sのモデルが無いので使わない（%s）", self._mr_name, e)
            except Exception as e:  # noqa: BLE001
                if self._mr_retries_left > 0:
                    self._mr_retries_left -= 1
                    logger.warning(
                        "%sのモデルを読み込めない（あと %d 回試す）: %s",
                        self._mr_name,
                        self._mr_retries_left + 1,
                        e,
                    )
                    return None
                self._mr_failed = True
                if self._mr_fatal:
                    logger.exception("%sのモデルを読み込めない（続けられない）", self._mr_name)
                    raise
                logger.exception("%sのモデルを読み込めない（縮退して続ける）: %s", self._mr_name, e)
            return self._mr_model

    def pre_warm(self) -> None:
        """起動時に読み始める。**起動は止めない**（別スレッドで走る）。

        読み終わる前に最初の呼び出しが来たら、`ensure()` が錠の前で待って同じものを返す
        （二重に読まない）。
        """
        threading.Thread(target=self.ensure, daemon=True, name=f"prewarm-{self._mr_name}").start()

    @property
    def ready(self) -> bool:
        """読み込み済みか。**読みにはいかない**（問い合わせが副作用を持たない）。"""
        return self._mr_model is not None

    @property
    def failed(self) -> bool:
        """読み込みに失敗したか（二度目を試さない印）。"""
        return self._mr_failed

    @property
    def device(self) -> str | None:
        """載せる先。**指定が無ければ `None`＝ライブラリに任せる。**

        環境変数を見るのは、テストが並列に走るためである（ワーカーごとにモデルを載せると
        GPU の VRAM を使い切ってプロセスごと落ちる。実測で `CUDA out of memory` が出た）。

        **型枠が `cuda`／`cpu` を決めてしまわない。** どこへ載せるかを一番よく知っているのは
        ライブラリのほうで、「明示せず任せる」は正当な選択である。実際、YOLO は device を
        渡しておらず（ultralytics 任せ）、埋め込みは「自動のときは device を渡さない
        （従来と同じ呼び方）」を実測から決めている。
        """
        if not self._mr_device_env:
            return None
        return os.environ.get(self._mr_device_env, "").strip() or None
