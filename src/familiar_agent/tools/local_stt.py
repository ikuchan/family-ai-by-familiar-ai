"""常時集音をローカルで行う（出-a）。

ElevenLabs の WebSocket（`realtime_stt.py`）はサーバー側で VAD を持ち、区間を切って
書き起こしを返していた。ローカル化すると**その両方を自分でやる**ことになる。

- 区間を切る … `silero_vad.VADIterator`（逐次・16kHz・**512 サンプル固定**）
- 書き起こす … `faster-whisper`（`stt.py` のモデルを共有する）

**口の形は `RealtimeSttClient` に揃える。** `on_committed`／`on_partial` を `asyncio.Queue`
で持ち、`connected` を返す。こうしておけば、`RealtimeSttSession` は担い手を差し替えるだけで
済む（配信のゲート＝`VoiceLoopGuard` も従来どおり効く）。

**部分結果は出さない。** 途中で何度も書き起こすと GPU を無駄に回す（発話 3 秒なら 6 回）。
代わりに発話の始まりを `on_speech_start` で知らせ、GUI が「聞いています」の印を出す。

**音量だけでは切れない**（実測：話した区間の下位 25% が無音と重なる）。息継ぎで切らないため、
無音が既定 1.0 秒続いてから区間を確定させる。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)

# silero-vad が要求するフレーム長（16kHz のとき）。**この値は動かせない。**
FRAME_SAMPLES = 512
_RATE = 16000
_BYTES_PER_SAMPLE = 2


def should_use_local(cfg) -> bool:
    """常時集音をローカルで行うか（`STT_ENGINE`）。"""
    return getattr(cfg, "engine", "whisper") == "whisper"


class LocalSttEngine:
    """マイクの音を受け、区間が終わったら書き起こしを配る。

    `feed()` は `MicCapture` のコールバックから呼ばれる。溜めて 512 サンプルずつ VAD へ
    渡し、無音が続いたら溜めた音を書き起こす。
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self.on_partial: asyncio.Queue[str] | None = None
        self.on_committed: asyncio.Queue[str] | None = None
        # 発話の始まりを知らせる差し込み口（GUI が「聞いています」を出す）。
        self.on_speech_start: Callable[[], None] | None = None
        self._vad = None
        self._buffer = bytearray()        # 512 サンプルに切り出す前の余り
        self._segment = bytearray()       # いま溜めている発話
        self._speaking = False
        self._segment_limit = int(cfg.max_segment_sec * _RATE) * _BYTES_PER_SAMPLE
        # 短い区間を持ち越すためのしきい値。**諦める時機は時刻で測る**（VAD は境目しか
        # 返さないので、無音のフレーム数は数えられない）。
        self._min_bytes = int(cfg.min_segment_sec * _RATE) * _BYTES_PER_SAMPLE
        self._held = bytearray()              # 次の発話まで持ち越している短い区間
        self._held_since: float | None = None  # 持ち越し始めた時刻

    @property
    def connected(self) -> bool:
        """ローカルなので、繋がっているかを問う相手がいない。常に真。"""
        return True

    async def connect(self) -> None:
        """VAD を用意する（`RealtimeSttClient.connect` と同じ位置に置く）。"""
        await asyncio.to_thread(self._ensure_vad)

    async def close(self) -> None:
        """溜めているものを捨てる。書き起こしかけは配らない。"""
        self._buffer.clear()
        self._segment.clear()
        self._held.clear()
        self._speaking = False
        self._held_since = None

    async def send_audio(self, pcm16le: bytes) -> None:
        """`RealtimeSttClient` と同じ名前の口（セッション側の分岐を減らす）。"""
        await self.feed(pcm16le)

    async def feed(self, pcm16le: bytes) -> None:
        """音を受け取り、512 サンプルずつ VAD へ渡す。"""
        if not pcm16le:
            return
        self._buffer.extend(pcm16le)
        step = FRAME_SAMPLES * _BYTES_PER_SAMPLE
        while len(self._buffer) >= step:
            frame = bytes(self._buffer[:step])
            del self._buffer[:step]
            await self._handle_frame(frame)

    # ── 中身 ───────────────────────────────────────────────────────────────

    async def _handle_frame(self, frame: bytes) -> None:
        """1 フレームを VAD へ渡し、区間の始まりと終わりに応じて溜める。

        **VAD は境目でしか値を返さない**（`start`／`end`。発話中も無音中も `None`）。
        無音の長さは VAD が内部で見ているので、こちらでフレームを数えてはいけない。
        """
        event = self._vad_step(frame)
        if event == "start" and not self._speaking:
            self._begin_speech()
        if not self._speaking:
            # 黙っているあいだ。持ち越しているものがあれば、諦める時機を計る。
            if self._held and self._held_since is not None:
                if time.monotonic() - self._held_since >= self._cfg.hold_give_up_sec:
                    await self._flush_held()
            return
        self._segment.extend(frame)
        if event == "end":
            await self._commit()
        elif len(self._segment) >= self._segment_limit:
            # 長すぎる区間は区切る。雑音が続いたときに溜め続けない。
            logger.info("STT: 発話が %.0f 秒を超えたので区切る", self._cfg.max_segment_sec)
            await self._commit(force=True)

    def _begin_speech(self) -> None:
        self._speaking = True
        # 持ち越している短い区間があれば、その続きとして溜める（文脈を繋ぐ）。
        self._segment = bytearray(self._held)
        self._held.clear()
        self._held_since = None
        if self.on_speech_start is not None:
            try:
                self.on_speech_start()
            except Exception:  # noqa: BLE001
                logger.debug("STT: 発話の始まりを知らせられなかった")

    async def _commit(self, *, force: bool = False) -> None:
        """区間の終わり。長ければ書き起こし、短ければ次の発話まで持ち越す。

        短い断片では whisper が文脈を掴めず崩れる（実測）。`force` は上限で区切るときだけ
        真にする（長さは足りているので持ち越す意味がない）。
        """
        audio = bytes(self._segment)
        self._segment.clear()
        self._speaking = False
        if not audio:
            return
        if not force and len(audio) < self._min_bytes:
            self._held = bytearray(audio)
            self._held_since = time.monotonic()
            logger.debug("STT: 区間が短いので持ち越す（%.1f 秒）",
                         len(audio) / (_RATE * _BYTES_PER_SAMPLE))
            return
        await self._write(audio)

    async def _flush_held(self) -> None:
        """持ち越していた区間を、諦めて単独で書き起こす。"""
        audio = bytes(self._held)
        self._held.clear()
        self._held_since = None
        if audio:
            logger.debug("STT: 次の発話が来ないので持ち越しを配る")
            await self._write(audio)

    async def _write(self, audio: bytes) -> None:
        """書き起こして配る。失敗しても落とさない。"""
        try:
            text = await asyncio.to_thread(self._transcribe, audio)
        except Exception as e:  # noqa: BLE001
            logger.exception("STT: 書き起こしに失敗した: %s", e)
            return
        if not text:
            return
        queue = self.on_committed
        if queue is not None:
            queue.put_nowait(text)

    def _ensure_vad(self):
        """VAD を1度だけ用意する（差し替え点）。

        **無音の窓は VAD が内部で持つ**（`min_silence_duration_ms`）。フレームごとの
        「無音かどうか」は外から見えないので、自分で数えることはできない。
        """
        if self._vad is not None:
            return self._vad
        from silero_vad import VADIterator, load_silero_vad

        started = time.monotonic()
        self._vad = VADIterator(
            load_silero_vad(),
            sampling_rate=_RATE,
            min_silence_duration_ms=int(self._cfg.vad_silence_sec * 1000),
        )
        logger.info(
            "STT: VAD を用意した（%.1f 秒・無音 %.1f 秒で区間の終わり）",
            time.monotonic() - started, self._cfg.vad_silence_sec,
        )
        return self._vad

    def _vad_step(self, frame: bytes) -> str | None:
        """1 フレームを VAD へ渡し、境目なら "start"／"end" を返す（差し替え点）。"""
        import numpy as np
        import torch

        vad = self._ensure_vad()
        samples = np.frombuffer(frame, dtype=np.int16).astype("float32") / 32768.0
        result = vad(torch.from_numpy(samples), return_seconds=False)
        if not result:
            return None
        return "start" if "start" in result else "end"

    def _transcribe(self, audio: bytes) -> str:
        """溜めた PCM をそのまま書き起こす（差し替え点）。

        モデルは `stt.py` と共有する（VRAM 1,936 MiB を二重に持たない）。

        **WAV に包んで渡してはいけない。** faster-whisper は容れ物を渡されると PyAV で
        復号し 16kHz へ再標本化する。その再標本化が `EAGAIN` を返したとき、PyAV が errno の
        説明文を ascii で読もうとして落ちる（`LANG=ja_JP.UTF-8` では説明文が日本語）。
        音はすでに 16kHz なので、float32 の配列を渡せば復号も再標本化も起きない。
        """
        import numpy as np

        from .stt import load_whisper_model

        model = load_whisper_model(self._cfg)
        if model is None:
            return ""

        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        started = time.monotonic()
        segments, info = model.transcribe(
            samples,
            language=(self._cfg.language or None),
            vad_filter=False,      # 区間は既に VAD で切ってある
        )
        text = "".join(seg.text for seg in segments).strip()
        logger.info(
            "STT: 書き起こした（%d 字・%.2f 秒・音声 %.1f 秒）",
            len(text), time.monotonic() - started,
            len(audio) / (_RATE * _BYTES_PER_SAMPLE),
        )
        return text
