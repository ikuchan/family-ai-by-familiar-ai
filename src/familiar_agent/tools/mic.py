"""Microphone capture — streams PCM 16 kHz 16-bit mono to an async callback."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import shutil
import subprocess
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

TARGET_RATE = 16000  # ElevenLabs Realtime STT expects 16 kHz PCM
CHANNELS = 1
_BLOCK_MS = 96  # 取り込むブロックの長さ（ミリ秒）。16kHz で 1,536 サンプル＝512×3 になり、
# silero-vad が要求する 512 サンプルで割り切れる（余りを持ち越さない）。


def _is_wsl2() -> bool:
    release = platform.release().lower()
    return bool(os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME")) or (
        "microsoft" in release or "wsl" in release
    )


def describe_sounddevice_input_failure(exc: Exception | None = None) -> str:
    """Return a user-facing microphone diagnosis for sounddevice failures."""
    detail = str(exc).strip() if exc else ""
    parts: list[str] = []
    if detail:
        parts.append(detail)

    if _is_wsl2():
        parts.append(
            "WSL2/WSLg hint: set PULSE_SERVER=unix:/mnt/wslg/PulseServer and install "
            "pulseaudio-utils plus libasound2-plugins. If `python -m sounddevice` shows no "
            "input devices, PortAudio cannot see the WSLg microphone bridge yet."
        )
    else:
        parts.append(
            "No default microphone input device is available to sounddevice. Try "
            "`python -m sounddevice` and check your OS microphone permissions."
        )

    return " ".join(part for part in parts if part)


def probe_sounddevice_input() -> tuple[bool, str]:
    """Best-effort check that sounddevice can see a default input device."""
    try:
        import sounddevice as sd
    except ImportError:
        return False, "sounddevice is not installed."

    try:
        devices = sd.query_devices()
    except Exception as exc:  # pragma: no cover - covered via query(kind="input") too
        return False, describe_sounddevice_input_failure(exc)

    if not devices:
        return False, describe_sounddevice_input_failure(
            RuntimeError("sounddevice did not enumerate any audio devices.")
        )

    try:
        info = sd.query_devices(kind="input")
    except Exception as exc:
        return False, describe_sounddevice_input_failure(exc)

    name = str(info.get("name", "default")).strip() or "default"
    sample_rate = int(info.get("default_samplerate", TARGET_RATE) or TARGET_RATE)
    return True, f"{name} @ {sample_rate} Hz"


class _Resampler:
    """素の標本化率から 16 kHz へ落とす（int16 モノラル）。

    **間引く前に帯域を切らなければならない。** このマイク（Yamaha YVC-300）は 48,000 Hz で、
    16,000 Hz へは正確に 3:1 である。以前は `np.interp` で位置を拾っていたが、それは
    3 サンプルおきに間引くだけで、**低域通過フィルタを通していなかった**。8 kHz より上の
    成分が折り返して band 内の雑音になり（エイリアシング）、常時集音の書き起こしが話した
    内容と全く違うものになっていた。

    `soxr.ResampleStream` は**フィルタの状態をまたいで保つ**ので、96 ミリ秒ずつ渡しても
    境目に段差が入らない。1 つの取り込みにつき 1 つ持つ（状態を混ぜない）。
    """

    def __init__(self, from_rate: int) -> None:
        self._stream = None
        if from_rate != TARGET_RATE:
            import soxr

            self._stream = soxr.ResampleStream(
                from_rate,
                TARGET_RATE,
                1,
                dtype="int16",
                quality="VHQ",
            )

    def process(self, pcm_bytes: bytes) -> bytes:
        if self._stream is None or not pcm_bytes:
            return pcm_bytes
        arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        return self._stream.resample_chunk(arr).tobytes()


def pick_input_device(devices: "list[dict]", apis: "list[dict]", wanted: str) -> "int | None":
    """名前に `wanted` を含む入力の機器の番号（`audio_devices.pick_device` の入力）。無ければ None。

    PipeWire を通る機器を直接の機器より先に選び、スピーカーの写しは選ばない（出-du 段 2・2026-11-23）。
    """
    from .audio_devices import pick_device

    return pick_device(devices, apis, wanted, kind="input")


def _pw_record_binary() -> "str | None":
    """`pw-record` の場所。無ければ None（差し替え点）。"""
    return shutil.which("pw-record")


def _pw_source_name(wanted: str) -> "str | None":
    """名前に `wanted` を含む、PipeWire（PulseAudio の口）の入力の名前。無ければ None（差し替え点）。

    PulseAudio の口に見える名前は、PipeWire の入力の口の名前そのもので、`pw-record --target` にそのまま渡せる。
    """
    try:
        import sounddevice as sd

        devices, apis = list(sd.query_devices()), list(sd.query_hostapis())
    except Exception:  # noqa: BLE001
        logger.warning("マイク：PipeWire の入力の口を探せなかった（直接の道で開く）", exc_info=True)
        return None
    from .audio_devices import is_pipewire

    i = pick_input_device(devices, apis, wanted)
    if i is None or not is_pipewire(apis, devices[i]):
        return None
    return str(devices[i]["name"])


def input_gain_from_env() -> float:
    """`.env` の `AUDIO_INPUT_GAIN`（倍率）。無い・読めない・0 以下なら 1.0（いまのまま）。

    マイク（Yamaha YVC-300）のハード音量は上限で、普通の声だと VAD にも whisper にも
    ほぼ無音に見えた（2026-09-16 実機）。取り込んだ PCM をソフトで増やす。
    """
    raw = os.environ.get("AUDIO_INPUT_GAIN", "").strip()
    if not raw:
        return 1.0
    try:
        gain = float(raw)
    except ValueError:
        logger.warning("AUDIO_INPUT_GAIN=%r を数として読めないので 1.0 にする", raw)
        return 1.0
    if gain <= 0:
        logger.warning("AUDIO_INPUT_GAIN=%r は 0 以下なので 1.0 にする", raw)
        return 1.0
    return gain


def apply_gain(pcm_bytes: bytes, gain: float) -> bytes:
    """int16 モノラルの PCM に倍率を掛ける。範囲を超えた分は飽和させる（折り返さない）。"""
    if gain == 1.0 or not pcm_bytes:
        return pcm_bytes
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) * gain
    return np.clip(arr, -32768, 32767).astype(np.int16).tobytes()


class MicCapture:
    """Capture audio from the default microphone using *sounddevice*.

    Automatically detects the device's native sample rate and resamples to
    16 kHz before handing PCM bytes to *on_audio*.  This avoids
    ``paInvalidSampleRate`` on devices whose native rate differs from 16 kHz
    (e.g. USB mics that default to 44 100 Hz).
    """

    def __init__(self, on_audio) -> None:  # noqa: ANN001 – Callable[[bytes], Awaitable]
        self._on_audio = on_audio
        self._stream: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._native_rate: int = TARGET_RATE
        # 開いた機器の名前（機器の見張りが、YAMAHA を開いたかを見る・知-bv）。
        self.device_name: str = ""
        # `pw-record` の道（出-du 段 2）：子のプロセスと、その音を読む糸。
        self._proc: "subprocess.Popen | None" = None
        self._reader: "threading.Thread | None" = None
        self._stopped = (
            False  # 止めた印（止めたあとの塊を渡さない・止めないうちの終わりを警告にする）
        )

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start capturing from the default input device.

        名前が合う PipeWire の入力の口があり `pw-record` があれば、そこから読む（出-du 段 2・2026-11-23・本人の決定イ）。
        YAMAHA を掴むのを PipeWire だけにし（パジュが直接掴むと、PipeWire が入力の口を開くとき「ビジー」でぶつかり、
        スピーカーの再生も始まらなくなった）、読み取りは別のプロセスにする（声の出口の `pw-play` と同じ・止めるのは子だけ）。
        無ければ、いままでどおり sounddevice で開く。
        """
        self._loop = loop
        self._stopped = False
        from .audio_devices import device_name

        wanted = device_name()
        binary = _pw_record_binary()
        source = _pw_source_name(wanted) if wanted and binary else None
        if binary and source:
            try:
                self._start_pw_record(binary, source)
                return
            except RuntimeError:
                # 耳が聞こえないまま始まりかけで残るより、直接の道で開く（取り合いで声が止まっても、出-du の見張りで抜ける）。
                logger.warning(
                    "マイク：pw-record で始められなかったので、直接の道で開く", exc_info=True
                )
        if wanted and binary:
            # 直接の機器を掴むと、PipeWire が入力の口を開くときにぶつかる（出-du）。黙って戻らない。
            logger.warning("マイク：PipeWire に名前の合う入力の口が無いので、直接の道で開く")

        import sounddevice as sd

        ok, detail = probe_sounddevice_input()
        if not ok:
            raise RuntimeError(detail)

        try:
            # 機器の名前（`audio_devices.device_name()`）の一部で選ぶ。
            device_idx: int | None = None
            audio_input_device = wanted
            if audio_input_device:
                device_idx = pick_input_device(
                    list(sd.query_devices()), list(sd.query_hostapis()), audio_input_device
                )
                if device_idx is None:
                    logger.warning(
                        "マイク：名前の合う機器が無いので、既定の入力で開く",
                    )

            # Use the device's native sample rate to avoid paInvalidSampleRate
            device_info = sd.query_devices(device=device_idx, kind="input")
            self.device_name = str(device_info.get("name", ""))
            self._native_rate = int(device_info["default_samplerate"])
            block_size = int(self._native_rate * _BLOCK_MS / 1000)
            # 取り込みごとに1つ持つ（フィルタの状態を前の取り込みと混ぜない）。
            resampler = _Resampler(self._native_rate)
            gain = input_gain_from_env()

            logger.info(
                "Microphone capture: device=%s native_rate=%d target_rate=%d gain=%.2f",
                device_info.get("name", "default"),
                self._native_rate,
                TARGET_RATE,
                gain,
            )

            def _callback(indata, frames, time_info, status):  # noqa: ANN001, ARG001
                if status:
                    logger.debug("Mic status: %s", status)
                self._deliver(apply_gain(resampler.process(bytes(indata)), gain))

            self._stream = sd.RawInputStream(
                samplerate=self._native_rate,
                blocksize=block_size,
                channels=CHANNELS,
                dtype="int16",
                device=device_idx,
                callback=_callback,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            raise RuntimeError(describe_sounddevice_input_failure(exc)) from exc

    def _deliver(self, pcm: bytes) -> None:
        """16 kHz にした音を、ループの側の `on_audio` へ渡す（取り込みの糸から呼ぶ）。"""
        loop = self._loop
        if loop is None or loop.is_closed() or self._stopped:
            return  # 止めたあとに届いた最後の塊は渡さない（起こし直した次の聞き取りへ混ぜない）

        def _hand() -> None:
            if not self._stopped:  # 渡す直前にもう一度（止めたのが、積んだあとだったとき）
                loop.create_task(self._on_audio(pcm))

        with contextlib.suppress(RuntimeError):  # 閉じる途中のループ（止めた直後に届いた最後の塊）
            loop.call_soon_threadsafe(_hand)

    def _start_pw_record(self, binary: str, source: str) -> None:
        """`pw-record` を起こし、標準出力の音を 96 ミリ秒ずつ読む糸を立てる。"""
        # 48 kHz で受け取る（YVC-300 の素の値。ほかの機器なら PipeWire がここへ合わせる）。16 kHz への再標本化は、
        # 帯域を切る質を保つため、いままでどおりこちらの soxr で行う。
        rate = 48_000
        self._native_rate = rate
        self.device_name = source
        resampler = _Resampler(rate)
        gain = input_gain_from_env()
        self._stopped = False
        from ..config import AgentConfig

        latency = int(AgentConfig.mic_latency_samples)
        proc = subprocess.Popen(
            [
                binary,
                "--raw",
                # 行き先が消えても、既定の入力（内蔵のマイクなど）へ黙って付け替えない（見張りが気づけなくなる）。
                "-P",
                "{ node.dont-fallback = true, node.dont-reconnect = true }",
                f"--target={source}",
                # 区切り（`AgentConfig.mic_latency_samples`・出-du 段 3）。既定の 100 ミリ秒だと声が割れた。
                f"--latency={latency}",
                f"--rate={rate}",
                f"--channels={CHANNELS}",
                "--format=s16",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            # 起こしてすぐ終わった（行き先が無い・PipeWire に届かない）なら、始められなかったとして上げる。
            code = proc.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            code = None
        if code is not None:
            with contextlib.suppress(Exception):
                if proc.stdout is not None:
                    proc.stdout.close()
            raise RuntimeError(f"pw-record がすぐ終わった（戻り値 {code}・入力の口 {source}）")
        self._proc = proc
        block = int(rate * _BLOCK_MS / 1000) * 2

        def _read() -> None:
            out = proc.stdout
            if out is None:
                return
            odd = b""
            try:
                while data := out.read(block):
                    data, odd = odd + data, b""
                    if len(data) % 2:  # 半端な 1 バイトは次の塊へ（int16 の途中で切れた）
                        data, odd = data[:-1], data[-1:]
                    self._deliver(apply_gain(resampler.process(data), gain))
            except Exception:  # noqa: BLE001
                logger.exception("Microphone capture: pw-record の音を読めなかった")
                return
            if self._stopped:
                logger.info("Microphone capture: pw-record の音が終わった（止めた）")
            else:
                logger.warning(
                    "Microphone capture: pw-record が止めないうちに終わった（戻り値 %s）",
                    proc.poll(),
                )

        self._reader = threading.Thread(target=_read, daemon=True, name="mic-pw-record")
        self._reader.start()
        logger.info(
            "Microphone capture: pw-record source=%s native_rate=%d latency=%d target_rate=%d gain=%.2f",
            source,
            rate,
            latency,
            TARGET_RATE,
            gain,
        )

    def stop(self) -> None:
        """Stop capturing."""
        if self._proc is not None:
            proc, self._proc = self._proc, None
            self._stopped = True
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(timeout=1.0)
            if self._reader is not None:
                self._reader.join(timeout=1.0)
                self._reader = None
            with contextlib.suppress(Exception):
                if proc.stdout is not None:
                    proc.stdout.close()
            logger.info("Microphone capture stopped（pw-record）")
            return
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Microphone capture stopped")
