"""TTS tool - voice of the embodied agent.

Built-in tools:
- say(text): speak aloud. 合成の担い手は TTS_ENGINE で決まる（gemini＝Gemini TTS・既定／
  sbv2＝ローカルの Style-Bert-VITS2／elevenlabs＝外部 API）。角括弧タグ [cheerful] を
  渡せるのは解する担い手だけ。
  When ELEVENLABS_API_KEY is unset, runs in display-only (silent) mode — text is shown but not spoken.
Config: ELEVENLABS_API_KEY, TTS_VOICE_ID, GO2RTC_URL, TTS_OUTPUT.

**主と控えの 2 段**（環-v・2026-09-24）。主が話せなかったら控え（既定は SBV2）で 1 回だけ
話し直す。主は外の API なのでネットが切れれば黙るが、家に居る相手からは壊れたのと区別が
つかない。控えはローカルで動くので、そこだけは代えられない。控えは起動時から温めておく。

担い手ごとに渡す文が違う（`_text_for_synth`）。ElevenLabs は漢字を読めないので全文を
ひらがなに、SBV2 は `pyopenjtalk` の読み違いだけを表で直し、Gemini はそのまま渡す。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

from ..voice_guard import VoiceLoopGuard, get_shared_voice_guard

logger = logging.getLogger(__name__)


def _write_pcm_as_wav(
    pcm_bytes: bytes, sample_rate: int = 16000, tmp_dir: str | None = None
) -> str:
    """Write raw 16-bit mono PCM bytes to a temp WAV file and return the path.

    Builds the 44-byte WAV/RIFF header without any external dependency.
    """
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    file_size = 36 + data_size  # RIFF chunk size = file_size - 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size,
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )

    suffix = ".wav"
    kwargs: dict = {"suffix": suffix, "delete": False}
    if tmp_dir:
        kwargs["dir"] = tmp_dir
    with tempfile.NamedTemporaryFile(**kwargs) as f:
        f.write(header)
        f.write(pcm_bytes)
        return f.name


def _write_tmp_audio(data: bytes, suffix: str = ".mp3") -> str:
    """Write raw audio bytes to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        return f.name


_GO2RTC_CACHE = Path.home() / ".cache" / "embodied-claude" / "go2rtc"
# On Windows the binary is go2rtc.exe; on other platforms there is no extension.
_GO2RTC_BIN = _GO2RTC_CACHE / ("go2rtc.exe" if sys.platform == "win32" else "go2rtc")
_GO2RTC_CONFIG = _GO2RTC_CACHE / "go2rtc.yaml"


def _ensure_go2rtc(api_url: str) -> None:
    """Start go2rtc if it's not already running."""
    try:
        urllib.request.urlopen(f"{api_url}/api", timeout=2)
        return  # already running
    except Exception:
        pass

    if not _GO2RTC_BIN.exists():
        logger.warning("go2rtc binary not found at %s", _GO2RTC_BIN)
        return
    if not _GO2RTC_CONFIG.exists():
        logger.warning("go2rtc config not found at %s", _GO2RTC_CONFIG)
        return

    logger.info("Starting go2rtc...")
    subprocess.Popen(
        [str(_GO2RTC_BIN), "-config", str(_GO2RTC_CONFIG)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    import time

    for _ in range(10):
        time.sleep(0.5)
        try:
            urllib.request.urlopen(f"{api_url}/api", timeout=1)
            logger.info("go2rtc started")
            return
        except Exception:
            continue
    logger.warning("go2rtc did not start in time")


# 自分で起こした合成サーバー。終了時に止めるために持つ（人が別に立てたものは触らない）。
# 角括弧タグ（[cheerful] など）を指示として解する担い手とモデル。ここに載っていないものには
# 渡さない——解さないモデルへ渡せば、そのまま音になるか、音素として崩れる。ElevenLabs でも
# 解するのは `eleven_v3` だけで、既定の flash は解さない。
TAG_AWARE_ENGINES = ("elevenlabs",)
TAG_AWARE_MODELS = ("eleven_v3",)
DEFAULT_ELEVENLABS_MODEL = "eleven_flash_v2_5"

#: Gemini TTS の既定（環-v・`根拠台帳` §45）。実機で詰まった 5 文を漢字のまま 5/5 で読み、
#: 感情も 3 つとも聞き分けられた組み合わせ。同じモデルでも声で変わる。
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash-lite-tts"
DEFAULT_GEMINI_VOICE = "Charon"

#: Gemini の逐次の返りは生 PCM（24kHz・16bit・mono）。一括のときだけ WAV で返る。
GEMINI_RATE = 24000


#: つなぎの道具（出-aq 段 2・2026-09-25）。**`say` とは別の道具にする。**
#: `say` は「返事」なので、つなぎを `say` で表すと主LLM は「もう返事をした」と受け取り、
#: 8 回中 7 回黙った（`(応答は既に送信済み)` と書いた）。別の道具なら 10/10 が本題を言う。
#: 定義は**規則ではなく、つなぎが何であるか**を書く——「〜するな」は置き場を変えても
#: 効かなかった（`根拠台帳` §48）。落とした後の漏れが 0 だった形（返事の前半・`say` は
#: その続き・重複は機械が消す）を使う。
FILLER_TOOL: dict = {
    "name": "filler",
    "description": (
        "Say the opening of your reply aloud while you are still working out the rest, so the "
        "person knows you heard them and are not ignoring them. The person has already heard "
        "it. Your `say` continues from there. If `say` begins with words you already spoke in "
        "`filler`, the system removes them before speaking, so the person never hears the same "
        "words twice."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "What to say first."}},
        "required": ["text"],
    },
}


_sbv2_proc: "subprocess.Popen | None" = None


def stop_sbv2_server() -> None:
    """自分で起こした合成サーバーを止める。

    **GPU を握り続けさせない。** モデルを載せたまま残ると、他の作業で GPU を使うときに
    邪魔になる。次の起動で読み込み（実測 2.8 秒）をやり直すことになるが、それは払える。

    自分が起こしていないなら何もしない。人が別に立てている場合がある。
    """
    global _sbv2_proc
    proc = _sbv2_proc
    _sbv2_proc = None
    if proc is None or proc.poll() is not None:
        return
    logger.info("SBV2 の合成サーバーを止める")
    with contextlib.suppress(Exception):
        proc.terminate()


def _sbv2_is_alive(url: str) -> bool:
    """合成サーバーが応えるか。"""
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def _spawn_sbv2(cfg) -> None:
    """合成サーバーを起こす（別プロセス・SBV2 専用の venv で走らせる）。

    本体の venv では動かない（Python と torch の版が違う）。`SBV2_PYTHON` が指す
    インタプリタで `scripts/sbv2_server.py` を実行する。
    """
    python = Path(os.path.expanduser(cfg.sbv2_python))
    script = Path(__file__).resolve().parents[3] / "scripts" / "sbv2_server.py"
    if not python.exists():
        logger.warning("SBV2 の python が見つからない（合成は使えない）: %s", python)
        return
    if not script.exists():
        logger.warning("SBV2 のサーバーが見つからない: %s", script)
        return
    env = dict(os.environ)
    env["SBV2_MODEL"] = cfg.sbv2_model
    env["SBV2_MODEL_DIR"] = os.path.expanduser(cfg.sbv2_model_dir)
    # url から待ち受け先を渡す（既定 127.0.0.1:5001）。
    host_port = cfg.sbv2_url.split("//", 1)[-1]
    if ":" in host_port:
        env["SBV2_HOST"], env["SBV2_PORT"] = host_port.split(":", 1)
    logger.info("SBV2 の合成サーバーを起こす（%s・モデル %s）", python, cfg.sbv2_model)
    global _sbv2_proc
    _sbv2_proc = subprocess.Popen(
        [str(python), str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def _gemini_client(api_key: str):
    """Gemini の口（差し替え点）。**鍵ごとに 1 つだけ持つ。**

    `genai.Client` は接続の設定を抱えるので、発話のたびに作ると支度を払い直すことになる。
    """
    global _gemini_clients
    client = _gemini_clients.get(api_key)
    if client is None:
        from google import genai

        client = genai.Client(api_key=api_key)
        _gemini_clients[api_key] = client
    return client


_gemini_clients: dict = {}


def ensure_sbv2_server(cfg, *, engine: str, output: str) -> None:
    """起動時に合成サーバーを起こす（使う構成のときだけ）。

    **待たない。** モデルの読み込みに十数秒かかるので、起動を塞がずに投げておく。最初の
    発話までに間に合わなければ、その1回だけ話せない（degrade して次から鳴る）。

    **控えに回っているときも起こす**（環-v）。主（Gemini）が話せなくなるのはネットが切れた
    ときで、そこからモデルを読むと 26.3 秒黙る。それでは控えの役を果たさない。

    使わない構成（別のエンジン・音を出さない）では起こさない。GPU と十数秒を無駄にしない。
    """
    if output == "silent":
        return
    if "sbv2" not in (engine, getattr(cfg, "fallback_engine", "")):
        return
    if _sbv2_is_alive(cfg.sbv2_url):
        return
    _spawn_sbv2(cfg)


class TTSTool:
    """Text-to-speech using ElevenLabs, played via go2rtc camera speaker and/or local speaker."""

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        go2rtc_url: str = "http://localhost:1984",
        go2rtc_stream: str = "tapo_cam",
        output: str = "local",
        voice_guard: VoiceLoopGuard | None = None,
        engine: str = "elevenlabs",
        sbv2_url: str = "http://127.0.0.1:5001",
        sbv2_style: str = "Neutral",
        sbv2_weight: float = 1.0,
        elevenlabs_model: str = DEFAULT_ELEVENLABS_MODEL,
        careful_model: str = "eleven_v3",
        speed: float = 0.9,
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        gemini_voice: str = DEFAULT_GEMINI_VOICE,
        gemini_api_key: str = "",
        fallback_engine: str = "",
    ) -> None:
        self.engine = engine
        # Gemini TTS と、主が話せなかったときの控え（環-v・2026-09-24）。
        self.gemini_model = gemini_model
        self.gemini_voice = gemini_voice
        self.gemini_api_key = gemini_api_key
        self.fallback_engine = fallback_engine
        self.elevenlabs_model = elevenlabs_model
        # じっくり読む声と、読み上げの速さ（環-u・2026-09-21）。
        self.careful_model = careful_model
        self.speed = float(speed)
        self.sbv2_url = sbv2_url
        self.sbv2_style = sbv2_style
        self.sbv2_weight = sbv2_weight
        self.api_key = api_key
        self.voice_id = voice_id
        self.go2rtc_url = go2rtc_url
        self.go2rtc_stream = go2rtc_stream
        # "local" = PC speaker only, "remote" = camera speaker only, "both" = both simultaneously
        self.output = output
        self._voice_guard = voice_guard or get_shared_voice_guard()
        # Serialize concurrent say() calls so audio never overlaps
        self._lock = asyncio.Lock()
        if self.api_key:
            _ensure_go2rtc(self.go2rtc_url)

    @property
    def understands_tags(self) -> bool:
        """この担い手が角括弧タグを指示として解するか。

        整え方・`say` の説明・規則の3箇所がこの1つの値を見る。担い手を切り替えたときに
        どれかだけが取り残されないようにするためである。
        """
        return self.engine in TAG_AWARE_ENGINES and self.elevenlabs_model in TAG_AWARE_MODELS

    def _clean_for_speech(self, text: str) -> str:
        """声にする前に整える。丸括弧のト書きは常に落とす（読み上げても意味が無い）。

        角括弧タグは、解する担い手にだけ残す。
        """
        from .._ui_helpers import clean_spoken_text, strip_stage_directions

        if self.understands_tags:
            return strip_stage_directions(text)
        return clean_spoken_text(text)

    def _elevenlabs_payload(self, text: str, *, careful: bool = False) -> dict:
        """ElevenLabs へ送る中身（環-u・2026-09-21）。

        `careful` は**じっくり読む声**（既定 `eleven_v3`）。漢字を読めるので**ひらがな化を
        通さない**。合成に 3.4 秒かかるので、主LLM が外への問い合わせの返りで話すときだけ使う
        （`core/voice_rules.careful_voice`）。`speed` は両方に掛ける（既定 0.9・1.0 は速すぎた）。
        """
        model = (
            getattr(self, "careful_model", "eleven_v3")
            if careful
            else getattr(self, "elevenlabs_model", DEFAULT_ELEVENLABS_MODEL)
        )
        body = text if careful else self._text_for_synth(text)
        return {
            "text": body,
            "model_id": model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": float(getattr(self, "speed", 0.9)),
            },
        }

    def _text_for_synth(self, text: str) -> str:
        """合成器へ渡す文。担い手ごとに、必要なぶんだけ直す。

        - `elevenlabs`：漢字を読めないので全文をひらがなに（環-t・出-ac）
        - `sbv2`：自前で読むが `pyopenjtalk` の読み違いが残るので**表の分だけ**当てる（環-v）
        - `gemini`：**何もしない**。漢字をそのまま読めることが選んだ理由である（環-v）

        声の門（`voice_guard`）へは元の文を渡す——書き起こし（漢字まじり）との照合に使うため。
        """
        from ..core.reading import fix_readings, for_speech

        if self.engine == "elevenlabs":
            return for_speech(text)
        if self.engine == "sbv2":
            return fix_readings(text)
        return text

    async def say(
        self, text: str, output: str | None = None, *, gain: float = 1.0, careful: bool = False
    ) -> str:
        """声に出す。合成の担い手は `engine` で決まる。`gain` はこの 1 回の再生にだけ掛ける倍率。

        `output`："local"＝PC のスピーカー／"remote"＝カメラのスピーカー（go2rtc）／
        "both"＝両方／"silent"＝出さない（実機テスト用・**合成も走らせない**）。

        同時に呼ばれても音が重ならないよう、`self._lock` で直列にする。
        """
        if output is None:
            output = self.output
        if output == "silent":
            return f"Said (silent): {text[:60]}"
        self._gain = gain  # この 1 回の再生にだけ効く（`_play_paths`／`_play_local` が読む）
        result = await self._say_with(self.engine, text, output, careful=careful)
        # 控えへ落ちる（環-v）。ネットが切れても話せるように、ローカルの SBV2 を残してある。
        # **1 回だけ**。控えでも駄目なら、主の言い分をそのまま返す（何が起きたかが残る）。
        # 控えは既存のテストが `__new__` で組む道具にも無い。他の後付けの属性（`speed`・
        # `careful_model`）と同じく `getattr` で受ける。
        fallback = getattr(self, "fallback_engine", "")
        if not result.startswith("Said:") and fallback and fallback != self.engine:
            logger.warning(
                "%s で話せなかったので控え（%s）へ落ちる：%s", self.engine, fallback, result
            )
            spare = await self._say_with(fallback, text, output, careful=careful)
            if spare.startswith("Said:"):
                return spare
        return result

    async def _say_with(self, engine: str, text: str, output: str, *, careful: bool) -> str:
        """担い手を 1 つ名指しで合成する。主にも控えにも同じ口を使う。"""
        if engine == "sbv2":
            return await self._say_sbv2(text, output)
        if engine == "gemini":
            return await self._say_gemini(text, output)
        return await self._say_elevenlabs(text, output, careful=careful)

    async def _say_gemini(self, text: str, output: str) -> str:
        """Gemini TTS で合成して鳴らす（環-v・2026-09-24）。

        **逐次で受け取る。** 最初の音までが 0.98 秒、音が全部そろうまでは長文で 3.9 秒で、
        待ちを決めるのは前者である（`根拠台帳` §45）。いまは全部溜めてから鳴らしているので
        差は出ないが、受け取り方は逐次のままにしておく（鳴らす側を後で直せる）。

        サーバーが返さなくても例外は投げない。話せなかったことだけを返す。
        """
        text = self._clean_for_speech(text)
        if not text:
            return "Said: (nothing to speak after cleaning)"
        if len(text) > 200:
            text = text[:197] + "..."

        async with self._lock:
            voice_guard = self._voice_guard or get_shared_voice_guard()
            self._voice_guard = voice_guard
            played_via: list[str] = []
            tmp_path: str | None = None
            voice_guard.on_tts_start(text)
            try:
                try:
                    audio, is_wav = await asyncio.to_thread(self._synth_gemini, text)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Gemini で合成できなかったので話せなかった: %s", e)
                    return f"話せなかった（合成に失敗）: {text[:40]}"
                if not audio:
                    return f"話せなかった（音が返らなかった）: {text[:40]}"
                tmp_path = (
                    _write_tmp_audio(audio, suffix=".wav")
                    if is_wav
                    else _write_pcm_as_wav(audio, sample_rate=GEMINI_RATE)
                )
                played_via = await self._play_paths(tmp_path, output)
                if not played_via:
                    return "話せなかった（鳴らせる出力が無い）"
                return f"Said: {text[:50]}... (via {', '.join(played_via)})"
            finally:
                voice_guard.on_tts_end(text, played=bool(played_via))
                if tmp_path is not None:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_path)

    def _synth_gemini(self, text: str) -> "tuple[bytes, bool]":
        """逐次に受け取って繋ぐ。返りは（音のバイト列, それが WAV か）。

        **`mime_type` で見分ける。** 逐次は生 PCM、一括は WAV で返る（2026-09-24 実測）。
        包み方を間違えると、雑音になるか無音になる。
        """
        from google.genai import types

        client = _gemini_client(self.gemini_api_key)
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.gemini_voice)
                )
            ),
        )
        started = time.monotonic()
        first: float | None = None
        buf: list[bytes] = []
        is_wav = False
        for chunk in client.models.generate_content_stream(
            model=self.gemini_model, contents=self._text_for_synth(text), config=config
        ):
            for part in (chunk.candidates[0].content.parts or []) if chunk.candidates else []:
                blob = getattr(part, "inline_data", None)
                if blob is None or not blob.data:
                    continue
                if first is None:
                    first = time.monotonic() - started
                is_wav = is_wav or "wav" in str(blob.mime_type or "")
                buf.append(blob.data)
        logger.info(
            "TTS: Gemini で合成した（%s・%s・最初の音 %.2f 秒・全部 %.2f 秒・%d 字）",
            self.gemini_model,
            self.gemini_voice,
            first if first is not None else float("nan"),
            time.monotonic() - started,
            len(text),
        )
        return b"".join(buf), is_wav

    async def _say_sbv2(self, text: str, output: str) -> str:
        """ローカルの Style-Bert-VITS2 で合成して鳴らす（出-a）。

        サーバーが落ちていても例外は投げない。話せなかったことだけを返す（機器は落ちる
        前提のもので、発話の失敗でターンごと壊すわけにはいかない）。
        """
        text = self._clean_for_speech(text)
        if not text:
            return "Said: (nothing to speak after cleaning)"
        if len(text) > 200:
            text = text[:197] + "..."

        async with self._lock:
            voice_guard = self._voice_guard or get_shared_voice_guard()
            self._voice_guard = voice_guard
            played_via: list[str] = []
            tmp_path: str | None = None
            voice_guard.on_tts_start(text)
            try:
                try:
                    wav = await self._synth_sbv2(text)
                except Exception as e:  # noqa: BLE001
                    logger.warning("SBV2 で合成できなかったので話せなかった: %s", e)
                    return f"話せなかった（合成に失敗）: {text[:40]}"
                tmp_path = _write_tmp_audio(wav, suffix=".wav")
                played_via = await self._play_paths(tmp_path, output)
                if not played_via:
                    return "話せなかった（鳴らせる出力が無い）"
                return f"Said: {text[:50]}... (via {', '.join(played_via)})"
            finally:
                voice_guard.on_tts_end(text, played=bool(played_via))
                if tmp_path is not None:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_path)

    async def _synth_sbv2(self, text: str) -> bytes:
        """合成サーバーへ投げて WAV を受け取る。"""
        import aiohttp

        payload = {"text": text, "style": self.sbv2_style, "weight": self.sbv2_weight}
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.sbv2_url}/synth", json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise OSError(f"SBV2 が {resp.status} を返した: {body[:80]}")
                return await resp.read()

    async def _play_paths(self, tmp_path: str, output: str) -> list[str]:
        """出来た音声ファイルを、指定された出力へ鳴らす。鳴った先の名前を返す。"""
        played_via: list[str] = []
        if output in ("remote", "both"):
            ok, msg = await asyncio.to_thread(
                _play_via_go2rtc, tmp_path, self.go2rtc_url, self.go2rtc_stream
            )
            if ok:
                played_via.append("camera")
            else:
                logger.warning("go2rtc playback failed: %s", msg)
        if output in ("local", "both") or (output == "remote" and not played_via):
            if await _play_local(tmp_path, gain=getattr(self, "_gain", 1.0)):
                played_via.append("local")
        return played_via

    async def _say_elevenlabs(self, text: str, output: str, *, careful: bool = False) -> str:
        """外部 API（ElevenLabs）で合成して鳴らす。SBV2 が動かないときの逃げ道。"""
        if not self.api_key:
            return f"(silent) {text}"

        import aiohttp

        # 丸括弧のト書き（'（静かに待つ）'）は落とし、[audio tags] は残す——eleven_v3 は
        # それを話し方の指示として解する。
        text = self._clean_for_speech(text)
        if not text:
            return "Said: (nothing to speak after cleaning)"
        if len(text) > 200:
            text = text[:197] + "..."

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}?output_format=pcm_16000"
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        payload = self._elevenlabs_payload(text, careful=careful)

        async with self._lock:
            voice_guard = getattr(self, "_voice_guard", None)
            if voice_guard is None:
                voice_guard = get_shared_voice_guard()
                self._voice_guard = voice_guard
            played_via: list[str] = []
            tmp_path: str | None = None
            voice_guard.on_tts_start(text)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status != 200:
                            err = await resp.text()
                            return f"TTS API failed ({resp.status}): {err[:80]}"
                        content_type = resp.headers.get("Content-Type", "")
                        audio_data = await resp.read()

                # ElevenLabs may return MP3 even when PCM was requested (model-dependent).
                # Detect by content-type and save to the correct format.
                is_mp3 = "mpeg" in content_type or audio_data[:3] in (
                    b"ID3",
                    b"\xff\xfb",
                    b"\xff\xf3",
                )
                if is_mp3:
                    tmp_path = _write_tmp_audio(audio_data, suffix=".mp3")
                else:
                    tmp_path = _write_pcm_as_wav(audio_data, sample_rate=16000)

                if output in ("remote", "both"):
                    ok, msg = await asyncio.to_thread(
                        _play_via_go2rtc, tmp_path, self.go2rtc_url, self.go2rtc_stream
                    )
                    if ok:
                        played_via.append("camera")
                    else:
                        logger.warning("go2rtc playback failed: %s", msg)
                        if output == "remote":
                            return f"TTS remote playback failed: {msg}"

                if output in ("local", "both") or (output == "remote" and not played_via):
                    local_ok = await _play_local(tmp_path, gain=getattr(self, "_gain", 1.0))
                    if local_ok:
                        played_via.append("local")

                if not played_via:
                    return "TTS playback failed (no working audio player found)"
                return f"Said: {text[:50]}... (via {', '.join(played_via)})"
            finally:
                voice_guard.on_tts_end(text, played=bool(played_via))
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "say",
                "description": (
                    "Speak text aloud. Use this to communicate with people in the room."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": (
                                "Text to speak. Can include ElevenLabs audio tags "
                                "like [cheerful], [warmly]."
                                if self.understands_tags
                                else "Text to speak. Plain text only — bracket tags "
                                "are stripped and have no effect."
                            ),
                        },
                        # 写真に誰が写っていたかの見立て（出-an・2026-09-22）。**口に出すなら
                        # 機械にも渡す**——推し量ること自体は禁じないが、推し量ったなら在席にも
                        # 使う。実機 15:57 の写真で測ると、欄が無ければ 6 回とも「パパ」と呼んで
                        # 機械へは何も渡らず、欄を足せば 6 回とも書く。確信度は受け側で 0.6 を
                        # 上限に掛ける（調停の見立てと同じ扱い・`core/seen_people`）。
                        "seen_people": {
                            "type": "array",
                            "description": (
                                "いま届いた写真に人が写っているなら、誰だと思うかを一人ずつ書く。"
                                "見た目と【一緒に暮らす人たち】の記述から推し量ってよい。"
                                "誰か分からない人は name を空にして、数に入れる。"
                                "写真が届いていない反復では書かない。"
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "家族の呼び方（分からなければ空）",
                                    },
                                    "confidence": {"type": "number", "description": "0.0〜1.0"},
                                },
                                "required": ["name", "confidence"],
                            },
                        },
                        # 想起した記憶をどう扱ったかの申告（課題5 E節 段2）。参照した MI だけ
                        # 再評価する、という設計の更新契機がこれ。W に出した id で指す。
                        "memory_verdicts": {
                            "type": "array",
                            # **4つの判定に、別々の引き金を与える**（出-h-い）。1文だけの
                            # 説明では、24回中9回で件数が欠け、判定は無難な `referred` へ
                            # 倒れていた（並んだ記憶が一斉に若返る形）。条件で分けると
                            # 欠落は 0 になり、`important` も出るようになった。
                            "description": (
                                "How each memory in the workspace was used. One entry for "
                                "every id listed there — leave none out. Copy each id exactly "
                                "as printed. Choose by these tests:\n"
                                "- `important`: you drew on it in your reply AND it matters "
                                "beyond this turn (the person asked about it, or it is "
                                "something you want to keep knowing about them).\n"
                                "- `referred`: you drew on it in your reply, but only for "
                                "this turn.\n"
                                "- `useless`: you looked at it and it was not worth "
                                "recalling here.\n"
                                "- `unused`: you did not draw on it at all. This is the "
                                "plain answer for memories your reply never touched — "
                                "most entries will be this."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    # **W の id を `enum` に入れない。** 道具の定義は安定部と
                                    # 同じキャッシュ範囲にあり（出-i）、想起のたびに変われば
                                    # 毎ターン書き直しになる（1000ターン 366円 → 738円）。
                                    # 実測では `enum` の有無で申告の成績は変わらなかった。
                                    "id": {"type": "string"},
                                    "verdict": {
                                        "type": "string",
                                        # 英語で返させる（判定語の精度が上がる）。
                                        # important=大事／useless=不要／referred=参照
                                        # ／unused=使わなかった
                                        "enum": ["important", "useless", "referred", "unused"],
                                    },
                                },
                                "required": ["id", "verdict"],
                            },
                        },
                    },
                    # **申告を必須にする**（出-h-い）。任意のままだと、規則側に日本語で
                    # 書いてあっても出ないことがある。記憶が育つ経路は申告1本しかない。
                    "required": ["text", "memory_verdicts"],
                },
            },
            FILLER_TOOL,
        ]

    async def call(self, tool_name: str, tool_input: dict) -> tuple[str, None]:
        # つなぎも声に出すことは同じ（出-aq 段 2）。ループは `filler` を横取りして背景で
        # 鳴らすが、道具として直接呼ばれても鳴らせるようにしておく。
        if tool_name in ("say", "filler"):
            result = await self.say(
                tool_input["text"],
                gain=float(tool_input.get("gain", 1.0) or 1.0),
                careful=bool(tool_input.get("careful", False)),
            )
            return result, None
        return f"Unknown tool: {tool_name}", None


def _pulse_env() -> dict[str, str] | None:
    """Build env dict with PULSE_SERVER/PULSE_SINK if set. Returns None if neither is set."""
    server = os.environ.get("PULSE_SERVER")
    sink = os.environ.get("PULSE_SINK")
    if not server and not sink:
        return None
    env = os.environ.copy()
    if server:
        env["PULSE_SERVER"] = server
    if sink:
        env["PULSE_SINK"] = sink
    return env


def _resolve_output_device() -> int | None:
    """Return sounddevice output device index from AUDIO_OUTPUT_DEVICE or AUDIO_INPUT_DEVICE env vars.

    Prefers AUDIO_OUTPUT_DEVICE; falls back to AUDIO_INPUT_DEVICE (both typically refer
    to the same USB speaker/mic combo like Yamaha YVC-300). Returns None to use the default.
    """
    name = (
        os.environ.get("AUDIO_OUTPUT_DEVICE", "").strip()
        or os.environ.get("AUDIO_INPUT_DEVICE", "").strip()
    )
    if not name:
        return None
    try:
        import sounddevice as sd

        for i, d in enumerate(sd.query_devices()):
            if name.lower() in d["name"].lower() and d["max_output_channels"] > 0:
                return i
    except Exception:
        pass
    return None


#: 再生はプロセス内で 1 本ずつ（出-ad・2026-09-19）。タイマーの音（1 秒の wav）と知らせの声（mp3）が同じ
#: 機器 `hw:1,0`（排他）を別スレッドから同時に開き、PortAudio（ALSA）が `Device unavailable` のあと
#: `double free or corruption` でプロセスごと落ちた（実機 11:39）。鍵を取ってから開く。音の 1 秒 → 声 →
#: 音の次の 1 秒、と交互になる。
_PLAYBACK_LOCK = threading.Lock()


async def _play_via_sounddevice(audio_path: str, gain: float = 1.0) -> bool:
    """Play WAV or MP3 file using sounddevice (pure Python, no system dependency).

    WAV: decoded by soundfile directly, resampled to the output device's native rate
    so we avoid PortAudio paInvalidSampleRate (e.g. Yamaha requires 48 kHz).
    MP3: decoded frame-by-frame with PyAV (av package), then played via sounddevice.
    再生は `_PLAYBACK_LOCK` で 1 本ずつ（同じ機器を同時に開かない）。
    """

    def _play() -> bool:
        with _PLAYBACK_LOCK:
            return _play_unlocked()

    def _play_unlocked() -> bool:
        if audio_path.lower().endswith(".mp3"):
            # On Windows prefer MCI (reliable, built-in) over PyAV+sounddevice
            if sys.platform == "win32" and _play_mp3_mci(audio_path):
                return True
            return _play_mp3_via_pyav(audio_path, gain)
        else:
            try:
                import numpy as np
                import sounddevice as sd
                import soundfile as sf
            except ImportError:
                return False
            try:
                data, samplerate = sf.read(audio_path)
                device_idx = _resolve_output_device()
                # Determine device's native sample rate; resample if needed.
                try:
                    dev_info = sd.query_devices(device_idx, kind="output")
                    native_rate = int(dev_info["default_samplerate"])
                except Exception:
                    native_rate = samplerate
                if native_rate != samplerate and native_rate > 0:
                    ratio = native_rate // samplerate
                    if ratio > 0:
                        data = (
                            np.repeat(data, ratio, axis=0)
                            if data.ndim > 1
                            else np.repeat(data, ratio)
                        )
                    play_rate = native_rate
                else:
                    play_rate = samplerate
                if gain != 1.0:
                    # タイマーの声だけ大きく（`TIMER_VOICE_GAIN`）。範囲を超えた分は飽和させる。
                    data = np.clip(data * gain, -1.0, 1.0)
                sd.play(data, play_rate, device=device_idx)
                sd.wait()
                return True
            except Exception as e:
                logger.warning("sounddevice/soundfile WAV playback failed: %s", e)
                return False

    return await asyncio.to_thread(_play)


def _play_mp3_mci(mp3_path: str) -> bool:
    """Play MP3 using Windows MCI (Media Control Interface) via ctypes. Windows only.

    MCI is built into Windows — no extra dependencies, supports MP3 natively.
    """
    try:
        import ctypes

        winmm = ctypes.windll.winmm  # type: ignore[attr-defined]
        alias = "familiar_tts"
        abs_path = os.path.abspath(mp3_path)
        winmm.mciSendStringW(f"close {alias}", None, 0, None)  # clean up any prior
        ret = winmm.mciSendStringW(f'open "{abs_path}" type mpegvideo alias {alias}', None, 0, None)
        if ret != 0:
            logger.warning("MCI open failed (ret=%d)", ret)
            return False
        winmm.mciSendStringW(f"play {alias} wait", None, 0, None)
        winmm.mciSendStringW(f"close {alias}", None, 0, None)
        return True
    except Exception as e:
        logger.warning("MCI playback failed: %s", e)
        return False


def _play_mp3_via_pyav(mp3_path: str, gain: float = 1.0) -> bool:
    """Decode MP3 with PyAV to s16 PCM, play via sounddevice.

    Uses s16 interleaved stereo (simpler than fltp planar) for cross-platform reliability.
    """
    try:
        import av
        import numpy as np
        import sounddevice as sd
    except ImportError:
        logger.warning("PyAV, numpy, or sounddevice not available for MP3 decoding")
        return False
    try:
        TARGET_RATE = 44100
        container = av.open(mp3_path)
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            container.close()
            return False

        # Resample to s16p mono @ TARGET_RATE — planar mono avoids stereo packing ambiguity
        resampler = av.AudioResampler(format="s16p", layout="mono", rate=TARGET_RATE)
        chunks_nd: list = []

        for frame in container.decode(audio_stream):
            if not isinstance(frame, av.AudioFrame):
                continue
            for rf in resampler.resample(frame):
                chunks_nd.append(rf.to_ndarray())  # shape: (1, n_samples)

        # Flush resampler
        for rf in resampler.resample(None):
            chunks_nd.append(rf.to_ndarray())

        container.close()
        if not chunks_nd:
            return False

        # Concatenate along samples axis → (1, total_samples) → flatten to (total_samples,)
        audio = np.concatenate(chunks_nd, axis=1).flatten().astype(np.float32) / 32768.0
        if gain != 1.0:
            audio = np.clip(audio * gain, -1.0, 1.0)
        # 出力機器は wav と同じ口で選ぶ（機器指定が無く、既定の出力＝PC のスピーカーへ出ていた・2026-09-19）
        sd.play(audio, TARGET_RATE, device=_resolve_output_device())
        sd.wait()
        return True
    except Exception as e:
        logger.warning("PyAV MP3 playback failed: %s", e)
        return False


async def _play_local(tmp_path: str, gain: float = 1.0) -> bool:
    """Play audio file on the local PC speaker. Returns True on success.

    Try order:
    1. afplay (macOS built-in)
    2. paplay (PulseAudio native — most reliable on WSL2/WSLg when PULSE_SERVER is set)
    3. mpv (auto audio backend selection)
    4. sounddevice (pure Python fallback, no system tools required)

    Note: file is always WAV (pcm_16000) so paplay needs no ffmpeg conversion.
    """
    pulse_env = _pulse_env()

    # --- afplay (macOS built-in) ---
    if sys.platform == "darwin":
        afplay = shutil.which("afplay")
        if afplay:
            try:
                proc = await asyncio.create_subprocess_exec(
                    afplay,
                    tmp_path,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()
                if proc.returncode == 0:
                    return True
                logger.warning(
                    "afplay failed (exit %d): %s",
                    proc.returncode,
                    stderr.decode(errors="replace")[:120],
                )
            except (FileNotFoundError, OSError) as e:
                logger.warning("Could not launch afplay: %s", e)

    # --- sounddevice with named output device (Linux: avoids ALSA hw exclusive-access
    #     conflict when mic is simultaneously open via PortAudio on the same device) ---
    if sys.platform != "darwin" and os.environ.get("AUDIO_INPUT_DEVICE", "").strip():
        sd_ok = await _play_via_sounddevice(tmp_path, gain=gain)
        if sd_ok:
            return True

    # --- paplay (PulseAudio native, WAV only) ---
    paplay = shutil.which("paplay")
    if paplay and tmp_path.lower().endswith((".wav", ".wave")):
        try:
            proc = await asyncio.create_subprocess_exec(
                paplay,
                tmp_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                env=pulse_env,
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                return True
            err = stderr.decode(errors="replace").strip()
            logger.warning("paplay failed (exit %d): %s", proc.returncode, err[:120])
        except (FileNotFoundError, OSError) as e:
            logger.warning("Could not launch paplay: %s", e)

    # --- mpv ---
    mpv = shutil.which("mpv")
    if mpv:
        try:
            proc = await asyncio.create_subprocess_exec(
                mpv,
                "--no-terminal",
                tmp_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                env=pulse_env,
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                return True
            logger.warning(
                "mpv failed (exit %d): %s", proc.returncode, stderr.decode(errors="replace")[:120]
            )
        except (FileNotFoundError, OSError) as e:
            logger.warning("Could not launch mpv: %s", e)

    # --- sounddevice (pure Python, no system dependency) ---
    return await _play_via_sounddevice(tmp_path, gain=gain)


def _play_via_go2rtc(file_path: str, go2rtc_url: str, stream_name: str) -> tuple[bool, str]:
    """Play audio file through camera speaker via go2rtc backchannel (sync, run in thread)."""
    try:
        abs_path = os.path.abspath(file_path)
        src = f"ffmpeg:{abs_path}#audio=pcma#input=file"
        url = (
            f"{go2rtc_url}/api/streams?dst={quote(stream_name, safe='')}&src={quote(src, safe='')}"
        )
        req = urllib.request.Request(url, method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())

        # Check if a sender was established (camera supports backchannel)
        has_sender = any(consumer.get("senders") for consumer in body.get("consumers", []))
        if not has_sender:
            return False, "go2rtc: no audio sender (camera may not support backchannel)"

        # Find ffmpeg producer ID to poll for completion
        ffmpeg_producer_id = None
        for p in body.get("producers", []):
            if "ffmpeg" in p.get("source", ""):
                ffmpeg_producer_id = p.get("id")
                break

        if ffmpeg_producer_id:
            import time

            for _ in range(60):
                time.sleep(0.5)
                try:
                    with urllib.request.urlopen(f"{go2rtc_url}/api/streams", timeout=5) as r:
                        streams = json.loads(r.read())
                    stream = streams.get(stream_name, {})
                    still_playing = any(
                        p.get("id") == ffmpeg_producer_id for p in stream.get("producers", [])
                    )
                    if not still_playing:
                        break
                except Exception:
                    break

        return True, f"played via go2rtc → {stream_name}"
    except Exception as exc:
        return False, f"go2rtc error: {exc}"
