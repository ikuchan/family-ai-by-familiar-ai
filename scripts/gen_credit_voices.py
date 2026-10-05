#!/usr/bin/env python
"""残高切れを知らせる録音 `src/familiar_agent/sounds/credit_<担い手>.wav` を作る（環-z・2026-10-05）。

LLM も TTS も切れているかもしれないときに鳴らすので、前もって録っておく（`設計方針_クレジット切れの知らせ`）。
いまの声（Style-Bert-VITS2・`.env` の `SBV2_*`）で「〇〇のクレジットが足りなくなりました。」を担い手ごとに読み上げる。
合成サーバーが動いていなければ起こし（アプリと同じ口 `tools/tts._spawn_sbv2`）、作り終えたら止める。

使い方：`uv run python scripts/gen_credit_voices.py`
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, "src")

# `.env` を自前で読む（この道具は familiar の起動経路を通らない）。
for _line in pathlib.Path(".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

import json  # noqa: E402

from familiar_agent.config import TTSConfig  # noqa: E402
from familiar_agent.core.credit import READINGS, sentence  # noqa: E402
from familiar_agent.tools import tts  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "src" / "familiar_agent" / "sounds"


def _synth(cfg: TTSConfig, text: str) -> bytes:
    payload = json.dumps(
        {"text": text, "style": cfg.sbv2_style, "weight": cfg.sbv2_weight}
    ).encode()
    req = urllib.request.Request(
        f"{cfg.sbv2_url}/synth", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main() -> int:
    cfg = TTSConfig()
    started_here = False
    if not tts._sbv2_is_alive(cfg.sbv2_url):
        tts._spawn_sbv2(cfg)
        started_here = True
        for _ in range(120):
            if tts._sbv2_is_alive(cfg.sbv2_url):
                break
            time.sleep(1)
        else:
            print("合成サーバーが 120 秒で起きなかった。")
            return 1
    try:
        for name in READINGS:
            text = sentence(name)
            wav = _synth(cfg, text)
            path = OUT / f"credit_{name}.wav"
            path.write_bytes(wav)
            print(f"{path.name}：{text}（{len(wav)} バイト）")
    finally:
        if started_here and tts._sbv2_proc is not None:
            tts._sbv2_proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
