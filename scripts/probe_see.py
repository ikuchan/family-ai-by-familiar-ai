"""実機のカメラで1枚撮り、意味づけ（VLM）が何を返すかをそのまま見る。

`see` は LLM が選ぶ道具なので、GUI で頼んでも選ばれるとは限らない。ここでは
`see` が通る区間（カメラで撮る → `extract_entities` へ通す）だけを、実コードの
まま直接叩く。

DB には触らない。カメラと VLM だけを使う。

    uv run python scripts/probe_see.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys

from familiar_agent.backend import create_scene_backend, create_utility_backend
from familiar_agent.config import AgentConfig
from familiar_agent.scene import _EXTRACT_SYSTEM, extract_entities
from familiar_agent.tools.camera import CameraTool

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s - %(message)s")


async def main() -> int:
    cfg = AgentConfig()
    cam_cfg = cfg.camera
    if not cam_cfg.host:
        print("カメラの設定が無い（CAMERA_HOST）。")
        return 1

    camera = CameraTool(
        cam_cfg.host, cam_cfg.username, cam_cfg.password, cam_cfg.port,
        preview=False,
        ptz_host=cam_cfg.ptz_host, ptz_username=cam_cfg.ptz_username,
        ptz_password=cam_cfg.ptz_password, ptz_port=cam_cfg.ptz_port,
    )
    backend = create_scene_backend(cfg) or create_utility_backend(cfg)
    print(f"意味づけの担い手 = {type(backend).__name__} "
          f"{getattr(backend, 'model', '')}")

    # 集音と同じで、RTSP は開くまで待つ。`__init__` が撮影スレッドを起こすので
    # （`camera.py:77`）、最初のフレームが届くまで数秒かかる。実機の `see` は
    # 起動から時間が経ってから呼ばれるため、この待ちは実機の条件を再現している。
    print("\n── 撮る（最初のフレームを待つ）──────────────")
    text, image_b64 = "", None
    for i in range(30):
        text, image_b64 = await camera.call("see", {})
        if image_b64:
            print(f"{i + 1} 秒目にフレームが届いた。")
            break
        await asyncio.sleep(1.0)
    print(f"カメラの返答: {text}")
    if not image_b64:
        print("30 秒待ってもフレームが来ない。VLM 以前（カメラか RTSP）の問題である。")
        return 1
    print(f"画像 = {len(image_b64)} 字（base64）")

    print("\n── VLM の生の返答 ──────────────")
    prompt = (f"{_EXTRACT_SYSTEM}\n\n"
              "Analyze this camera image directly and extract all visible entities.")
    raw = await backend.complete_with_image(prompt, image_b64)
    print(repr(raw)[:3000])

    print("\n── 見立て ──────────────")
    s = (raw or "").strip()
    if not s:
        print("空。VLM が何も返していない。")
    elif s.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", s)
        try:
            n = len(json.loads(body).get("entities", []))
            print(f"コードフェンス付きの JSON。剥がせば通る（{n}件）。→ 診断どおり。")
        except Exception as e:  # noqa: BLE001
            print(f"コードフェンス付きだが、剥がしても読めない: {e}")
    elif s.startswith("{"):
        print("素の JSON。実機ではフェンスが付いていない。→ 別の原因を探す必要がある。")
    else:
        print("JSON でもフェンスでもない。説明文かもしれない。→ 修正の中身が変わる。")

    print("\n── いまの extract_entities の結果 ──────────────")
    print(await extract_entities(str(text), backend, image_b64=image_b64))
    # 撮影スレッドを畳んでから戻る。畳まないと解釈器の終了と競って
    # `FATAL: exception not rethrown` が出る。
    camera.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
