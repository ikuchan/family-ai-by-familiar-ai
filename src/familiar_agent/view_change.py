"""見えが変わったかの足切りと、前と今の見比べ（知-i 段 2・`設計詳細_活性_O書込_知覚在席` §4）。

足切りは安く、見比べは高い。足切りが拾うのは、人が映っていない（枠が 0 個）・見えが落ち着いている
（移動平均からの隔たりが小さい）・前の見えから遠い・その定点をしばらく見比べていない回だけ。
物が変わったのか明るさだけなのかは、前（左）と今（右）を並べた 1 枚を見せて軽い視覚のモデルに聞く。
"""

from __future__ import annotations

import base64
import io
import json
import logging

from .visual_norm import cosine_distance

logger = logging.getLogger(__name__)

KINDS = ("物", "明るさ", "なし")

_PROMPT = """\
同じカメラで同じ向きから撮った部屋の写真を、左右に並べてある。左が前、右が今。
人のことは考えない。置いてある物が増えた・減った・動いた・替わったかを見る。
明るさ・日の当たり方・照明の入り切り・画面の映りだけの違いは「明るさ」とする。
出力は JSON だけ：{"kind": "物" | "明るさ" | "なし", "what": "物が変わったときだけ、何がどう変わったかを 1 文で"}
"""


def is_candidate(
    *,
    ref: list[float] | None,
    seen: list[float],
    surprise: float | None,
    boxes: int,
    checked_at: float | None,
    now: float,
    threshold: float,
    settled: float,
    cooldown_sec: float,
) -> bool:
    """見比べに回すか。前の見えが無い・枠がある・落ち着いていない・近い・最近見比べた、なら回さない。"""
    if ref is None or boxes > 0 or surprise is None or surprise > settled:
        return False
    if checked_at is not None and now - checked_at < cooldown_sec:
        return False
    return cosine_distance(ref, seen) >= threshold


_LONG_EDGE = 1024  # 並べる前に 1 枚ずつ長い辺をここまで縮める（送る量を抑える）


def side_by_side(before: bytes, after: bytes) -> str:
    """前（左）と今（右）を同じ高さで並べた JPEG を base64 で。重いので `compare` がループの外で呼ぶ。"""
    from PIL import Image

    a = Image.open(io.BytesIO(before)).convert("RGB")
    b = Image.open(io.BytesIO(after)).convert("RGB")
    a.thumbnail((_LONG_EDGE, _LONG_EDGE))
    b.thumbnail((_LONG_EDGE, _LONG_EDGE))
    h = min(a.height, b.height)
    a = a.resize((max(1, a.width * h // a.height), h))
    b = b.resize((max(1, b.width * h // b.height), h))
    out = Image.new("RGB", (a.width + b.width, h))
    out.paste(a, (0, 0))
    out.paste(b, (a.width, 0))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def can_judge(backend) -> bool:
    """画像を見せて聞けるモデルか（CLI などの後ろ盾は画像を受けない）。"""
    return callable(getattr(backend, "complete_with_image", None))


async def compare(backend, before: bytes, after: bytes) -> tuple[str, str] | None:
    """前と今の画像（JPEG のバイト列）を並べて見比べる。並べられない（壊れた画像）ときも None。"""
    import asyncio

    try:
        image_b64 = await asyncio.to_thread(side_by_side, before, after)
    except Exception:  # noqa: BLE001
        logger.warning("見比べる画像を並べられなかった", exc_info=True)
        return None
    return await judge(backend, image_b64)


async def judge(backend, image_b64: str) -> tuple[str, str] | None:
    """並べた 1 枚を見せて、変わったものの種類と、物なら何が変わったかを返す。読めなければ None。"""
    try:
        raw = str(await backend.complete_with_image(_PROMPT, image_b64, max_tokens=300) or "")
    except Exception:  # noqa: BLE001
        logger.warning("見えの見比べを呼べなかった（前の見えは置き換えない）", exc_info=True)
        return None
    try:
        got = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except ValueError:
        logger.debug("見えの見比べの返りが JSON でない（%d 字）", len(raw))
        return None
    kind = str(got.get("kind", "")).strip()
    if kind not in KINDS:
        return None
    return kind, str(got.get("what", "")).strip() if kind == "物" else ""
