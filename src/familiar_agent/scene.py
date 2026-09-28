"""World model — persistent scene entity tracking and change detection.

Phase 1 of the familiar-ai roadmap.

Architecture:
- extract_entities(): calls the utility backend to parse a scene description
  into a structured list of {label, category, confidence} dicts.
- read_photo(): reads a photo into what is seen and who is in it (出-au 段 5-7a).

（場面の実体を DB に追う仕組みは、作るだけで読まれなかったので環-ab で外した。）
"""

from __future__ import annotations

import logging
from typing import Any

from .core.structured_ask import read_json


logger = logging.getLogger(__name__)

#: 実体の一覧は短い JSON なので、これで足りる（`ask_json` の既定と揃える）。
_EXTRACT_MAX_TOKENS = 512

_EXTRACT_SYSTEM = """\
You are a scene-analysis assistant. Given a description of what an AI agent sees,
extract the distinct entities (people, objects, locations/features) and return them
as JSON with the key "entities".

Each entity must have:
  - "label": short singular noun ("person", "chair", "window")
  - "category": one of "person", "object", "location"
  - "confidence": float 0.0–1.0

Return ONLY the JSON object. Example:
{
  "entities": [
    {"label": "chair", "category": "object", "confidence": 0.9},
    {"label": "person", "category": "person", "confidence": 0.8}
  ]
}
"""


_READ_PHOTO_PROMPT = """\
この写真を見て、写っているものと、写っている人を書く。

[一緒に暮らす人たち（見た目の手がかり）]
{family}

- entities：写っているもの。一つずつ {{"label": "日本語の名前"}}。
- people：写っている人。一人ずつ {{"name": "呼び方", "confidence": 0.0〜1.0}}。見た目と上の記述から誰だと思うかを
  推し量ってよい。**誰か分からない人は name を空にして、数に入れる。** 人が写っていなければ空の並び。

次の形の JSON だけを返す（他には何も書かない）:
{{"entities": [], "people": []}}
"""


async def read_photo(
    image_b64: str, backend: Any, *, family_md: str
) -> "tuple[list[str], list[dict]]":
    """写真を読み、(見えたもののラベル, 写っている人の見立て) を返す（出-au 段 5-7a・`設計方針_判定の段` v0.4 §2.2.2）。

    Jev は写真を見られないので、読み取りの結果をシステムの状態（O の記録と在席）として残し、判定はその文を読む。
    読めない・失敗なら ([], [])。
    """
    if not image_b64 or not hasattr(backend, "complete_with_image"):
        return [], []
    try:
        raw = await backend.complete_with_image(
            _READ_PHOTO_PROMPT.format(family=family_md or "（記述なし）"), image_b64
        )
    except Exception as exc:  # noqa: BLE001
        _log_unusable("", str(exc))
        return [], []
    data = read_json(str(raw or ""))
    if not isinstance(data, dict):
        _log_unusable(str(raw or ""), "JSON として読めない")
        return [], []
    labels = [
        str(e.get("label", "")).strip()
        for e in data.get("entities") or []
        if isinstance(e, dict) and str(e.get("label", "")).strip()
    ]
    people = [
        {"name": str(p.get("name", "") or ""), "confidence": float(p.get("confidence", 0.0) or 0.0)}
        for p in data.get("people") or []
        if isinstance(p, dict)
    ]
    return labels, people


async def extract_entities(
    description: str, backend: Any, image_b64: str | None = None
) -> list[dict]:
    """Call the utility backend to extract structured entities from a scene description.

    When image_b64 is provided and the backend supports complete_with_image(),
    the raw camera image is sent directly to the backend (useful for local VLMs via
    Ollama) and description is used only as fallback.

    Returns a list of dicts with keys: label, category, confidence.
    Returns [] on any parse error.
    """
    text_prompt = f"{_EXTRACT_SYSTEM}\n\nScene description:\n{description}"
    data = None
    if image_b64 and hasattr(backend, "complete_with_image"):
        vision_prompt = (
            f"{_EXTRACT_SYSTEM}\n\n"
            "Analyze this camera image directly and extract all visible entities."
        )
        # 画像の経路は口を通さない（`complete_with_image` は口の対象外）。返答が素の
        # JSON とは限らないのは同じなので、コードフェンスは剥がす。
        try:
            raw = await backend.complete_with_image(vision_prompt, image_b64)
        except Exception as exc:  # noqa: BLE001
            _log_unusable("", str(exc))
            raw = ""
        if raw:
            data = read_json(str(raw))
            if data is None:
                _log_unusable(raw, "JSON として読めない")
    if data is None:
        # **ここは記録を自分で持つ**（出-d）。「空だったのか説明文だったのか」で対応が
        # 変わるので、先頭を添えて残す（実機で `see` が失敗し続けたとき、VLM が何を返した
        # のか分からなかった）。読み取り自体は共通の `read_json` に任せる。
        try:
            # `max_tokens` は必須（渡していなかったので、この経路は Anthropic の
            # バックエンドで必ず例外になっていた・2026-09-04 に実測で見つけた）。
            raw_text = str(await backend.complete(text_prompt, _EXTRACT_MAX_TOKENS) or "")
        except Exception as exc:  # noqa: BLE001
            _log_unusable("", str(exc))
            return []
        data = read_json(raw_text)
        if data is None:
            _log_unusable(raw_text, "JSON として読めない")
            return []
    entities = data.get("entities", [])
    if not isinstance(entities, list):
        _log_unusable(str(data), "entities が配列でない")
        return []
    return [e for e in entities if isinstance(e, dict) and "label" in e]


# 記録に載せる返答の長さ。情景の説明は長く、全文を warning で出すとログが埋まる。
_REPLY_HEAD_CHARS = 200


def _log_unusable(raw: Any, reason: str) -> None:
    """意味づけに使えない返答を、先頭を添えて残す。

    実機で `see` が「意味づけは何も返さなかった」を出し続けたとき、失敗が debug で
    例外の型しか出ておらず、**VLM が何を返したのかが分からなかった**。空文字なのか、
    説明文なのか、JSON もどきなのかで対応が変わる。
    """
    text = "" if raw is None else str(raw)
    head = text[:_REPLY_HEAD_CHARS] if text.strip() else "（空）"
    logger.warning("情景の意味づけに使えない返答（%s）：%s", reason, head)
