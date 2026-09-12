"""主LLM の生成時間を、モデル・写真の有無・思考の深さで比べる（出-k-い）。

実機と同じ system 文（静的核＋自己認識＋家族＋規則）・同じ道具（`say`）・同じ問い
（「何が見えますか？」）を、条件ごとに N 回投げ、**1 回の生成秒数・返事の字数・出力トークン**を
表にする。本番 DB は読まない（自己認識は `capability_state` の要約か `ME.md`、家族は `FAMILY.md`）。

使い方:
  uv run python scripts/measure_main_llm.py                      # 既定の2モデル・各条件 3 回
  uv run python scripts/measure_main_llm.py -n 5 -m claude-haiku-4-5-20251001 -m claude-sonnet-4-5
  uv run python scripts/measure_main_llm.py --no-photo           # 写真なしだけ

費用の目安：写真つき 1 回 ≈ 1,000 トークン入力。Haiku で 0.2 円、Sonnet で 0.6 円。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_QUESTION = "何が見えますか？"
_WORKSPACE = """[いまの作業状態]
[過去の記憶（証拠つき）: conf<0.55 は不確か]:
- 2026-09-12 21:41 id:a1b2c3d4e5f6 (適合度:0.95) conf:0.62 相手が言った: 何が見えますか？
- 2026-09-12 21:42 id:b2c3d4e5f6a7 (適合度:0.90) conf:0.58 わたしが見た: 出入り口を見た。見えたもの：chair、dining table
- 2026-09-12 21:42 id:c3d4e5f6a7b8 (適合度:0.88) conf:0.60 わたしの求め: 「何が見えますか？」と聞かれ、1番：see「目の前を見る」の結果が届いた：（見えたものは、作業状態の『わたしが見た』の行にある）
"""


def _latest_capture() -> Path | None:
    d = Path.home() / ".familiar_ai" / "captures"
    files = sorted(d.glob("*.jpg"), key=lambda p: p.stat().st_mtime) if d.exists() else []
    return files[-1] if files else None


def _system() -> tuple[str, str]:
    from familiar_agent.capability_state import load_summary
    from familiar_agent.loop.prompt import build_event_system_prompt

    me = ""
    try:
        me = load_summary()
    except Exception:  # noqa: BLE001
        me = ""
    me = me or (ROOT / "ME.md").read_text(encoding="utf-8")
    fam = (ROOT / "FAMILY.md").read_text(encoding="utf-8") if (ROOT / "FAMILY.md").exists() else ""
    return build_event_system_prompt(
        self_understanding=me,
        family_md=fam,
        present_ctx='(present :speaker "ゆうすけ")',
        pi_ctx="(inner-state :mood 落ち着いている :drive 普通)",
        workspace_ctx=_WORKSPACE,
        iter_ctx="(iteration :n 2 :max 5 :thinking-round 1)",
    )


def _say_def() -> list[dict]:
    from familiar_agent.tools.tts import TTSTool

    return TTSTool.get_tool_definitions(SimpleNamespace(understands_tags=False))  # type: ignore[arg-type]


async def _one(backend, system, tools, content, effort) -> tuple[float, int, int, str]:
    t = time.perf_counter()
    result, _ = await backend.stream_turn(
        system=system,
        messages=[backend.make_user_message(content)],
        tools=tools,
        max_tokens=1024,
        on_text=None,
        effort=effort,
    )
    sec = time.perf_counter() - t
    say = next((tc for tc in result.tool_calls if tc.name == "say"), None)
    text = str(say.input.get("text", "")) if say else (result.text or "")
    return sec, len(text), int(result.output_tokens or 0), text


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", action="append", dest="models")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--no-photo", action="store_true")
    args = ap.parse_args()
    models = args.models or ["claude-haiku-4-5-20251001", "claude-sonnet-4-5"]

    from familiar_agent.backends import create_backend
    from familiar_agent.backends.shared import _supports_adaptive_thinking
    from familiar_agent.bootstrap import load_app_bootstrap
    from familiar_agent.config import AgentConfig

    load_app_bootstrap()
    system = _system()
    tools = _say_def()
    photo = None if args.no_photo else _latest_capture()
    image_b64 = base64.b64encode(photo.read_bytes()).decode() if photo else None
    print(
        f"写真: {photo or '（なし）'}  system 文: {len(system[0]) + len(system[1])} 字  各 {args.n} 回"
    )
    print()
    print(
        "| モデル | 条件 | effort | 秒（平均／最小／最大） | 返事の字数（平均） | 出力トークン（平均） |"
    )
    print("|---|---|---|---|---|---|")
    samples: list[str] = []
    for model in models:
        os.environ["MODEL"] = model
        os.environ["PLATFORM"] = "anthropic"
        backend = create_backend(AgentConfig())
        conds: list[tuple[str, str | None, object]] = [("文字だけ", None, _QUESTION)]
        if image_b64:
            conds.append(
                (
                    "写真つき",
                    None,
                    [
                        {"type": "text", "text": _QUESTION},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                    ],
                )
            )
            if _supports_adaptive_thinking(model):
                conds.append(("写真つき", "low", conds[-1][2]))
        for label, effort, content in conds:
            secs, chars, toks = [], [], []
            failed = ""
            for _ in range(args.n):
                try:
                    sec, n_chars, n_tok, text = await _one(backend, system, tools, content, effort)
                except Exception as e:  # noqa: BLE001
                    failed = str(e)[:100]
                    break
                secs.append(sec)
                chars.append(n_chars)
                toks.append(n_tok)
                samples.append(
                    f"[{model} {label} effort={effort or '-'}] {sec:.2f}s {n_chars}字: {text[:80]}"
                )
            eff = effort or ("-" if not _supports_adaptive_thinking(model) else "既定(high)")
            if failed or not secs:
                print(f"| {model} | {label} | {eff} | 失敗：{failed} | | |")
                continue
            print(
                f"| {model} | {label} | {eff} | {statistics.mean(secs):.2f}／{min(secs):.2f}／{max(secs):.2f} "
                f"| {statistics.mean(chars):.0f} | {statistics.mean(toks):.0f} |"
            )
    print()
    print("返事の例:")
    for s in samples:
        print("  " + s)


if __name__ == "__main__":
    asyncio.run(main())
