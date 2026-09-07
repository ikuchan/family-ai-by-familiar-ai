"""到達不能だった2つの経路を撤去した。

**brief-turn（軽量返信モード）** — `core/brief_turn.py` の3つの関数は `agent.py` が
staticmethod として束ねていたが、**呼ぶ箇所が src にも tests にも0件**だった。型注釈も
撤去済みの `social_policy` module を指していた（`test_no_phantom_imports.py`）。

**遅延配信のポーリング** — `should_deliver_deferred_result()` は、CUI・GUI・TUI が毎周回
問い合わせていたゲートである。#12a でその3つのポーリングを撤去し、完了は
**完了キュー→O→次反復**へ移した（`test_deferred_to_completion_queue.py`）。ゲート自体は
`InformationProcessing._delivery_block_reason()` が引き継いでいる（在席・静穏時間・
「黙っていて」の依頼）。関数だけが残り、`_last_social_decision` を代入する箇所も0件で、
その分岐には到達しなかった。

撤去の証明は数え上げでなく**旧名で引いて0件**である。
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).parent.parent
_SRC = _ROOT / "src/familiar_agent"


def _hits(name: str) -> list[str]:
    out: list[str] = []
    for f in sorted(_SRC.rglob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if name in line:
                out.append(f"{f.relative_to(_SRC)}:{i}: {line.strip()}")
    return out


def test_the_brief_turn_module_is_gone():
    assert not (_SRC / "core/brief_turn.py").exists()


def test_no_brief_turn_name_survives():
    for name in (
        "brief_turn",
        "brief_reply",
        "is_candidate_brief_turn",
        "_BRIEF_REPLY_MAX_ITERATIONS",
        "_BRIEF_REPLY_MAX_TOKENS",
        "_BRIEF_REPLY_TOOL_NAMES",
    ):
        assert _hits(name) == [], f"{name} が残っている"


def test_no_deferred_polling_name_survives():
    for name in ("should_deliver_deferred_result", "_last_social_decision"):
        assert _hits(name) == [], f"{name} が残っている"


def test_the_gate_it_carried_lives_on_in_the_loop():
    """撤去で在席・静穏時間の判定まで失っていないこと。"""
    from familiar_agent.loop.event_loop import InformationProcessing

    assert hasattr(InformationProcessing, "_delivery_block_reason")
