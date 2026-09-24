"""能力（REST 内省の層 4）の器——一覧と要約（記-a-と・2026-09-14）。

- **一覧**：`capabilities.yaml`（repo）だけ。**実行時に書き換えない**し、写しも持たない。
  書くのは**機能を作るとき**である（`CLAUDE.md`：ファイルに置くのは既定値と人の入力だけ）。
  機械が書き直す道は 環-y（2026-09-24）で撤去した——動いているアプリがリポジトリを
  書き換えることになり、出来を誰も見ないため（本人の決定）。
- **要約**：`agent_state.capability_summary`。`ME.md`（人が書いた人格）に、実装から導いた
  「できること」を足した一枚で、システム文の `[あなたは誰か]` に載る。

要約の作り直しは `loop/rest_capabilities.py`（REST の 1 パスの最後）が行う。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import psycopg2.extras

from .db import get_db

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).parent.parent.parent / "capabilities.yaml"
_STATE_KEY = "capability_summary"


def load_manifest() -> str:
    """Return raw YAML text of capabilities.yaml, or empty string if missing."""
    try:
        return _MANIFEST_PATH.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Could not read capabilities.yaml: %s", e)
        return ""


#: 道具の日本語の呼び名（出-al・2026-09-22）。**人に言える言い方**で持つ——`get_family_schedule`
#: と言われても相手には伝わらない。一覧の `summary` は英語なので、ここを正本にする。
TOOL_NAMES_JA: "dict[str, str]" = {
    "get_family_schedule": "家族の予定を見る道具（カレンダー）",
    "get_house_rules": "家の決まりを見る道具",
    "search_notion": "家の目次・日次記録を探す道具（Notion）",
    "get_journal": "日ごとの記録を読む道具",
}


def missing_tools(manifest: str, tools: "set[str]") -> "list[tuple[str, str]]":
    """一覧が持っている道具のうち、いま取れないものを（道具名, 日本語の呼び名）で返す。

    繋がっていない道具は候補から黙って消えるだけで、**無いという事実がどこにも残らない**。
    実機 17:50 はそのせいで「調べたけど出てこない」を 3 回繰り返した（出-al）。
    """
    out: list[tuple[str, str]] = []
    for line in manifest.splitlines():
        stripped = line.strip()
        if not stripped.startswith("enabled_tool:"):
            continue
        tool = stripped.split(":", 1)[1].strip()
        if tool and tool not in tools:
            out.append((tool, TOOL_NAMES_JA.get(tool, tool)))
    return out


def live_tool_names(agent) -> "list[str]":
    """いま実際に取れる道具の名前（出-al・2026-09-22）。

    繋がっていなければ空。設定ファイルに書いてあることとは別で、**能力はこちらで数える**。
    """
    mcp = getattr(agent, "_mcp", None)
    if mcp is None:
        return []
    try:
        return [str(d.get("name", "")) for d in mcp.get_tool_definitions() if d.get("name")]
    except Exception:  # noqa: BLE001
        logger.warning("いま取れる道具を数えられなかった", exc_info=True)
        return []


def filter_enabled(manifest: str, env: dict | None = None, tools: "set[str] | None" = None) -> str:
    """有効な能力だけを残した manifest を返す。

    門は 3 つある。

    - `enabled: true` ——いつでも有効。
    - `enabled_env: CAMERA_HOST` ——環境変数があるときだけ。「**条件つき**」であって
      「有効」ではない。見ないと、繋がっていない身体を能力として語ることになる
      （`ME.md`「カメラ：無い」に対し要約が「I can see ... using a camera」になっていた）。
    - `enabled_tool: get_family_schedule` ——**その道具がいま取れるときだけ**（出-al・
      2026-09-22）。MCP 由来の能力にこれを使う。環境変数では実態を表せない——`MCP_CONFIG`
      を設定せず既定パス `~/.familiar-ai.json` を使う機体では、道具が動いていても環境変数が
      無いので一覧から落ちていた。逆に、設定に書いてあってもサーバーが落ちていれば道具は無い。

    `tools` を渡さない呼び方では、道具の門を持つ能力は**残さない**。知らないときに
    「使える」と言うより、落ちて気づくほうがよい。

    yaml を解析せず行単位で扱うのは、`detail: >` の折り返しを保ったまま項目だけを落とす
    ためで、整形し直すと生成側へ渡る文面が変わる。
    """
    import os as _os

    environ = _os.environ if env is None else env
    out: list[str] = []
    block: list[str] = []
    keep = True

    def _flush() -> None:
        if keep:
            out.extend(block)
        block.clear()

    for line in manifest.splitlines(keepends=True):
        is_item = line.lstrip().startswith("- id:")
        if is_item:
            _flush()
            keep = True  # 既定は残す（条件の記載が無ければ有効）
        stripped = line.strip()
        if stripped.startswith("enabled:"):
            keep = stripped.split(":", 1)[1].strip().lower() == "true"
        elif stripped.startswith("enabled_env:"):
            keep = bool(environ.get(stripped.split(":", 1)[1].strip()))
        elif stripped.startswith("enabled_tool:"):
            keep = stripped.split(":", 1)[1].strip() in (tools or set())
        if block or is_item:
            block.append(line)
        else:
            out.append(line)  # 先頭の `capabilities:` など
    _flush()
    return "".join(out)


def build_self_understanding_prompt(*, me_md: str, manifest: str) -> str:
    """自己認識を1枚に組ませるプロンプト（案B）。

    `ME.md` は**逐語で**渡し、変えずに残すよう指示する。要約させると丁寧さの規則のような
    細かい指定が静かに落ち、人が書いた人格が生成物に上書きされる。生成が担うのは
    「できること」の部分だけで、実装が変われば自己認識がそこに追随する。
    """
    return (
        "あなた自身についての説明を1枚にまとめる。出力はその文章だけとし、前置きを書かない。\n\n"
        "次の【私について】は人が書いたあなたの人格である。**一字も変えずそのまま**先頭に写す。\n"
        "言い換え・要約・整形をしない。\n\n"
        "そのうえで、下の能力一覧から、あなたが実際にできることを「## 私にできること」という\n"
        "見出しの節にして続ける。一人称で、10〜20行の箇条書きにする。一覧に無いことは書かない。\n"
        "【私について】に「無い」と書かれている体については、できると書かない。\n"
        "**人に話す言葉で書く。内部の仕組みの名前を使わない**（「予測誤差」「ソーシャルポリシー」\n"
        "「モジュール」のような語は、自分を語る言葉として不自然で、会話へ漏れる）。\n"
        "その仕組みが外から見て何をもたらすかを書く（例：内部で気分を更新する仕組み →\n"
        "「そのときの気分で受け答えが変わる」）。外から見て何も変わらないものは書かない。\n"
        "**「ユーザー」と呼ばない。**あなたは家族と暮らしている。相手は家族であり、利用者ではない。\n"
        "**同じことを書かない。**似た能力はまとめて1行にする。行数より、重ならないことを優先する。\n\n"
        "【私について】\n" + me_md + "\n\n"
        "【能力一覧】\n" + manifest + "\n"
    )


def load_capabilities() -> str:
    """能力の一覧（YAML 文字列）＝ `capabilities.yaml`（環-y・2026-09-24）。

    **置き場は 1 つだけにする。** 以前は DB（`agent_state.capabilities`）を先に引き、
    無ければファイルへ落ちる作りだったが、DB の行は**一度も書かれなかった**——書く口は
    `regenerate_manifest` 1 箇所で、それが一度も走らなかったためである。10 日のあいだ、
    空の器を経由して同じファイルを読んでいた。
    """
    return load_manifest()


def load_summary() -> str:
    """Return the AI-written capability summary from agent_state, or ''."""
    try:
        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT value_json FROM agent_state WHERE state_key = %s",
                    (_STATE_KEY,),
                )
                row = cur.fetchone()
        if row:
            import json

            return str(json.loads(row["value_json"]))
    except Exception as e:
        logger.warning("Could not load capability summary: %s", e)
    return ""


def save_summary(text: str) -> None:
    """Persist the AI-written capability summary to agent_state."""
    try:
        import json

        now = datetime.now(timezone.utc).isoformat()
        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_state (state_key, value_json, updated_at)"
                    " VALUES (%s, %s, %s)"
                    " ON CONFLICT (state_key) DO UPDATE"
                    "   SET value_json = EXCLUDED.value_json,"
                    "       updated_at = EXCLUDED.updated_at",
                    (_STATE_KEY, json.dumps(text), now),
                )
            conn.commit()
    except Exception as e:
        logger.warning("Could not save capability summary: %s", e)
