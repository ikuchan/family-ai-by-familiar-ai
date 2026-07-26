"""Capability manifest loader and AI self-understanding storage.

The agent periodically reads capabilities.yaml, asks the LLM to write
a first-person capability summary, and stores it in agent_state.
That summary is injected into the variable system prompt each turn.

Auto-regeneration:
  During ``rest`` desire turns the agent calls ``should_regenerate_manifest()``
  and, if the YAML is older than ``_MANIFEST_MAX_AGE_SECONDS``, spawns
  ``_regenerate_capability_manifest()`` in agent.py to rewrite the file.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import psycopg2.extras

from .db import get_db

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).parent.parent.parent / "capabilities.yaml"
_STATE_KEY = "capability_summary"
_REFRESH_EVERY_N_TURNS = 50
_MANIFEST_MAX_AGE_SECONDS = 86400  # regenerate at most once per day

_SRC = Path(__file__).parent
_ROOT = _SRC.parent.parent

_SECRET_KEYWORDS = frozenset({"API_KEY", "PASSWORD", "SECRET", "TOKEN", "WEBHOOK"})

_KEY_MODULES = [
    "desires.py",
    "relationship.py",
    "appraisal.py",
    "social_policy.py",
    "self_narrative.py",
    "mcp_client.py",
    "interoception.py",
    "default_mode.py",
    "prediction.py",
    "workspace.py",
    "memory_worker.py",
    "meta_monitor.py",
    "tape.py",
    "concern_engine.py",
]


def load_manifest() -> str:
    """Return raw YAML text of capabilities.yaml, or empty string if missing."""
    try:
        return _MANIFEST_PATH.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Could not read capabilities.yaml: %s", e)
        return ""


def filter_enabled(manifest: str, env: dict | None = None) -> str:
    """有効な能力だけを残した manifest を返す。

    `enabled_env: CAMERA_HOST` は「**条件つき**」であって「有効」ではない。環境変数が
    実際に設定されているかを見ないと、繋がっていない身体を能力として語ることになる
    （`ME.md`「カメラ：無い」に対し要約が「I can see ... using a camera」になっていた）。

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
            keep = True                     # 既定は残す（条件の記載が無ければ有効）
        stripped = line.strip()
        if stripped.startswith("enabled:"):
            keep = stripped.split(":", 1)[1].strip().lower() == "true"
        elif stripped.startswith("enabled_env:"):
            keep = bool(environ.get(stripped.split(":", 1)[1].strip()))
        if block or is_item:
            block.append(line)
        else:
            out.append(line)                # 先頭の `capabilities:` など
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


def should_refresh(turn_index: int) -> bool:
    """True on turn 0 (no summary yet) or every N turns thereafter."""
    if turn_index == 0:
        return not bool(load_summary())
    return turn_index % _REFRESH_EVERY_N_TURNS == 0


# ---------------------------------------------------------------------------
# Auto-regeneration helpers (used during rest desire turns)
# ---------------------------------------------------------------------------


def should_regenerate_on_startup() -> bool:
    """Return True when capability_summary is absent.

    Called on turn 0 to decide whether to regenerate the full YAML (and then
    refresh the summary) rather than just refreshing the summary from an
    existing YAML.  Returns False when a summary is already stored so normal
    startup does not re-run the expensive LLM generation step.
    """
    return not bool(load_summary())


def should_regenerate_manifest(max_age_seconds: int = _MANIFEST_MAX_AGE_SECONDS) -> bool:
    """Return True if capabilities.yaml is missing or older than max_age_seconds."""
    if not _MANIFEST_PATH.exists():
        return True
    age = datetime.now().timestamp() - _MANIFEST_PATH.stat().st_mtime
    return age > max_age_seconds


def _module_docstring(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return ast.get_docstring(tree) or ""
    except Exception:
        return ""


def collect_manifest_context() -> str:
    """Collect tool definitions, module docstrings, .env config, and MCP servers."""
    parts: list[str] = []

    # Tool files
    tools_dir = _SRC / "tools"
    tool_lines: list[str] = []
    for f in sorted(tools_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        doc = _module_docstring(f)
        tool_lines.append(f"### tools/{f.name}\n{doc[:350]}" if doc else f"### tools/{f.name}")
    parts.append("## Built-in tools\n" + "\n\n".join(tool_lines))

    # Key modules
    mod_lines: list[str] = []
    for name in _KEY_MODULES:
        f = _SRC / name
        if not f.exists():
            continue
        doc = _module_docstring(f)
        mod_lines.append(f"### {name}\n{doc[:350]}" if doc else f"### {name}")
    parts.append("## Key modules\n" + "\n\n".join(mod_lines))

    # .env (secrets redacted)
    env_file = _ROOT / ".env"
    env_lines: list[str] = []
    if env_file.exists():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                env_lines.append(line)
            elif "=" in line:
                key, _, val = line.partition("=")
                if any(kw in key.upper() for kw in _SECRET_KEYWORDS):
                    env_lines.append(f"{key}=<redacted>")
                else:
                    env_lines.append(f"{key}={val}")
    parts.append("## .env (secrets redacted)\n" + "\n".join(env_lines))

    # MCP config
    mcp_path_str = os.environ.get("MCP_CONFIG", "")
    mcp_path = Path(mcp_path_str) if mcp_path_str else Path.home() / ".familiar-ai.json"
    mcp_lines: list[str] = []
    if mcp_path.exists():
        try:
            data = json.loads(mcp_path.read_text(encoding="utf-8"))
            for name, cfg in data.get("mcpServers", {}).items():
                cmd = cfg.get("command", "")
                args = " ".join(str(a) for a in cfg.get("args", []))
                mcp_lines.append(f"- {name}: {cmd} {args}".strip())
        except Exception:
            pass
    parts.append("## MCP servers\n" + ("\n".join(mcp_lines) if mcp_lines else "(none)"))

    return "\n\n".join(parts)


def build_generation_prompt(context: str, existing_yaml: str) -> str:
    """Return the LLM prompt that generates a fresh capabilities.yaml."""
    existing_section = (
        f"\n\nExisting capabilities.yaml (preserve IDs where applicable):\n{existing_yaml}"
        if existing_yaml
        else ""
    )
    return (
        "Generate a `capabilities.yaml` for the familiar-ai embodied companion agent.\n\n"
        "Format each capability as:\n"
        "  - id: snake_case_id\n"
        "    summary: one-sentence description\n"
        "    detail: >\n"
        "      2-4 sentences. Name key classes, tools, env vars.\n"
        "    enabled: true          # always-on\n"
        "    # OR enabled_env: ENV_VAR\n"
        "    # OR enabled: false\n\n"
        "Rules:\n"
        "- Cover ALL visible capabilities: built-in tools, core modules, MCP servers, hardware.\n"
        "- One entry per MCP server (id = mcp_<name>).\n"
        "- enabled_env for anything requiring a specific env var.\n"
        "- enabled: false only for explicitly unfinished features.\n"
        "- Output ONLY valid YAML starting with 'capabilities:'. No fences. No commentary.\n\n"
        f"{context}"
        f"{existing_section}"
    )


def save_manifest(yaml_content: str) -> None:
    """Write yaml_content to capabilities.yaml, stripping accidental markdown fences."""
    text = yaml_content.strip()
    for fence in ("```yaml", "```yml", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    _MANIFEST_PATH.write_text(text + "\n", encoding="utf-8")
    logger.info("capabilities.yaml regenerated (%d chars)", len(text))
