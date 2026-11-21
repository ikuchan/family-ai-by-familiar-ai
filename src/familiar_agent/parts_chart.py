"""部品どうしのつながりの図を、コードから作る（環-bb・2026-11-21 本人「部品の図も作る」「設計の部品で」）。

部品は `設計図_Mermaid` の全体図の部品（T の TIF・GMD・B、I の AIF・DIF・CORE・RES・W・O・C・SS）と、実物にある内省（REST）・
画面（UI）・起動と基盤（BOOT）。どのファイルがどの部品かは下の対応表で決める（ファイルの名前で決める例外と、フォルダで決める
既定）。矢印は、git に入っているソースの import（関数の中のものも含み、型の注釈のためだけのものは外す）を読み、部品をまたぐものを、import している（ファイル → ファイル）の組の数つきで描く。図は手で書かず、設計図の
文書の「①-2」に、ここが出す文字をそのまま貼る（`tests/test_parts_chart.py` が一致を確かめる）。

使い方：`uv run python -m familiar_agent.parts_chart`
"""

from __future__ import annotations

import ast
import collections
import functools
import subprocess
from pathlib import Path

_SRC = Path(__file__).resolve().parent

#: 部品の名前（日本語：英語（頭字）：働き）と、属する機構（T・I・外）。
PARTS: "dict[str, tuple[str, str]]" = {
    "TIF": ("自律活動IF：Tonic Interface（TIF）：T の出入り口", "T"),
    "GMD": ("外界監視・雰囲気・情動（GMD）：気づき・気分・欠乏", "T"),
    "B": ("揮発記憶：Buffer（B）：直近の状態", "T"),
    "AIF": ("非同期IF：Async Interface（AIF）：T 接続", "I"),
    "DIF": ("機器IF：Device Interface（DIF）：カメラ・スピーカー・マイク・道具", "I"),
    "CORE": ("メイン：Loop Manager（CORE）：反復・調停・生成・決まり", "I"),
    "RES": ("資源ハンドラ（RES）：LLM・Jev・MCP・調べもの", "I"),
    "W": ("作業記憶：Workspace（W）", "I"),
    "O": ("エピソード記憶：Observation（O）：記憶と DB", "I"),
    "C": ("設定：Config（C）", "I"),
    "SS": ("自己状態：Self-state（SS）：パジュの状態図", "I"),
    "REST": ("内省：REST：一人の時間の振り返り", "外"),
    "UI": ("画面：UI", "外"),
    "BOOT": ("起動と基盤：Boot（BOOT）：起動・DB の用意", "外"),
}

#: ファイル（`src/familiar_agent` からの相対・拡張子なし）→ 部品。フォルダの既定より先に見る。
_EXACT = {
    "tif": "TIF",
    "loop/tonic": "TIF",
    "mood_register": "GMD",
    "emotion_pad": "GMD",
    "drive_register": "GMD",
    "occupancy_sensor": "GMD",
    "occupancy_map": "GMD",
    "visual_norm": "GMD",
    "time_decay": "GMD",
    "core/drive_autonomy": "GMD",
    "core/drive_dynamics": "GMD",
    "core/drive_satisfaction": "GMD",
    "core/inner_state": "GMD",
    "core/solitude": "GMD",
    "core/bond_cue": "GMD",
    "core/occupancy_rules": "GMD",
    "core/diffuse": "B",
    "diffuse_store": "B",
    "io/aif": "AIF",
    "io/oif": "O",
    "loop/workspace": "W",
    "core/context_parts": "W",
    "store/context": "W",
    "loop/paju_state": "SS",
    "loop/paju_state_chart": "SS",
    "loop/paju_state_lean": "SS",
    "silence_state": "SS",
    "config": "C",
    "config_overrides": "C",
    "settings_schema": "C",
    "core/settings": "C",
    "env_reload": "C",
    "capability_state": "C",
    "mcp_client": "RES",
    "core/jev_judges": "RES",
    "core/structured_ask": "RES",
    "core/llm_protocol": "RES",
    "core/model_resource": "RES",
    "tools/deep_research": "RES",
    "tools/deferred_fetch": "RES",
    "tools/deferred_search": "RES",
    "tools/page_text": "RES",
    "tools/memory": "O",
    "memory_worker": "O",
    "person_memory_manager": "O",
    "relationship": "O",
    "core/memory_tree": "O",
    "core/mental_item": "O",
    "core/recall_options": "O",
    "gui": "UI",
    "gui_state_chart": "UI",
    "tui": "UI",
    "_ui_helpers": "UI",
    "diagnostics": "UI",
    "_i18n": "UI",
    "main": "BOOT",
    "bootstrap": "BOOT",
    "setup": "BOOT",
    "db": "BOOT",
    "db_migrations": "BOOT",
    "sqlite_migrations": "BOOT",
    "__init__": "BOOT",
    "camera_discovery": "DIF",
    "realtime_stt_session": "DIF",
    "voice_guard": "DIF",
    "scene": "DIF",
    "poses": "DIF",
    "agent": "CORE",
    "routines": "CORE",
    "errors": "CORE",
}
#: フォルダ → 部品（既定）。`loop/rest*` は内省。
_FOLDER = {
    "backends": "RES",
    "recognition": "DIF",
    "store": "O",
    "core": "CORE",
    "loop": "CORE",
    "io": "DIF",
    "tools": "DIF",
}


def part_of(module: str) -> "str | None":
    """ファイル（`loop/event_loop` の形。パッケージは `store/__init__`）の部品。決まらなければ None。"""
    if module in _EXACT:
        return _EXACT[module]
    if module == "loop/rest" or module.startswith("loop/rest_"):
        return "REST"
    if "/" in module:
        return _FOLDER.get(module.split("/")[0])
    return None


#: 図から外すファイル（図を作る道具そのもの）。
_SKIP = {"parts_chart"}


@functools.cache
def _modules() -> "dict[str, Path]":
    """git に入っているソース（手元の試しのファイルで図が変わらない）。git が読めなければ、置いてあるものを読む。"""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "*.py"], cwd=_SRC, capture_output=True, text=True, check=True
        ).stdout.split("\0")
        paths = [_SRC / p for p in listed if p]
    except (OSError, subprocess.CalledProcessError):
        paths = list(_SRC.rglob("*.py"))
    out = {}
    for path in sorted(paths):
        name = path.relative_to(_SRC).with_suffix("").as_posix()
        if name not in _SKIP:
            out[name] = path
    return out


def unassigned() -> "list[str]":
    """どの部品にも入らないファイル（試験は空であることを確かめる）。"""
    return [m for m in _modules() if part_of(m) is None]


def _resolve(base: "list[str]", dotted: str, names: "list[str]") -> "list[str]":
    """`base`（パッケージの道）＋ `dotted` が指すファイル。モジュールを名前で読み込む形（`from ..store import clock`）も解く。"""
    mods = _modules()
    parts = base + [p for p in dotted.split(".") if p]
    path = "/".join(parts)
    found = []
    rest = not names  # 名前の無い import（`import familiar_agent.x`）は、その道そのもの
    for name in names:
        sub = f"{path}/{name}" if path else name
        if sub in mods:
            found.append(sub)
        else:
            rest = True  # モジュールでない名前は、その道（モジュールか包みの __init__）から来る
    if rest:
        for cand in (path, f"{path}/__init__" if path else "__init__"):
            if cand in mods:
                found.append(cand)
                break
    return found


def _runtime_nodes(tree: ast.AST) -> "list[ast.AST]":
    """木の節を並べる。`if TYPE_CHECKING:` の中（型の注釈のためだけの import・動くときは読まない）は外す。"""
    out: "list[ast.AST]" = []
    stack: "list[ast.AST]" = [tree]
    while stack:
        node = stack.pop()
        out.append(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and "TYPE_CHECKING" in ast.unparse(child.test):
                stack.extend(child.orelse)
                continue
            stack.append(child)
    return out


@functools.cache
def imported_modules(module: str) -> "frozenset[str]":
    """ファイル `module` が import している、この包みの中のファイル（関数の中の import も含む）。"""
    path = _modules()[module]
    pkg = module.split("/")[:-1]
    found: "set[str]" = set()
    for node in _runtime_nodes(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            names = [a.name for a in node.names]
            mod = node.module or ""
            if node.level:
                if node.level - 1 > len(pkg):
                    continue  # 包みより深い相対 import は Python でも読めない。当て推量しない
                base = pkg[: len(pkg) - (node.level - 1)]
                found.update(_resolve(base, mod, names))
            elif mod == "familiar_agent" or mod.startswith("familiar_agent."):
                found.update(_resolve([], mod[len("familiar_agent") :], names))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("familiar_agent."):
                    found.update(_resolve([], alias.name[len("familiar_agent") :], []))
    found.discard(module)
    return frozenset(found)


@functools.cache
def edges() -> "dict[tuple[str, str], int]":
    """部品 → 部品の、import している（ファイル → ファイル）の組の数（同じ部品の中は数えない）。"""
    count: "collections.Counter[tuple[str, str]]" = collections.Counter()
    for module in _modules():
        src = part_of(module)
        for target in imported_modules(module):
            dst = part_of(target)
            if src and dst and src != dst:
                count[(src, dst)] += 1
    return dict(count)


def mermaid() -> str:
    """部品の図を Mermaid の文字にする（上から下・矢印のそばに、import しているファイルの組の数）。末尾は改行で終わる。"""
    lines = ["flowchart TD"]
    for group, title in (
        ("T", "自律機構：Tonic（T）"),
        ("I", "情報処理機構：Information-processing（I）"),
    ):
        lines.append(f'  subgraph {group}["{title}"]')
        lines += [f'    {k}["{name}"]' for k, (name, g) in PARTS.items() if g == group]
        lines.append("  end")
    lines += [f'  {k}["{name}"]' for k, (name, g) in PARTS.items() if g == "外"]
    for (a, b), n in sorted(edges().items()):
        lines.append(f"  {a} -->|{n}| {b}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(mermaid(), end="")
