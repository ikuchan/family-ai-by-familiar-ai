"""出-dm・出-dn・出-dp：ログに家族の言葉と名前を出さない（2026-11-21）。

ログに出してよいのは、時刻、決まった文、件数だけ。言葉は `core.redact.chars` で字数に、名前と曲名は `tag` で短い印にする。
見張り：INFO 以上のログで、言葉を切って出す書式（`%.40s` など）や `%r` を使う所は、下の許しの一覧（家族の言葉が入らない
と確かめたもの）だけにする。新しく足した所がこの試験で落ちたら、`chars`・`tag` で包むか、入らない理由を書いて許しに足す。
"""

from __future__ import annotations

import ast
import pathlib
import re

from familiar_agent.core.redact import chars, tag

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "familiar_agent"
_LEVELS = {"info", "warning", "error", "exception", "critical"}
_SPEC = re.compile(r"%-?\d*\.\d+[sr]|%r")

#: 家族の言葉が入らない書式の始め（ファイルごと）。理由：記録の番号、見えた物の名前、設定の値、サーバーの名前と誤り、
#: 外のサイトの返り、定点の名前。
_ALLOWED = {
    "config_overrides.py": ("Config の調整を拒否",),
    "core/structured_ask.py": (
        "形のある答えを読めなかった",
    ),  # 字数と最初の 1 字（`{` か否か）だけ
    "loop/event_loop.py": (
        "event-loop 意味づけが返った",
        "event-loop %s見えたもの",
    ),
    "loop/finals.py": (  # 記録の id の頭
        "event-loop 最終版の継起を書いた",
        "event-loop 記憶の答えは使わない",
        "event-loop 記憶の答えで答える",
    ),
    "mcp_client.py": ("MCP connect", "Failed to connect to MCP server"),
    "occupancy_map.py": ("定点の一覧に無い名前",),
    "poses.py": ("定点の設定を読めなかった", "定点が可動範囲の外"),
    "store/observations.py": ("Ruri の数値を書けなかった", "append_and_reembed"),
    "store/ruri_backfill.py": ("Ruri で数値にできなかった",),
    "tools/memory.py": ("note_lookup_started failed",),
    "tools/mic.py": ("AUDIO_INPUT_GAIN",),
    "tools/realtime_stt.py": ("Realtime STT websocket closed",),
    "tools/deferred_fetch.py": ("deferred fetch の道具が使えなかった",),
    "tools/deferred_search.py": (
        "deferred search ページを読めなかった",
        "deferred search %s が使えなかった",
        "deferred search 両方の道具",
    ),
    "loop/rest_season.py": ("rest 季節の層",),
}


def _found() -> list[str]:
    out = []
    for p in sorted(_SRC.rglob("*.py")):
        rel = p.relative_to(_SRC).as_posix()
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
                continue
            if n.func.attr not in _LEVELS or not n.args:
                continue
            a0 = n.args[0]
            if not (isinstance(a0, ast.Constant) and isinstance(a0.value, str)):
                continue
            if _SPEC.search(a0.value) and not a0.value.startswith(_ALLOWED.get(rel, ())):
                out.append(f"{rel}:{n.lineno} {a0.value[:50]}")
    return out


def test_no_log_prints_cut_words_outside_the_allow_list() -> None:
    assert _found() == []


#: 家族の言葉や名前が入りうる引数の名前。INFO 以上のログでは `chars`・`tag`・`len` で包む。
_SENSITIVE = {
    "query",
    "utterance",
    "transcript",
    "title",
    "artist",
    "person",
    "who",
    "speaker",
    "body",
    "text",
}
_WRAPS = {"chars", "tag", "len", "_names"}
#: 包まずに出てくるが、ログに中身は出ないか、家族の言葉ではないもの（書式の始め）。
_SENSITIVE_ALLOWED = {
    "loop/event_loop.py": (
        "声の似かた",
        "話していいと言われたので黙るのを解く",
    ),  # 空かどうかを見るだけ。出すのは印
    "loop/rest_season.py": ("rest 季節の層",),  # 外のサイトの返り
    "tools/stt.py": ("STT: ElevenLabs error",),  # サービスの誤りの文
}


def _bare(node: ast.AST) -> list[str]:
    hits: list[str] = []

    def walk(n: ast.AST, wrapped: bool) -> None:
        if isinstance(n, ast.Call):
            f = n.func
            name = (
                f.id
                if isinstance(f, ast.Name)
                else (f.attr if isinstance(f, ast.Attribute) else "")
            )
            inner = wrapped or name in _WRAPS
            for a in [*n.args, *(k.value for k in n.keywords)]:
                walk(a, inner)
            walk(f, wrapped)
            return
        if isinstance(n, ast.Name) and n.id in _SENSITIVE and not wrapped:
            hits.append(n.id)
        if isinstance(n, ast.Attribute):
            if n.attr in _SENSITIVE and not wrapped:
                hits.append(n.attr)
            walk(n.value, wrapped)
            return
        for c in ast.iter_child_nodes(n):
            walk(c, wrapped)

    walk(node, False)
    return hits


def test_no_log_passes_family_words_unwrapped() -> None:
    """`%s` でそのまま出す形も見る（言葉を切る書式だけを見ると、`%s` の言葉や名前を見落とす）。"""
    found = []
    for p in sorted(_SRC.rglob("*.py")):
        rel = p.relative_to(_SRC).as_posix()
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
                continue
            if (
                n.func.attr not in _LEVELS
                or not n.args
                or getattr(n.func.value, "id", "") != "logger"
            ):
                continue
            a0 = n.args[0]
            if not (isinstance(a0, ast.Constant) and isinstance(a0.value, str)):
                continue
            if a0.value.startswith(_SENSITIVE_ALLOWED.get(rel, ())):
                continue
            bare = [x for a in n.args[1:] for x in _bare(a)]
            if bare:
                found.append(f"{rel}:{n.lineno} {a0.value[:40]} ← {','.join(bare)}")
    assert found == []


def test_chars_keeps_only_the_length_and_tag_is_short_and_stable() -> None:
    assert chars("夏祭りはいつ") == "6 字" and chars(None) == "0 字"
    assert tag("なまえ") == tag("なまえ") != tag("べつのなまえ")  # 同じ起動のあいだは同じ印
    assert (
        tag("なまえ").startswith("#") and len(tag("なまえ")) == 5 and "なまえ" not in tag("なまえ")
    )
    assert tag("") == "-"
