"""Locale detection and string translations for familiar-ai."""

from __future__ import annotations

import json
import locale
import os
from pathlib import Path

__all__ = ["_LANG", "_t", "BANNER"]

_LOCALES_DIR = Path(__file__).parent / "locales"

_VERSION = "v0.1"


def _detect_lang() -> str:
    """Detect language from environment. Returns a key present in _T, or 'en'."""
    raw = (
        os.environ.get("LANG")
        or os.environ.get("LC_ALL")
        or os.environ.get("LC_MESSAGES")
        or os.environ.get("LANGUAGE")
        or locale.getlocale()[0]
        or ""
    )
    lang = raw.split(":")[0]  # LANGUAGE can be colon-separated list

    # Traditional Chinese variants must be checked before generic "zh"
    if any(lang.startswith(p) for p in ("zh_TW", "zh_HK", "zh_MO", "zh-TW", "zh-HK", "zh-MO")):
        return "zh-tw"

    # Map locale prefix → i18n key. Order matters only for ambiguous prefixes.
    _PREFIX_MAP: list[tuple[str, str]] = [
        ("ja", "ja"),
        ("zh", "zh"),
        ("fr", "fr"),
        ("de", "de"),
        ("es", "es"),
        ("pt", "pt"),
        ("ru", "ru"),
        ("ko", "ko"),
        ("hi", "hi"),
        ("uk", "uk"),
        ("it", "it"),
        ("nl", "nl"),
        ("pl", "pl"),
        ("tr", "tr"),
        ("vi", "vi"),
        ("th", "th"),
        ("id", "id"),
        ("ms", "ms"),
        ("ar", "ar"),
        ("fa", "fa"),
        ("he", "he"),
        ("bn", "bn"),
        ("ta", "ta"),
        ("te", "te"),
        ("kn", "kn"),
        ("ml", "ml"),
        ("pa", "pa"),
        ("ur", "ur"),
        ("sw", "sw"),
        ("af", "af"),
        ("sv", "sv"),
        ("no", "no"),
        ("da", "da"),
        ("fi", "fi"),
        ("cs", "cs"),
        ("sk", "sk"),
        ("hu", "hu"),
        ("ro", "ro"),
        ("el", "el"),
        ("bg", "bg"),
        ("hr", "hr"),
        ("sr", "sr"),
        ("ca", "ca"),
        ("fil", "fil"),
        ("lt", "lt"),
        ("lv", "lv"),
        ("et", "et"),
        ("sl", "sl"),
        ("hy", "hy"),
        ("as", "as"),
        ("az", "az"),
        ("be", "be"),
        ("bs", "bs"),
        ("ceb", "ceb"),
        ("ny", "ny"),
        ("gl", "gl"),
        ("ka", "ka"),
        ("gu", "gu"),
        ("ha", "ha"),
        ("is", "is"),
        ("ga", "ga"),
        ("jv", "jv"),
        ("kk", "kk"),
        ("ky", "ky"),
        ("ln", "ln"),
        ("lb", "lb"),
        ("mk", "mk"),
        ("mr", "mr"),
        ("ne", "ne"),
        ("ps", "ps"),
        ("sd", "sd"),
        ("so", "so"),
        ("cy", "cy"),
    ]
    for prefix, code in _PREFIX_MAP:
        if lang.lower().startswith(prefix):
            return code
    return "en"


_LANG = _detect_lang()


def _load_locale(lang: str) -> dict[str, str]:
    """Load strings for *lang* from locales/{lang}.json, falling back to en."""
    path = _LOCALES_DIR / f"{lang}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    en_path = _LOCALES_DIR / "en.json"
    if en_path.exists():
        return json.loads(en_path.read_text(encoding="utf-8"))
    return {}


def _build_T_from_locales() -> dict[str, dict[str, str]]:
    """Reconstruct the legacy _T dict from all locale JSON files (for backward compat)."""
    result: dict[str, dict[str, str]] = {}
    for path in sorted(_LOCALES_DIR.glob("*.json")):
        lang = path.stem
        strings = json.loads(path.read_text(encoding="utf-8"))
        for key, value in strings.items():
            result.setdefault(key, {})[lang] = value
    return result


_EN_STRINGS: dict[str, str] = _load_locale("en")
# Per-language string cache; keyed by lang code so monkeypatching _LANG still works.
_LOCALE_CACHE: dict[str, dict[str, str]] = {}

# Backward-compatible export — built from JSON files, same structure as before
# 文字列の正本は `locales/{lang}.json`（Phase 6-3）。`_T` は試験と古い呼び手のために、そこから組み直す。
# 以前は locales が無いときの控えとして約 2,800 行の文字の表を持っていたが、locales は包みに入っていて読めないことが無く、
# 表のほうが古くなっていた（JSON に 9 個ある語が無かった）ので外した（2026-11-23）。
_T: dict[str, dict[str, str]] = _build_T_from_locales()


def _t(key: str, **kwargs: str) -> str:
    # Read _LANG dynamically so monkeypatching _LANG in tests takes effect.
    strings = _LOCALE_CACHE.setdefault(_LANG, _load_locale(_LANG))
    text = strings.get(key) or _EN_STRINGS.get(key, key)
    return text.format(**kwargs) if kwargs else text


def _make_banner(include_commands: bool = True) -> str:
    """Build a startup banner. CJK/emoji go outside the ASCII box to avoid width issues."""
    subtitle = _t("banner_subtitle")
    lines = [
        "🦉  Familiar AI",
        "╔══════════════════════════════════════╗",
        f"║  version {_VERSION:<29}║",
        "╚══════════════════════════════════════╝",
        f"  {subtitle}",
    ]
    if include_commands:
        lines.append(f"  {_t('repl_commands')}")
    return "\n".join(lines) + "\n"


BANNER = _make_banner()
