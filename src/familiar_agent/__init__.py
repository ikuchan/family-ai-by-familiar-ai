"""Embodied agent - a real-world exploration AI."""
from __future__ import annotations

import re


def _format_version(raw: str) -> str:
    """Convert a PEP 440 version string to v0.HASH format.

    Examples:
      "0.6.0+g30bfd72"           -> "v0.30bfd"
      "0.6.1.dev3+g30bfd72"      -> "v0.30bfd"
      "0.6.1.dev3+g30bfd72.d*"   -> "v0.30bfd*"  (dirty)
      "0.6.0"                    -> "v0.6.0"      (exact tag, no hash)
    """
    dirty = ".d" in raw
    m = re.search(r"\+g([0-9a-f]+)", raw)
    if m:
        h = m.group(1)[:5]
        return f"v0.{h}{'*' if dirty else ''}"
    return f"v{raw}"


try:
    from importlib.metadata import version as _pkg_version
    __version__ = _format_version(_pkg_version("familiar-ai"))
except Exception:
    try:
        from familiar_agent._version import __version__ as _v
        __version__ = _format_version(_v)
    except Exception:
        __version__ = "v0.?????"
