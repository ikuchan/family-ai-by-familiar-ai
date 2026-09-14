"""Generate capabilities.yaml by introspecting the project with an LLM.

Reuses collect_manifest_context() / build_generation_prompt() from capability_state so the
CLI and the REST pass (loop/rest_capabilities.py, layer 4) stay in sync. **This script is the
only writer of capabilities.yaml**: the file is the default kept in git; the running agent keeps
its current list in the DB (agent_state.capabilities) and never writes the file.

Usage:
    uv run python scripts/generate_capabilities.py
    uv run python scripts/generate_capabilities.py --overwrite   # force regenerate
    uv run python scripts/generate_capabilities.py --dry-run     # print to stdout only
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "capabilities.yaml"

sys.path.insert(0, str(ROOT / "src"))

from familiar_agent.capability_state import (  # noqa: E402
    build_generation_prompt,
    collect_manifest_context,
    load_manifest,
)
from familiar_agent.core.helpers import strip_code_fence  # noqa: E402


def _load_api_key() -> tuple[str, str]:
    """Return (platform, api_key) from .env."""
    env_file = ROOT / ".env"
    values: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                values[k.strip()] = v.strip()
    platform = values.get("PLATFORM", "anthropic").lower()
    api_key = values.get("API_KEY", os.environ.get("API_KEY", ""))
    return platform, api_key


async def _call_llm(prompt: str) -> str:
    platform, api_key = _load_api_key()
    if not api_key:
        print("ERROR: API_KEY not found in .env", file=sys.stderr)
        sys.exit(1)

    if platform == "anthropic":
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)
        resp = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        from anthropic.types import TextBlock

        for block in resp.content:
            if isinstance(block, TextBlock):
                return block.text.strip()
        return ""

    print(f"ERROR: unsupported platform '{platform}' for this script", file=sys.stderr)
    sys.exit(1)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing file")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout, do not write file")
    args = parser.parse_args()

    if OUT.exists() and not args.overwrite and not args.dry_run:
        print(
            "capabilities.yaml already exists.\n"
            "Use --overwrite to regenerate, or --dry-run to preview.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Collecting project context…", file=sys.stderr)
    context = collect_manifest_context()
    existing_yaml = load_manifest()

    prompt = build_generation_prompt(context, existing_yaml)

    print("Calling LLM…", file=sys.stderr)
    yaml_content = await _call_llm(prompt)

    if args.dry_run:
        print(yaml_content)
    else:
        OUT.write_text(strip_code_fence(yaml_content).strip() + "\n", encoding="utf-8")
        print(f"Written → {OUT}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
