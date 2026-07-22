"""FAMILIAR_ENV_FILE を素の .env の上に重ねて読む（実機テスト用の .env.quiet 方式）。

素の .env（秘密鍵など）を先に読み、指定ファイルのキーがそれを上書きする。これで
.env.quiet は上書きしたい数行だけで済む。
"""

from __future__ import annotations

import os


def test_env_overlay_layers_base_then_override(tmp_path, monkeypatch):
    from familiar_agent import bootstrap

    base = tmp_path / ".env"
    base.write_text("FAM_OVL_A=base\nFAM_OVL_B=base\n", encoding="utf-8")
    overlay = tmp_path / ".env.quiet"
    overlay.write_text("FAM_OVL_B=overlay\n", encoding="utf-8")

    monkeypatch.setattr(bootstrap, "_base_env_path", lambda: base)
    monkeypatch.setenv("FAMILIAR_ENV_FILE", str(overlay))
    for k in ("FAM_OVL_A", "FAM_OVL_B"):
        monkeypatch.delenv(k, raising=False)

    try:
        bootstrap.load_app_bootstrap()
        assert os.environ["FAM_OVL_A"] == "base"     # base の値が継がれる
        assert os.environ["FAM_OVL_B"] == "overlay"  # overlay が勝つ
    finally:
        os.environ.pop("FAM_OVL_A", None)
        os.environ.pop("FAM_OVL_B", None)
