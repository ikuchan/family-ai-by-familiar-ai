"""設定画面の保存先は素の `.env`（環-n・2026-09-18）。

`FAMILIAR_ENV_FILE=.env.quiet` で起動した状態で設定画面を保存すると、上書き用の雛形（git 管理下・数行だけの約束）
に設定一式（鍵 5 つ・画面の既定値 9 行）が書き出された（実機 18:27）。読む側（`bootstrap`：素の `.env` を読んで
上書き file を重ねる）と書く側（`resolve_env_path()`＝上書き file）が一致していなかった。
保存先を素の `.env`（`bootstrap.settings_env_path()`）に固定し、`reload_env` も読む側と同じ順で 2 つを読む。
"""

from __future__ import annotations

import inspect
import os

from familiar_agent import bootstrap, env_reload


def test_settings_are_saved_to_the_base_env_even_with_an_overlay(monkeypatch, tmp_path):
    base = tmp_path / ".env"
    overlay = tmp_path / ".env.quiet"
    base.write_text("A=1\n", encoding="utf-8")
    overlay.write_text("B=2\n", encoding="utf-8")
    monkeypatch.setattr(bootstrap, "_base_env_path", lambda: base)
    monkeypatch.setenv("FAMILIAR_ENV_FILE", str(overlay))
    assert bootstrap.resolve_env_path() == overlay  # 読む側の重ね先はそのまま
    assert bootstrap.settings_env_path() == base  # 書く側は素の .env


def test_reload_reads_the_base_then_the_overlay(monkeypatch, tmp_path):
    base = tmp_path / ".env"
    overlay = tmp_path / ".env.quiet"
    base.write_text("A=1\nB=1\n", encoding="utf-8")
    overlay.write_text("B=2\n", encoding="utf-8")
    monkeypatch.setattr(bootstrap, "_base_env_path", lambda: base)
    monkeypatch.setenv("FAMILIAR_ENV_FILE", str(overlay))
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    r = env_reload.reload_env(
        bootstrap.settings_env_path()
    )  # GUI と同じ呼び方（素の .env を明示で）
    assert r.found and os.environ["A"] == "1" and os.environ["B"] == "2"  # 上書きが勝つ
    base.write_text("A=9\nB=1\n", encoding="utf-8")
    env_reload.reload_env(base)  # 保存直後に呼ばれる形
    assert os.environ["A"] == "9" and os.environ["B"] == "2"
    monkeypatch.delenv("A", raising=False)
    env_reload.reload_env()  # path 無し＝上書き file だけ。素の .env は読まない（環-r）
    assert "A" not in os.environ and os.environ["B"] == "2"


def test_the_gui_saves_to_the_settings_path_and_reloads_with_it():
    from familiar_agent import gui

    src = inspect.getsource(gui)
    assert "_ENV_PATH: Path = settings_env_path()" in src
    assert "_ENV_PATH: Path = resolve_env_path()" not in src
    assert "reload_env(settings_env_path())" in src  # 素の .env を暗黙に読ませない（環-r）
