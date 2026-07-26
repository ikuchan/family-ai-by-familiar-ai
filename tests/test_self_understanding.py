"""自己認識を1枚にする（案B）：ME.md を素材に、実装から導いた「できること」を足す。

`ME.md`（人が書く人格・話し方・体）と `capability_summary`（実装から自動生成）が
同じことを2箇所で述べ、食い違っていた（`ME.md`「カメラ：無い」に対し要約は
「I can see ... using a camera」）。

- **人格は逐語で残す**。要約させると丁寧さの規則のような細かい指定が静かに落ち、
  人が書いたものが生成物に上書きされる。生成が担うのは「できること」だけ。
- **有効条件は実際に評価する**。`enabled_env: CAMERA_HOST` は「条件つき」であって
  「有効」ではない。環境変数が設定されているかを見ないと、無い身体を能力として語る。
"""

from __future__ import annotations

from familiar_agent.capability_state import (
    build_self_understanding_prompt,
    filter_enabled,
)

_YAML = """capabilities:
  - id: memory
    summary: Store and recall observations
    enabled: true
  - id: camera_vision
    summary: Capture and interpret images from a camera
    enabled_env: CAMERA_HOST
  - id: unfinished
    summary: Not done yet
    enabled: false
"""


def test_capability_needing_an_unset_env_var_is_excluded():
    out = filter_enabled(_YAML, env={})
    assert "memory" in out
    assert "camera_vision" not in out      # CAMERA_HOST が無いので使えない
    assert "unfinished" not in out


def test_capability_is_included_when_its_env_var_is_set():
    out = filter_enabled(_YAML, env={"CAMERA_HOST": "192.168.0.10"})
    assert "camera_vision" in out


def test_me_md_is_carried_verbatim_into_the_generation_prompt():
    me = "名前： パジュ\n話し方：標準語。丁寧さは相手で決める。"
    prompt = build_self_understanding_prompt(me_md=me, manifest=_YAML)
    assert me in prompt                    # 逐語で渡す
    assert "そのまま" in prompt             # 変えずに残せと指示している
