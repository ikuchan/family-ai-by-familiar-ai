"""#11 段階1：イベント駆動ループの system プロンプト組み立て（案B・クリーン最小）。"""

from __future__ import annotations

from familiar_agent.loop.prompt import EVENT_SYSTEM_PROMPT, build_event_system_prompt


def test_static_prompt_has_body_net_no_legs_and_family_bond():
    p = EVENT_SYSTEM_PROMPT
    assert "search_deferred" in p          # net（インターネット）を body-tool に
    assert "legs" not in p and "walk" not in p  # 足なし
    assert "family-bond" in p              # 家族の一員・切れない関係の identity
    assert "見捨て" in p                    # 見捨てない/見捨てられない
    assert "no-raw-internal-metrics" in p  # 生の内部指標を出さない（social_policy から移植）
    assert "one-output-per-iteration" in p  # 1反復1出力


def test_build_includes_self_knowledge_present_pi_and_w():
    out = build_event_system_prompt(
        self_understanding="[ME] ぼくの口調\n\n## 私にできること\n- 能力の要約",
        family_md="[FAMILY] パパ・ママ",
        present_ctx="(present :speaker \"パパ\")",
        pi_ctx="[内部状態(PI)] 気分: おだやか / 欲求: SEEKING 高",
        workspace_ctx="[想起]昔の話",
    )
    # 返りは (安定部, 可変部)。安定部は反復ごとに変わらないので backend がキャッシュする。
    stable, variable = out
    for needle in ("[ME] ぼくの口調", "[FAMILY] パパ・ママ", "能力の要約", "family-bond"):
        assert needle in stable
    for needle in ("(present", "[内部状態(PI)]", "[想起]昔の話"):
        assert needle in variable


def test_build_omits_retired_layers():
    out = build_event_system_prompt(
        self_understanding="me", family_md="fam",
        present_ctx="", pi_ctx="pi", workspace_ctx="w",
    )
    joined = "\n".join(out)
    assert "[Interaction policy]" not in joined   # social_policy は載せない
    assert "interoception" not in joined           # 撤去対象は載せない


def test_self_understanding_is_one_block_not_two():
    # 人格（人が書く）と能力（実装から導く）を別々に注入すると、同じことを2箇所で述べて
    # 食い違う（ME.md「カメラ：無い」に対し要約が「I can see ... using a camera」）。
    # 案B：生成の時点で1枚にまとめ、注入も1本にする。
    import inspect

    sig = inspect.signature(build_event_system_prompt)
    assert "self_understanding" in sig.parameters
    assert "me_md" not in sig.parameters and "capabilities" not in sig.parameters
