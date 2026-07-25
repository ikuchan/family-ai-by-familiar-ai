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
        me_md="[ME] ぼくの口調",
        family_md="[FAMILY] パパ・ママ",
        capabilities="能力の要約",
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
        me_md="me", family_md="fam", capabilities="cap",
        present_ctx="", pi_ctx="pi", workspace_ctx="w",
    )
    joined = "\n".join(out)
    assert "[Interaction policy]" not in joined   # social_policy は載せない
    assert "interoception" not in joined           # 撤去対象は載せない
