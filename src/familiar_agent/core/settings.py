"""層 3「設定値」の登録の表（記-a-に・2026-09-14・`設計方針_REST内省_設定値を調整する`）。

**登録が無い値は動かせない。** 登録には範囲・刻み・見る計測（計測ログの種別）・規則（機械か LLM か）・
根拠を必ず添える。`config_overrides.RANGES` はこの表から導く（範囲を二重に持たない）。接続情報
（鍵・URL・機器 id）と式の骨格（`rate`・`theta_fire`）は登録しない——`config_overrides.is_protected` が
守る。値は `agent_state.config_overrides`（DB）にあり、既定は `config.py`。`.env` は読まない。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Setting:
    field: str  # 完全名（`MemoryConfig.recall_half_life_days`）
    lo: float
    hi: float
    step: float  # 1 晩に動かせる幅
    measure_kind: str  # 見る計測ログの種別（`調停`・`申告`・`気分`・`欲求`・`層1`・`蒸留`）
    rule: str  # `機械`（閾値の比較で決まる）／`LLM`（数字を見て提案する）
    note: str  # 根拠（`根拠台帳` の節など）


_MOOD_AXES = ("p", "pn", "a", "dom")
_DRIVE_AXES = ("seeking", "rest", "bond", "safety", "esteem")

REGISTRY: tuple[Setting, ...] = (
    Setting(
        "MemoryConfig.distill_min_a0",
        0.20,
        0.70,
        0.05,
        "蒸留",
        "LLM",
        "蒸留の材料から外す新規性の下限。実測分布 最小 0.143・p10 0.469・p25 0.604（根拠台帳）",
    ),
    Setting(
        "MemoryConfig.recall_half_life_days",
        1.0,
        30.0,
        1.0,
        "層1",
        "LLM",
        "観測の半減期 HL。10 日＝「昨日は必ず・10 日前はあまり」（出来事を畳む v0.2・根拠台帳 §34）",
    ),
    Setting(
        "MemoryConfig.info_target_bits",
        16384.0,
        1048576.0,
        16384.0,
        "層1",
        "LLM",
        "層 1 の定常値 I*。2^17 ≈ 150 kbit〔仮〕（出来事を畳む §4・§6）。規則は ろ-に（核の固め）の後に接続",
    ),
    Setting(
        "MemoryConfig.core_same_cos",
        0.9,
        1.0,
        0.01,
        "層1",
        "LLM",
        "同一とみなすコサイン。0.98 以上は同文（2026-09-15 実測・出来事を畳む §3c）。規則は未接続",
    ),
    Setting(
        "MemoryConfig.core_bundle_cos",
        0.3,
        0.8,
        0.05,
        "層1",
        "LLM",
        "束ねの半径 τ。二山の谷 0.5（2026-09-15 実測・出来事を畳む §3c）。規則は未接続",
    ),
    Setting(
        "MemoryConfig.core_bundle_min",
        2.0,
        6.0,
        1.0,
        "層1",
        "LLM",
        "固めてよい束の最小の大きさ m（出来事を畳む §3c）。規則は未接続",
    ),
    Setting(
        "MemoryConfig.core_bundles_per_night",
        1.0,
        20.0,
        1.0,
        "層1",
        "LLM",
        "1 晩に固める束の上限（出来事を畳む §3c）。規則は未接続",
    ),
    Setting(
        "MemoryConfig.diffuse_far_share",
        0.0,
        1.0,
        0.1,
        "申告",
        "機械",
        "関連想起の 2 並びの配分。参照された側へ寄せる（出来事を畳む v0.2）",
    ),
    Setting(
        "AgentConfig.arbiter_timeout_sec",
        1.0,
        10.0,
        0.5,
        "調停",
        "LLM",
        "調停の時間切れ。1.0 未満は普通の会話が間に合わず、10.0 超は遅延が実用にならない（課題8 記-a-に）",
    ),
    Setting(
        "MemoryConfig.recent_exchanges_arbiter",
        1,
        10,
        1,
        "申告",
        "機械",
        "直近の窓（軽量LLM）。窓の端に参照が 20% 以上なら +1、末尾が一度も参照されなければ −1（課題5 D 章）",
    ),
    Setting(
        "MemoryConfig.recent_exchanges_main",
        1,
        10,
        1,
        "申告",
        "機械",
        "直近の窓（主LLM）。規則は軽量LLM と同じ（課題5 D 章）",
    ),
    *(
        Setting(
            f"InnerStateConfig.mood_{axis}_{q}",
            0.0,
            1.0,
            0.05,
            "気分",
            "機械",
            "気分の言葉の境目＝その軸の実測分位。等頻度を保つように取り直す（課題5 C2 節）",
        )
        for axis in _MOOD_AXES
        for q in ("p10", "p30", "p70", "p90")
    ),
    *(
        Setting(
            f"InnerStateConfig.drive_p70_{axis}",
            0.0,
            1.0,
            0.05,
            "欲求",
            "機械",
            "欲求の弱い言及の境目＝その軸の実測 p70。等頻度を保つように取り直す（課題5 C2 節）",
        )
        for axis in _DRIVE_AXES
    ),
)

_BY_FIELD = {s.field: s for s in REGISTRY}


def get(field: str) -> "Setting | None":
    return _BY_FIELD.get(field)


def ranges() -> dict[str, tuple[float, float]]:
    """`config_overrides.RANGES` の中身（この表から導く）。"""
    return {s.field: (s.lo, s.hi) for s in REGISTRY}
