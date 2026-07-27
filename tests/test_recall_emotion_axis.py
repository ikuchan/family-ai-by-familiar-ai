"""感情軸（いまの気分に近い記憶）の一次絞り。

$e = \\exp(-D^2/(2\\sigma^2))$、$D^2 = \\sum_i \\lambda_i (\\mathrm{logit}(x_{obs,i}) -
\\mathrm{logit}(x_{mood,i}))^2$。**ロジット空間での重み付きユークリッド距離**なので、
$\\sqrt{\\lambda_i}$ を畳み込んだ4次元ベクトルを持てば、pgvector の L2 距離がそのまま
$D$ になる。関連軸で既に使っている仕組みで、新しい依存は増えない。

出発点は**そのターンの mood** なので、気分が動けば候補も変わる（活性軸と違い、軸として
機能する）。$\\lambda_i$ は格納値へ畳み込む（案イ）ので、変えたら一括で作り直す。
"""

from __future__ import annotations

import inspect
import math

from familiar_agent.emotion_pad import pad_to_search_vector
from familiar_agent.store.observations import ObservationStore
from familiar_agent.tools.memory import _emotion_match


def test_search_vector_is_the_logit_of_each_axis():
    # 中立 0.5 はロジットで 0。
    assert pad_to_search_vector((0.5, 0.5, 0.5, 0.5)) == [0.0, 0.0, 0.0, 0.0]


def test_search_vector_clamps_the_ends():
    # 0 と 1 はロジットが発散するので ε で寄せる（活性導出と共通の 0.001）。
    v = pad_to_search_vector((0.0, 1.0, 0.5, 0.5))
    assert all(math.isfinite(x) for x in v)
    assert v[0] < -6.0 and v[1] > 6.0


def test_l2_distance_between_search_vectors_matches_the_scored_distance():
    # ベクトル間の L2 距離が、採点が使う D と一致する（λ=1 のとき）。
    obs = (0.8, 0.15, 0.55, 0.6)
    mood = (0.5, 0.5, 0.5, 0.5)
    vo, vm = pad_to_search_vector(obs), pad_to_search_vector(mood)
    d = math.sqrt(sum((a - b) ** 2 for a, b in zip(vo, vm)))
    # 採点は e=exp(-D²/(2σ²))。σ=1 なら D は逆算できる。
    e = _emotion_match(obs, mood, sigma=1.0)
    assert abs(math.sqrt(-2.0 * math.log(e)) - d) < 1e-6


def test_lambdas_are_folded_into_the_stored_vector():
    # 重み付き距離を素の L2 にするため、√λ を畳み込む（案イ）。
    v1 = pad_to_search_vector((0.8, 0.5, 0.5, 0.5), lambdas=(1.0, 1.0, 1.0, 1.0))
    v4 = pad_to_search_vector((0.8, 0.5, 0.5, 0.5), lambdas=(4.0, 1.0, 1.0, 1.0))
    assert abs(v4[0] - 2.0 * v1[0]) < 1e-9      # √4 = 2 倍


def test_by_emotion_searches_the_nearest_in_that_space():
    src = inspect.getsource(ObservationStore.by_emotion)
    assert "emotion_vec" in src
    assert "<->" in src                          # pgvector の L2 距離
    assert "o.superseded_by IS NULL" in src      # 死んだ記録は候補にしない
    assert "s.person_id = %s" in src             # 視点スコープは他軸と揃える


def test_by_emotion_returns_the_same_columns_as_the_other_axes():
    # 採点器は軸を区別せず行を扱うので、列が揃っていないと落ちる。
    import re

    def cols(fn):
        return set(re.findall(r"o\.(\w+)", inspect.getsource(fn)))

    assert cols(ObservationStore.by_vector) - {"id"} <= cols(ObservationStore.by_emotion)
