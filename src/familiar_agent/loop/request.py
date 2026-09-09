"""求め RQ：1つの求めが始まってから閉じるまでの状態（環-e-に・に-5-に）。

`InformationProcessing` は 30 個の可変状態を1つの名前空間へ平らに置いていた。設計の当初案は
**振る舞い**（生成器・動作器・想起）で file を割ることだったが、実測すると 30 個中 14 個が
その境界をまたぐ。**所有を分けるのは振る舞いではなく寿命である**（`モジュール分割設計`
に-2 の結論・Out of the Tar Pit / Statecharts）。

寿命は3つある。

| 寿命 | 中身 | 持ち主 |
|---|---|---|
| 装置 | 起動から終了まで（`_agent`・口・駆動体・キュー） | `InformationProcessing` |
| **求め** | 始まってから閉じるまで | **ここ** |
| 反復 | 1反復（`_w_id_map`） | 引数と返り値で渡す |

**`None` にしない。** 求めが無いことは、いまも `request_id is None` が表している。持ち主の
側で `if self._req else` を 40 箇所へ入れれば、寿命を表すどころか読みにくくなる。

**丸ごと作り直すことはしない。** 求めの寿命に見えて、実は求めをまたぐものがある——
`_request_generation` は打ち切りの検出に使う単調増加で、戻せば打ち切りが効かなくなる。
束ごとに、**いまと同じリセット点をそのまま保つ**。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Request:
    """1つの求めの寿命だけ生きる状態。

    - `iterations`／`iterations_capped`：**決める反復**を数える（環-h ⑤）。主LLM の返りで
      0 へ戻るので、この数は求めの長さを表さない。長さを表すのは考えた回数のほうである
    """

    iterations: int = 0
    iterations_capped: bool = False
