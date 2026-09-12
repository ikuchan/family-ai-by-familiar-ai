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
| 反復 | 1反復（W と、その12桁の対応表） | 引数と返り値で渡す（属性に置かない） |

**`None` にしない。** 求めが無いことは、いまも `request_id is None` が表している。持ち主の
側で `if self._req else` を 40 箇所へ入れれば、寿命を表すどころか読みにくくなる。

**丸ごと作り直すことはしない。** 求めの寿命に見えて、実は求めをまたぐものがある——
`_request_generation` は打ち切りの検出に使う単調増加で、戻せば打ち切りが効かなくなる。
束ごとに、**いまと同じリセット点をそのまま保つ**。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Lookup:
    """1件の調べもの（環-g・段は）。

    以前は「どの動作で」「何という語で」が**6つの入れ物に3通りで**入っていた。
    `_inflight`（数）と `_in_flight_lookups`（列）は名前も意味もほぼ同じで、5箇所で
    別々に動かしていた。1件を1つの器にすれば、**飛行中の数は導出になり**、釣り合いを
    手で守らずに済む。

    `generation` は投げたときの求めの世代。打ち切ったあとに届いた完了を捨てるのに使う。
    """

    index: int
    action: str
    query: str
    generation: int
    result: "str | None" = None

    @property
    def in_flight(self) -> bool:
        """まだ結果が届いていないか。"""
        return self.result is None


@dataclass
class Request:
    """1つの求めの寿命だけ生きる状態。

    - `iterations`／`iterations_capped`：**決める反復**を数える（環-h ⑤）。主LLM の返りで
      0 へ戻るので、この数は求めの長さを表さない。長さを表すのは考えた回数のほうである
    - 列（`lookups`・`said_fillers`・`speech_to_deliver`・`turn_records`）は
      **`default_factory` で求めごとに別の入れ物にする**。既定値を1つにすると、次の求めへ
      前の一言や前のターンの記録が残る
    """

    # この求めの起点として O に書いた記録の id。版はこれを親に持つ。
    # **`None` は「いま生きている求めが無い」を表す。** 器そのものは常に在る。
    request_id: str | None = None
    # 求めそのものの文面。**どの版にも入れる。** 前の版は畳まれて辿れなくなるので、
    # 各版が単独で「何を聞かれたか」を持たないと、求めの文脈が失われる。
    request_text: str = ""
    # いま生きている版の id。次の版がこれを畳む（1本の鎖）。
    live_version_id: str | None = None
    # この反復の手がかり。**いま生きている記録の内容**である。求めの始まりでは来た事実
    # （発話・情動・機器）で、版が書かれたあとはその版の content になる。想起のクエリ・
    # 調停の入力・user メッセージの3つがこれを読む。
    cue: str = ""
    # 人の言葉そのもの（情動・機器で始まった求めでは空）。
    utterance: str = ""
    # 反復の起点。種別＝発話｜情動｜機器｜完了。情動や機器で起きた反復には人の発話が
    # 無いので、起点の内容を手がかり・調停の入力・user メッセージに使う。
    trigger_kind: str = "発話"
    # **決める反復**（軽量LLM が司る反復）を数える。上限に達した反復は recall を渡さない。
    # 主LLM の返りを受けた時点で 0 へ戻すので、**この数は求めの長さを表さない**（環-h ⑤）。
    # 長さを表すのは考えた回数（`_thinking_round`）のほうである。
    iterations: int = 0
    iterations_capped: bool = False
    # この求めで投げた調べもの（1件＝1つの `Lookup`）。**飛行中も届いた分も同じ列**に並ぶ
    # （`result` が `None` なら飛行中）。主LLM も `action="主LLM"` としてここへ並ぶ（環-h）。
    #
    # 通し番号は求めの中で1から振る。いま調べものを識別しているのは語だけで、同じ語を2回
    # 投げると区別できない。版の content へ「1番：… 2番：…」と列挙し、届いた完了を番号で
    # 対応づけるために振る。求めをまたいだ突き合わせは要らないので、一意な id ではなく
    # 通し番号で足りる。
    lookups: list[Lookup] = field(default_factory=list)
    # この求めのあいだに言ったつなぎ（言った順）。次のつなぎを、繰り返しでなく続きとして
    # 自然につなぐために見せる。
    said_fillers: list[str] = field(default_factory=list)
    # 配る保留（「いつ・何を言いたかったか」）。W へ流し、**求めが閉じたら**捨てる
    # （`_finish` と打ち切り）。
    speech_to_deliver: list[str] = field(default_factory=list)
    # このターンが作った記録と、その役割（観測 id, 役割）。**一つの並びが二つの用を
    # 賄う**：拡散想起の母集合（共起の関係）へ載せる id と、やりとりの関係の項。
    # 役割は 起点・版・見た・つなぎ・答え・独白（`_note_record`）。つなぎは共起に載せない
    # （中身が無く、育てる価値がない）。中断はこの求めで閉じるが、次の求めの共起には
    # 載る（打ち切った調査と言い直した問いの共起は、たどる価値がある）。
    turn_records: list[tuple[str, str]] = field(default_factory=list)
    # 主LLM が `see` を出したときの思考の深さ。see の帰りの反復は調停を飛ばして主LLM へ
    # 戻すので、そのとき引き継ぐ（空＝まだ見ていない・既定を当てる）。
    see_effort: str = ""
    # いまのやりとりが、その並びのどこから始まったか。**やりとりは並びの一区間**である。
    # 母集合への持ち越しは打ち切りでも消さないが、やりとりは打ち切りで区切る。二つの用は、
    # 区切りの規則が違う。
    exchange_start: int = 0
