# familiar-ai 直近の進め方と進捗（v0.4）

> v0.4：C-2（situated の役割整理）を実装済み（設計整理のみ・実行時の挙動不変）に更新。`situated_embeddings` の二役割（1=視点シフト検索・本人／2=在席者相関 p・他者・自分除外）を台帳へ固定し、現行で生きているのは役割1のみ・役割2＝p 軸は Phase 2、AGENT_SELF situated は自己の中立視点と明記した。ソースコードは変更なし。段階C（C-1・C-2）が済み、次は MI 集約段の調査。

> v0.3：C-1 を実装済みに更新。situated 相関の読み出し層 `_read_observations_by_situated` を新設（第一段・未接続）し、`recall_day_summaries` をその層へ付け替えて所有者絞りを撤去（第二段）。付け替え前の反証確認で、フォールバック二関数（`_recall_keyword_fallback`、`_recall_recency_fallback`）は主 situated 経路が0件のときだけ発火するため同じ situated 相関へ寄せると恒常的に空になると判明し、C-1 対象から外して別課題へ申し送った。C-1 の付け替え対象は `recall_day_summaries` の1本に絞った。

> v0.2：C-1 の対象から `recall_on_this_day` を外し、W の5軸に収まる三関数（`_recall_keyword_fallback`、`_recall_recency_fallback`、`recall_day_summaries`）に絞った。日付一致（周期一致の記念日想起）は W の5軸に受け皿がなく、trigger 側の時刻起因の情動発火に属するため、本体ごと trigger 段へ申し送りとした。完了条件と次の一歩もこれに合わせた。

> Phase 1（O/MI モデルの確立）の作業について、直近の進め方と現在地の記録。詳しい方法論は「出力に関する指示」に、段取りは課題8 に、値の根拠は課題5 と計測台帳にある。ここはその上での現在地のスナップショットである。

## 直近の進め方（サイクル）

一項目ずつ、次の流れで進めている。

1. 現状を実物のソースで調べる。反証側（間違っていたら何が見えるか）も対にして確かめ、事実と推測を分ける。
2. 設計方針を原因と対応案の形で出し、承認を得る。仮置きの値や未決は勝手に決めず質問して決める。
3. 承認後、TDD 改造方針を md で出す。目的と注意点、テスト先行、grep 完了条件、既存テスト修正の要否、DB とマイグレーション、コメント方針、最後にユーザーへの全体テスト依頼を含める。
4. ユーザーがコードとテストを当て、全体テストを回し、緑を報告する。
5. 緑を確認してから、実装済みの内容だけを課題8 と設計図と別紙と用語一覧へ差し替えで反映する。

段階B までは薄い縦切りで進めた。新規モジュールを未接続で足し、既存の生きた経路には結ばず、外部挙動を変えない形にした。器と関数を先に作り、発火やループや mood 値への接続は後続に置いている。

## 進捗（課題8 v0.15 の Phase 1 段取りに沿って）

### 段階A（MI 構造の確立）

- A-1：器の導入（`PrimitiveMentalItem` と `MentalItem`）実装済み。
- A-2：読み出し属性と器の対応表の調査完了。
- A-3-1：活性の (a0,n) 保存形式。列 `activation_a0` と `activation_n`、導出関数 `_derive_activation`（step は課題5 で 0.33 に確定。当初 0.7 を修正済み）実装済みで未接続。
- A-3 の Phase 1 残務：`recall_count` と `last_recalled_at` を新しさ（t 軸）の若返りへ対応づけ、記述で完了。
- A-3 の残り（導出値の想起接続、n 増減ほか）は Phase 2。

### 段階B（T レジスタ）

- B-1：mood（PAD）レジスタ `MoodPAD`（`mood_register.py`。M_rest=(0.5,0.5,0.5,0.5) へ半減期600秒で収束。state_key `mood_pad`）実装済みで未接続。
- B-2：drive（5欲求）レジスタ `AiDrivers`（`drive_register.py`。器のみ。state_key `drive5`）実装済みで未接続。
- B-3：PI 構築 `build_primitive` と PI→MI 拡張 `expand_to_mental`（`tif.py`。emotion に `MoodPAD`、drive に `AiDrivers` を載せる）実装済みで未接続。
- B-2 の蓄積 dynamics と、B-3 の Nudge および発火接続は後続段。

### 段階C（C-1・C-2 実装済み）

- C-1：観測想起経路の所有者絞りの撤去。代替の相関経路を先に作り、そのあと所有者絞りを外す二段で進めた。両段とも実装済み。
  - 第一段【実装済み・未接続】：situated 相関の読み出し層 `_read_observations_by_situated(person_id, n, columns, *, kind=None, keywords=())` を新設。`situated_embeddings s` を `observations o` に JOIN し `s.person_id` で紐づける（所有者に依らない母集合）。順序は timestamp DESC でベクトル類似度は使わない。テスト10件。
  - 第二段【実装済み】：`recall_day_summaries` を `_read_observations_by_kind`（所有者絞り）から `_read_observations_by_situated`（situated 相関・kind="day_summary"）へ付け替え、所有者絞りを撤去。母集合が所有者から在席者相関へ変わる（戻り値の形は不変）。既存 day_summary テスト4件を相関の意味論へ更新。
  - フォールバック二関数 `_recall_keyword_fallback`、`_recall_recency_fallback` は C-1 対象から外した。付け替え前の反証確認で、両関数は主 situated 経路（`recall` の cosine 検索・min_score=0）が0件のときにだけ発火し、その0件は「その person の situated 行が無い」ときに限ると判明した。同じ situated 相関へ寄せると発火条件と母集合が一致して恒常的に空になる。所有者絞りのまま「situated 行を持たない観測」を拾う役目を残す。「situated 行を持たない観測をフォールバックがどう扱うか」は C-1 と別課題として申し送り。
  - `recall_on_this_day` は C-1 の対象外。本体が周期一致（今日と同じ月日の記念日想起）で、W の5軸に受け皿がない。周期一致は trigger 側の時刻起因の情動発火に属し、その機構は未実装。person 絞りだけを situated へ寄せると日付一致を W 想起の一部として実装で確定させ、trigger 段の設計判断を先取りする。よって日付一致の本体ごと trigger 段へ申し送り、C-1 では触らない。当面は現状維持（`observations.person_id = self._person_id` のまま動く。外部挙動不変）。
  - 主 recall はすでに situated 相関で引くので対象外。
  - `self_model`、`curiosities`、`semantic_facts`、`behavior_policies` は後続の MI 集約段へ回す。
  - 完了条件（達成済み）：`recall_day_summaries` から `observations.person_id` の所有者絞りが消えた。書き込み側の `delete_day_summaries_for_date`（削除の所有者指定）とフォールバック二関数は対象外。`recall_on_this_day` の所有者絞りは trigger 段の完了条件へ移す。
- MI 集約段（新設・後続）：`self_model`、`curiosities`、`semantic_facts`、`behavior_policies` を MI へ集約し、person は situated 相関で結ぶ。自己モデルの専用 MI 化を含む。person 縛りの撤去はこの段。置き場所（段階C の C-3 か段階D か）は未確定。
- C-2【実装済み・設計整理のみ】：situated_embeddings の二役割（1=視点シフト検索・`s.person_id=問う人`・本人／2=在席者相関 p・在席他者・自分除外・[D-在席相関]）を台帳へ固定。現行コードで生きているのは役割1のみで、役割2（p 軸のスコアリング）は 5軸スコアラごと Phase 2。AGENT_SELF の situated 行は自己の中立視点（役割1の自己スコープ・`perspective_vec` 不在なら素の記憶ベクトル）で p 軸の自分除外とは別物。ソースコードは変更せず、位置づけの固定のみ。実機テスト（節目③）は p 軸実装後へ保留。

### C-1 で定めた設計上の理解

- 所有者（`observations.person_id`）は書き込み時の出所（話し手の空間）であって、想起の絞りではない。想起は situated 相関（視点）で引くのが意図。所有者相関という種別を situated に足す案は落とした。
- 在席者横断の視点相関の読み出しは、これから視点相関の実装として作る（今はなくてよい）。

## 直近のドキュメント整備（差し替え待ち）

- 用語一覧 v0.24：B スライスの実装名（`MoodPAD`、`AiDrivers`、`tif` の関数、各 state_key）を該当行へ併記。
- 別紙（設計詳細_活性_O書込_知覚在席）v0.4：B-2 反映。プロジェクトが v0.3 のため差し替え漏れの再提示。

## 次の一歩

C-1・C-2 は実装済み（段階C の Phase 1 スライスが済んだ）。次は MI 集約段の設計に入る。`self_model`、`curiosities`、`semantic_facts`、`behavior_policies` を MI へ集約し、person は situated 相関で結ぶ段。まず対象4種の現在の読み出し（`recall_self_model`、`recall_curiosities`、および `semantic_facts`／`behavior_policies` の読み出し経路）を実物のソースで調べ、MI 集約と situated 相関化の可否と順序を腑分けする調査に入る。置き場所（段階C の C-3 か段階D か）はこの調査で確定する。
