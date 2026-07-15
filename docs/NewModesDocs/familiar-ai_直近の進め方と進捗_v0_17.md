# familiar-ai 直近の進め方と進捗（v0.17）

> v0.17：v0.16 の「次の一歩」の誤りを訂正。課題11k は課題5 v0.23（v0.22 改訂・2026-07-11）で既に設計クローズ済みであり、「まず課題11k を詰める」という前提が成り立たない。クローズの内容は、e 軸（感情一致）の関数形をガウシアン $e=\exp(-D^2/(2\sigma^2))$（起点 $\sigma=1.0$・$\lambda_i=1.0$・距離 D は全軸ロジット空間の軸重み付き PAD 距離・端クランプ $\varepsilon=0.001$）に確定し、観測の値踏み（O の emotion(PAD)）は評価器(LLM)が PAD を直接出す（機械写像 φ を作らない）と決めたこと。したがって P-3（感情の PAD 化・e 軸）の塞ぎは設計未決ではなく実装である。具体的には、評価器が観測 emotion を4軸 PAD で出すこと、`observations.emotion` を文字列から PAD へ移すこと（O書込）、`_compute_final_score` に e 軸を加えること、mood レジスタへ接続すること。これらは設計が決まっており、知覚（[D-知覚]）や新ループ（課題8）には依存しない。「次の一歩」節をこの実情に合わせて下に書き換えた。

> v0.16：P-2（参照申告で n 増減・freshness）を**新ループ待ちで保留**。精査の結果、参照申告の案a（フルLLM が `note_referenced` ツールを ReAct で呼ぶ）は、[D-反復出力]（1反復1出力・ターン内多段なし＝現行の主LLM↔ツールループを撤去）が消す機構に依存すると判明。案b（構造化出力での申告）が新設計と整合だが、出力ループ自体が [D-反復出力] で作り直されるため、P-2 は新ループ（課題8）が入るまで保留とする。参照申告・再評価・全候補一律強化の撤去は新ループ上で実装する。

> v0.15：Phase 2 に着手（入口＝P＝想起から）。**スライス P-1＝活性の想起接続**を実装。`_compute_final_score` の `importance` を `_derive_activation(activation_a0, activation_n)` へ差し替え、recall クエリを activation_a0/n の取得へ、`_compute_final_score` の呼び出しも更新。**二重時間減衰を解消**＝`importance` の日次減衰（`_generate_day_summary` の `decay_importance_async`）を撤去し、時間減衰は t 軸（time_score）へ一元化・a 軸は (a0,n) のイベント駆動（[D-想起合成]）。挙動変化（二重減衰の解消）＝**実機確認が要る**（UC⑥・節目①相当）。テスト2件＋回帰緑（時間減衰の順序は維持）。Phase 2 の背骨（5軸スコアラ）の第一歩。次は P-2（参照申告で n 増減・freshness 若返り）／P-3（感情 PAD 化）。

> v0.14：REST 内省サイクルの構造を確定（折衷型・課題10・設計図 v0.46）。起動＝T の純粋欠乏発火（日次）。1パス＝読み込み→蒸留（自己エピソード・per-person 関係サマリ）→open 棚卸し（孤児 Warn・消さない）→Config 自己調整（範囲内・人の設定不変）。**圧縮系（near-dup 統合・situated relation_key 語彙の増減）は同じパス内で量ベース**（たまっていれば実施）。平均ベクトル再推定はさらに低頻度。すべて版履歴で可逆、機械（距離/冗長度/Warn）とLLM（蒸留/棚卸し/命名/値提案）を切り分け。具体値（発火欲求・蓄積レート・量/滞留閾値・頻度）は課題5/10。実装は Phase 2。

> v0.13：situated の relation_key 語彙の増減設計を確定（[D-在席相関/V2]・設計図 v0.45）。**REST が relation_key（関係の種類）を育て・畳む**——増やす＝既存 relation_concept と遠い関与が繰り返し出たら新関係語を立てる（命名 LLM・距離判定 機械）、減らす＝近い concept を統合／使われない relation_key を間引く（版履歴で可逆）。初期3種（presence/speaker/subject）は基幹の錨で対象外。閾値・回数・失効は課題10/5。これは vector で関係を表す狙い（open-vocabulary の自己管理）の実体。実装は Phase 2（REST 依存）。

> v0.12：系統B の読み出し器を実装（未接続の薄い縦切り）。`memory.py` に `_read_supersede_chain(head_id, columns)` を新設＝現行版を起点に `superseded_by` を `WITH RECURSIVE` でさかのぼり版チェーンを再構成する dumb な読み出し（採点なし・既存経路から未接続・テスト4件）。MIデータモデル v0.07 §7 に反映。系統B 畳み込み本体は REST 依存で Phase 2。

> v0.11：MI 集約段の設計が一通り確定したので実装へ着手。**situated V2 の schema 器を Phase 1 分として実装**＝スライス1（`relation_key` 列・マイグレーション 022）とスライス2（UNIQUE を `(obs_id,person_id,relation_key)` へ・023・`_upsert_situated_embedding` に relation_key 既定 presence）。いずれも生成 presence のみで挙動不変。RED→GREEN、自分で回した回帰・ruff・mypy は緑（全体テストはユーザー実行中）。**slice-3 以降（視点列から presence/speaker/subject の関係生成・person_id 削除・旧 `_remember` 撤去）は、書き込みが視点列を実質埋めていない＝在席検出・話者帰属（[D-知覚]）依存で Phase 2 へ申し送り**と判明。設計図 v0.44・課題8 v0.18 に反映。

> v0.10：論点1（c）＝系統A の対応づけを確定。self_model→自己認識 MI 自己エピソード部（REST 蒸留・能力部は capability_summary・self_narrative_log 廃止で pinned へ）、curiosity→cue／SEEKING の open 意図 O（自己認識 MI でない）、方針は二分（自己認識 MI 方針＝核＋Config／behavior_policies＝信念 MI・REST が間接蒸留）。MIデータモデル v0.06 の付録A に反映。これで MI 集約段の設計（系統A・系統B・situated V2・自己認識 MI 構築規約）が一通り揃った。

> v0.9：自己認識 MI＝システムプロンプトの構築規約を追加（[D-自己認識分離]）。**プロンプトキャッシュ**整合のため、不変度順（核→Config→自己エピソード/policy）に前から並べ、毎ターンの可変分（W・mood・在席者・ユーザー入力）は messages 側へ置く。各区画に文字数上限（値は課題5）。キャッシュ非対応バックエンドでも無害。設計図 v0.42・用語一覧 v0.25 に反映。

> v0.8：論点2（situated V2 の生成規則・移行）を確定。観測の既存視点列（`writer_id`／`subject_id`／`participants_json`）が関係エッジの素材と判明。関係初期集合＝`presence`（←participants_json）／`speaker`（←writer_id）／`subject`（←subject_id＋content 抽出）。**旧 `_remember` の複製モデル（scope speaker/witnessed/scene・kind utterance/witnessed/scene）を撤去し単一 O＋関係エッジへ一本化**（複数名対応の根本課題への回答）。移行は既存観測1件を視点列から複数関係エッジへ展開（person_id はフォールバックのみ・列削除）。設計図 [D-在席相関/V2]（v0.41）・gap v0.4・MIデータモデル v0.05 に反映。これで situated V2 の設計（構造＋生成規則＋移行）が確定、残るは β 分離の測定（課題7）と実装の置き場所（課題8）。

> v0.7：系統A の調査で「self_model／curiosity の書き＝DEFAULT／読み＝AGENT_SELF のスコープ不一致」を発見。これは旧実装（複数名対応の試み）と新設計の gap で、系統A も REST 蒸留の自己認識 MI へ収束する Phase 2 寄りと判明（先の「系統A は Phase 1 で contained」は撤回）。ここから論点2（所有権・相関）を議論し、situated V2 の構造を確定：`observations.person_id` を削除し person↔MI は situated だけが担う／situated は型つき関係エッジ（`(obs_id,person_id)` に複数行・`UNIQUE` 撤去・在席関係／会話主体が並ぶ）／関係の種別は vector で表す（open-vocabulary）＋帳簿用 `relation_key` TEXT 列／分離が難しい関係は独立 vector 行で関係だけ引ける／p 軸は在席関係の行。設計図 [D-在席相関/V2]・gap v0.3 に反映。生成規則・移行写像・β 分離は次段。

> v0.6：系統B 畳み込みの confidence を確定。**信頼度は数値属性を持たず MI の content に自然文注記として書くにとどめる**（検索の5軸に入れないので機械可読スカラ不要）。数値導出案（activation 同型の (c0,m)・確証+1／反証−1／使われない decay をユースケース①〜⑥でシミュレート）は検討のうえ撤回。信頼度の更新は REST 内省が結末を読み content を書き換え supersede する形で Phase 2 寄り。旧 semantic_facts／behavior_policies・固定キー投影・confidence・adjust・memory_revisions は撤去対象。これで系統B の設計（supersede＋confidence）が確定（`MIデータモデル` §7〔設計確定・実装未着手〕）。

> v0.5：MI 集約段の設計を会話で開始（実装未着手）。対象4種を二系統に腑分けした。系統A＝`self_model`／`curiosity`（observations に kind で入る自己スコープ・C-1 と同型で contained）、系統B＝`semantic_facts`／`behavior_policies`（observations から `_project_observation` が固定キー投影する意味・信念層・key／revisions／confidence を持つ）。ユーザー方針で系統B の畳み込みを先に議論。supersede について、identity キーは足さず supersede チェーンの再帰想起で identity と revisions を賄い、W 取り込みは `superseded_by IS NULL` とする方向を確定（`MIデータモデル` §7 に反映）。書き込み側の紐づけ（old_id 同定）は類似度／REST へ寄せ Phase 2 寄り。confidence の写し先は次に議論。

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

Phase 2 の想起（入口 P）で、すぐ入るスライスと塞がっているスライスを次のように分ける。

- P-2（参照申告で n 増減）は新ループ（課題8・[D-反復出力]）待ちで保留。
- P-3（感情の PAD 化・e 軸）は設計クローズ済み（課題11k・課題5 v0.23）で、残りは実装のみ。評価器(LLM)が観測 emotion を4軸 PAD で出す、`observations.emotion` を文字列から PAD へ移す（O書込）、`_compute_final_score` に e 軸（ガウシアン）を加える、mood レジスタへ接続する。知覚や新ループに依存しないため、次に着手できるスライスである。
- P-4（5軸合成）は e 軸と p 軸に加え、r 軸が平均中心化（課題7・未実装）に依存する。
- p 軸は知覚（[D-知覚]）待ち。

したがって次の一歩は P-3 の実装である。着手前に、どの薄い縦切りから始めるか（書き込み側の PAD 化から入るか、読み出し側の e 軸から入るか）を設計方針として会話で決める。

現状のコードは、観測の `emotion` を文字列で保存し（`tools/memory.py`・既定 `"neutral"`）、`_compute_final_score` は `cosine × time_score × _derive_activation(a0,n)` で e 軸の項を持たない。`appraisal.py` は現在のターンの情動を VAD（−1〜1）で機械算出するが、観測ごとの PAD としては保存していない。P-3 はこの書き込み経路の PAD 移行と読み出し側の e 軸追加が本体になる。

（以下は Phase 1 完了時点の旧メモ）C-1・C-2 は実装済み（段階C の Phase 1 スライスが済んだ）。MI 集約段の設計を会話で進行中（実装未着手）。系統B（`semantic_facts`／`behavior_policies`）は設計確定（キーレス supersede・content 注記・REST 更新・旧2テーブル撤去・`MIデータモデル` §7）で実装本体は Phase 2 寄り。系統A（`self_model`／`curiosity`）の対応づけも確定（self_model→自己認識 MI 自己エピソード部・curiosity→cue O・方針二分・MIデータモデル v0.06 付録A）で、更新は REST 依存の Phase 2 寄り。自己認識 MI のシステムプロンプト構築規約（不変度順・messages 分離・文字数上限）も確定（[D-自己認識分離]）。situated V2 の構造・生成規則・移行も確定（[D-在席相関/V2]・gap v0.4）。situated V2 の schema 器（スライス1・2）を実装し、Phase 1 分はここで一区切り（全体テスト緑待ち）。slice-3 以降は視点列を埋める知覚（[D-知覚]）依存で Phase 2。**Phase 1 で残る MI 集約段の実装可能面はほぼ尽きた**（集約本体は REST 依存）。次の候補：(A) 系統B の読み出し器（キーレス supersede チェーンの再帰想起ヘルパー・母集合を変えず未接続の薄い縦切り）、(B) 別の設計論点や別課題（Phase 2 の p 軸設計・REST 詳細＝課題10 等）へ移る。全体テスト緑を確認してから、どちらへ進むか決める。
