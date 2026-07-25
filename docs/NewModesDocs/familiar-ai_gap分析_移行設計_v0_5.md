# familiar-ai gap 分析・移行設計（旧構成 → 新構成）（v0.5）

v0.4：situated V2 の生成規則・移行を確定。関係初期集合＝presence/speaker/subject（視点列 participants_json/writer_id/subject_id から生成）。旧 `_remember` 複製モデル（scope speaker/witnessed/scene・kind utterance/witnessed/scene）の撤去を申し送りへ追加。移行写像を「既存観測1件→複数関係エッジ展開」へ精緻化。

v0.3：situated V2（型つき関係エッジ・[D-在席相関/V2]）を反映。person_id 保持メモを「`observations.person_id` 削除＋situated だけが person↔MI を担う（型つき関係エッジ・`relation_key` 帳簿列・`UNIQUE(obs_id,person_id)` 撤去・独立 vector 行）」へ更新。課題8 申し送りも V2 へ。生成規則・移行写像・β 分離は次段。

課題6 gap 文書化の承認用ドキュメント。旧実装の ~20 ストア（DB テーブル22個）＋旧感情系を、新設計（O 一元化・T レジスタ・W 派生・共通 MI）へどう移すかの対応表。設計レベルの対応と確定先（[D-…]）を示し、コード撤去・マイグレーション・テストの詳細は課題8 に送る。

v0.2：課題7 のコード確認を反映。GlobalWorkspace 行を精緻化（記憶ストアでなく coalition 競合＋ignition 機構→ W へ直接転用は不可・調停/発火へ）。

0. 前提と読み方

新設計の骨格：記憶は O に一元化（append＋supersede）、W は O からの派生ビュー（毎ターン想起で構築・退避/eviction/fade なし）、T 側は数値レジスタ（drive/mood/norm/presence）、記憶は 共通 MI（PI＝emotion(PAD)/drive、MI＝PI＋id/content/vector/supersedes/activation）。
確定先の表記：各対応の根拠ブロックを [D-…] で示す。値の確定先は課題5（パラメータ）・課題7（実測初期値）・課題11(k)（PAD 化）等。
実装（撤去・新テーブル・マイグレーション・既存テスト修正）は課題8 で TDD として行う。本書は「何を何へ移すか」を確定する承認用で、コード手順は含めない。

1. 旧 DB テーブル（22個）→ 新構成 対応表
| 旧 | 旧の役割 | 新構成での扱い | 確定先 |
|---|---|---|---|
| observations | 観測・記憶の本体 | O（単一エピソード記憶＝MI の本体） | [D-記憶単一化]／[D-O書込]／[D-MIモデル] |
| obs_embeddings | 観測の埋め込み | MI.vector（関連軸の素材） | [D-想起合成]／[D-MIモデル] |
| situated_embeddings | 人視点別の埋め込み | 相関サブテーブル（MI×person の situated・在席者相関 p の素材・person 別視点） | [D-在席相関]／[D-想起合成] |
| memory_activation | 活性（重要度） | MI.activation（(a0,n) から導出・on-read） | [D-活性]／[D-想起合成] |
| memory_events | イベントログ | O への取込の前段（O に吸収） | [D-O書込] |
| memory_jobs | materialize ジョブ | O 投影の非同期ジョブ（保守器/REST） | [D-O書込] |
| memory_revisions | 版履歴 | MI.supersedes（版履歴） | [D-記憶単一化]／[D-MIモデル] |
| memory_links | 明示の連想リンク | 廃止（代替＝[D-WR拡散想起]。連想は vector 関連・明示関係は content・読み側は旧実装で未結線） | [D-WR拡散想起]／[D-MIモデル] |
| episodes | エピソード | O に統合 | [D-記憶単一化] |
| episode_memories | エピソード記憶 | O に統合 | [D-記憶単一化] |
| mental_state_log | mood/affect ログ | M（PAD・T レジスタ）。mood の永続は agent_state | [D-B分離]／[D-活性] |
| agent_state | drive/mood/能力要約 等 | T レジスタ（drive→PI.drive／mood→PI.emotion）＋ capability_summary→SS・自己認識 MI | [D-B分離] |
| pending_speech | 未発話 | O の open 意図（W 派生・退避なし） | [D-記憶単一化]／[D-単一想起] |
| unfinished_business | 未完了 | O の open 意図（気がかり・salience） | [D-気がかり統合]／[D-単一想起] |
| persons | 人物 identity | I 自前の人物 identity（InsightFace person_id・identity は B に持たない） | [D-知覚]／[D-B定点] |
| relationship_state | 関係メタ（trust/intimacy 等） | 廃止＋移管（§3 参照） | [D-在席相関] |
| scene_entities | 知覚エンティティ | T(G) 知覚（norm・在席＝private）。驚いた出来事は O | [D-知覚]／[D-B分離] |
| scene_events | 知覚イベント | 同上（scene の出来事・驚きは O） | [D-知覚] |
| self_narrative_log | 自己物語の日記 | 廃止＋移管（§3 参照） | [D-在席相関] |
| semantic_facts | 意味事実 | 観測→意味の昇格（O の外・O は出来事のみ） | [D-O書込] |
| behavior_policies | 行動方針 | 観測→方針の昇格（policy・自己認識 MI policy） | [D-O書込]／[D-想起合成] |
| exploration_state | 探索状態 | 廃止＋機能移管（§3 参照） | [D-知覚]／[D-記憶単一化] |
2. 旧 実行時状態クラス → 新構成 対応表
| 旧 | 旧の役割 | 新構成での扱い | 確定先 |
|---|---|---|---|
| GlobalWorkspace（workspace.py） | 作業記憶（名）／実体は coalition 競合＋ignition | 作業集合・broadcast の概念→W／競合・ignition 実装→調停・発火（記憶ストアでないため W へ直接転用は不可・課題7 で確認） | [D-記憶単一化]／[D-I内部] |
| PendingSpeechStore | 未発話キュー | O の open 意図（W 派生） | [D-単一想起] |
| ConcernEngine／Concern | 気がかり（最大5・自前 decay） | O の open 意図＋salience | [D-気がかり統合]（課題11j） |
| AttentionSchema | 注意・焦点 | MI.activation／salience（焦点は W 構築で表す） | [D-活性] |
| AffectiveState／MentalStateBus／Affect | 感情状態 | M（PAD・T レジスタ） | [D-B分離]／[D-活性] |
| PredictionEngine／PredictionSignal | 予測・期待 | norm（T(G) private・予測の驚き） | [D-知覚]／[D-B分離] |
| RelationshipTracker | 関係追跡 | 廃止＋移管（§3） | [D-在席相関] |
| PersonMemoryManager | 人物記憶 | I 自前 person identity | [D-知覚]／[D-B定点] |
| SelfState | 自己状態 | SS（自己状態・H 即時読み） | [D-I内部] |
| SelfNarrative | 自己物語 | 廃止＋移管（§3） | [D-在席相関] |
| CapabilityState | 能力自己理解 | 自己認識 MI（能力） | [D-想起合成]／[D-自己認識分離] |
| SocialPolicyEngine／SocialState | 社会的方針 | social policy／配信ゲート | [D-値踏み] |
| AppraisalEngine | 値踏み | 評価器（軽量LLM） | [D-値踏み]／[D-I内部] |
| DefaultModeProcessor | アイドル自発想起 | 廃止（自発活性は T 発火・W 空きの拡散は WR） | [D-WR拡散想起] |
| ExplorationTracker | 探索追跡 | 廃止＋機能移管（§3） | [D-知覚] |
| TAPE（tape.py） | 事前プラン＋replan | 廃止（[D-反復出力]「1反復1出力」で置換） | [D-反復出力] |
| DecayState／HeartbeatState | tick/減衰 | T-tick／freshness（時刻基準） | [D-周期]／[D-活性] |
| desires.py（DEFAULT_DESIRES） | 旧15欲求 | D（5欲求 drive） | §5・[D-発火] |
3. 廃止・移管した5件（未マップ store の確定）

tape：廃止。事前の多段アクションプラン＋ループ中の block 判定・replan は、新ループ（1反復1出力・ターン内多段なし）の前提と衝突する。計画性は open 意図（O・想起で再会）＋フルLLM の行動組み立て＋調停の drive-serving 選択で代替し、専用プランレイヤは新設しない（[D-反復出力]）。
memory_links：廃止。連想は vector 関連（r）へ一元化、明示したい関係は content に書く。MI 間の構造関係は supersedes（版履歴）のみ。旧実装は読み側（辿り取得）が未結線で、廃止しても現挙動は不変。連想の拡散は WR からの想起で代替（[D-WR拡散想起]）。
exploration_state：廃止＋機能移管。探索履歴・未探索ヒント＝③見た定点の印（O の MI・activation on-read 減衰・最も薄れた定点を選ぶ）、novelty＝取込時算出、見回りの動機＝SEEKING、警戒＝[D-行動選択]、カメラ位置＝SS／DIF、カメラ読み＝[D-知覚]。テーブルは旧実装で未結線。
self_narrative_log：廃止＋移管。I が体験・対話・行動したことだけを O に書くので、その日の O はすべて自己の体験（自己エピソードでない O は存在しない）。よって person_id で自己を絞らず、REST 内省でその日の O を日付で読み返し、フルLLM が一人称の自己エピソード要約へ蒸留して 自己認識 MI の「自己エピソード部分」を supersede 更新（自己認識 MI＝能力＋方針＋自己エピソード部分・pinned）。meta_monitor の自己一貫性チェックも REST 内省へ（[D-在席相関]）。
relationship_state：廃止＋移管。関係内容（傾向・好み・境界・履歴・evidence）＝O の MI（相手の person_id・相関サブテーブルで在席時に想起）。trust／intimacy は専用スカラを持たず、在席者相関＋感情想起で W に集まる関係記憶から評価器/フルLLM が都度導出。social ゲート（言及可否・関係記憶想起・積極度）＝[D-値踏み]・配信ゲート・自己認識 MI policy。REST が per-person の関係サマリを蒸留（自己エピソード部分と同型）（[D-在席相関]）。

4. 旧フィールド・旧 kind の扱い

旧フィールド廃止：state_type／source／status／actionable_when／target／persist／pose／meta／urgency／novelty。MI は kind を持たず、意味は content に置き LLM が解釈する（[D-MIモデル]）。
旧 kind → 新所属・表現：MIデータモデルの移行早見表（付録A）に従う。分類は格納先でなく content の解釈で表す。
person_id 保持メモ（[D-在席相関/V2] で更新）：**`observations.person_id` は削除**し、person と MI の結びつきは situated だけが担う（既存データは所有者 person を写像で situated へ移す）。situated は「MI×person の型つき関係エッジ」へ精緻化＝`(obs_id, person_id)` に複数行を許し（`UNIQUE` 撤去）、在席関係／会話主体など複数関係が並ぶ。関係の種別は vector で表し（open-vocabulary）、帳簿用 `relation_key` TEXT を1列持つ（検索に使わない）。分離が難しい関係は内容を混ぜない独立 vector 行で「関係だけ」を引ける。所有者フィルタは廃し、p 軸（在席者相関・自分除外）は在席関係の行を使う（[D-在席相関]）。

5. 旧感情系・旧欲求 → 新（PAD・5欲求）

旧感情系 → PAD：AffectiveState/mental_state の感情を M（PAD）／MI の emotion(PAD) へ。emotion 文字列→PAD の写像（畳み込み関数 φ）は課題11(k)。全軸 [0,1]・中立0.5・両側（双極 P を P/Pn に分離）。
旧15欲求 → 5欲求：SEEKING／REST／BOND／SAFETY／ESTEEM（PAD 4軸とマズロー5段階に対応）。DEFAULT_DESIRES 全15件の集約マッピングは確定済み（6-2）。ESTEEM は旧システムに出所がない新規追加軸（移行でなく新設計の gap）。蓄積レート等の値は課題5 B。
失敗（agency_error）→ 情動：I の activation には入れず T 側の mood 変調（失敗→mood Pn↑/Dom↓→ESTEEM 変調・間接経路）。経路は確定、PAD 写像値は課題11(k) 据え置き。

6. 課題8 への申し送り（実装時）

テーブル撤去：memory_links／exploration_state／self_narrative_log／relationship_state、および関連クラス（DefaultModeProcessor／ExplorationTracker／SelfNarrative／RelationshipTracker／TAPE）の撤去・置換。
新規：WRDB（[D-WR拡散想起]）のマイグレーションとテスト。
相関サブテーブル＝situated_embeddings を型つき関係エッジへ（[D-在席相関/V2]）：`relation_key` 列追加・`UNIQUE(obs_id, person_id)` 撤去・関係は vector（open-vocabulary）・独立 vector 行の許容。関係の初期集合＝`presence`（←`participants_json`）／`speaker`（←`writer_id`）／`subject`（←`subject_id`＋content 抽出）。β 分離可能性は課題7 計測へ。
旧 `_remember` の複製モデル（`scope` speaker/witnessed/scene で観測を人ごとストアへ重複保存・kind `utterance`/`witnessed`/`scene`）を撤去し、単一 O＋situated 関係エッジへ一本化（複数名対応の根本課題への回答・[D-記憶単一化]／[D-在席相関/V2]）。
`observations.person_id` の削除と既存データの situated 写像：既存観測1件を `participants_json`＋`writer_id`＋`subject_id` から複数の関係エッジへ展開（`person_id` は presence/speaker のフォールバックのみ）。所有者絞りの撤去は済（C-1）だが列削除は V2 で行う。
appraisal／social_policy の trust/intimacy 依存を、想起した関係記憶からの評価器導出へ置換。
旧フィールド・旧 kind 参照の撤去は、旧名で grep して残存ゼロを完了条件にする。
DB 更新を伴うため、既存テストの修正要否を検討し、マイグレーション方法をテストに含める。
全体テストは `./scripts/run_tests.sh` で自分（Claude Code）が回す。

---

## 更新履歴

> v0.5：一行に潰れていた「旧 DB テーブル → 新構成」対応表と「旧クラス → 新構成」対応表を Markdown テーブルへ復元（内容は保持）。旧運用の記述「全体テストはユーザー実施」を現行運用へ訂正。
