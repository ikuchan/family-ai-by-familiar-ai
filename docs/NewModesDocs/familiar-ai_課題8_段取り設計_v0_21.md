# familiar-ai 課題8 段取り設計（段階的 TDD 改造の順序と依存）（v0.21）

> v0.21：Phase 2 の締めの分割単位を `モジュール分割設計 v0.1` として別文書に切り出し、本書からは参照する。目標構成は設計のコンポーネント名に合わせた `core/`／`store/`／`loop/`／`io/`／`legacy/`。

> v0.20：**Phase 2 の締めに「境界の切り出しと生存確認の不変条件」を新設**。`agent.py`（4,024行）と `memory.py`（2,593行）を一度に読める大きさへ分解する。2026-07-20 に見つけた3件の不具合（say ターンの永続化漏れ・感情語の飽和・時刻の9時間ずれ）の検討から、関数の長さより**端から端まで誰も確かめていなかったこと**が効いていると判断し、不変条件を先頭に置いた。ループ構造の作り替えは Phase 5 のまま前倒ししない（撤去される構造への投資を避ける）。`memory.py` の分解は v0.6 で Phase 2 へ申し送った宿題の履行、`agent.py` の副作用境界の切り出しは新設。

> v0.19：実機テストの起動コマンドを訂正。`./run.sh` は `--gui` を付けない CLI 起動で、`run_gui` が呼ばれず GUI のアイドルループが回らない（`main.py:459,500`）。普段の運用と同じ `./run-gui.sh`（`uv run familiar --gui`）に改めた。段取り自体は変えていない。

> 確定した新設計を現行コードへ落とす順序と依存を決める文書。**本書は段取り（フェーズ順序・依存・チェックポイント）の設計方針であり、各フェーズの TDD 手順・コードは承認後に別途出す。** 一項目ずつ確認しながら進める原則に従い、各フェーズ末に確認を挟む。

> v0.18：MI 集約段の設計確定（系統A・系統B・situated V2・自己認識 MI）を締めに反映。situated V2 の schema 器＝スライス1（relation_key 列・022）／スライス2（UNIQUE を relation_key 込みへ・023・upsert キー化）を Phase 1 分として実装（生成 presence のみ・挙動不変）。slice-3 以降（関係生成・person_id 削除・旧 `_remember` 撤去）は視点列を埋める知覚（[D-知覚]）依存で Phase 2 へ申し送り。集約全体の更新機構は REST 内省依存で Phase 2 寄り。

> v0.17：C-2（situated の役割整理）を実装済み（設計整理のみ・実行時の挙動不変）に更新。`situated_embeddings` の二役割（1=視点シフト検索・本人／2=在席者相関 p・他者・自分除外）を台帳へ固定し、現行で生きているのは役割1のみ・役割2＝p 軸は 5軸スコアラごと Phase 2、AGENT_SELF situated は自己の中立視点（役割1の自己スコープ）と明記。ソースコードは変更なし。節目③（在席者に応じた想起変化）は p 軸実装後へ保留。次のコードは MI 集約段。

> v0.16：C-1（person_id 所有者絞りの撤去）を実装済みに更新。situated 相関の読み出し層 `_read_observations_by_situated` を新設（第一段・未接続・テスト10件）し、`recall_day_summaries` をその層へ付け替えて所有者絞りを撤去（第二段・母集合が在席者相関へ変わる・戻り値の形は不変・既存 day_summary テスト4件を相関の意味論へ更新）。反証確認で、フォールバック二関数は主 situated 経路が0件のときだけ発火するため同じ situated 相関へ寄せると恒常的に空になると判明し、C-1 対象から外して別課題へ申し送った。次のコードは C-2（situated の役割整理）または MI 集約段。

> v0.15：B-3（PI 構築と PI→MI 拡張）を実装済みに更新。`tif.py` を新設＝`build_primitive`（M と D から PrimitiveMentalItem を構築）と `expand_to_mental`（PI→MI 拡張）の純関数。emotion に MoodPAD、drive に AiDrivers を流用し A-1 器は無変更。実際の発火とループには未接続で外部挙動不変。DB 不使用でマイグレーションなし。新規テスト3件。Nudge と N_PAD と発火接続は後続段。段階B の Phase 1 スライス（B-1／B-2／B-3）が揃い、次のコードは段階C の C-1。
> v0.14：B-2（drive（5欲求）レジスタ）を器のみ実装済みに更新。`drive_register.py` を新設＝5欲求（SEEKING／REST／BOND／SAFETY／ESTEEM）の器 AiDrivers（各軸 [0,1]・静止0.0）、agent_state（state_key drive5）への load_drives／save_drives。器と永続化のみで蓄積と放電と mood 変調（dynamics）は未実装。生きた15欲求 DesireSystem（agent_state["desires"]）にも as_coalition にも未接続で外部挙動不変。新規マイグレーションなし。新規テスト5件。dynamics と PI.drive surface と旧15→新5 移行は後続段。次のコードは B-3（PI 構築と MI 拡張）。
> v0.13：B-1（mood（PAD）レジスタの器）を実装済みに更新。`mood_register.py` を新設＝4軸 PAD の器 MoodPAD、各軸を M_rest=(0.5,0.5,0.5,0.5) へ半減期600秒で収束させる純関数 decay_to_rest、agent_state（state_key mood_pad）への load_mood／save_mood。emotion→PAD 写像 φ（課題11k・未確定）にも既存 mood にも未接続で外部挙動不変。新規マイグレーションなし（agent_state の新キー1つ）。新規テスト7件。次のコードは B-2（drive レジスタ）。
> v0.12：B-1 の平静を全軸0.5中立化へ修正（旧記載 (P=0,Pn=0,A=0,Dom=0.5) → M_rest=(0.5,0.5,0.5,0.5)・課題5 と発火/mood 別紙に整合）。あわせて B-1 を薄い縦切りに明記＝PAD レジスタの器と M_rest への指数収束（HL_M=600秒）と agent_state 永続化だけを先に作り、emotion→PAD 写像 φ（課題11k・写像値 未確定）の接続は後続とする。既存の単一ラベル mood と AffectiveState は撤去せず残す（外部挙動不変）。
> v0.11：A-3 の Phase 1 残務を決定と記述で確定（コード変更なし）。観測列 recall_count・last_recalled_at（017）を調査し、旧 recall の `_compute_final_score` と `_mark_recalled` の中だけで使われることを grep で確認。対応づけを確定＝recall_count は新しさ（t 軸）の若返り回数（半減期倍化）、last_recalled_at は若返りの起点リセットで、activation の n（評価由来）とは別系統・想起回数からは引き継がない。更新トリガは機械想起からフルLLM 参照申告へ移す。二列は旧 recall スコアリングに閉じるため Phase 1 は現状維持（外部挙動不変）とし、再編は 5軸スコアラの Phase 2 で行う。よって A-3 の Phase 1 残務は完了し、次のコードは段階B の B-1。詳細は活性の別紙 v0.2 §4。
> v0.10：A-3 の最初の一歩 A-3-1（活性の (a0,n) 保存形式）を実装済みに更新。マイグレーション 2026-07-02-021 で observations に activation_a0（REAL・既定1.0）と activation_n（INTEGER・既定0）を追加、importance を NULL 防御後に activation_a0 へ簡易移行（importance 列は残す）。memory.py に導出関数 _derive_activation（既定 step=0.33・値は課題5）を新設。導出値は想起スコアへ未接続で recall は従来どおり importance を読むため外部挙動は不変。新規テスト7件（純導出5・マイグレーション DB2）。recall_count・last_recalled_at の役割再編、想起スコアへの接続、n 増減の評価機構は A-3 の残り（Phase 2）へ申し送り。次の一歩は A-3 の残り。
> v0.9：A-2 を調査完了として記録。観測を読み出すコードが返していた属性（memory_id／summary／emotion／timestamp／date／time／direction／kind／source_kind／image_path／importance／superseded_by／検索メタ）を洗い、器（MentalItem）への対応表を作成。当初想定との食い違い（挙げた旧属性名が observations カラムに無い・date/time は 016 で削除済み・A-2 の対象は DB カラムでなく読み出しコードが返す属性）を記録。recall_count／last_recalled_at は A-3 へ、視点4列は段階 C へ申し送り。次の一歩は A-3（activation の (a0,n) 導出）。
> v0.8：A-1 を実装済みに更新（器 `PrimitiveMentalItem`／`MentalItem` と変換関数 `_row_to_mental_item` を導入・`recall_self_model` で器を組み立てる経路を通す・返り値は従来のまま・アクセス層無変更・新規3テスト＋既存115件・全体テスト通過）。次の一歩は A-2（MI 属性最小化の準備＝調査）。
> v0.7：Phase 1 節に詳細ステップ（薄い縦切り A-1〜D-2）を織り込み、**TUI 実機テスト（`./run.sh`）の必要タイミングを各ステップに明文化**（節目①〜⑤の一覧表つき）。読み出し層寄せ4本は段階 A-0（完了）として位置づけ。Phase 1 のことは本文書に一元化（別文書を作らない）。最初の一歩は A-1（observations を MI として読む・スキーマ変更なし）。各ステップは予定であり実装が進むごとに進捗を追記。
> v0.6：Phase 1 の読み出し層寄せを締め（4本目＝`recent_feelings`・層を `str | tuple` へ初拡張・後方互換・全テスト通過）。単純 kind 絞りの読み出しは層に集約。残る複雑な recall（別テーブル・日付演算・複合条件）は層に寄せず Phase 2 で「取り出しメソッド＋5軸スコアラ」へ分解する方針を確定（渡し方＝専用メソッドで構造化値・LIKE は situated 近傍へ）。データモデル本体は未着手。
> v0.5：Phase 1 の3本目を実装済みとして記録（`recall_day_summaries` を層経由へ・在席者スコープ `self._person_id` でも層無変更のまま機能・層の再利用性を3パターンで実証・追加テスト4件＋既存97件非回帰・全体テスト通過）。次は kind 単純絞りでない recall（semantic_facts・on_this_day 等）で層適用の可否を別途検討。
- **進捗【実装済み・4本目＝読み出し層寄せの締め】**：`recent_feelings`（`kind IN ('feeling','conversation')`）を寄せるため、層の kind 引数を **`str | tuple[str,...]`** に拡張（複数値は `kind IN (...)`）。**層自体を変える初の一本**だが、str 分岐 SQL は従来とバイト同一で後方互換（既存3本は無変更・str 経路の回帰テストで保証・RED でタプル経路の失敗＝`operator does not exist: text = record` を確認）。追加テスト6件＋既存15件＋非回帰97件・全体テスト通過。**これで単純な kind 絞りの observations 読み出しは層に集約され、Phase 1 の読み出し層寄せは一区切り**（curiosities／self_model／day_summaries／recent_feelings の4本＋層の単一・複数 kind 対応）。
- **読み出し層寄せの締め＝残る recall の申し送り**：層に寄せないものを理由付きで確定。別テーブルを読む（`recall_semantic_facts`＝semantic_facts／`recall_behavior_policies`＝behavior_policies／`recall_revisions`＝memory_revisions）、日付・時刻演算（`recall_on_this_day`＝EXTRACT MONTH/DAY／時刻近傍）、集約・存在確認・dedup・複合条件（situated 近傍を含む主 recall 系）は、いずれも「kind と person_id で新しい順に読む」形から外れ、検索・絞り込みロジックを含む。これらは**Phase 2 で「層の取り出しメソッド＋5軸スコアラ」に分解**する。渡し方の骨格（確定）：層は生 SQL/WHERE/LIKE を受けず、取り出しパターンごとの専用メソッド（`by_kind` 実装済み・将来 `by_vector`＝situated 近傍・`by_date` 等）で**構造化値**を受ける。LIKE 検索は O 一元化後に situated 近傍（r 軸採点）へ置き換わる見込みで、専用メソッドを残すかは Phase 2 で再考。データモデル本体は未着手。
> v0.4：Phase 1 の次の一本を実装済みとして記録（`recall_self_model` を層経由へ・emotion 込み経路で層の再利用性を実証・層コード無変更・追加テスト4件＋既存97件非回帰・全体テスト通過）。次の一本は `recall_day_summaries` 等。データモデル本体は未着手。
> v0.3：Phase 1 の最初の一本を実装済みとして記録（読み出しアクセス層の実体化＝`_read_observations_by_kind` 新設・`recall_curiosities` を層経由へ・スキーマ変更なし・外部挙動不変・全テスト pass）。次の一本は emotion 込みの `recall_self_model` の層寄せ。データモデル本体は未着手。
> v0.2：未決4点を確定（Phase 1 から着手・薄い縦切り・スコアはフラグ排他切替・min_score は Phase 2 後）。Phase 0 は実機完了。

## 1. 依存の事実（コード根拠）

- **スコアは1関数に集約**：`final_score = cosine × time_score × importance`（memory.py:70/:86）。ここが新5軸への切替点。
- **recall 系統は多数**：主 `recall`（:761）に加え、`recall_self_model`／`recall_curiosities`／`recall_semantic_facts`／`recall_behavior_policies`／`recall_day_summaries`／`recall_divergent`／`recall_on_this_day`／`recall_revisions` 等（同期・async 併せ十数個）。**1経路へ統合**が要る。
- **旧系統の規模**（gap 文書の段階移行対象・ファイル数）：workspace 15／prediction 8／interoception 7／social_policy 3／appraisal 3／concern_engine 3／attention_schema 1。
- **廃止/移管ストアの結線数**：self_narrative 15／tape 11／pending_store 11／GlobalWorkspace 6／relationship_state 4／memory_links 3／exploration_state 0（既に未結線）。
- **現行先行作業**：BUG-1（冪等化＋purge・作業中）、bge-m3 移行（チケット v0.2）。新アーキとは独立で、再測定の前提。

## 2. 段取りの原則

- **recall 一本化が起点**：旧 cosine 純積と新5軸を**並走させない**。新スコアラへはクリーンに切り替える（フラグで旧/新を持つにしても、同時に二重採点しない）。
- **現行先行（Phase 0）は新アーキと独立**に進む（BUG-1・bge-m3）。再測定はこの後。
- **基盤（O/MI データモデル）が先、recall 一本化が次**：5軸は O の上で動くため、O モデルが無いと意味を持たない。
- **旧系統の撤去は最後**：新経路が通って実証されてから、grep 残存ゼロを完了条件に撤去する。
- **DB 変更はマイグレーション＋テスト＋ロールバック**を各フェーズに含める。カラム・ストア廃止は「旧名 grep で0件」を完了条件にする。
- **一項目ずつ**：各フェーズはさらに小ステップに割り、ステップ毎にユーザー確認。全体テストはユーザー実施。

## 3. フェーズ順序（提案）

### Phase 0：現行先行（並行・Claude Code 進行中）
- 内容：BUG-1（`save_async_with_id` 冪等化＋既存重複 purge）、bge-m3 移行（EMBEDDING_MODEL/DIM・e5 プレフィックス除去・situated 列 384→1024・全再埋め込み・mu 再計算）。
- 依存：相互順序＝BUG-1 完了 → bge-m3 移行。新アーキ各フェーズの前提ではないが、**再測定（c_lo/c_hi）の前提**。
- 完了条件：重複検索0件／全 situated 1024次元／旧モデル名・旧次元の残存ゼロ。
- チェックポイント：再測定を回せる状態。

### Phase 1：基盤＝O / MI データモデル（最大・最重要）
- 内容：O 一元化（observations → O・全 O＝自己体験）、MI 構造と自己認識 MI、T レジスタ、相関サブテーブル（person 別・situated 先行形を活用）、W 派生ビュー（退避/eviction/fade なし）。スキーマ＋マイグレーション中心。**recall は旧のまま温存**（まだ切替えない）。
- 依存：Phase 0 の bge-m3（O の埋め込みは bge-m3）。
- テスト観点：O への移行で既存観測が欠落・重複しない／T レジスタの読み書き／相関サブテーブルの person 別想起。
- チェックポイント：旧 recall が新 O 上でも従来どおり動く（回帰）。

#### Phase 1 の詳細ステップ（薄い縦切りの計画）

大枠の順序＝**MI 構造 → T レジスタ → 相関サブテーブル → O 一元化と撤去**（MI 構造が土台。O 一元化も PI 化も相関も「MI とは何か」が決まらないと移せない）。各ステップは1つの薄い縦切り＝pytest で通る最小の一本とし、「調査 → 設計方針（原因と対応案）→ 承認 → TDD 改造 → pytest 確認 → ユーザーの全体テスト」で進める。**未実装のステップは予定**であり、実装が済んだら各ステップに進捗を追記する。

**実機テスト（`./run-gui.sh`＝`uv run familiar --gui`・普段の運用と同じ GUI で対話）は、下表の「実機」印のステップ後に必ず挟む。** pytest が「壊れていない」を、実機が「自然に振る舞う」を担保する二重確認。全ステップでなく、記憶・感情・想起の体感が変わる節目に絞る。試すシナリオは各ステップの設計方針を詰める段で具体化する（下表は暫定案・ユースケース②〜⑥も素材に使える）。

段階A（MI 構造の確立）：
- **A-0【完了】読み出しアクセス層寄せ**＝下の実装済み進捗（最初の一本〜4本目）。`_read_observations_by_kind` に4本を寄せ、単一・複数 kind に対応済み。段階A の接続点。
- **A-1【実装済み・最初の一歩】observations を器（MentalItem）として読む**（スキーマ変更なし）。基底クラス `PrimitiveMentalItem`（emotion／drive・A-1では未設定）と拡張クラス `MentalItem`（＋id／content／vector／supersedes／activation）を `memory.py` に導入。観測行から器を組み立てる変換関数 `_row_to_mental_item` を新設。`recall_self_model` の取得列に id／superseded_by／importance を加え、器を組み立てる経路を通す。**器は返り値には使わず、返り値の辞書は従来どおり**（外部挙動を保つ・器の利用は次の一本に分ける）。emotion・drive・vector は未設定（None）で、感情のPAD化・埋め込み取り込みは後続の段階。アクセス層 `_read_observations_by_kind` は無変更（取得列を増やしただけ・本体は git diff で無差分を確認）。テスト（`tests/test_mental_item.py` 新規3件）：変換関数の組み立て／継承関係／`recall_self_model` の外部挙動保存。既存関連テスト非破壊（115 passed）・全体テスト通過。**実機テスト：不要**（内部表現のみ・体感不変）。
- **A-2【調査完了】MI 属性最小化の準備**。器の属性を最小化する段階 B に備え、既存コードが観測を読み出すときに呼び出し側へ返していた属性を洗い、それぞれが新しい器（MentalItem）のどの属性に対応するかの対応表を作る調査。この段では廃止しない。実機テストは不要（調査のみ）。

  **当初想定と実環境の食い違い（記録）**：着手前に廃止候補として挙げていた名前（state_type・source・status・target・persist・pose・meta・urgency・novelty）は、observations テーブルのカラムには存在しなかった。これらは設計 [D-MIモデル] が廃止と定めた旧属性の名だが、現行の observations の実カラムではない。また当初 A-2 の対象に含めていた date・time カラムは、マイグレーション 016 が timestamp を TIMESTAMPTZ へ正規化した際に既に削除済みで、本番データベースに存在しない。A-2 の対象は「観測を読み出すコードが返していた属性」であり、DB のカラムそのものではない。この取り違えを記録に残す。

  **観測を読み出すコードが返していた属性と、器（MentalItem）への対応表**（対象は observations を読む経路のみ。semantic_facts・behavior_policies・完了キュー等の別テーブル経路は含めない）：

  | 返り値の属性 | 由来（memory.py） | 器への対応 |
  |---|---|---|
  | memory_id／id | row["id"] | 器 `id` |
  | summary | row["content"] | 器 `content` |
  | emotion | row["emotion"] | 器 `emotion`（現状は感情ラベル文字列。PAD 化は段階 B・[D-MIモデル]） |
  | timestamp | row["timestamp"] | 器の属性ではなく store メタデータ。ここから date・time を導出 |
  | date | `_ts_to_date(timestamp)` | 器に持たない。timestamp からの派生（DB でも 016 で削除済み） |
  | time | `_ts_to_time(timestamp)` | 器に持たない。timestamp からの派生（同上） |
  | direction | row["direction"] | 器に写さない。kind の言い換え（本番で kind と行ごとに一対一を確認） |
  | kind | row["kind"] | 器は kind を持たない設計（意味は content へ・[D-MIモデル]）。ただし読み出しの絞り込みに現用のため当面残す（整理は段階 B 以降） |
  | source_kind | row["kind"] | kind の重複（同じ row["kind"] を別キーで再掲）。器に持たない |
  | image_path | row["image_path"] | 器に写さない。将来の顔認識・画像想起（DIF 知覚）まで休眠。廃止の実装はその仕組みができてから（本番で 2624 件中 1 件のみ値を持つ） |
  | activation の素 | row["importance"] | 器 `activation`（現 importance の一般化）。(a0,n) 導出は A-3 |
  | supersedes の素 | row["superseded_by"] | 器 `supersedes` |
  | score／confidence／retrieval_method | 検索経路が算出 | 記憶そのものの属性でなく想起（5軸スコアリング）の産物。器に持たず Phase 2 のスコアラが付す |

  vector（器の属性）は、この読み出し経路では返していない（埋め込みは別テーブル situated_embeddings。器への取り込みは後続の段階）。drive（基底 PI の属性）も観測の読み出しにはなく、T レジスタ由来で段階 B の PI 構築で入る。

  **後続段階への申し送り**：recall_count・last_recalled_at（017 で追加）は activation の導出と新しさ減衰を担うため A-3（activation の (a0,n) 導出）で役割を再編する。writer_id・subject_id・participants_json・scope（012 で追加した視点列。読み出しの絞り込みには未使用）は段階 C（person_id 所有者絞りの撤去と situated 相関整理）で相関サブテーブルへ整理する。
- **A-3【一部実装済み】activation の (a0,n) 導出**。importance を activation へ一般化し、(a0,n) 保存・on-read 導出の器を作る（値の確定は課題5）。
  - **A-3-1【実装済み・A-3 の最初の一歩】(a0,n) 保存形式と導出関数**。マイグレーション `2026-07-02-021_activation_a0_n.py` で observations に `activation_a0`（REAL NOT NULL・既定1.0）と `activation_n`（INTEGER NOT NULL・既定0）を追加。既存行は `importance` を NULL 防御（1.0）してから `activation_a0 = importance` で簡易移行（`importance` 列は削除せず残す）。`memory.py` に導出関数 `_derive_activation(a0, n, *, floor=0.0, c=2.0, epsilon=0.001, step=0.33)` を新設（a0 を [floor,c] で正規化しロジットで無限区間へ写し、n·step を足してロジスティックで [floor,c] へ戻す・n=0 で a=a0）。**導出値は想起スコアへ未接続**で、`recall` は従来どおり `cosine × time_score × importance` で `importance` を読む（`_derive_activation` の呼び出しは定義とテストのみで稼働経路ゼロ）。よって**外部挙動は不変**。step の既定は課題5 で確定した 0.33（取込 a0=0.75 から評価5回で実用上限1.5 に達する効き幅・キーワード引数で差し替え可・Config 配線は後続）。テスト（`tests/test_activation_derive.py` 新規7件）：純導出5件（n=0 恒等・n に単調・両端 floor/C へ漸近・±1 対称・a0=0.75 で5回1.5到達）とマイグレーション DB2件（列追加確認・importance→activation_a0 移行確認）。**実機テスト：不要**（未接続で体感不変）。
  - **A-3 の Phase 1 残務【記述で完了・コード変更なし】recall_count・last_recalled_at の役割再編（対応づけの確定）**。A-2 で申し送った二列（017 で追加）を調査し、旧 recall の `_compute_final_score`（`final_score = cosine × time_score × importance`）と `_mark_recalled` の中だけで使われることを grep で確認（機械想起 conversation で `recall_count += 1` と `last_recalled_at = now()`、spontaneous で `last_recalled_at` のみ更新・system は無更新）。確定した対応づけ：`recall_count` は新しさ（t 軸）の若返り回数（`time_decay` の reinforce＝半減期倍化）、`last_recalled_at` は若返りの起点リセットに対応する。**activation の `n`（評価由来の正味デルタ）とは別系統で、想起回数からは引き継がない**。更新トリガは機械想起からフルLLM の参照申告へ移す（[D-想起合成]「機械想起では activation も freshness も触らない」）。二列は旧 recall スコアリングに閉じているため、Phase 1 では現状維持（旧 recall が読む・外部挙動不変）とし、コード再編は 5軸スコアラを載せる Phase 2 で行う。詳細は活性の別紙 v0.2 §4。
  - **A-3 の残り【予定・Phase 2】**：導出値を想起スコアへ接続（Phase 2 の5軸スコアラで importance の代わりに導出 a を使う）、段2 の評価による `n` の増減機構、参照した MI の新しさ（t 軸）の若返り、解決時の a0 再測定。上の recall_count・last_recalled_at の再編コードもここに含む。テスト観点：(a0,n) からの導出が単調・可逆／既存 importance 読み出しと互換。**実機テスト：必要（節目①）**＝記憶を問う対話（ユースケース⑥）で、想起される記憶が従来と大きくずれないか体感確認（導出が想起へ効く段で行う）。

段階B（T レジスタの分離）：
- **B-1【実装済み・薄い縦切り】mood（PAD）レジスタの器**。`mood_register.py` を新設し、4軸 PAD の器 `MoodPAD`（P／Pn／A／Dom・全軸 [0,1]・中立0.5・`clipped`／`to_json_dict`／`from_json_dict`）、各軸を平静 M_rest=(0.5,0.5,0.5,0.5) へ半減期 HL_M=600秒（課題5 確定）で収束させる純関数 `decay_to_rest`（`rest+(x−rest)·2^(−経過/HL_M)`・DB 非依存）、agent_state への読み書き `load_mood`／`save_mood`（state_key `mood_pad`・self_state と同じ upsert・行が無ければ中立を返す）を置いた。収束先は各軸0.5の中点で、time_decay の floor 減衰とは別式。**emotion 文字列→PAD 写像 φ（課題11k・写像値 未確定）にも既存 mood にも未接続**＝agent.py の `_mood`／`_decayed_mood` と AffectiveState と mental_state_log は無変更で外部挙動不変（撤去は Phase 6）。φ 接続（旧データ移行と実行時 PAD 設定・評価器 LLM が P/Pn/Dom 直接出力・A は機械 A）は課題11k 確定後。新規マイグレーションなし（agent_state の新キー1つ・DDL なし）。テスト：`tests/test_mood_register.py` 新規7件（収束を上からと下からの両側・両軸独立漸近・範囲クリップ・agent_state 保存/読み出し往復・キー不在時の中立既定）。**実機テスト：不要**（未接続で体感不変・B-2 とまとめて節目②で確認）。
- **B-2【一部実装済み・薄い縦切り・器のみ】drive（5欲求）レジスタの器**。`drive_register.py` を新設し、5欲求（SEEKING／REST／BOND／SAFETY／ESTEEM・集約は課題6-2 確定）の器 `AiDrivers`（各軸 float・[0,1]・静止（既定）0.0・`clipped`／`to_json_dict`／`from_json_dict`）、agent_state への読み書き `load_drives`／`save_drives`（state_key `drive5`・self_state と同じ upsert・行が無ければ全0.0 を返す）を置いた。**器と永続化だけ**で、蓄積と放電と mood 変調（dynamics）は作っていない。**既存の生きた15欲求 `DesireSystem`（agent_state["desires"]）にも `as_coalition` 経路にも未接続**＝desires.py と既存の "desires" キーは無変更で外部挙動不変（撤去は Phase 6）。旧15→新5 のデータ移行はしない（マッピングは課題6-2 で記録済み・移行は挙動切替の段）。新規マイグレーションなし（agent_state の新キー1つ・DDL なし）。テスト：`tests/test_drive_register.py` 新規5件（静止既定＝全0.0・範囲クリップ・JSON 往復・agent_state 保存/読み出し往復・キー不在時の全0.0 既定）。**実機テスト：不要**（未接続で体感不変）。
  - **B-2 の残り【予定・後続段】**：蓄積 dynamics（[D-活性] の `drive_i ← clip(drive_i + rate·mult·learn·g_{D,i}(M)·P_T, 0, 1)`・共通レートとバイアス b_i と変調行列 C_{ij} は課題5 B と発火・mood 別紙の暫定値）、全放電（発火時 −放電量）、mood M による変調 g_{D,i}(M)、PI.drive への surface、旧15→新5 のデータ移行と挙動切替。dynamics は mood レジスタ（B-1）の値と発火に依存するため後続。**実機テスト：必要（節目②）**＝数回の対話で、感情が自然に変化し平静へ戻るか・欲求が極端に振れないか体感確認（dynamics と surface が効く段で行う）。
- **B-3【実装済み・薄い縦切り】PI 構築と PI→MI 拡張（構築関数のみ）**。`tif.py` を新設し、`build_primitive(emotion: MoodPAD, drive: AiDrivers) -> PrimitiveMentalItem`（発火ペイロード構築・emotion←M・drive←D）と `expand_to_mental(pi, *, id, content, vector=None, supersedes=None, activation=None) -> MentalItem`（I 取り込みでの PI→MI 拡張・pi の emotion と drive を引き継ぎ id 等を付与）の二つの純関数を置いた。emotion に B-1 の `MoodPAD`、drive に B-2 の `AiDrivers` を値として流用し、A-1 の器（`PrimitiveMentalItem`／`MentalItem`／`_row_to_mental_item`）は無変更（注釈 `object | None` のまま）。**実際の発火とループには未接続**で、両関数は稼働経路から呼ばれない＝外部挙動不変。DB 不使用の純関数（マイグレーションなし）。テスト：`tests/test_tif.py` 新規3件（`build_primitive` が M と D を運ぶ・`expand_to_mental` の引き継ぎと付与・`MentalItem` が `PrimitiveMentalItem` のサブクラス）。**実機テスト：不要**（未接続で体感不変）。
  - **B-3 の残り【予定・後続段】**：I→T Nudge を MI として発し T が emotion をフィルタ（[D-T境界]）、N_PAD（W の MI の emotion を activation 重みで合成）、発火とループへの `build_primitive`／`expand_to_mental` の接続。いずれも想起（W・Phase 2）と MI.emotion の PAD 化（課題11k）と発火に依存するため後続。

段階C（相関サブテーブルの整理）：
- **C-1【実装済み】person_id 所有者絞りの撤去**。代替の相関経路を先に作り、そのあと所有者絞りを外す二段で進めた。第一段：situated 相関の読み出し層 `_read_observations_by_situated(person_id, n, columns, *, kind=None, keywords=())` を新設（`situated_embeddings s` を `observations o` に JOIN し `s.person_id` で紐づける・所有者に依らない母集合・timestamp DESC・ベクトル類似度は使わない・未接続・テスト10件）。第二段：`recall_day_summaries` をこの層へ付け替え、`observations.person_id` 所有者絞りを撤去（母集合が在席者相関へ変わる・戻り値の形は不変・既存 day_summary テスト4件を相関の意味論へ更新）。フォールバック二関数 `_recall_keyword_fallback`／`_recall_recency_fallback` は C-1 対象から外した（主 situated 経路が0件のときだけ発火し、その0件は「その person の situated 行が無い」ときに限るため、同じ situated 相関へ寄せると恒常的に空になる。所有者絞りのまま「situated 行を持たない観測」を拾う役目を残す）。「situated 行を持たない観測のフォールバック扱い」は別課題へ申し送り。完了条件（達成）：`recall_day_summaries` から `observations.person_id` 所有者絞りが消えた（書き込み側 `delete_day_summaries_for_date` とフォールバック二関数は対象外）。**実機テスト：不要**（C-2 とまとめて節目③）。
- **C-2【実装済み・設計整理のみ・実行時の挙動不変】situated の役割整理**。situated_embeddings（obs_id, person_id, vector）を MI×person 相関の先行形として、二役割で整理・固定した。役割1＝視点シフト検索（`s.person_id=問う人`・本人・その視点に寄せた母集合とスコア）、役割2＝在席者相関 p（W 想起の第5軸・在席**他者**・**自分除外**・noisy-OR・[D-在席相関]）。**現行コードで生きているのは役割1のみ**（主 `recall` のコサイン検索と C-1 の `_read_observations_by_situated`）。**役割2（p 軸のスコアリング）は 5軸スコアラごと Phase 2**（現 `_compute_final_score` は p を含まない）。AGENT_SELF の situated 行は自己の中立視点（役割1の自己スコープ・`perspective_vec` 不在なら素の記憶ベクトル）で、p 軸の自分除外とは別物。ソースコードは変更せず、位置づけを「store と I/F」台帳と本節に固定した。**実機テスト：p 軸が入る Phase 2 まで保留**（現行は話し手ひとりの視点シフトのみで、在席者に応じた想起変化は p 軸実装後にしか観測できない。旧記載の節目③はこの保留に置き換え）。

段階D（O 一元化の残りと撤去・最後）：
- **D-1【予定】旧テーブルの O 統合**。episodes・episode_memories・memory_events・pending_speech・unfinished_business 等を O へ統合、semantic_facts・behavior_policies は O 外の昇格処理へ。1テーブルずつ薄く。テスト観点：統合で既存データが欠落・重複しない（移行の要）。マイグレーション＋ロールバック。**実機テスト：必要（節目④・テーブルごと）**＝統合したテーブルに関わるシナリオ（例：記憶を問う対話・②〜⑤の該当機能）で、統合前と挙動が変わらないか確認。
- **D-2【予定】旧テーブル・旧クラスの撤去**。memory_links・exploration_state・self_narrative_log・relationship_state と関連クラス（DefaultModeProcessor・ExplorationTracker・SelfNarrative・RelationshipTracker・TAPE 等）を撤去。**新経路が通って実証されてから**。完了条件：旧名・旧クラスが grep で0件。**実機テスト：必要（節目⑤・Phase 1 総合）**＝主要シナリオを一通り試し、Phase 1 前と体感が劣化していないか総合確認。

**TUI 実機チェックポイント一覧（節目のみ）**：

| 節目 | 直前ステップ | 試すシナリオ（暫定案） | 見るポイント |
|---|---|---|---|
| ① | A-3 導出の想起接続（A-3-1 は未接続で対象外） | 記憶を問う対話（UC⑥） | 想起される記憶が従来と大きくずれないか |
| ② | B-2（感情・欲求が動く） | 数回の対話 | 感情が自然に変化し平静へ戻るか・欲求が極端に振れないか |
| ③ | Phase 2 の p 軸（在席者相関） | 在席者を想定した対話 | 想起が在席者に応じて変わるか（C-2 は設計整理のみで役割2＝p 軸未実装のため、この観測は p 軸実装後へ保留） |
| ④ | D-1（O 統合・テーブルごと） | 統合テーブル該当シナリオ | 統合前と挙動が変わらないか |
| ⑤ | D-2（撤去・Phase 1 総合） | 主要シナリオ一通り | Phase 1 前と体感が劣化していないか |

**A-1・A-2 完了、A-3-1 実装済み、A-3 の Phase 1 残務も記述で完了、B-1・B-2・B-3 も実装済み**（段階B の Phase 1 スライス完了＝B-1 `mood_register.py`／B-2 `drive_register.py`（`AiDrivers`）／B-3 `tif.py`（`build_primitive`・`expand_to_mental`）・いずれも器または構築関数で未接続）。A-3 の残りは Phase 2、B-2 の残り（蓄積 dynamics ほか）と B-3 の残り（Nudge・N_PAD・発火接続）は後続段。**C-1（person_id 所有者絞りの撤去）実装済み**（`_read_observations_by_situated` 新設＋`recall_day_summaries` 付け替え・フォールバック二関数は別課題へ申し送り）。**C-2（situated の役割整理）実装済み・設計整理のみ**（二役割を台帳へ固定・役割2＝p 軸は Phase 2・コード変更なし・実機テストは Phase 2 まで保留）。**MI 集約段の設計は一通り確定**（系統A＝self_model→自己認識 MI 自己エピソード部／curiosity→cue O、系統B＝キーレス supersede＋content 注記の信念 MI、situated V2＝型つき関係エッジ、自己認識 MI＝核/Config/自己エピソード/policy＋プロンプトキャッシュ整合の構築規約・MIデータモデル §7／[D-在席相関/V2]／[D-自己認識分離]）。**いずれも更新機構が REST 内省に依存するため実装本体は Phase 2 寄り**（REST 詳細＝周期・閾値は課題10）。**Phase 1 で入ったのは situated V2 の schema 器のみ**＝スライス1（`relation_key` 列・2026-07-12-022）とスライス2（UNIQUE を relation_key 込みへ・2026-07-12-023・`_upsert` キー化）で、いずれも生成 presence のみ・挙動不変。**slice-3 以降（視点列から presence/speaker/subject の関係生成・person_id 削除・旧 `_remember` 撤去）は、書き込みが視点列を実質埋めていない＝在席検出・話者帰属（[D-知覚]）が入る Phase 2 に依存するため申し送り**。着手時は実環境の最新ソースを確認してから、調査 → 設計方針 → 承認 → TDD 改造の順で進める。

#### Phase 1 実装済み進捗（読み出し層寄せ＝段階 A-0）

- **進捗【実装済み・最初の一本】**：薄い縦切りの起点として、読み出しアクセス層の最初の実体を実装。`memory.py` に `_read_observations_by_kind(kind, person_id, n, columns)` を新設（observations を kind／person_id で絞り新しい順に n 件読む機械的読み出し・採点や想起判断は持たない＝課題13c の dumb な層）し、最も単純な `recall_curiosities` をこの層経由へ付け替え（生 SQL 除去・外部挙動不変・スキーマ変更なし）。新規テスト7件＋既存110件 pass・全体テスト通過。次の一本＝emotion を含む `recall_self_model` を同じ層へ寄せ、`columns` が emotion 込み経路でも機能するか確認（layer の再利用性を1本広げる）。データモデル本体（O 一元化・T レジスタ・相関サブテーブル・emotion の PAD 化）は未着手。
- **進捗【実装済み・次の一本】**：`recall_self_model`（emotion を返す）を `_read_observations_by_kind` 経由へ付け替え。`columns=("content","timestamp","emotion")` を渡すだけで**層のコードは無変更**（git diff で層ゼロ差分を確認）＝emotion 込み経路で層の再利用性を実証。外部挙動不変・スキーマ変更なし。追加テスト4件（emotion の distinct 値が変わらず返ることを含む・実スキーマは `emotion TEXT NOT NULL DEFAULT 'neutral'` のため NULL ケースは検証対象外と判断）＋既存97件 非回帰・全体テスト通過。次の一本は `recall_day_summaries` 等の残る単純 recall。
- **進捗【実装済み・3本目】**：`recall_day_summaries` を層経由へ付け替え。person_id が固定の AGENT_SELF_ID でなく **`self._person_id`（在席者スコープ・実行時に決まる値）**でも**層のコードは無変更**（git diff で層ゼロ差分）＝層の再利用性を「AGENT_SELF_ID 固定・emotion 込み・在席者スコープ」の3パターンで実証。外部挙動不変・スキーマ変更なし。追加テスト4件（在席者スコープの分離テストを含む・person_id は persons への FK があるため既存の予約 ID＝DEFAULT_PERSON_ID/AGENT_SELF_ID を使用）＋既存97件 非回帰・全体テスト通過。次の一本は `recall_semantic_facts`・`recall_on_this_day` 等、kind 単純絞りでない recall（クエリ検索・日付演算を含む）で、層をそのまま使えるか別途検討が要る。

### Phase 2：recall 一本化＝5軸スコアラ（起点の切替）
- 内容：旧 cosine 純積（:86）→ 新5軸（埋め込み平均中心化 → r 段階的関連係数〔ハード veto なし〕／t 新しさ／e 感情一致／a activation／p 在席者相関 noisy-OR／加重平均 M／min_score 足切り）。recall_* 群を1経路へ統合。**旧スコアと並走させずクリーン切替**。
- 依存：Phase 1（O モデル）＋ Phase 0（bge-m3 ベクトル）。p の在席入力は Phase 3 まで暫定（在席ゼロ扱い可）。
- テスト観点：5軸スコアの単調性・min_score 足切り・r 段階化（関連を veto しない）・noisy-OR の在席ゼロ時の分母処理・recall_* 統合後に各用途が従来近い結果を返す。
- チェックポイント：**ここで再測定の min_score を確定**（c_lo/c_hi は Phase 0 後に測れるが、min_score は5軸実装後）。

#### Phase 2 の締め：境界の切り出しと生存確認の不変条件〔決定・未着手〕

Phase 2 を閉じる前に、`agent.py`（4,024行）と `tools/memory.py`（2,593行）を、**一度に読める大きさ**へ分解する。目的は二つある。バグを減らすことと、人と LLM の双方が全体を把握できる状態に戻すことである。

**この段でやること／やらないこと**

- **やる**：副作用の境界を切り出す（永続化・時刻・発話・ストアアクセス）。これらは新ループになっても残る概念なので、Phase 5 で捨てる投資にならない。
- **やる**：端から端まで通る少数の不変条件をテストに置く。
- **やらない**：ターン内イテレーションの入れ子（`run()` の制御構造）の作り替え。[D-反復出力] で撤去が決まっており、撤去される構造へ投資しない（Phase 5 のまま）。

**根拠**

2026-07-20 に3件の不具合を見つけた。`say()` で話したターンが永続化されない（`run()` 795行の約590行目に埋もれた条件式）、感情語の一致を辞書サイズで割っていて値踏みゲートを越えない（`appraisal.py`）、観測 `timestamp` が9時間ずれる（`memory.py` の1行）。関数の長さが直接の原因なのは1件目だけで、3件に共通するのは**誰も端から端まで確かめていなかったこと**である。最後の書き込みが 2026-06-29 で、3週間気づかなかった。よって不変条件を先に置く。

置き場所を Phase 2 の締めにするのは、Phase 3 と Phase 4 が同じ2ファイルへ積み増すためである。Phase 3 は話者帰属を書き込み経路（`memory.py`）へ供給し、在席ゼロ時の発話ゲート（`agent.py` の `run()`）に触る。Phase 4 は `say()` の呼び出し口に触る。分解しないまま積むと同じ失敗を繰り返す。

なお `memory.py` の分解は新しい作業ではなく、**v0.6 で Phase 2 へ申し送った宿題の履行**である（「残る複雑な recall は Phase 2 で『取り出しメソッド＋5軸スコアラ』へ分解」）。`agent.py` の副作用境界の切り出しは、どのフェーズにも割り当てが無かった作業として、ここに新設する。

**分割の単位**：`モジュール分割設計 v0.1` に切り出した。判断の基準は「設計が定めたコンポーネント（[D-I内部]／[D-周期]／[D-B分離]）に名前を合わせる」と「変わりそうな判断を隠す単位で切る（Parnas 1972）」の二つで、依存は内側へ向ける（Ports and Adapters）。目標構成は `core/`／`store/`／`loop/`／`io/`／`legacy/`。撤去予定を `legacy/` へ隔離しておくことで、Phase 6 の撤去をディレクトリの削除に近づける。

**順序**

1. **生存確認の不変条件**。分解の安全網。これ無しに2ファイルを切り刻むのは、壊れても分からない状態で大工事をすることになる。粒度は着手時に詰める。
2. **`store/` の切り出し**。O とアクセス層（`by_vector`／`by_date` 等の専用メソッド・構造化値で受ける）、および時刻の一元化（`clock.py`）。
3. **`core/recall_score.py`**（W 構築の5軸スコアラ）→ **`loop/evaluator.py`**（設計に名前のある実体）→ **`loop/persistence.py`**（永続化の呼び出し口）→ **`legacy/` への隔離**。
4. **`min_score` の是正**（生コサインの閾値 → 合成スコアの床）。Phase 2 のチェックポイント。境界が整ってから行う。

**実機テスト**：必要。分解の前後で体感が変わらないことを確認する（挙動不変が完了条件）。

### Phase 3：知覚入力
- 内容：在席（YOLO／InsightFace／DINOv2）、声紋話者帰属（ECAPA-TDNN）、VAD（silero・**512 サンプル@16kHz 単位**）＋発話バッファ → STT（faster-whisper int8/medium）と話者同定へ分配、在席相関 p の実入力結線、顔×声＝融B。
- 依存：Phase 2（p 軸がスコアラにある）。InsightFace の onnxruntime CUDA/cuDNN 解決（実機セットアップ）。
- テスト観点：VAD のフレーム条件（512）／発話バッファの終端確定／話者同定の閾値（enrollment 後）／在席ゼロ時の発話ゲート。
- チェックポイント：「誰が話したか」が相関サブテーブルへ供給される。

### Phase 4：音声出力
- 内容：TTS＝Style-Bert-VITS2（jvnv-M2-jp）、PAD→(style, style_weight) 写像、voice_guard 維持。STT/TTS の provider 抽象（`tools/tts.py`／`tools/stt.py`）。GPU 利用時 BERT float32。
- 依存：Phase 1（感情 PAD が MI/T にある）。Phase 3 とは独立に進められる。
- テスト観点：PAD→style 写像の代表点（平常/喜び/落ち着き/ささやき）／voice_guard の自発話聞き返し防止／合成失敗時のフォールバック。

### Phase 5：挙動（反復抑止・REST・拡散想起）
- 内容：[D-反復出力]（1反復1出力）、REST（近重複統合・内省・自己エピソード supersede・per-person 蒸留）、WR 拡散想起（memory_links 代替）。
- 依存：Phase 1（O/T）＋ Phase 2（recall）。BUG-1 の冪等化と整合（恒久の反復抑止）。
- テスト観点：同一意図の再発火が止まる／REST が日次 O を正しく統合／WR の seed と想起 MI 選択。

### Phase 6：旧系統の撤去
- 内容：workspace／prediction／interoception／social_policy／appraisal／concern_engine／attention_schema の撤去、廃止ストア（tape／memory_links／self_narrative／relationship_state→案A／pending_store→O open 意図／GlobalWorkspace→調停・発火）の移管完了・撤去。
- 依存：上記すべてが通って実証された後。
- 完了条件：各旧名を grep して**残存ゼロ**（数えた件数でなく0件を証明）。
- テスト観点：撤去後に新経路だけで全機能が成立する回帰。

## 4. 並行可否

- **並行可**：Phase 0（現行先行）は全体と並行。Phase 3（知覚入力）と Phase 4（音声出力）は Phase 2 後なら相互に並行可。
- **直列必須**：Phase 0(bge-m3) → Phase 2／Phase 1 → Phase 2 → (3,4,5) → Phase 6。撤去（6）は最後。

## 5. リスク

- Phase 1（O 移行）と Phase 2（スコア切替）が最大の山。データ移行の欠落・次元/型不整合・旧新スコアの取り違えに注意。各々マイグレーションのドライラン→確認→実行の2段。
- 旧系統の結線が深い（workspace 15・self_narrative 15）。撤去時に隠れ参照が残ると壊れるため、grep 0件を厳守。
- 現行先行（Phase 0）が未完だと Phase 2 の再測定が回せない。順序を守る。

## 6. 決定事項（承認済み・2026-06-29）

Phase 0（BUG-1・bge-m3 移行）は実機で完了。残る段取りの判断は次のとおり確定。

1. **着手フェーズ＝Phase 1（O/MI データモデル）から**。5軸は O の上で動くため基盤が先（Phase 2 は Phase 1 の後）。
2. **Phase 1 の刻み＝薄い縦切り**。最小の O＋MI で一通り通してから広げる（O 移行の欠落・不整合を小さく検出するため）。
3. **スコア切替＝フラグで旧/新を排他切替（並走しない）**。旧 cosine 純積と新5軸を同時に二重採点しない（取り違え防止）。
4. **min_score 確定＝Phase 2 実装後に再測定**。5軸スコアが動かないと測れないため、Phase 2 のチェックポイントで確定（c_lo/c_hi は暫定 0.25/0.50 を先行使用）。

---

承認済みにつき、最初に着手するフェーズ＝**Phase 1（O/MI データモデル）の TDD 改造方針**を、原因と対応案＋テスト＋マイグレーション＋完了条件の形で起こす。Phase 1 は薄い縦切りで、さらに小ステップに割って一項目ずつ確認しながら進める。
