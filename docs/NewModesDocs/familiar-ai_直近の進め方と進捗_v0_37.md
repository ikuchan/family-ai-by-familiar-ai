# familiar-ai 直近の進め方と進捗（v0.37）

> v0.37：**顔認識の GPU 実行を実機で通した**（環境修正）。既定 PyPI の `onnxruntime-gpu` が CUDA 13 版（`libcudart.so.13` を要求）で、環境の CUDA 12.8（torch と共通）と不一致のため `import onnxruntime` ごと失敗し、顔認識が黙って無効化されていた。(1) insightface が引き込む CPU 版 `onnxruntime` を uv の `override-dependencies` で無効化（GPU 版と同じ import 名前空間を奪い合う事故を防ぐ）、(2) `onnxruntime-gpu` を公式 CUDA-12 索引（1.28.0・`python_version>='3.11'` マーカー付き＝requires-python は 3.10 のまま）へ固定。さらに onnxruntime は nvidia pip ホイールの CUDA/cuDNN を自動 load しないため `libcublasLt.so.12` が見つからず CPU に落ちていたので、`face._get_model` で FaceAnalysis 構築前に `onnxruntime.preload_dlls()` を呼ぶようにした。結果 buffalo_l の全モデルが `CUDAExecutionProvider`（CUDA 先頭）で動作。旧 deepface/resemblyzer と GUI(PySide6) は `uv sync --all-extras` で復元。確認手順（`nvidia-smi`／`ort.get_available_providers()`／各セッションの `get_providers()`）で GPU 実行を検証。次は GUI（気分・感情＋在席・話者・カメラ）。

> v0.36：**認識を InsightFace/ECAPA-TDNN へ載せ替え**（挙動変化・実機依存）。顔を deepface→InsightFace(ArcFace)、声を resemblyzer→ECAPA-TDNN(speechbrain) にし、`RecognitionHint` の I/F と公開 API（`recognize_face_async`・`register_face`・`VoiceIdentifier`）を保った（呼び出し元は無変更）。判定ロジックは `recognition/embedding_store.py`（`best_match`＝cosine 最大＋しきい値・人ごと埋め込み pkl）へ寄せ、実モデルは遅延シングルトンで1回ロード（CUDA→CPU フォールバック）。しきい値は `RecognitionConfig`（env）＝認識 顔0.35/声0.25・自動切替 顔0.45/声0.35（いずれも仮置き・実機調整）。載せ替えで生 cosine の尺度が変わり `AUTO_SWITCH_THRESHOLD=0.75` のままだと話者自動切替が発火しないため、`apply_hint` を **source 別しきい値**へ変えた（顔・声で別値・text/auto は従来 0.75）。旧 deepface/resemblyzer は optional のまま残す（当面）。実モデルは重く GPU 依存なのでテストは埋め込みをモックし判定・保存・per-source 切替のみ検証、実モデル統合は実機確認に回す。次は GUI（気分・感情＋在席・話者・カメラ）。

> v0.35：**`min_score` を是正**（挙動変化）。生コサインの SQL 足切りから**合成 final score の soft 床**へ付け替え、根拠台帳 §3–4 の確定方針（r は段階化・無関係排除は合成スコアが担う）にコードを一致させた。store の `by_vector` から `min_cosine` を撤去して素取得に戻し（grep 0件）、床は `recall` が採点後に `score >= min_score` で課す。床を課すと `LIMIT n` の後で n を割るため、`min_score>0` のとき候補を n×3（上限20）過剰取得し、絞って上位 n を返す。既定 `RECALL_MIN_SCORE` を **0.05 起点**にした（台帳 §4・確定は実データのスコア分布から）。既定 0.05 は本番の想起集合を変えるので実機確認へ申し送り。テストは、古い観測（t を落とし M<1 で composite<cosine）で床の中間を突いて「合成床は切り生コサイン床は残す」を分離、過剰取得の件数、store 署名から `min_cosine` 消滅を確認。全体緑。これで **Phase 2 の締め（想起の背骨＋境界切り出し）が一通り終わり**、次は Phase 3（知覚・在席）へ。

> v0.34：Phase 2 の締めの切り出しを進め、**`agent.py` から評価器を `loop/evaluator.py` へ分離**した（挙動保存）。感情・要約・相手気分・整合性チェックの4メソッドと `A_GATE`・PAD 評価関数・各プロンプトを移し、`agent.py` は薄い委譲だけ残す。`_evaluator` は内部欲求ターンの backend スワップに追随する派生プロパティにした。永続化（`loop/persistence.py`）は `run()` ごと作り替える Phase 5 へ送り切り出さない（分割設計 v0.4）。あわせて **timestamp のタイムゾーンずれを是正**（挙動変化）：`timestamp::date`・`EXTRACT`・psycopg2 が返す datetime はセッションの TimeZone で解釈されるが、既定 UTC のままだと、ローカルの「今日」で引いた記憶が UTC 早朝帯（JST 00:00〜09:00）で前日扱いになり漏れ、表示時刻も9時間ずれていた（マイグレーション028 で値を真 UTC へ揃えたことで露呈した宿題）。`Database.conn()` の接続確立時に一度 `SET TIME ZONE INTERVAL` でローカルオフセットへ固定し、全 timestamptz 読みを生活時間へ寄せた（オフセット導出は `store/clock.py` の `local_utc_offset()` に閉じる・DST のある地域では接続存続中にずれうるが JST は DST なし）。テストは日付境界（ローカル今日 00:30＝UTC 前日）を跨ぐ instant で絞り込みと表示を検証し、既存のアクセス層テストが naive datetime を挿入セッションの TZ で解釈させていた曖昧さを tz 付きへ改めた。全体 1,474 件＋不変条件8件緑。次は **`min_score` の是正**（生コサインの閾値→合成スコアの足切り・挙動変化・設計を会話で決める）。

> v0.33：Phase 2 の締めの境界切り出しのうち、**`store/` の切り出しを完了**（S1〜S6d）。`tools/memory.py` は 2,594 行から 1,238 行へ減り、SQL は `store/`（と撤去予定の `legacy/`）にだけ残る。途中で継承（mixin）をやめ**合成へ組み替えた**（C1〜C3）。mixin が MRO で層の本体を覆い隠す事故が出たためで、各層は `StoreContext` から共有の道具を受け取り依存を引数に出す。生存確認の不変条件8件を先に置いたことでこの事故に気づけた。詳細は `モジュール分割設計 v0.2`。残るは `agent.py` の副作用境界の切り出し（`loop/persistence.py`・`loop/evaluator.py`）と `min_score` の是正。次は agent.py の切り出しへ。

> v0.32：スライス3 の実機確認から入って**3件の不具合を掘り当て、いずれも修正した**。(1) `say()` で話したターンが永続化されない（`run()` に埋もれた条件式・記憶が3週間書かれていなかった）、(2) 感情語の一致を辞書サイズで割っていて値踏みゲート A_GATE=0.25 を越えず、評価器が一度も起動していなかった、(3) 観測 `timestamp` が tz なしの `datetime.now()` で書かれ9時間ずれていた（マイグレーション028 で既存2,646件を補正）。あわせて **auto-say を機構ごと撤去**し、発話を `say()` へ一本化した（話すかの判断をモデルへ委ねる設計と矛盾する機構だった）。修正後の実機で、評価器が起動して観測 PAD が中立を脱し（tender／relieved／happy）、mood が p=0.596 まで動いた。**感情ループが端から端まで繋がった**。次は Phase 2 の締めとして、`agent.py`（4,024行）と `memory.py`（2,593行）の境界切り出しと生存確認の不変条件（課題8 v0.20 で新設）。スライス3 の実機確認（シナリオ1〜3）は未了で、通常の利用の中で行う。

> v0.31：スライス3 の実機確認に入ったところで、**会話しても記憶が一切書かれない不具合**を見つけた。真因は永続化の判定が本文テキストだけを見ていることで、エージェントは `say()` ツールだけで話すため `final_text` が `"(no response)"` になり、観測・会話 summary・mood・メンタル状態がまとめて飛ばされる。最後の書き込みは 2026-06-29 で、スライス3 より前から壊れていた。詳細と設計方針は `不具合調査_say発話ターンが永続化されない v0.2` にある。**想起の実機確認はこれが直るまで成り立たない**（mood が中立から動かず、気分一致想起を観測できない）。作業は「止血 → auto-say の撤去 → 新ループ（[D-反復出力]）への移行設計」の順に分けて進めると決めた。3つ目は `agent.py`（4,047行・`run()` が795行）の分解を含む。

> v0.30：スライス3（e 軸のスコア接続）を実装し、**想起スコアがハイブリッド合成になった**（挙動変化）。`_compute_final_score` を純積から `r^{w_r} × (w_t·t+w_e·e+w_a·a)/(w_t+w_e+w_a)` へ替え、新設の `_stretch_relevance` で r を伸長し、既存 `_emotion_match` を e として接続した。e の基準は**今の気分**で、mood は想起1回につき1つ読み全候補に共通に使う。**mood の読みは DB ロックの外に置いた**：`load_current_mood` は再入不可の `db.lock` を取るので、ロック内から呼ぶと平均中心化 C2 と同型のデッドロックになる。あわせて既定値を課題5 v0.24 へ揃えた（半減期 7日→3日・t_floor 0.25→0.001）。p 軸は知覚待ちで項ごと持たない。テスト13件新規（伸長5・合成6・recall 接続2）＋既存4件を式へ更新、全体1401件緑。**全体テストで初回は赤**になり、原因は実装ではなくテスト側の欠陥だった：`_NOW` をモジュール読み込み時に固定していたため、実行までに828秒空いて t が 1.0 から減衰し、手計算の期待値とずれた。呼び出し時刻で取る形に直し、20秒の遅延を挟んでも通ることで原因が消えたのを確かめた。min_score が生コサインの閾値である点と候補集合の切り方は変えていない（別スライスへ申し送り）。次は実機確認（P-1・W2b-2・スライス3 をまとめて）である。

> v0.29：本文（進捗・次の一歩）を現状へ揃えた。v0.18 以降の実装（P-1、P-3 のスライス1と書き込み PAD 化 W1a〜W2b、mood の PAD 化 mood-a〜c、平均中心化 C1／C2）は版メモにしか書かれておらず、本文の「次の一歩」は P-3 着手前のままだったので、Phase 2 の節を新設して実装済みの内容へ差し替えた。あわせて c_lo=0.0／c_hi=1.0 の決定（計測台帳 v0.7）を反映した。実装の変更はない。

> v0.28：平均中心化 C2 を実装し、**中心化（C1／C2）が完了**（挙動変化）。純関数 `_situated_vector(mem_vec, p_vec, mu)` を新設し、`normalise(mem_vec + ALPHA·p_vec − mu)`（mu が None なら従来式）へ一本化。**書き込み（`_upsert_situated_embedding`）と問い合わせ（`recall` のクエリ）の両方をこの関数に通す**ことで、片側だけ中心化して別空間になる事故を構造で防いだ（直書きの合成式は grep で0件）。mu は `_embedding_mu` が遅延1回だけ読みインスタンスへ持つ。マイグレーション027 で既存 situated を同じ式へ一括再計算（mu 未推定なら何もしない）。**実装中にデッドロックを1件発見して修正**：`db.lock` は再入不可の `threading.Lock` で、`_materialize_save_event` はロックを保持したまま situated 生成を呼ぶため、そこから `load_embedding_mean` が同じロックを取り直して保存経路が停止していた。`load_embedding_mean(dim, conn=None)` と `_embedding_mu(conn=None)` で呼び出し元の接続を受け取る形にして解消（90秒ハング→0.84秒）。テスト6件（純関数2・書き込み中心化・mu 無しフォールバック・書込と問合せが同空間・backfill）＋全体回帰1388件緑。次はハイブリッド合成（e 軸込み）だが、その前に **c_lo/c_hi を実データで決める**：中心化後のコサインがどれだけ散らばるかで、伸長が要る（案1）か中心化コサインをそのまま r にする（案2）かが決まる。本番 DB を読み取り専用で計測する。

> v0.27：スライス3（e 軸をスコアへ）の調査で、**e は加算部の一項なので純積からハイブリッド合成への切り替えが必要**（乗算で足すと拒否権になり設計と逆）、かつ**ハイブリッドの r 伸長は平均中心化が前提**（生コサインは異方性で mean≈0.88・窓0.016）と判明。方針は (い)＝**平均中心化を先に**を選択。中心化は C1（器・未接続）→ C2（適用と backfill・挙動変化）に分け、**C1 を実装**。マイグレーション026 で `embedding_means` を新設し（`scope`／`scope_key` で複数行＝将来の person 別中心化やクラスタ別平均を行追加で置ける・`vector` は BYTEA で次元非依存・`dim` で埋め込みモデル変更時の取り違え防止・`timestamptz`・`generated always as identity`）、既存 `obs_embeddings` から global の mu を一度推定して保存（0件なら行を作らない）。`tools/memory.py` に読み出し `load_embedding_mean(dim)`（行なし・次元不一致は `None`＝中心化しないフォールバック）を追加。未接続で挙動不変。テスト5件＋全体回帰緑。mu は Config（範囲つきのつまみ）ではなく統計量なので専用ストアに置き、再推定の起動を REST が持つ形にする。次は C2（situated 書き込みと recall クエリの両方で mu を引く＋既存 situated の一括再計算）。

> v0.26：mood の PAD 化 mood-c を実装し、**感情ループの上半分（W→N_PAD→M）を接続**（挙動変化）。`mood_register.py` に `_load_mood_with_updated_at`・`decay_and_nudge`（純＝`decay_to_rest`→`compute_n_pad`→`nudge_toward`）・`nudge_current_mood`（自己接続＝現 mood と updated_at を読み経過で減衰し nudge して save）を新設。`agent.py` の post-response pipeline で、評価器（`_emotion_for_turn`）の後に、想起記憶（PAD, activation）＋現ターン感情 E_cur（重み＝既定 a0=1.0）＋自己認識 MI フラット項から `nudge_current_mood` を呼ぶ。`memories` を pipeline へ配線。課題5 の「W は現在も含む・現在/過去で重み付けず」に沿い、現ターンの感情も W の一員として nudge に入る（1ターン遅れではない）。評価器のベースは直前 mood。decay は updated_at からの実経過（初回は経過0）。テスト（`decay_and_nudge` 純2件・`nudge_current_mood` DB2件）＋全体回帰緑。これで `load_current_mood` が実 mood を返し始め、評価器ベースが生きる。次はスライス3（e 軸をスコアへ＝合成式をハイブリッドへ切り替え）。ただし調査で、e は加算部の一項なので純積からハイブリッドへの切替が必要で、r 軸の伸長は平均中心化（課題7・未実装）依存と判明。方針は会話で決定中。

> v0.25：mood の PAD 化 mood-b を実装。recall が nudge の入力（各記憶の PAD と activation 重み）を露出する。SELECT に PAD 4列（emotion_p/pn/a/dom）を足し、返り dict に `"emotion_pad"`（`MoodPAD`）と `"activation"`（`_derive_activation(a0,n)`）を追加。追加フィールドのみで既存消費者は無視するため挙動不変。テスト3件（PAD 露出・activation 露出・既存キー不変）＋全体回帰緑。これで mood-c が `[(m["emotion_pad"], m["activation"]) for m in memories]` を `compute_n_pad` へ渡せる。次は mood-c（ターンで recall 後・pipeline 前に N_PAD→decay（updated_at からの経過）→nudge→save・接続・挙動変化）。

> v0.24：mood の PAD 化（案A で後回しにした感情ループの上半分＝W→mood）に着手し、mood-a（未接続）を実装。`mood_register.py` に、W の感情トーン N_PAD を activation 加重平均で作る `compute_n_pad`（自己認識 MI のフラット項 (0.5,0.5,0.5,0.5)・重み `SELF_KNOWLEDGE_MI_WEIGHT=2.0`＝課題5 の C を常に含むので W が空でも中立を返す）と、課題5 の式で mood を動かす `nudge_toward`（`A_M←max(A_M,A_N)`／`X_M←X_M+A_N(X_N−X_M)`・X＝p,pn,dom・A_N＝N_PAD.a）を新設。どちらも純関数・未接続で挙動不変。テスト8件＋全体回帰緑。未決は 1=自己認識 MI フラット項を含める・2=減衰の経過は agent_state の updated_at・3=nudge の接続点は recall 後 pipeline 前、で確定済み。次は mood-b（recall が各記憶 dict に PAD と activation を載せる）→ mood-c（ターンで N_PAD→decay→nudge→save・接続・挙動変化）。

> v0.23：W2b-2 を実装し、**書き込み PAD 化（W2）が完了**（実行時に接続・挙動変化）。`mood_register.py` に自己接続の `load_current_mood()`（読みだけ）、`agent.py` に定数 `A_GATE=0.25`・評価器プロンプト `_EMOTION_PAD_PROMPT`・モジュール関数 `_evaluate_emotion_pad(backend, text, mood, arousal)`（arousal<A_GATE は評価器を呼ばず P/Pn/Dom＝M、以上は固定順3数値を正規表現で拾い [0,1] クランプ、失敗は mood フォールバック、A 軸は機械 arousal）・インスタンスメソッド `_emotion_for_turn`（PAD 評価＋`label_from_pad` でラベル派生）を新設。`_run_post_response_pipeline` が `_emotion_for_turn` を呼び、生観測と会話 summary に `emotion_pad` を保存、派生ラベルを既存消費者へ渡す。`arousal=affect.arousal` を配線。旧 `_infer_emotion`／`_EMOTION_PROMPT`（ラベル直出し）を撤去（src grep 0件）。挙動変化＝静かなターン（A<0.25）は評価器 LLM を起動せず mood、ラベルは PAD 派生。テスト（`_evaluate_emotion_pad` 4件・`load_current_mood` 2件）と mock 差し替え（4ファイル・旧機構テスト3件撤去）＋全体回帰緑。W2b-2 の実機確認シナリオを文書へ追記（実施は P-1〜P-4 とまとめてスライス3 後）。次は mood の PAD 化（案A で後回しにした分・W の N_PAD で mood を nudge）→ スライス3（e 軸をスコアへ）。

> v0.22：W2b（接続・挙動変化）を W2b-1（書き込み配管・不変）と W2b-2（評価器接続・挙動変化）に分け、W2b-1 を実装。`save`／`save_with_id` に任意引数 `emotion_pad: MoodPAD | None` を足し、payload へ `to_json_dict()` で載せる。`_materialize_save_event` が `from_json_dict`（未指定は中立 `MoodPAD()`）で戻し、INSERT に PAD 4列（emotion_p/pn/a/dom）を書く。PAD は payload（JSON）経由で遅延マテリアライズ（`materialize_now=False`・memory_events）も通る。呼び出し側はまだ `emotion_pad` を渡さない（`agent.py` で0件）ので既定は中立0.5＝列既定と同値で外部挙動不変。マイグレーション不要（列は W1a 済み）。テスト3件（PAD 付き保存・PAD 無しは0.5・遅延 payload 往復）＋全体回帰緑。次は W2b-2（評価器を PAD 出力へ・A_gate と arousal 配線・解析・`label_from_pad` でラベル派生・生観測と会話 summary に PAD を渡す・旧 `_infer_emotion`／`_EMOTION_PROMPT` 撤去）。ここで挙動が変わる（静かなターンは評価器 LLM 非起動・ラベルは PAD 派生）。実機確認は P-1〜P-4 とまとめてスライス3 の後・知覚の前（実機確認シナリオ文書へ追記して溜める）。

> v0.21：W2（評価器が PAD を直接出力）を W2a（未接続の追加）と W2b（接続・挙動変化）に分け、W2a を実装。(1) `emotion_pad.py` を新設＝PAD↔ラベルの**生きた正本** `LABEL_PAD`（マイグレーション025 の `_LABEL_PAD` は凍結写しで値一致）と、PAD→ラベルの逆引き `label_from_pad`（ユークリッド最近傍で12ラベルへ量子化）。(2) Y＝`_row_to_mental_item` が観測行の PAD 列を `MoodPAD` として `MentalItem.emotion` に載せる（`row.get` 既定0.5で安全・`recall_self_model` の columns に PAD 列を追加）。これで評価器の PAD・行の列・MI 器の emotion が同じ `MoodPAD` で一本化（B-3 の tif.py と型が揃う）。`label_from_pad` は W2a では未接続（呼び出しは W2b）で外部挙動不変。純関数テスト（正本網羅・凍結写し一致・逆引き）と `test_mental_item` の PAD 版更新＋全体回帰緑。逆引き距離は e 軸の logit（`_emotion_match`）でなくユークリッドにした（12点への量子化には十分・emotion_pad を軽く保つ）。次は W2b（評価器を PAD 出力へ差し替え・A_gate と A の配線・観測への PAD 保存・ラベル派生）。

> v0.20：書き込み PAD 化の W1b を実装。マイグレーション025 で、既存観測の PAD を確定した12ラベル→4軸 PAD の写像で埋める（移行専用・一回限り・実行時 φ ではない）。表に無いラベルは既定0.5のまま。写像値の正本はマイグレーション025 の `_LABEL_PAD`（両価 moved/nostalgic は Pn を上げ、proud は Dom=0.90、鎮静系 relieved/sad/nostalgic/tender は A を 0.25〜0.35）に一元化し、設計ドキュメントからは値を複製せず参照する（runtime パラメータではないため課題5 には置かない）。PAD 列は依然としてスコア・recall・書き込み経路から未参照（実行時は未接続）。テスト4件（写像適用・表外は既定維持・neutral 明示更新・写像表の網羅）＋全体回帰緑。次は W2（評価器が P/Pn/Dom を直接出力し新規観測に PAD を保存・A 軸は機械・A_gate ゲート・評価器プロンプト＝自己認識 MI の変更を伴う）。

> v0.19：P-3 の書き込み PAD 化を、mood 化に先行して進めることに決めた（案A）。理由は、mood の PAD レジスタを nudge で駆動するには入力が PAD である必要があり、それには評価器が PAD を出す書き込み側が先に要るため。実行時の機械写像 φ を作らない（課題5 v0.23）という制約とも整合する。書き込み PAD 化は W1a（列追加）→ W1b（既存行を一回限りの label→PAD 写像で埋める）→ W2（評価器が P/Pn/Dom を直接出力）の3段に分ける。**W1a を実装**：マイグレーション024 で `observations` に感情 PAD 列 `emotion_p`／`emotion_pn`／`emotion_a`／`emotion_dom`（案B・`double precision NOT NULL DEFAULT 0.5`・各列 CHECK 0..1）を追加。既存行・新規行とも既定0.5で、評価器・スコア・recall は無変更・列は未参照（外部挙動不変）。文字列 emotion 列は残す。テスト4件（列追加・既定0.5・CHECK 下限/上限）＋全体回帰緑。W1b の label→PAD 写像値（12ラベル・4軸）は確定済み（moved/nostalgic は Pn を上げた両価・proud は Dom=0.90・鎮静系は A を 0.25〜0.35）。次は W1b の実装。

> v0.18：P-3（感情の PAD 化・e 軸）のスライス1を実装。想起スコアの e 軸を計算する純関数 `_emotion_match`（`tools/memory.py`）を新設した。課題5 v0.23 で確定したガウシアン $e=\exp(-D^2/(2\sigma^2))$ を、既存 `_derive_activation` と同じロジット＋ε の流儀で書いた（各 PAD 軸を ε で両端へ寄せロジットで元空間へ戻し、軸重み λ_i つき二乗距離を作る）。σ・λ_i・ε は課題5 の起点値で Config 差し替え可。`_compute_final_score` にも recall 経路にも未接続で外部挙動不変、DB 変更なし。単体テスト7件（完全一致で1・距離単調・範囲(0,1]・対称・端クランプで有限・軸重みで低下・σ で寛容）と全体回帰が緑。次の候補はスライス2（mood の PAD 化・一回限りの写像で既存行を埋める）か、書き込み側の PAD 列（案B＝軸ごとの数値列）。e 軸の体感シナリオはスライス3（e 軸のスコア接続）の改造方針に含める。

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

### Phase 2（想起・入口 P）

段階C までで Phase 1 の実装可能面が尽きたため、Phase 2 の想起から着手した。以下はいずれも実装済みで、全体回帰は緑である。

- P-1【実装済み・挙動変化】：活性の想起接続。`_compute_final_score` の `importance` を `_derive_activation(a0,n)` へ差し替え、`importance` の日次減衰を撤去して時間減衰を t 軸へ一元化した。
- P-3 スライス1【実装済み・未接続】：e 軸の純関数 `_emotion_match`（ガウシアン $e=\exp(-D^2/(2\sigma^2))$・全軸ロジット空間の軸重み付き PAD 距離）。スコアへは未接続。
- P-3 書き込み PAD 化【実装済み・挙動変化】：マイグレーション024 で `observations` に PAD 4列を追加（W1a）、025 で既存行を一回限りの label→PAD 写像で埋め（W1b）、`emotion_pad.py` に PAD↔ラベルの正本と逆引き `label_from_pad` を新設（W2a）、`save` 系へ `emotion_pad` を通し（W2b-1）、評価器を PAD 直接出力へ差し替えた（W2b-2）。旧 `_infer_emotion`／`_EMOTION_PROMPT` は撤去済み（src の grep で0件）。挙動変化は、arousal が `A_GATE=0.25` 未満の静かなターンで評価器 LLM を起動せず mood を使う点と、ラベルが PAD 派生になる点である。
- mood の PAD 化【実装済み・挙動変化】：`compute_n_pad`・`nudge_toward`（mood-a）、recall が各記憶に `emotion_pad` と `activation` を載せる（mood-b）、post-response pipeline で `nudge_current_mood` を呼ぶ（mood-c）。これで感情ループの上半分（W→N_PAD→M）が接続され、`load_current_mood` が実 mood を返す。
- 平均中心化【実装済み・挙動変化】：マイグレーション026 で `embedding_means`（`scope`／`scope_key` で複数行・`vector` は BYTEA・`dim` つき）を新設し global の mu を推定（C1）。純関数 `_situated_vector(mem_vec, p_vec, mu)` を新設し、situated の書き込みと recall のクエリの両方をこの関数へ通して片側だけ中心化する事故を構造で防ぎ、マイグレーション027 で既存 situated を再計算した（C2）。実装中に `db.lock`（再入不可）を保持したまま `load_embedding_mean` が同じロックを取り直すデッドロックを1件発見し、接続を引数で渡す形にして解消した（90秒ハング→0.84秒）。
- スライス3（e 軸のスコア接続）【実装済み・挙動変化】：合成を純積からハイブリッドへ切り替え、`_stretch_relevance`（新設）で r を伸長し、`_emotion_match` を e として接続した。基底プロファイルは在席者ゼロの $(1,1,1,1.5)$。合成係数7つ（$c_{lo}$・$c_{hi}$・$w_r$・$w_t$・$w_e$・$w_a$・$\sigma$）を `MemoryConfig` へ出し、半減期と $t_{floor}$ の既定を課題5 v0.24 へ揃えた。これで想起の軸は r・t・a・e の4本になり、`_compute_final_score` が設計式と一致した。
- c_lo／c_hi【決定済み】：中心化後の本番 `obs_embeddings` 2,525件を読み取り専用で計測し、無作為ペアの p50 が 0.4800→−0.0235、窓幅が 0.0918→0.2427（約2.6倍）になることを確認した。値域が [0,1] にほぼ収まるため **c_lo=0.0・c_hi=1.0**（伸長式は実装するが係数を効かせない）と決めた。根拠は計測台帳 v0.7 §3 にある。

## 直近のドキュメント整備（差し替え待ち）

- 用語一覧 v0.24：B スライスの実装名（`MoodPAD`、`AiDrivers`、`tif` の関数、各 state_key）を該当行へ併記。
- 別紙（設計詳細_活性_O書込_知覚在席）v0.4：B-2 反映。プロジェクトが v0.3 のため差し替え漏れの再提示。

## 次の一歩

次の一歩は**境界の切り出しと生存確認の不変条件**である（課題8 v0.20・Phase 2 の締め）。順序は、不変条件 → `memory.py` の分解 → `agent.py` の副作用境界 → `min_score` の是正。ループ構造の作り替えは Phase 5 のまま前倒ししない。永続化の不具合はすでに修正済みで、詳細は `不具合調査_say発話ターンが永続化されない v0.3` にある。並行して、以下の実機確認を通常の利用の中で行う。P-1（活性の想起接続）・W2b-2（評価器の PAD 出力）・スライス3（e 軸の接続）はいずれも挙動変化を伴い、実機確認シナリオ文書へ溜めたまま実施していない。想起順が体感として破綻しないかは実データと実会話でしか見えないので、知覚（[D-知覚]）へ進む前にここで確かめる。とくにスライス3 は、気分一致想起が効いているか、e が拒否権になっていないか、半減期短縮（7日→3日）で古い記憶が沈みすぎていないかの3点を対で見る。

実機確認のあと、コードで次に着手できるのは **min_score の意味の是正**である。現行は生コサインに対する SQL の閾値だが、課題5 は合成スコアの床と定めており食い違っている。スライス3 では挙動変化の切り分けのために触らなかった。

残りのスライスは次のとおり塞がっている。

- P-2（参照申告で n 増減）は新ループ（課題8・[D-反復出力]）待ちで保留。案a が依存する主LLM↔ツールループ自体を新ループが作り直すため、その上で実装する。
- P-4（5軸合成）は p 軸を待つ。r・t・a・e の4軸はスライス3 で揃った。
- p 軸（在席者相関）は知覚（[D-知覚]）待ち。situated V2 の slice-3 以降（視点列からの関係生成・`person_id` 削除・旧 `_remember` 撤去）も同じく知覚が視点列を埋めることに依存する。
- MI 集約段の本体（系統A・系統B の畳み込み）は REST 内省（課題10）待ち。

実機確認は P-1・W2b-2・スライス3 をまとめて、知覚の前に行う。シナリオは実機確認シナリオ文書（v0.3）にある。

（以下は Phase 1 完了時点の旧メモ）C-1・C-2 は実装済み（段階C の Phase 1 スライスが済んだ）。MI 集約段の設計を会話で進行中（実装未着手）。系統B（`semantic_facts`／`behavior_policies`）は設計確定（キーレス supersede・content 注記・REST 更新・旧2テーブル撤去・`MIデータモデル` §7）で実装本体は Phase 2 寄り。系統A（`self_model`／`curiosity`）の対応づけも確定（self_model→自己認識 MI 自己エピソード部・curiosity→cue O・方針二分・MIデータモデル v0.06 付録A）で、更新は REST 依存の Phase 2 寄り。自己認識 MI のシステムプロンプト構築規約（不変度順・messages 分離・文字数上限）も確定（[D-自己認識分離]）。situated V2 の構造・生成規則・移行も確定（[D-在席相関/V2]・gap v0.4）。situated V2 の schema 器（スライス1・2）を実装し、Phase 1 分はここで一区切り（全体テスト緑待ち）。slice-3 以降は視点列を埋める知覚（[D-知覚]）依存で Phase 2。**Phase 1 で残る MI 集約段の実装可能面はほぼ尽きた**（集約本体は REST 依存）。次の候補：(A) 系統B の読み出し器（キーレス supersede チェーンの再帰想起ヘルパー・母集合を変えず未接続の薄い縦切り）、(B) 別の設計論点や別課題（Phase 2 の p 軸設計・REST 詳細＝課題10 等）へ移る。全体テスト緑を確認してから、どちらへ進むか決める。
