# 旧版ドキュメント（OldDocs）

**ここに置いてあるのは、撤去した旧 ReAct 経路の記録である。いまの実物ではない。**

`agent.py` の `run()` が 735 行の ReAct ループとして回り、感情・社会方針・メタ監視などが
それぞれの module として並んでいた時代の説明である。その経路は **#12a（環-c）で撤去**した。
いまのターンの中身は、イベント駆動ループ（`loop/event_loop.py`）が持つ。

**捨てずに残すのは、当時どう考えていたかが分かるためである。** 撤去したものの台帳は
`docs/NewModesDocs/` の `設計図_Mermaid`「撤去済み（#12a）」節にある。

| file | 何の記録か |
|---|---|
| `technical.md` | ターンの11段のパイプラインと、各層（内受容・値踏み・社会方針・メタ監視）の説明 |
| `architecture.md` | Neighbor Intelligence Stack への層の対応づけ |
| `CHANGES.md` | fork の v0.5 → v0.6 の変更点（SQLite → PostgreSQL 等・2026-06-16 まで）。その後の変更は git の履歴と `NewModesDocs` の各更新履歴にある |
| `HowToDecideWhomToTalk.md` | 旧 desire（`share_memory`）を前提にした「誰に話すか」の設計メモ（2026-06-14）。desire は 環-d で撤去した |
| `future-model.md` | Chronos-Neighbor 構想（2026-04・英語） |
| `handson-guide.md` | 2026-04-04 のハンズオンの資料 |

## いまの設計はどこにあるか

| 知りたいこと | 見る場所 |
|---|---|
| 構造の正本 | `docs/NewModesDocs/` の `設計図_Mermaid` |
| 語と略語 | 同 `用語_略語一覧` |
| 段階・順序・現在地 | 同 `課題8_段取り設計` |
| ループの設計判断 | 同 `設計方針_ループの語を束ねる`・`設計方針_主LLMを投げっぱなしにする` |
| module 構成 | `docs/ソースツリー.md`（実物から生成） |

（2026-09-10 に `technical.md`・`architecture.md` を、2026-09-17 に残り 4 file を `docs/` 直下から移した。）
