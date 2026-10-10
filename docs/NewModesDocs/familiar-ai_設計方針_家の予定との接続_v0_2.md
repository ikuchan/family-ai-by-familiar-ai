# familiar-ai 設計方針：家の予定との接続（v0.2）

> 知-j（予定の確認）の設計。家の記録（`obsidian-memo`）と対になる、**家の予定（カレンダー）**への入口。
> 向こう側（`ObsidianMemo`）の設計は `10_Wiki/参照/Vault保管先の検討.md`（2026-08-28）にあり、
> 2026-09-13 に資料一式（設計・実体・契約）として渡された。ここはパジュ側の判断だけを書く。

## 1. 事実（渡された資料と実物から）

| | |
|---|---|
| 予定はどこにあるか | **Google カレンダー。Vault には無い。** 向こうの設計は「Todo・予定は Notion／Google カレンダーが一次、Vault は派生」「予定だけは腐らせない＝直読」「`get_schedule` は memo_mcp から**外す**（責務が混ざる）」 |
| カレンダーの 2 ティア | ファミリー（`family01088600973735344679@group.calendar.google.com`）は**全員**、個人・仕事・妻のは本人だけ。道具は `get_family_schedule`（全員）と `get_schedule_yusuke`（Yusuke だけ・名前でゲート） |
| ゲート | 在席ゲートは入れない。ゲートは**話者**だけ（2026-08-28 の本人の決定） |
| こちらの現状 | 予定を読む仕組みは無かった。「今日の予定は？」に recall を 2 回して空振り（2026-09-13 実機） |
| MCP の同期の道具 | `house_rules` は動作の表に載っていたが、**主LLM が `get_house_rules` を呼んでも捨てられていた**（`_LOOKUP_ACTIONS` に道具名が無く、`_dif.lookup` は検索と取得しか知らない）。実機で一度も呼ばれた記録が無い |

## 2. 決めたこと（2026-09-13）

1. **カレンダー MCP はこちらで作る**：`mcp/calendar_mcp/`（依存ゼロ・stdio・`memo_mcp` と同じ作法＝`initialize`／`ping`／`tools/list`／`tools/call`、stdout は JSON-RPC 専用、道具の失敗は result の `isError`）。
2. **取り方は ICS 直読から**（イ）：ファミリーカレンダーの秘密の ICS（`FAMILY_CALENDAR_ICS`・`~/.familiar-ai.json` の `env`・git に置かない）を HTTPS で読む。認証コード無し・読み取り専用。**更新の遅れは実測してから**、実害があれば Google Calendar API＋エージェント専用アカウント（ロ）へ。
3. **道具は `get_family_schedule(days=1)` だけを先に**。返りの先頭に【いま】（JST の日付・曜日・時刻）と【出典】（カレンダー名・取得時刻）を付け、腐りが分かる形にする（`get_house_rules` に倣う）。繰り返し（`RRULE`）は `DAILY|WEEKLY|MONTHLY|YEARLY`・`UNTIL`・`COUNT`・`BYDAY`・`INTERVAL`・`EXDATE` を展開し、それ以外は**黙って落とさず**題名だけ出す。個人ティア `get_schedule_yusuke` は 知-f の後。
4. **パジュ側**：MCP の同期の道具を**動作名と道具名の両方**で引ける表 `_MCP_LOOKUPS` を置き、`_run_lookup_body` が `DIF.call_tool` で呼んで完了として積む（`recall` と同じ）。`_ACTIONS` に `family_schedule`、調停の候補にも `house_rules`／`family_schedule`（繋がっているときだけ・`extra_actions`）。`capabilities.yaml` に 1 項目（本文はコピーしない）。
5. **Notion は別の課題（知-k）**。中身は基本すべて家族ティア（同日の決定）。

## 3. 起動

```json
{ "mcpServers": { "family-calendar": {
    "type": "stdio",
    "command": "/path/to/family-ai-by-familiar-ai/.venv/bin/python",
    "args": ["-m", "calendar_mcp"],
    "cwd": "/path/to/family-ai-by-familiar-ai/mcp",
    "env": { "FAMILY_CALENDAR_ICS": "https://calendar.google.com/calendar/ical/.../private-.../basic.ics" }
}}}
```

`cd mcp && python -m calendar_mcp --selftest` で initialize → tools/list → tools/call を 1 往復ずつ流す（実際に ICS を取る）。ターミナルからは `env` が渡らないので、サーバーは `~/.familiar-ai.json` の `family-calendar` の `env` も読む。

**⚠ URL は「非公開 URL（iCal 形式）」**（末尾が `basic.ics`）。カレンダーの共有リンク（`…/calendar/u/0?cid=…`）ではない（2026-09-13 に取り違えがあった）。Google カレンダー（PC）→ マイカレンダーの「︙」→「設定と共有」→「カレンダーの統合」にある。形が違えば道具はその旨を返す。

## 3b. 予定を先読みして記憶に置く（情-q・2026-10-10・`6b22c3b`）

BOND で話しかけるとき、思い出すことが家族の予定に寄るように、予定を記憶（O）に置く（本人の決定）。T が起動直後に
1 回、そのあと 1 時間に 1 回（本人の決定・パジュへのメモと同じ）、`get_family_schedule(days=7)`（7 日分・本人の決定）を
読む（`loop/schedule_watch`）。時刻の行（`【いま】`・`【出典】`）を外した本文を前回（`agent_state.family_schedule`）と
比べ、変わっていれば機器の記録「予定」（「家族のこれから 7 日の予定：…」・1,500 字まで）として O に書く。パジュへのメモは
初回は覚えるだけだが、予定は初回も書く（O に無いと思い出せない）。日付が進むと 7 日の範囲が動くので、少なくとも
1 日 1 回は書き直す。話しかけはしない。BOND の発火で予定をどう使うかは `設計詳細_発火_mood機構` §2-d。

「今日の予定は？」と聞かれたときは、これまでどおり道具で読み直す（記憶の予定は最大 1 時間古い）。

## 4. 確かめること（実機）

- 「今日の予定は？」→ 調停が `family_schedule` を 1 手で投げ、次の反復で予定が返る。
- 「明日の予定は？」→ `days=2` で明日が出る。
- カレンダーに予定を足してから何分で ICS に出るか（更新の遅れ）を測り、`根拠台帳` に残す。

## 更新履歴

> v0.2：§3b 予定の先読みと記憶への記録を足した（情-q・`6b22c3b`・2026-10-10）。
>
> v0.1：新規（2026-09-13）。渡された `obsidian-memo` の資料一式を読み、予定はカレンダー MCP をこちらで作ると決めた（ICS 直読から）。
