# familiar-ai 設計方針：Notion との接続（v0.1）

> 知-k（Notion を読む口）。家の記録（`obsidian-memo`）・家の予定（`家の予定との接続`）と並ぶ、
> **家の記録の目次と日次記録**への入口。

## 1. 事実（2026-09-13・鍵で読み取りだけ試した）

| | |
|---|---|
| 向こうの Notion の位置づけ | **下流**。`10_Wiki` の索引 DB（ページ名・abstract・パス・種別・確度・更新）と日次記録 DB（対象日・朝の気分・肉体疲労・脳疲労・体感・睡眠分・睡眠スコア）。`memo_mcp/notion_api.py` は **Vault → Notion の一方向同期**で、読む口は無い |
| この鍵から見えるもの | データソース 2 つだけ：`ObsidianMemo 索引（10_Wiki）` 70 行・`ObsidianMemo 日次記録` 28 行。Todo の DB は共有されていない。索引のページに本文（ブロック）は無い（本文は Vault） |
| ティア | **基本すべて家族ティア**（2026-09-13 の本人の決定）。名前に人を入れない |
| 鍵 | `NOTION_TOKEN`（`~/.config/obsidian-memo.env`・memo_mcp と同じ置き場・git に置かない）。`Notion-Version: 2026-03-11` |

## 2. 決めたこと

1. **実体はこちらで作る**：`mcp/notion_mcp/`（依存ゼロ・stdio・memo_mcp／calendar_mcp と同じ作法）。汎用の Notion MCP は採らない（道具が多く、名前でゲートが掛けられず、鍵で全部見える）。
2. **道具は 2 本**：`search_notion(query, limit=5)`（`POST /v1/search`・ページ名／abstract／パス／種別・確度／更新日）と `get_journal(days=7)`（日次記録のデータソースを `日付` で絞る）。どちらも先頭に【いま】【出典】。**本文は返さない**（索引のページに本文は無く、Vault にある。深く読むのは 知-g-い）。
3. **パジュ側**：`_MCP_LOOKUPS` に `notion_search`／`search_notion`（**query が要る**・見出し「Notion で『…』を探す」）と `journal`／`get_journal`。調停の候補にも繋がっているときだけ（`extra_actions`）。`capabilities.yaml` に 1 項目。
4. 日次記録のデータソース id は `~/.familiar-ai.json` の `notion-memo` の `env`（`NOTION_JOURNAL_DATA_SOURCE`）。鍵は置かない。

## 3. 起動

```json
{ "mcpServers": { "notion-memo": {
    "type": "stdio",
    "command": "/path/to/family-ai-by-familiar-ai/.venv/bin/python",
    "args": ["-m", "notion_mcp"],
    "cwd": "/path/to/family-ai-by-familiar-ai/mcp",
    "env": { "NOTION_JOURNAL_DATA_SOURCE": "<日次記録のデータソース id>" }
}}}
```

`cd mcp && python -m notion_mcp --selftest`（実際に Notion を読む・鍵は `~/.config/obsidian-memo.env`）。2026-09-13 に疎通：「サッカー」で 3 件、日次記録 直近 3 日。

## 更新履歴

> v0.1：新規（2026-09-13）。
