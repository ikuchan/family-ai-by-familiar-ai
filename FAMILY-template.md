# FAMILY-template.md
# Copy this file to FAMILY.md and describe the people who live with you.
# FAMILY.md is gitignored — this is yours alone.
# The AI uses this information to recognise family members and tailor responses.

# 一緒に暮らす人たち

## （名前を変えてください）

- **名前**：
- **呼び方**：（AIがどう呼ぶか — 例：お父さん、太郎くん）
- **英字**：（任意 — 例：Taro Tanaka。先頭の語を小文字にしたもの（taro）が、その人だけの道具の名前の末尾（`ask_vault_taro`）になる。書かなければその人だけの道具は無い）
- **関係**：（例：家族の父、長男）
- **外見の特徴**：（例：眼鏡をかけている、短髪）
- **よく着る服**：（任意）
- **性格・傾向**：（任意 — 例：夜型、話好き）

## （もう一人）

- **名前**：
- **呼び方**：
- **関係**：
- **外見の特徴**：
- **性格・傾向**：

## 話者の指定方法

チャット画面で以下のように入力すると、AIはその人として受け取ります。

```
[太郎] ただいま
@花子: 今日のご飯は？
/speaker 太郎
```

AIは話者ごとに信頼度・親密度・好みを個別に記憶します。
顔認識を有効にするには、`~/.familiar_ai/faces/<名前>/` に写真を置いてください。
