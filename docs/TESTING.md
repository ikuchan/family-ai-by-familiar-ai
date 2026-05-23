# テスト手順

ソフトウェアテストとハードウェアテストの両方をカバーする。

---

## ソフトウェアテスト

### 前提

```bash
uv sync --group dev
```

### 1. DB 接続テスト

```bash
uv run python -c "
import os
os.environ['DATABASE_URL'] = 'postgresql://familiar:familiar@localhost:5432/familiar_ai'
from familiar_agent.db import get_db
from familiar_agent.db_migrations import apply_migrations, default_migration_dir
db = get_db()
with db.lock:
    n = apply_migrations(db.conn(), default_migration_dir())
    db.commit()
print(f'OK — {n} migrations applied')
"
```

期待出力: `OK — 12 migrations applied`

---

### 2. pgvector 動作確認

```bash
uv run python -c "
import os, numpy as np
os.environ['DATABASE_URL'] = 'postgresql://familiar:familiar@localhost:5432/familiar_ai'
from familiar_agent.db import get_db, vec_to_sql, sql_to_vec

db = get_db()
with db.lock:
    conn = db.conn()
    with conn.cursor() as cur:
        # ベクトルの挿入・検索テスト
        cur.execute(\"CREATE TEMP TABLE vec_test (v vector(4))\")
        cur.execute(\"INSERT INTO vec_test VALUES (%s::vector)\", (vec_to_sql([1,0,0,0]),))
        cur.execute(\"INSERT INTO vec_test VALUES (%s::vector)\", (vec_to_sql([0,1,0,0]),))
        cur.execute(\"SELECT v::text, 1-(v<=>%s::vector) AS score FROM vec_test ORDER BY v<=>%s::vector LIMIT 1\",
                    (vec_to_sql([1,0,0,0]),vec_to_sql([1,0,0,0])))
        row = cur.fetchone()
    conn.rollback()
print(f'pgvector OK — top score: {float(row[\"score\"]):.4f}')  # 期待: 1.0000
"
```

---

### 3. 人物登録・視点ベクトルテスト

```bash
uv run python << 'EOF'
import os
os.environ['DATABASE_URL'] = 'postgresql://familiar:familiar@localhost:5432/familiar_ai'

from familiar_agent.tools.memory import ObservationMemory
from familiar_agent.person_memory_manager import PersonMemoryManager, AGENT_SELF_ID

mem = ObservationMemory()
mgr = PersonMemoryManager(mem)

# 人物登録
alice_id = mgr.register_person("test_alice", "テストアリス")
bob_id   = mgr.register_person("test_bob",   "テストボブ")
print(f"alice: {alice_id[:8]}  bob: {bob_id[:8]}")

# 存在を通知
import asyncio
async def test():
    await mgr.person_arrived(alice_id)
    await mgr.person_arrived(bob_id)
    print(f"present: {mgr.get_present_ids()}")
    assert alice_id in mgr.get_present_ids()
    assert bob_id   in mgr.get_present_ids()

    # speaker セット
    await mgr.set_speaker(alice_id)
    assert mgr.current_speaker_id == alice_id
    print("speaker OK")

    # メモリ書き込み (speaker scope)
    store = mgr.get_speaker_memory()
    ok = store.save("テスト記憶: 猫が好き", kind="observation", emotion="happy")
    assert ok, "save failed"
    print("save OK")

    # recall
    results = store.recall("猫", n=3)
    assert results, "recall returned nothing"
    print(f"recall OK — top: {results[0]['summary'][:50]}")

asyncio.run(test())
print("All tests passed")
EOF
```

---

### 4. マルチスコープ書き込みテスト

```bash
uv run python << 'EOF'
import os, asyncio
os.environ['DATABASE_URL'] = 'postgresql://familiar:familiar@localhost:5432/familiar_ai'

from familiar_agent.tools.memory import ObservationMemory, MemoryTool
from familiar_agent.person_memory_manager import PersonMemoryManager

mem = ObservationMemory()
mgr = PersonMemoryManager(mem)
tool = MemoryTool(mgr)

async def test():
    alice_id = mgr.register_person("scope_alice", "スコープテスト")
    bob_id   = mgr.register_person("scope_bob",   "ボブスコープ")

    await mgr.person_arrived(alice_id)
    await mgr.person_arrived(bob_id)
    await mgr.set_speaker(alice_id)

    # scope=witnessed でアリスとボブ両方に書き込む
    result, _ = await tool.call("remember", {
        "content": "今日は晴れていい天気ですね",
        "scope": "witnessed",
        "emotion": "happy",
    })
    print(f"remember: {result}")
    assert "ボブスコープ" in result or "witnessed" in result or "目撃" in result, \
        f"Expected witnessed write, got: {result}"

    # bob の記憶から「晴れ」を検索
    bob_store = mgr.get_memory_for(bob_id)
    results = bob_store.recall("晴れ 天気", n=3)
    assert results, "Bob should have witnessed memory"
    print(f"Bob recall OK: {results[0]['summary'][:60]}")

    print("Scope test passed")

asyncio.run(test())
EOF
```

---

### 5. 自動テスト実行

```bash
uv run pytest tests/ -v --tb=short
```

テストファイルの置き場: `tests/test_memory.py`, `tests/test_person.py`

---

## ハードウェアテスト

### カメラテスト (Tapo C220 / USB)

#### RTSP 接続確認

```bash
# ffmpeg で1フレーム取得
ffmpeg -y -rtsp_transport tcp \
  -i "rtsp://USER:PASS@192.168.1.xxx/stream1" \
  -frames:v 1 /tmp/test_frame.jpg

# 取得できたか確認
ls -la /tmp/test_frame.jpg
```

期待: ファイルサイズ > 0

#### OpenCV (USB ウェブカム) 確認

```bash
uv run python -c "
import cv2
cap = cv2.VideoCapture(0)
assert cap.isOpened(), 'Camera not found'
ret, frame = cap.read()
cap.release()
assert ret, 'Frame capture failed'
cv2.imwrite('/tmp/test_usb_frame.jpg', frame)
print(f'USB camera OK — frame shape: {frame.shape}')
"
```

---

### 顔認識テスト (deepface が必要)

```bash
# 顔画像の登録
uv run python -c "
from familiar_agent.recognition.face import register_face
register_face('alice', '/path/to/alice_photo.jpg')
print('Face registered')
"

# フレームから認識
uv run python << 'EOF'
import os, asyncio
os.environ['DATABASE_URL'] = 'postgresql://familiar:familiar@localhost:5432/familiar_ai'

from familiar_agent.tools.memory import ObservationMemory
from familiar_agent.person_memory_manager import PersonMemoryManager
from familiar_agent.recognition.face import recognize_face_async

mem = ObservationMemory()
mgr = PersonMemoryManager(mem)

async def test():
    hint = await recognize_face_async("/tmp/test_frame.jpg", mgr)
    if hint:
        name = mgr.get_person_name(hint.person_id)
        print(f"Recognized: {name} (conf={hint.confidence:.2f})")
    else:
        print("No face recognized (or deepface not installed)")

asyncio.run(test())
EOF
```

---

### マイク / STT テスト

```bash
# 録音テスト (sounddevice)
uv run python -c "
import sounddevice as sd, soundfile as sf, numpy as np
duration, sr = 3, 16000
print('Recording 3 seconds...')
audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()
sf.write('/tmp/test_audio.wav', audio, sr)
print(f'Recorded: /tmp/test_audio.wav ({audio.shape[0]} samples)')
"

# STT テスト
uv run python -c "
import whisper
model = whisper.load_model('tiny')
result = model.transcribe('/tmp/test_audio.wav', language='ja')
print(f'STT result: {result[\"text\"]}')
"
```

---

### 声紋認識テスト (resemblyzer が必要)

```bash
uv run python << 'EOF'
import os, asyncio
os.environ['DATABASE_URL'] = 'postgresql://familiar:familiar@localhost:5432/familiar_ai'

from familiar_agent.tools.memory import ObservationMemory
from familiar_agent.person_memory_manager import PersonMemoryManager
from familiar_agent.recognition.voice import VoiceIdentifier

mem = ObservationMemory()
mgr = PersonMemoryManager(mem)
vi  = VoiceIdentifier(mgr)

alice_id = mgr.register_person("alice", "アリス")

# 声を登録
vi.register_voice(alice_id, "/path/to/alice_sample.wav")
print("Voice registered")

# 認識テスト
async def test():
    hint = await vi.identify_async("/tmp/test_audio.wav")
    if hint:
        name = mgr.get_person_name(hint.person_id)
        print(f"Speaker identified: {name} (conf={hint.confidence:.2f})")
    else:
        print("Speaker not identified (confidence below threshold or resemblyzer not installed)")

asyncio.run(test())
EOF
```

---

### PresenceWatcher 統合テスト

```bash
uv run python << 'EOF'
import os, asyncio, logging
logging.basicConfig(level=logging.INFO)
os.environ['DATABASE_URL'] = 'postgresql://familiar:familiar@localhost:5432/familiar_ai'

from familiar_agent.tools.memory import ObservationMemory
from familiar_agent.person_memory_manager import PersonMemoryManager
from familiar_agent.recognition.presence_watcher import CameraPresenceWatcher

mem = ObservationMemory()
mgr = PersonMemoryManager(mem)
watcher = CameraPresenceWatcher(mgr, interval_sec=3.0, absent_threshold_sec=15.0)

async def run():
    # 存在変化のコールバックを登録
    async def on_switch(old, new):
        print(f"Speaker switched: {old} → {new}")
    mgr.on_switch(on_switch)

    await watcher.start()
    print("Watcher running — 15 seconds...")
    await asyncio.sleep(15)
    await watcher.stop()
    print(f"Final present: {[mgr.get_person_name(p) for p in mgr.get_present_ids()]}")

asyncio.run(run())
EOF
```

---

## トラブルシューティング

| 症状 | 確認事項 |
|---|---|
| `psycopg2.OperationalError` | `DATABASE_URL` の設定、PostgreSQL の起動を確認 |
| `pgvector` 関連エラー | `CREATE EXTENSION vector` が実行済みか確認 |
| `No module named 'deepface'` | `uv sync --group recognition` を実行、または顔認識をスキップ |
| カメラ取得失敗 | `ffmpeg` のインストール、`CAMERA_HOST` の設定、ネットワーク接続を確認 |
| STT 無音 | マイクのデバイス番号を確認: `python -c "import sounddevice; print(sounddevice.query_devices())"` |
| `situated_embeddings` が空 | 書き込みが1件以上あるか確認。`SELECT COUNT(*) FROM situated_embeddings;` |
