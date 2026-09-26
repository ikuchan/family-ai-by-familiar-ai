"""Global pytest configuration for familiar-ai."""

from __future__ import annotations

import os
import time

# Must be set before any familiar_agent imports so Database singleton picks up correct URL.
os.environ.setdefault("FAMILIAR_EMBEDDING_PREWARM", "0")

# 並列実行（pytest-xdist）ではワーカーごとに別 DB を使う。autouse の clean_db が
# 共有テーブルを truncate するため、1 DB を並列共有すると互いにデータを消し合う。
# `PYTEST_XDIST_WORKER`（gw0/gw1…）を DB 名に反映し、非並列（未設定/master）は従来の
# `familiar_test` を使う。DATABASE_URL は Database singleton が import 時に拾うので、
# familiar_agent の import より前にここで確定させる。
_PG_HOST = "postgresql://familiar:familiar@localhost:5433"
_BASE_DB = "familiar_test"
_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
_DB_NAME = f"{_BASE_DB}_{_WORKER}" if (_WORKER and _WORKER != "master") else _BASE_DB
os.environ["DATABASE_URL"] = f"{_PG_HOST}/{_DB_NAME}"

import psycopg2  # noqa: E402
import pytest  # noqa: E402

_TEST_DB_URL = os.environ["DATABASE_URL"]


def _ensure_worker_db_and_schema() -> None:
    """ワーカー別 DB を用意し schema を張る（非並列の base DB は作成不要）。

    無ければ `familiar_test` へ管理接続して `CREATE DATABASE`（familiar は CREATEDB 権限）。
    並列起動時は複数ワーカーが同時に template1 を触って `being accessed` になり得るので
    retry する。作成後 apply_migrations で全 schema を張る（直接 psycopg2 で引くテストが
    最初に来ても表があるように、遅延適用でなくここで確定させる）。
    """
    if _DB_NAME != _BASE_DB:
        for attempt in range(10):
            try:
                admin = psycopg2.connect(f"{_PG_HOST}/{_BASE_DB}")
                admin.autocommit = True
                with admin.cursor() as cur:
                    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (_DB_NAME,))
                    if not cur.fetchone():
                        cur.execute(f'CREATE DATABASE "{_DB_NAME}"')
                admin.close()
                break
            except psycopg2.Error:
                time.sleep(0.3 * (attempt + 1))
        else:
            return  # 作成できなければ諦める（そのワーカーの DB テストは失敗する）
    from familiar_agent.db_migrations import apply_migrations, default_migration_dir

    conn = psycopg2.connect(_TEST_DB_URL)
    apply_migrations(conn, default_migration_dir())
    conn.commit()
    conn.close()


try:
    _ensure_worker_db_and_schema()
except psycopg2.Error:
    # DB が起動していないときは収集ごと落とさない（DB を使わない純テストは走れる）。
    # DB を使うテストは autouse clean_db／各テストの接続で従来どおり失敗する。
    pass

# Reserved person IDs (mirrors migration 010)
_AGENT_SELF_ID = "00000000-0000-0000-0000-000000000000"
_DEFAULT_PERSON_ID = "00000000-0000-0000-0000-000000000001"

_TRUNCATE_TABLES = [
    "timers",
    "situated_memories",
    "obs_embeddings",
    "episode_memories",
    "memory_salience",
    "memory_jobs",
    "memory_events",
    "episodes",
    "pending_speech",
    "observations",
    "persons",
    "mental_state_log",
    "agent_state",
]


def _reset_db_singleton() -> None:
    """Close and clear the Database singleton so the next test gets a fresh connection."""
    try:
        import familiar_agent.db as db_module

        with db_module._INSTANCE_LOCK:
            if db_module._INSTANCE is not None:
                try:
                    db_module._INSTANCE.close()
                except Exception:
                    pass
                db_module._INSTANCE = None
    except Exception:
        pass


def _truncate_all() -> None:
    """Truncate all test data tables. Silently skips tables that don't exist yet."""
    try:
        conn = psycopg2.connect(_TEST_DB_URL)
        conn.autocommit = True  # each statement is its own transaction
        with conn.cursor() as cur:
            for table in _TRUNCATE_TABLES:
                try:
                    cur.execute(f"TRUNCATE TABLE {table} CASCADE")
                except Exception:
                    pass  # table may not exist yet
            # Re-insert reserved persons removed by TRUNCATE CASCADE
            now = "2026-01-01T00:00:00"
            for pid, name, display in [
                (_AGENT_SELF_ID, "__self__", "Agent self"),
                (_DEFAULT_PERSON_ID, "default", "Default Person"),
            ]:
                try:
                    cur.execute(
                        "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
                        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                        (pid, name, display, now, now),
                    )
                except Exception:
                    pass
        conn.close()
    except Exception:
        pass


def assert_database_url_untouched() -> None:
    """`DATABASE_URL` がテスト DB のままか。違えば**その場で落とす**（環-r・2026-09-19）。

    `reload_env()` が本物の `.env` を読み、`DATABASE_URL` が本番（5432）に変わったまま後のテストが走り、
    本番 `familiar_ai` に 168 行の観測が書かれた（00:22〜01:21 JST・片付け済み）。テストは
    `os.environ["DATABASE_URL"]` から直に接続するものが多く、変わった瞬間から本番へ書く。番人は
    各テストの後ろで見て、変えたテストの名前で落とす（次のテストが本番へ書く前に止める）。
    """
    now = os.environ.get("DATABASE_URL")
    if now != _TEST_DB_URL:
        os.environ["DATABASE_URL"] = _TEST_DB_URL  # 後ろのテストを守ってから落とす
        raise RuntimeError(
            f"DATABASE_URL がテスト中に変わった（本番へ書く前に止めた）：{now!r} → テスト DB に戻した。"
            "本物の .env を読み込んだテストを直す（reload_env／load_dotenv に path を渡す・_base_env_path を差し替える）"
        )


@pytest.fixture(autouse=True)
def clean_db():
    """Isolate each test: reset singleton + truncate tables before and after."""
    _reset_db_singleton()
    _truncate_all()
    yield
    assert_database_url_untouched()  # 本番へ書く前に止める（環-r）
    _reset_db_singleton()  # close open transactions before TRUNCATE
    _truncate_all()


@pytest.fixture(autouse=True)
def _measure_log_in_tmp(tmp_path, monkeypatch):
    """計測ログ（`rest_logs/measure.log`）を試験ごとの一時 dir に向ける。

    層 3（`rest_settings.adjust_settings`）は `measure.read_rows()`／`rotate()` を既定の場所
    （`~/.cache/familiar-ai`）で呼ぶ。向け先を変えないと、全体テストが**実機の計測ログを読んで
    改名する**（2026-09-14・19:53 と 20:33 に 2 回起きた）。`setup(base_dir=…)` を自分で呼ぶ
    試験はそちらが優先される。
    """
    from familiar_agent.core import measure

    monkeypatch.setattr(measure, "default_base_dir", lambda: tmp_path)
    yield


@pytest.fixture(autouse=True)
def _no_real_camera_thread(monkeypatch):
    """`CameraTool` を作っても、本物の裏方スレッド（映像の取り込み）を立てない。

    `CameraTool` は作った瞬間にスレッドを立てて映像の接続を開きに行く。届かないアドレスで作る
    試験が 1 件ごとに 1 本（計 17 本）を残し、30 秒の時間切れがプロセスの終了と重なると、全件通過の
    あとに終了コード 134 で落ちた（2026-09-25・全体テストで 3 回・`test_no_real_camera_thread_in_tests.py`）。
    """
    from familiar_agent.tools.camera import CameraTool

    monkeypatch.setattr(CameraTool, "start", lambda self: None)
    yield


@pytest.fixture(autouse=True)
def _tests_open_the_window(request, monkeypatch):
    """ループの仕組みを確かめる試験では、会話入力の窓の判定を「受けて窓を開ける」にする（出-au 段 1-2）。

    出-au で、キーボードも名前（ウェイクワード）が無ければ窓の外として捨てるようにした。ループの試験の多くは
    名前の無い素の文を `push_utterance` に渡しており、以前のキーボードの扱い（いつでも受けて窓を開ける）を
    前提にしている。門そのものを確かめる試験は `@pytest.mark.real_window`（ファイルなら `pytestmark`）を付けて、
    本物の判定を使う。
    """
    if request.node.get_closest_marker("real_window") is not None:
        yield
        return
    from familiar_agent.loop.event_loop import InformationProcessing

    def _admits(self, trigger):
        self._wake_window().open(self._arrival(trigger))
        return True

    monkeypatch.setattr(InformationProcessing, "_window_admits", _admits)
    yield


@pytest.fixture(autouse=True)
def _async_stuck_guard(monkeypatch):
    """`asyncio.run` が終わらなければ、待っていたタスクの場所を載せて失敗させる（環-aa・2026-09-26）。

    全体テストが 14 回中 3 回、`test_event_loop.py` の別々の試験で 120 秒の上限に達して止まった。止まったときに
    生きていたのはメインのスレッドだけで、**どのタスクが何を待っていたかは残らなかった**（再現を狙った 9 回では
    出なかった）。次に止まったとき必ず原因が残るようにする。秒数は `TEST_ASYNC_STUCK_SEC`（既定 90・pytest の
    上限 120 より短く）。pytest-asyncio の試験（`asyncio.run` を使わない）は包まない。
    """
    import asyncio
    import io

    real_run = asyncio.run

    def guarded(main, *args, **kwargs):
        async def guard():
            task = asyncio.ensure_future(main)
            limit = float(os.environ.get("TEST_ASYNC_STUCK_SEC", "90"))
            done, _ = await asyncio.wait({task}, timeout=limit)
            if done:
                return task.result()
            trace = io.StringIO()
            for t in asyncio.all_tasks():
                if t is asyncio.current_task():
                    continue
                trace.write(f"-- {t!r:.200}\n")
                t.print_stack(limit=15, file=trace)
            task.cancel()
            await asyncio.wait({task}, timeout=1.0)
            raise TimeoutError(
                f"asyncio.run が {limit:g} 秒で終わらなかった。待っていたタスク：\n{trace.getvalue()}"
            )

        return real_run(guard(), *args, **kwargs)

    monkeypatch.setattr(asyncio, "run", guarded)
    yield
