"""モデル資源（MR）の型枠が、重みを持つものの共通の関心事を引き受ける（出-c）。

**同じ形が3回、少しずつ違って書かれていた。** `_EmbeddingModel`・`VisualEncoder`・
`PersonDetector` は同じ問題を同じ順序で解いていながら、失敗フラグの名前（`_failed` と
`_load_failed`）も、読み込み口の形（`pre_warm()` ／ `_ensure_model() -> bool` ／
`-> Any`）も、載せる先の決め方（環境変数／`torch.cuda.is_available()`／指定なし）も
揃っていなかった。並行制御に至っては、3つのうち1つにしか無い。

型枠が引き受けるのは**推論そのものではなく、モデルを持つことに伴う関心事**である。
失敗の約束は**宣言で分ける**——致命なら例外、そうでなければ縮退（出-b が「埋め込みは
致命」と決めた結果が、コードの一行に出る形にする）。
"""

from __future__ import annotations

import threading

import pytest

from familiar_agent.core.model_resource import ModelResource


class _Fake(ModelResource):
    """読み込みの回数を数えるだけの模型。"""

    def __init__(self, *, fails: bool = False, **kw) -> None:
        super().__init__(**kw)
        self.loads = 0
        self._fails = fails

    def _load(self):
        self.loads += 1
        if self._fails:
            raise RuntimeError("重みが取れない")
        return object()


# ── ① 読めないときは縮退する（既定） ───────────────────────────────────────

def test_a_missing_model_degrades_instead_of_raising() -> None:
    """重みが取れない環境でも、他の機能は動き続ける。"""
    r = _Fake(name="試し", fails=True)
    assert r.ensure() is None
    assert r.ready is False


# ── ② 致命だと宣言したものは例外を投げる（反証側） ─────────────────────────

def test_a_fatal_resource_raises() -> None:
    """`fatal=True` は「これが無ければ続けられない」の宣言（出-b・埋め込みが該当）。"""
    r = _Fake(name="試し", fatal=True, fails=True)
    with pytest.raises(RuntimeError):
        r.ensure()


# ── ③ 二度目以降は試さない ─────────────────────────────────────────────────

def test_a_failed_load_is_not_retried() -> None:
    """失敗を記憶する。毎回試すと、重みの無い環境で呼ぶたび重くなる。"""
    r = _Fake(name="試し", fails=True)
    for _ in range(5):
        assert r.ensure() is None
    assert r.loads == 1


def test_a_successful_load_happens_once() -> None:
    r = _Fake(name="試し")
    first = r.ensure()
    assert r.ensure() is first
    assert r.loads == 1
    assert r.ready is True


# ── ④ 並行して呼んでも読み込みは1回だけ ───────────────────────────────────

def test_concurrent_callers_load_only_once() -> None:
    """いま `VisualEncoder` と `PersonDetector` には並行制御が無い。

    在席の常駐と想起が同時に触れば、重いモデルを二重に読む。型枠が引き受ける。
    """
    import time

    class _Slow(_Fake):
        def _load(self):
            time.sleep(0.05)          # 読み込みに時間がかかるものを模す
            return super()._load()

    r = _Slow(name="試し")
    got: list = []
    threads = [threading.Thread(target=lambda: got.append(r.ensure())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert r.loads == 1, f"{r.loads} 回読み込んでいる"
    assert len({id(g) for g in got}) == 1, "スレッドごとに別のモデルを掴んでいる"


# ── ⑤ 載せる先の決め方 ─────────────────────────────────────────────────────

def test_the_device_comes_from_the_environment_first(monkeypatch) -> None:
    """環境変数が最優先。テストは並列で走るので、GPU の奪い合いを断つために要る。"""
    monkeypatch.setenv("試し_DEVICE", "cpu")
    r = _Fake(name="試し", device_env="試し_DEVICE")
    assert r.device == "cpu"


def test_an_unset_device_means_leave_it_to_the_library(monkeypatch) -> None:
    """**指定が無ければ `None`＝ライブラリに任せる。**

    型枠が `cuda`／`cpu` のどちらかを必ず決めてしまうと、「明示せず任せる」という選択が
    できなくなる。これは埋め込み固有の都合ではない——YOLO も実際には device を渡して
    おらず（ultralytics 任せ）、埋め込みは「自動のときは device を渡さない（従来と同じ
    呼び方）」を実測（GPU の VRAM を使い切ってワーカーが落ちた）から決めている。
    どこへ載せるかを知っているのはライブラリのほうである。
    """
    monkeypatch.delenv("試し_DEVICE", raising=False)
    assert _Fake(name="試し", device_env="試し_DEVICE").device is None


def test_a_resource_without_a_device_setting_is_also_none() -> None:
    """環境変数の名前すら渡さないものは、当然 `None`（YOLO がこれ）。"""
    assert _Fake(name="試し").device is None


# ── ⑥ YOLO を型枠へ移しても、挙動は変わらない ─────────────────────────────

def test_the_person_detector_conforms_to_the_type() -> None:
    from familiar_agent.recognition.person_detector import PersonDetector

    assert issubclass(PersonDetector, ModelResource)


def test_the_person_detector_still_reports_nobody_when_the_model_is_missing() -> None:
    """読めないときは「見えなかった」（0人）。誤って「居る」と言うより安全である。"""
    import asyncio

    from familiar_agent.recognition.person_detector import PersonDetector

    d = PersonDetector(model_name="存在しない重み.pt")
    assert asyncio.run(d.count(object())) == 0


# ── ⑦ 機器やライブラリが無いのは「永続的な失敗」──────────────────────────

class _NoLibrary(_Fake):
    def _load(self):
        self.loads += 1
        raise ImportError("insightface が入っていない")


class _Flaky(_Fake):
    """2回目で読めるようになるもの（VRAM の一時不足などを模す）。"""

    def _load(self):
        self.loads += 1
        if self.loads < 2:
            raise RuntimeError("いま VRAM が足りない")
        return object()


def test_a_missing_library_is_never_retried_even_with_retries() -> None:
    """カメラやマイクが無い構成では、ライブラリごと入っていないことがある。

    これは**永続的な失敗**なので、何度試しても同じである。再試行すると、呼ぶたびに
    数秒を失うだけになる（`load_whisper_model` は書き起こしのたびに呼ばれる）。
    """
    r = _NoLibrary(name="試し", retries=5)
    for _ in range(4):
        assert r.ensure() is None
    assert r.loads == 1, f"永続的な失敗を {r.loads} 回試している"


def test_a_missing_weight_file_is_never_retried() -> None:
    class _NoWeights(_Fake):
        def _load(self):
            self.loads += 1
            raise FileNotFoundError("重みが無い")

    r = _NoWeights(name="試し", retries=5)
    r.ensure(); r.ensure()
    assert r.loads == 1


def test_a_temporary_failure_is_retried_up_to_the_limit() -> None:
    """一時的な失敗は待って試す。いまは一度失敗すると再起動まで二度と読まない。"""
    r = _Flaky(name="試し", retries=2)
    assert r.ensure() is None      # 1回目は失敗
    assert r.ensure() is not None  # 2回目で読めた
    assert r.loads == 2


def test_retries_are_exhausted_and_then_remembered() -> None:
    """使い切ったら記憶する。**永久に試し続けない。**"""
    r = _Fake(name="試し", fails=True, retries=2)
    for _ in range(6):
        assert r.ensure() is None
    assert r.loads == 3, f"上限（1+2）を超えて {r.loads} 回試している"


# ── ⑧ 先読み（起動時・別スレッド・起動を止めない）─────────────────────────

def test_pre_warm_loads_in_the_background() -> None:
    """起動時に読み始めて、最初の呼び出しを待たせない。**起動は止めない。**"""
    import time

    class _Slow(_Fake):
        def _load(self):
            time.sleep(0.05)
            return super()._load()

    r = _Slow(name="試し")
    started = time.monotonic()
    r.pre_warm()
    assert time.monotonic() - started < 0.04, "先読みが起動を止めている"

    assert r.ensure() is not None    # 読み終わるまで待って受け取る
    assert r.loads == 1, "先読みと本読みで二重に読んでいる"
