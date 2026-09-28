"""GUI の待ちの表示は、埋め込みの準備が済めば「考え中」になる（環-ab・2026-09-28）。

「初期化中…」を出すのは、`_turn_count == 0`（まだ 1 度も話していない）か、埋め込みの準備がまだのときだった。
`_turn_count` を増やしていたのは旧 `run()` で、撤去後はいつも 0 のまま——GUI はいつ話しかけても待ちのあいだ
「初期化中…」と出していた。表示を切り替える合図（`on_phase`）も、渡す先の `agent.run` がもう呼ばない。
起動直後かどうかは、埋め込みの準備だけで決める。
"""

from __future__ import annotations

import inspect


def test_the_waiting_text_does_not_look_at_the_turn_count():
    from familiar_agent.gui import FamiliarWindow

    src = inspect.getsource(FamiliarWindow._run_agent)
    assert "_turn_count" not in src
    assert "is_embedding_ready" in src  # 起動直後の判断は残る
