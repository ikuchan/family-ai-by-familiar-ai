"""版の content を組み立てる（設計方針『求めの版チェーン』V2）。

求めは1本の版チェーンとして進む。各版が求めの状態であり、content にはその時点の全体
（何を聞かれ、いま何を待ち、何が届いたか）が書かれる。

```
版1  「昨日の天気覚えてる？」と聞かれた
版2  「昨日の天気覚えてる？」と聞かれ、1番：search「昨日 天気」を起動中
版3  「昨日の天気覚えてる？」と聞かれ、1番：search「昨日 天気」の結果が届いた：…
```

並行する調査は**求めの中の通し番号**で列挙し、鎖は分岐させない。状態は「求め」の状態で
あって個々の調査の状態ではないので、3件飛んでいても求めの状態はひとつである。

ここでは content を組み立てる処理だけを見る。版を実際に書いて前の版を畳むのは V3 である。
"""

from __future__ import annotations

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip(origin: str = "昨日の天気覚えてる？"):
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._origin_text = origin
    return ip


def test_version_one_is_just_the_request() -> None:
    """版1 は求めそのもの。まだ何も調べていない。"""
    ip = _ip()
    assert ip._version_content() == "「昨日の天気覚えてる？」と聞かれた"


def test_in_flight_is_listed_with_its_index() -> None:
    """飛行中の調査を通し番号つきで並べる。"""
    ip = _ip()
    ip._in_flight_lookups = [("search_deferred", "昨日 天気", 1)]
    got = ip._version_content()
    assert "「昨日の天気覚えてる？」と聞かれ" in got, "求めが落ちている"
    assert "1番：search_deferred「昨日 天気」を起動中" in got, f"飛行中が並んでいない: {got}"


def test_parallel_lookups_are_all_listed() -> None:
    """並行する調査は全部並べる（鎖は分岐させない）。"""
    ip = _ip("今日の天気は？")
    ip._in_flight_lookups = [
        ("search_deferred", "今日 天気", 1),
        ("fetch_deferred", "example.com/a", 2),
    ]
    got = ip._version_content()
    assert "1番：search_deferred「今日 天気」を起動中" in got
    assert "2番：fetch_deferred「example.com/a」を起動中" in got


def test_arrived_results_are_listed_with_their_index() -> None:
    """届いた結果は通し番号つきで並べる。"""
    ip = _ip()
    ip._lookup_results = [(1, "search_deferred", "昨日 天気", "晴れだった")]
    got = ip._version_content()
    assert "1番：search_deferred「昨日 天気」の結果が届いた：晴れだった" in got, (
        f"結果が並んでいない: {got}"
    )


def test_arrived_and_in_flight_are_both_listed() -> None:
    """届いた分と待っている分が混在しても、両方を番号で並べる。"""
    ip = _ip("今日の天気は？")
    ip._lookup_results = [(1, "search_deferred", "今日 天気", "リンク一覧")]
    ip._in_flight_lookups = [("fetch_deferred", "example.com/a", 2)]
    got = ip._version_content()
    assert "1番：search_deferred「今日 天気」の結果が届いた：リンク一覧" in got
    assert "2番：fetch_deferred「example.com/a」を起動中" in got


def test_the_result_body_is_not_truncated_here() -> None:
    """結果は content の組み立てでは切らない。

    切るなら書き込みの上限で切る。ここで切ると、どこで短くなったのかが追えなくなる。
    """
    long_body = "あ" * 3000
    ip = _ip()
    ip._lookup_results = [(1, "recall", "語", long_body)]
    assert long_body in ip._version_content(), "組み立てで切っている"


def test_aborted_version() -> None:
    """打ち切りも版のひとつ。何を打ち切ったかを残す。"""
    ip = _ip()
    ip._in_flight_lookups = [("search_deferred", "昨日 天気", 1)]
    got = ip._version_content(aborted=True)
    assert "打ち切った" in got, f"打ち切りが分からない: {got}"
    assert "1番：search_deferred「昨日 天気」" in got, "何を打ち切ったかが残っていない"


def test_origin_is_carried_in_every_version() -> None:
    """どの版にも求めそのものが入る（前の版は畳まれて辿れなくなる）。"""
    ip = _ip("たいきのサッカーは？")
    ip._lookup_results = [(1, "recall", "サッカー", "水曜")]
    ip._in_flight_lookups = [("search_deferred", "会場", 2)]
    for kw in ("たいきのサッカーは？", "1番：recall「サッカー」の結果が届いた", "2番："):
        assert kw in ip._version_content(), f"{kw} が入っていない"
