"""聞き取れていない書き起こしで I のループを起こさない。

実機（GUI・マイク）で、周囲の会話を拾った書き起こしがそのままターンを起こした。

    （聞き取り不能） え、サボさんもこうやってさ、なんていうの？あのー、…

`_looks_like_audio_event` は「括弧を外すと**何も残らない**」ときだけ真になるので、
印の後ろに文が続くと通り抜ける。結果、聞き返し→その声をまた拾う→また聞き返す、で
35秒に7回喋った。

STT 自身が「聞き取れていない」と印を付けているものを LLM へ渡す理由がない。**印を
含むなら落とす**（ループを起こす前に捨てる）。
"""

from __future__ import annotations

from familiar_agent.realtime_stt_session import should_skip_stt


def test_marker_with_trailing_speech_is_dropped():
    text = "（聞き取り不能） え、サボさんもこうやってさ、なんていうの？あのー、選手の練習やけんさ"
    assert should_skip_stt(text) is True


def test_marker_alone_is_dropped():
    assert should_skip_stt("（聞き取り不能）") is True


def test_marker_in_the_middle_is_dropped():
    assert should_skip_stt("それでね（聞き取り不能）だと思うよ") is True


def test_plain_speech_is_kept():
    assert should_skip_stt("今日の天気を調べて") is False


def test_ordinary_parentheses_are_kept():
    # 括弧を使うだけの発話まで落とすと、普通の話が届かなくなる。
    assert should_skip_stt("あれ（きのう話したやつ）どうなった？") is False
