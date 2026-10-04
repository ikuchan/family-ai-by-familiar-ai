"""名乗りで話者を付ける（知-w・2026-09-19）。在席があるときだけ。

「パジュ、こんにちはパパだよ」がカメラに映らない位置からの声で入口に飲まれた（実機 11:32）。在席（居るか）は
カメラだけで決める（マイクは証拠にしない・変えない）。**カメラが人を見ているとき**は、声の名乗り（「パパだよ」
「僕はたいき」）から話者を付ける（`/speaker` と同じ効き：`set_active`・`_speaker_set_at`・PMM 同期）。
名乗りの読みは調停（`Decision.speaker_claim`）、検めは機械（在席あり・家族の名前か呼び名に一致）。
道具の帰りの反復では読まない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core.speaker_claim import resolve_claim
from familiar_agent.config import RecognitionConfig
from familiar_agent.loop import arbiter
from tests._arbiter_fakes import decide, jev_says, writer_says
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent

FAMILY = "## ゆうすけ\n- **名前**：ゆうすけ\n- **呼び方**：パパ\n- **英字**：Yusuke Ikunaga\n\n## たいき\n- **名前**：たいき\n- **呼び方**：たいき\n- **英字**：Taiki Ikunaga\n"


def test_a_claim_resolves_to_a_family_name_or_nothing():
    assert resolve_claim("パパ", FAMILY) == "パパ"
    assert (
        resolve_claim("ゆうすけ", FAMILY) == "パパ"
    )  # 名前で名乗っても呼び方に寄せる（/speaker パパ と同じ表記）
    assert resolve_claim("たいき", FAMILY) == "たいき"
    assert resolve_claim("太郎", FAMILY) is None
    assert resolve_claim("", FAMILY) is None


#: 呼び方が**複数**の家族。実物の `FAMILY.md` はこの形（「パパ、ゆうすけ、おとうさん、Papa、father」）。
FAMILY_ALIASES = (
    "## ゆうすけ\n- **名前**：ゆうすけ\n- **呼び方**：パパ、ゆうすけ、おとうさん、Papa、father\n"
    "\n## たえこ\n- **名前**：たえこ\n- **呼び方**：ママ、たえこ、おかあさん\n"
)


def test_any_one_of_the_ways_we_call_them_resolves():
    """**個々の呼び方**で当たる（出-ae-は・2026-09-22）。

    直す前は呼び方の一覧まるごととしか比べておらず、いちばん自然な「パパだよ」「ママだよ」で
    話者が付かなかった。当たるのは `名前` の欄と一致する言い方だけだった。
    """
    assert resolve_claim("パパ", FAMILY_ALIASES) == "パパ"
    assert resolve_claim("おとうさん", FAMILY_ALIASES) == "パパ"
    assert resolve_claim("Papa", FAMILY_ALIASES) == "パパ"
    assert resolve_claim("ママ", FAMILY_ALIASES) == "ママ"
    assert resolve_claim("おかあさん", FAMILY_ALIASES) == "ママ"


def test_the_name_field_still_resolves():
    assert resolve_claim("ゆうすけ", FAMILY_ALIASES) == "パパ"
    assert resolve_claim("たえこ", FAMILY_ALIASES) == "ママ"


def test_what_comes_back_is_one_name_not_the_whole_list():
    """返すのは**呼びかけに使う名前**（先頭）。一覧のまま `set_active` へ入れない。"""
    got = resolve_claim("ゆうすけ", FAMILY_ALIASES)
    assert "、" not in str(got)


def test_someone_outside_the_family_still_resolves_to_nothing():
    assert resolve_claim("太郎", FAMILY_ALIASES) is None


def test_the_arbiter_carries_the_claim_and_drops_it_on_a_tool_return():
    def claim(**kw):
        return asyncio.run(
            decide(
                jev=jev_says("light", claimed="パパ"),
                writer=writer_says({"text": "パパ、おかえり"}),
                utterance="パパだよ",
                family_md=FAMILY,
                **kw,
            )
        )

    assert claim().speaker_claim == "パパ"
    # 道具の帰りでは、Jev の答えに混じっていても読まない（機械の守り・情-n）。
    assert claim(tool_return=True).speaker_claim == ""

    # 名乗りは Jev に問う（出-au 段 5-7d）。道具の帰りでは問わない。
    def asked(**kw):
        return arbiter.Arbiter(jev=None, writer=None)._questions(
            arbiter.ArbiterInput(utterance="パパだよ", workspace_ctx="", **kw)
        )

    assert "名乗" in asked()["claims"]["instructions"] and "claimed" in asked()
    assert "claims" not in asked(tool_return=True)


def _ip(*, present: float, score: "float | None" = 0.9):
    """名乗った本人の声への似かた（`_claim_score`）を差し替える。知-ai から、名乗りは声が本人に当たるときだけ付く。"""
    a = _agent(stream_returns=[])
    a._family_md = FAMILY
    a._occupancy = MagicMock(return_value=present)
    a._persons.active_name = "推定話者"
    a._sync_pmm_speaker = AsyncMock()
    a.config.recognition = RecognitionConfig()
    ip = InformationProcessing(a)
    ip._req.voice = [1.0]  # 声の特徴がある発話
    ip._claim_score = lambda pid, voice: score  # type: ignore[method-assign]
    ip._learn_voice = MagicMock()  # type: ignore[method-assign]
    return ip, a


def test_a_claim_whose_voice_matches_sets_the_speaker_like_the_command():
    """在席（カメラ）は見ない（知-ai）。声が本人に当たれば `/speaker` と同じ効き。"""
    ip, a = _ip(present=0.0)
    d = arbiter.Decision(branch="light", text="x", speaker_claim="ゆうすけ")
    asyncio.run(ip._apply_speaker_claim(d))
    a._persons.set_active.assert_called_once_with("パパ")
    assert isinstance(a._speaker_set_at, float)
    a._sync_pmm_speaker.assert_awaited_once_with("パパ")


def test_a_short_voice_or_unknown_name_sets_nothing():
    ip, a = _ip(present=1.0, score=0.29)  # 声が本人に届かない
    asyncio.run(
        ip._apply_speaker_claim(arbiter.Decision(branch="light", text="x", speaker_claim="パパ"))
    )
    a._persons.set_active.assert_not_called()
    ip, a = _ip(present=1.0)
    asyncio.run(
        ip._apply_speaker_claim(arbiter.Decision(branch="light", text="x", speaker_claim="太郎"))
    )
    a._persons.set_active.assert_not_called()
    a._persons.active_name = "パパ"
    asyncio.run(
        ip._apply_speaker_claim(arbiter.Decision(branch="light", text="x", speaker_claim="パパ"))
    )
    a._persons.set_active.assert_not_called()  # 既に同じ話者なら何もしない


# 名乗りの預かり（知-w-ろ・在席が無い名乗りを 30 秒預かる）は知-ai で撤去した。名乗りは在席を見ず、声が本人に
# 当たるときだけ付く（`test_voice_without_occupancy`）。
