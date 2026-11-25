package jp.paju.core

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * 機体から届いた頼み（c6）を、端末のほかのアプリへ渡す時機。時計はすぐ渡す（前に出ない）。Spotify は求めが閉じてから渡す
 * （端-c c6b 段 1）。渡すと Spotify が前に出てパジュの画面が止まり、つながりが切れる。先に渡すと、そのあとの返事と `closed` を受け取れない。
 */
class OrderTest {
    private val hello = FromMachine.Hello("m", "v", listOf("パジュ"), listOf("ママ"), false)
    private val timer = FromMachine.Timer(180, "パスタ")
    private val alarm = FromMachine.Alarm(7, 0, "起こす")
    private val music = FromMachine.Music("spotify:playlist:0")
    private val none = emptyList<FromMachine>()

    private fun waiting(): Talk = Talk().also { it.on(hello); it.sent() }

    @Test
    fun theClockIsHandedOverAtOnce() {
        val t = waiting()
        assertEquals(listOf<FromMachine>(timer), t.on(timer))
        assertEquals(listOf<FromMachine>(alarm), t.on(alarm))
        assertEquals(none, t.on(FromMachine.Closed))
    }

    @Test
    fun spotifyIsHandedOverWhenTheRequestCloses() {
        val t = waiting()
        assertEquals(none, t.on(music))
        assertEquals(none, t.on(FromMachine.Say("開くね", "開くね")))
        assertEquals(listOf<FromMachine>(music), t.on(FromMachine.Closed))
        assertEquals(none, t.on(FromMachine.Closed)) // 渡したものは 2 度渡さない
    }

    @Test
    fun spotifyIsDroppedOnlyWhenTheLinkIsCut() {
        val cut = waiting()
        cut.on(music)
        cut.disconnected()
        cut.on(hello)
        cut.sent()
        assertEquals(none, cut.on(FromMachine.Closed))
    }

    @Test
    fun spotifySurvivesARefusedWordAndTheTimeout() {
        // 機体はもう「開く」と言っている。別の言葉が断られても、返事を待つ 3 分が過ぎても、求めが閉じたら渡す
        val refused = waiting()
        refused.on(music)
        refused.sent()
        refused.on(FromMachine.Refused)
        assertEquals(listOf<FromMachine>(music), refused.on(FromMachine.Closed))

        val slow = Talk()
        slow.on(hello)
        val n = slow.sent()
        slow.on(music)
        slow.timedOut(n)
        assertEquals(listOf<FromMachine>(music), slow.on(FromMachine.Closed))
    }

    @Test
    fun spotifyOfTheNextRequestWaitsForItsOwnClose() {
        val t = waiting()
        t.sent() // 2 つめの言葉
        assertEquals(none, t.on(FromMachine.Closed)) // 1 つめが閉じた
        t.on(music) // 2 つめの求めの頼み
        assertEquals(listOf<FromMachine>(music), t.on(FromMachine.Closed))
    }
}
