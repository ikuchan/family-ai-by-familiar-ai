package jp.paju.core

import kotlin.test.Test
import kotlin.test.assertEquals

/** 話す画面の状態（課題9 §7.4「端末での断り方」）：つながっていない・話せる・グレー・返事を待つ。 */
class TalkTest {
    private fun hello(busy: Boolean) = FromMachine.Hello("m", "v", listOf("パジュ"), listOf("ママ"), busy)

    @Test
    fun followsTheWindowAndTheRequest() {
        val t = Talk()
        assertEquals(Screen.OFFLINE, t.screen)
        t.on(hello(busy = true))
        assertEquals(Screen.BUSY, t.screen)
        t.on(FromMachine.Window(busy = false))
        assertEquals(Screen.FREE, t.screen)
        t.sent()
        assertEquals(Screen.WAITING, t.screen)
        t.on(FromMachine.Say("はい", "はい"))
        assertEquals(Screen.WAITING, t.screen)
        t.on(FromMachine.Closed)
        assertEquals(Screen.FREE, t.screen)
        t.sent()
        t.on(FromMachine.Refused)
        assertEquals(Screen.FREE, t.screen)
        t.sent()
        t.disconnected()
        assertEquals(Screen.OFFLINE, t.screen)
    }

    @Test
    fun aBusyWindowGreysEvenWhileWaiting() {
        val t = Talk()
        t.on(hello(busy = false))
        t.sent()
        t.on(FromMachine.Window(busy = true))
        assertEquals(Screen.BUSY, t.screen)
    }

    @Test
    fun theSpeakerStaysUntilChanged() {
        val t = Talk(speaker = "ママ")
        t.on(hello(busy = false))
        t.disconnected()
        assertEquals("ママ", t.speaker)
        t.speaker = "パパ"
        assertEquals("パパ", t.speaker)
    }

    @Test
    fun waitingEndsAfterTheTimeoutButOnlyForTheSameWords() {
        // 機体の返事が来ないまま（求めが例外で終わったなど）でも、返事を待ち続けない（c2）
        val t = Talk()
        t.on(hello(busy = false))
        val first = t.sent()
        val second = t.sent()
        t.timedOut(first)  // 前の言葉の時間切れは、あとの言葉の待ちを解かない
        assertEquals(Screen.WAITING, t.screen)
        t.timedOut(second)
        assertEquals(Screen.FREE, t.screen)
    }
}
