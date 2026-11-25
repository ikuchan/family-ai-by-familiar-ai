package jp.paju.core

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/** 考え中の合図を端末で鳴らす（課題9 端-c c4・b1）。音と間隔は機体と同じ（見本の `beeps`・`beep_interval_sec`）。 */
class BeepTest {
    private fun hello() = FromMachine.Hello("m", "v", listOf("パジュ"), listOf("ママ"), false)

    @Test
    fun theSoundsAndThePaceAreTheMachines() {
        @Suppress("UNCHECKED_CAST")
        val wire = Json.parse(File(System.getProperty("paju.wire")).readText()) as Map<String, Any?>
        assertEquals(wire["beeps"], Beeps.FILES)
        assertEquals((wire["beep_interval_sec"] as Double * 1000).toLong(), Beeps.INTERVAL_MS)
    }

    @Test
    fun beepsFromTheCueUntilTheReply() {
        val t = Talk()
        t.on(hello())
        t.sent()
        assertNull(t.beep)
        t.on(FromMachine.Cue("recalling"))
        assertEquals("react_beep_recalling.wav", t.beep)
        t.on(FromMachine.Cue("deciding"))
        assertEquals("react_thinking.wav", t.beep)
        t.on(FromMachine.Say("はい", "はい")) // 答えを言うあいだは鳴らさない（機体と同じ）
        assertNull(t.beep)
        t.on(FromMachine.Cue("checking")) // 返事のあとにまた考えれば、また鳴らす
        assertEquals("react_beep_checking.wav", t.beep)
        t.on(FromMachine.Closed)
        assertNull(t.beep)
    }

    @Test
    fun stopsWhenTheTalkEndsAnyOtherWay() {
        for (end in listOf<(Talk) -> Unit>({ it.on(FromMachine.Refused) }, { it.disconnected() }, { it.timedOut(1) })) {
            val t = Talk()
            t.on(hello())
            t.sent()
            t.on(FromMachine.Cue("noting"))
            end(t)
            assertNull(t.beep)
        }
    }

    @Test
    fun anUnknownKindIsSilent() {
        val t = Talk()
        t.on(hello())
        t.sent()
        t.on(FromMachine.Cue("recalling"))
        t.on(FromMachine.Cue("新しい子"))
        assertNull(t.beep)
    }

    @Test
    fun aLateCueAfterTheWaitEndedIsSilent() {
        val t = Talk()
        t.on(hello())
        val serial = t.sent()
        t.timedOut(serial) // 3 分待って話せる画面に戻した
        t.on(FromMachine.Cue("checking"))
        assertNull(t.beep)
    }
}
