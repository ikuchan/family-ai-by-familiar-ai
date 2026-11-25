package jp.paju.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/** 機体の端末の口が捨てる言葉は、アプリが送らない（空・2000 字を超える言葉・32 字を超える話者）。 */
class WireLimitsTest {
    @Test
    fun doesNotSendWhatTheMachineWouldDrop() {
        assertNull(Wire.encodeSay("   ", "ママ"))
        assertNull(Wire.encodeSay("あ".repeat(2001), "ママ"))
        assertNull(Wire.encodeSay("おはよう", "あ".repeat(33)))
    }

    @Test
    fun countsLettersAsTheMachineDoes() {
        // 機体（Python の len）は絵文字を 1 字と数える
        val emoji = "😀"
        assertNotNull(Wire.encodeSay(emoji.repeat(2000), ""))
        assertNull(Wire.encodeSay(emoji.repeat(2001), ""))
        assertNotNull(Wire.encodeSay("はい", emoji.repeat(32)))
        assertNull(Wire.encodeSay("はい", emoji.repeat(33)))
    }

    @Test
    fun sendsTrimmedWordsAtTheLimit() {
        assertEquals("""{"type":"say","text":"${"あ".repeat(2000)}","speaker":"ママ"}""", Wire.encodeSay(" ${"あ".repeat(2000)} ", " ママ "))
    }
}
