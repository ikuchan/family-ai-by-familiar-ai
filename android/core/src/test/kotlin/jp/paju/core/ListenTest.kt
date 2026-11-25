package jp.paju.core

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/** 声の書き起こしから、機体へ送る言葉を選ぶ（課題9 端-c c3・§7.4「会話が続くあいだ」）。 */
class ListenTest {
    private val names = listOf("パジュ")

    @Test
    fun sendsWhatStartsWithTheName() {
        val ear = Listen()
        assertEquals("パジュ、明日の天気は？", ear.heard(" パジュ、明日の天気は？ ", names, Screen.FREE, 0))
    }

    @Test
    fun doesNotSendTalkThatIsNotToPaju() {
        val ear = Listen()
        assertNull(ear.heard("明日の天気は？", names, Screen.FREE, 0))
        assertNull(ear.heard("ねえパジュ", names, Screen.FREE, 0))
    }

    @Test
    fun theNameAloneWaitsForTheWordsThatFollow() {
        val ear = Listen()
        assertNull(ear.heard("パジュ", names, Screen.FREE, 0))
        assertEquals("明日の天気は？", ear.heard("明日の天気は？", names, Screen.FREE, 9_000))
    }

    @Test
    fun theWaitEndsAfterTheMachineWindow() {
        val ear = Listen()
        assertNull(ear.heard("パジュ", names, Screen.FREE, 0))
        assertNull(ear.heard("明日の天気は？", names, Screen.FREE, Listen.WINDOW_MS))
    }

    @Test
    fun talkGoesOnWithoutTheNameAfterAReply() {
        val ear = Listen()
        ear.replied(100_000)
        assertEquals("それでね", ear.heard("それでね", names, Screen.FREE, 105_000))
        ear.sent(105_000) // 送れた言葉も会話を延ばす
        assertEquals("あとね", ear.heard("あとね", names, Screen.FREE, 114_000))
        assertNull(ear.heard("おやすみ", names, Screen.FREE, 125_000))
    }

    @Test
    fun wordsNotSentDoNotKeepTheTalkGoing() {
        val ear = Listen()
        assertEquals("パジュ、調べて", ear.heard("パジュ、調べて", names, Screen.FREE, 0))
        // 送れなかった（長すぎた・切れていた）ので sent を呼ばない。名前なしの声は送らない
        assertNull(ear.heard("それでね", names, Screen.FREE, 1_000))
    }

    @Test
    fun whileWaitingOnlyTheWordsRightAfterTheLastSendAreSent() {
        // 返事を待つあいだのそばの話し声（テレビ・家族どうし）を送り続けない
        val ear = Listen()
        assertEquals("パジュ、調べて", ear.heard("パジュ、調べて", names, Screen.FREE, 0))
        ear.sent(0)
        assertEquals("やっぱり明日の", ear.heard("やっぱり明日の", names, Screen.WAITING, 5_000))
        assertNull(ear.heard("それでさ", names, Screen.WAITING, 60_000))
    }

    @Test
    fun cuttingEndsTheTalk() {
        val ear = Listen()
        ear.replied(0)
        ear.cut() // 切れた・グレーになった（声を聞かないうちに）
        assertNull(ear.heard("それでね", names, Screen.FREE, 4_000))
    }

    @Test
    fun nothingIsSentWhenPajuCannotTalk() {
        val ear = Listen()
        ear.replied(0)
        assertNull(ear.heard("パジュ、おはよう", names, Screen.BUSY, 1_000))
        assertNull(ear.heard("パジュ、おはよう", names, Screen.OFFLINE, 1_000))
        // グレーのあいだに会話は切れる。話せるようになっても、名前なしの声は送らない
        assertNull(ear.heard("それでね", names, Screen.FREE, 2_000))
    }

    @Test
    fun theNameRepeatedIsNotSentEvenInTheConversation() {
        val ear = Listen()
        ear.replied(0)
        assertNull(ear.heard("パジュパジュ", names, Screen.FREE, 1_000))
        assertNull(ear.heard("  ", names, Screen.FREE, 1_000))
    }

    @Test
    fun theWindowIsAsLongAsTheMachineWindow() {
        @Suppress("UNCHECKED_CAST")
        val wire = Json.parse(File(System.getProperty("paju.wire")).readText()) as Map<String, Any?>
        assertEquals((wire["window_sec"] as Double * 1000).toLong(), Listen.WINDOW_MS)
    }
}
