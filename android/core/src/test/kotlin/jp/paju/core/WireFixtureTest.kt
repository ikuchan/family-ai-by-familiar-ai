package jp.paju.core

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/** 機体の試験（tests/test_device_wire_is_shared.py）と同じ見本を読み、口の形と名前の合図が機体とずれていないかを見る。 */
class WireFixtureTest {
    @Suppress("UNCHECKED_CAST")
    private val wire = Json.parse(File(System.getProperty("paju.wire")).readText()) as Map<String, Any?>

    @Test
    fun readsWhatTheMachineSends() {
        val fm = (wire["from_machine"] as List<String>).map { Wire.decode(it) }
        assertEquals(
            listOf(
                FromMachine.Hello(model = "m", voice = "v", names = listOf("パジュ"), speakers = listOf("ママ", "パパ"), busy = false),
                FromMachine.Hello(model = "m", voice = "v", names = listOf("パジュ"), speakers = listOf("ママ", "パパ"), busy = true),
                FromMachine.Say(text = "[cheerful] おはよう！", display = "おはよう！"),
                FromMachine.Cue(kind = "recalling"),
                FromMachine.Closed,
                FromMachine.Refused,
                FromMachine.Window(busy = true),
                FromMachine.Window(busy = false),
            ),
            fm,
        )
    }

    @Test
    fun writesWhatTheMachineTakes() {
        @Suppress("UNCHECKED_CAST")
        for (s in wire["to_machine"] as List<Map<String, String>>) {
            assertEquals(s["wire"], Wire.encodeSay(s["typed"]!!, s["speaker"]!!))
        }
    }

    @Test
    fun hearsTheNameByTheMachineRule() {
        @Suppress("UNCHECKED_CAST")
        val names = wire["names"] as List<String>
        @Suppress("UNCHECKED_CAST")
        for (c in wire["wake"] as List<Map<String, String>>) {
            assertEquals(Heard.valueOf(c["heard"]!!.uppercase()), Wake.hear(c["text"]!!, names), c["text"])
        }
    }

    @Test
    fun dropsWhatItCannotRead() {
        assertNull(Wire.decode("not json"))
        assertNull(Wire.decode("""{"type": "unknown"}"""))
        assertNull(Wire.decode("""["say"]"""))
        assertNull(Wire.decode("""{"type": "say", "text": 3}"""))
        assertNull(Wire.decode("[".repeat(100_000)))  // 深すぎる入れ子でも落ちずに捨てる
        assertNull(Wire.decode("""{"type": "cue", "kind": "\u+041"}"""))
    }
}
