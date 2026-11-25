package jp.paju.core

/** 機体から届く字（課題9 §6.6）。 */
sealed interface FromMachine {
    /** つないだすぐあと。返事の声の設定、パジュの呼び名（名前を呼ぶ合図）、話者のボタンに並べる家族の呼び名、いま話せないか。 */
    data class Hello(
        val model: String,
        val voice: String,
        val names: List<String>,
        val speakers: List<String>,
        val busy: Boolean,
    ) : FromMachine

    /** 返事。`text` は声にする字（声のタグを残す）、`display` は画面の字。 */
    data class Say(val text: String, val display: String) : FromMachine

    /** 考え中の合図の種類が変わった（b1）。 */
    data class Cue(val kind: String) : FromMachine

    /** 求めが閉じた（返事の無いまま閉じても届く）。 */
    data object Closed : FromMachine

    /** 送った言葉を断った（窓が家かほかの端末のもの）。 */
    data object Refused : FromMachine

    /** この端末が話せないかが変わった。 */
    data class Window(val busy: Boolean) : FromMachine
}

/** 端末の口の字の読み書き。上限は機体の `io/device_link.py` と同じ。 */
object Wire {
    const val TEXT_MAX = 2000
    const val NAME_MAX = 32

    /** 機体の字を読む。読めない字・知らない種類は null（機体と同じく捨てる）。 */
    fun decode(raw: String): FromMachine? {
        val m = try {
            Json.parse(raw) as? Map<*, *>
        } catch (e: IllegalArgumentException) {
            null
        } ?: return null
        return when (m["type"]) {
            "hello" -> {
                val v = m["voice"] as? Map<*, *> ?: return null
                FromMachine.Hello(
                    model = v["model"] as? String ?: return null,
                    voice = v["voice"] as? String ?: return null,
                    names = strings(m["names"]) ?: return null,
                    speakers = strings(m["speakers"]) ?: return null,
                    busy = m["busy"] as? Boolean ?: return null,
                )
            }
            "say" -> FromMachine.Say(
                text = m["text"] as? String ?: return null,
                display = m["display"] as? String ?: return null,
            )
            "cue" -> FromMachine.Cue(m["kind"] as? String ?: return null)
            "closed" -> FromMachine.Closed
            "refused" -> FromMachine.Refused
            "window" -> FromMachine.Window(m["busy"] as? Boolean ?: return null)
            else -> null
        }
    }

    /** 言葉と押した話者を、機体へ送る字にする。機体が捨てる言葉（空・長すぎる）は null。 */
    fun encodeSay(text: String, speaker: String): String? {
        val t = text.trim(::isPySpace)
        val p = speaker.trim(::isPySpace)
        if (t.isEmpty() || letters(t) > TEXT_MAX || letters(p) > NAME_MAX) return null
        return """{"type":"say","text":${Json.quote(t)},"speaker":${Json.quote(p)}}"""
    }

    private fun strings(v: Any?): List<String>? =
        (v as? List<*>)?.map { it as? String ?: return null }

    /** 機体（Python の `len`）と同じ数え方。絵文字も 1 字。 */
    private fun letters(s: String): Int = s.codePointCount(0, s.length)

    /** 機体（Python の `str.strip`）が落とす空白と同じ字。 */
    private fun isPySpace(c: Char): Boolean =
        c in '\u0009'..'\u000d' || c in '\u001c'..'\u0020' || c == '\u0085' || c == '\u00a0' || c == '\u1680' ||
            c in '\u2000'..'\u200a' || c == '\u2028' || c == '\u2029' || c == '\u202f' || c == '\u205f' || c == '\u3000'
}
