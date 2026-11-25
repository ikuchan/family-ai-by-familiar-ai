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

    /** 端末の時計アプリでタイマーを掛ける（端-c c6a・b2）。`seconds` は 1〜86400。 */
    data class Timer(val seconds: Int, val label: String) : FromMachine

    /** 端末の時計アプリでアラームを掛ける（端-c c6a・b2）。時と分は端末の時刻（過ぎていれば翌日に鳴るのは時計アプリ）。 */
    data class Alarm(val hour: Int, val minute: Int, val label: String) : FromMachine

    /** 端末の Spotify を開いて示す（端-c c6b 段 1）。`uri` は Spotify の URI で、空なら Spotify を開くだけ。 */
    data class Music(val uri: String) : FromMachine
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
            "timer" -> FromMachine.Timer(
                seconds = whole(m["seconds"], 1..86_400) ?: return null,
                label = m["label"] as? String ?: return null,
            )
            "alarm" -> FromMachine.Alarm(
                hour = whole(m["hour"], 0..23) ?: return null,
                minute = whole(m["minute"], 0..59) ?: return null,
                label = m["label"] as? String ?: return null,
            )
            "music" -> FromMachine.Music((m["uri"] as? String)?.takeIf(::isSpotify) ?: return null)
            else -> null
        }
    }

    /** Spotify の URI（`spotify:` のあとに空白の無い ASCII の字が 1 字以上・機体の `_device_music` と同じ）か空だけを受ける。
     * 機体の字でほかのアプリやページを開かせない。 */
    private fun isSpotify(uri: String): Boolean =
        uri.isEmpty() || (uri.length > PREFIX.length && uri.startsWith(PREFIX) && uri.all { it in '!'..'~' })

    private const val PREFIX = "spotify:"

    /** 言葉と押した話者を、機体へ送る字にする。機体が捨てる言葉（空・長すぎる）は null。 */
    fun encodeSay(text: String, speaker: String): String? {
        val t = text.trim(::isPySpace)
        val p = speaker.trim(::isPySpace)
        if (t.isEmpty() || letters(t) > TEXT_MAX || letters(p) > NAME_MAX) return null
        return """{"type":"say","text":${Json.quote(t)},"speaker":${Json.quote(p)}}"""
    }

    /** 範囲の中の整数（Json の数は Double で届く）。時計アプリが受ける範囲（Android の AlarmClock）の外は null。 */
    private fun whole(v: Any?, range: IntRange): Int? {
        val d = v as? Double ?: return null
        return d.toInt().takeIf { it.toDouble() == d && it in range }
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
