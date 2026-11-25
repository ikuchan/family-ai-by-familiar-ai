package jp.paju.core

/**
 * 声の書き起こしから、機体へ送る言葉を選ぶ（課題9 端-c c3）。
 *
 * 名前で始まる言葉は送る。名前だけなら送らず、続く言葉を名前なしで受ける。返事のあとも、会話が続くあいだ（§7.4・家なら名前で
 * 開いた窓、端末なら返事から同じ長さ）は名前なしの言葉を送る。機体は端末に、その端末の窓が開いているかを送らないので、窓の長さ
 * （[WINDOW_MS]・機体の `WAKE_WINDOW_SEC` と見本で揃える）をアプリが自分で数える。返事を待つあいだも、名前なしの言葉は送った言葉
 * から同じ長さだけ受ける（そばの話し声を送り続けない）。話せなくなったら（つながっていない・グレー・[cut]）会話を切る。
 * 時刻は呼ぶ側が渡す（ミリ秒・話し始めた時刻）。会話を延ばすのは、送れた言葉（[sent]）と返事（[replied]）だけ。
 */
class Listen {
    private var until = Long.MIN_VALUE

    fun heard(text: String, names: List<String>, screen: Screen, now: Long): String? {
        if (screen == Screen.OFFLINE || screen == Screen.BUSY) {
            cut()
            return null
        }
        val words = text.trim()
        if (words.isEmpty()) return null
        return when (Wake.hear(words, names)) {
            Heard.NAMED -> words
            Heard.NAME_ONLY -> null.also { until = now + WINDOW_MS }
            Heard.NONE -> if (now < until && !Wake.repeats(words, names)) words else null
        }
    }

    /** 言葉を機体へ送れた。そこから会話が続く。 */
    fun sent(now: Long) {
        until = now + WINDOW_MS
    }

    /** 返事が届いた・求めが閉じた。そこから会話が続く。 */
    fun replied(now: Long) {
        until = now + WINDOW_MS
    }

    /** 話せなくなった（切れた・窓がほかの口のものになった・画面を閉じた）。会話を切る。 */
    fun cut() {
        until = Long.MIN_VALUE
    }

    companion object {
        /** 会話が続くあいだ（機体の窓 `core/wake_window.WAKE_WINDOW_SEC` と同じ・見本の `window_sec`）。 */
        const val WINDOW_MS = 10_000L
    }
}
