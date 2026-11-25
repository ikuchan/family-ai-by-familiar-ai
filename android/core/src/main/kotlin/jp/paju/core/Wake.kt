package jp.paju.core

import java.text.Normalizer

/** 書き起こしの聞き分け。 */
enum class Heard {
    /** 名前で始まり、言葉が続く。言葉を送る。 */
    NAMED,

    /** 名前だけ。送らず、続く言葉を聞く。 */
    NAME_ONLY,

    /** パジュに話しかけていない。送らない。 */
    NONE,
}

/**
 * 名前を呼ぶ合図（課題9 §4 の 4・§8.3）：端末の中の書き起こしに、パジュの呼び名があるか。
 *
 * 機体の門と同じ決まりで聞く（`core/silence_rules.names_me` の 1 字違いを許さない側と、`core/wake_strictness`）。
 * ゆるい読みで比べ、文頭の空白と記号だけを飛ばす（「ねえパジュ」は当たらない）。名前だけの繰り返しは送らない。
 * 機体と違い、1 字違いの名前は聞き分けない（Jev に確かめる道が端末に無いため）。
 */
object Wake {
    private const val TAIL = "うぅ"
    private val small = "ぁぃぅぇぉゃゅょっゎ".zip("あいうえおやゆよつわ").toMap()

    fun hear(text: String, names: List<String>): Heard {
        val loosed = names.map(::loose).filter { it.isNotEmpty() }.sortedByDescending { it.length }
        val s = loose(text)
        val head = skipHead(s)
        val name = loosed.firstOrNull { head.startsWith(it) } ?: return Heard.NONE
        if (isRepeat(s, loosed)) return Heard.NONE
        val rest = head.substring(name.length)
        return if (rest.all(::isFiller)) Heard.NAME_ONLY else Heard.NAMED
    }

    /** ゆるい読み：カタカナはひらがなに、長音と濁点・半濁点は落とし、小書きは大きく。 */
    fun loose(s: String): String {
        val b = StringBuilder()
        for (c in Normalizer.normalize(s.lowercase(), Normalizer.Form.NFD)) {
            if (c == '゙' || c == '゚' || c == 'ー' || c == '〜' || c == '～') continue
            val h = if (c.code in 0x30A1..0x30F6) (c.code - 0x60).toChar() else c
            b.append(small[h] ?: h)
        }
        return b.toString()
    }

    private fun isMark(c: Char): Boolean = c.isWhitespace() || when (Character.getType(c).toByte()) {
        Character.CONNECTOR_PUNCTUATION, Character.DASH_PUNCTUATION, Character.START_PUNCTUATION,
        Character.END_PUNCTUATION, Character.INITIAL_QUOTE_PUNCTUATION, Character.FINAL_QUOTE_PUNCTUATION,
        Character.OTHER_PUNCTUATION, Character.MATH_SYMBOL, Character.CURRENCY_SYMBOL,
        Character.MODIFIER_SYMBOL, Character.OTHER_SYMBOL, Character.SPACE_SEPARATOR,
        Character.LINE_SEPARATOR, Character.PARAGRAPH_SEPARATOR -> true
        else -> false
    }

    private fun isFiller(c: Char): Boolean = c in TAIL || isMark(c)

    private fun skipHead(s: String): String = s.dropWhile(::isMark)

    private fun isRepeat(s: String, loosed: List<String>): Boolean {
        var count = 0
        var i = 0
        while (i < s.length) {
            if (isFiller(s[i])) { i++; continue }
            val hit = loosed.firstOrNull { s.startsWith(it, i) } ?: return false
            count++
            i += hit.length
        }
        return count >= 2
    }
}
