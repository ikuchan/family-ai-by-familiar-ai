package jp.paju.core

/**
 * 端末の口（課題9 §6.6）の字を読み書きする、小さな JSON。
 *
 * 口の字は平たい（入れ子は hello の voice だけ）ので、部品を入れずに標準の Kotlin で読む。読めた値は
 * Map・List・String・Boolean・Double・null になる。読めなければ [IllegalArgumentException]（入れ子が深すぎるときも）。
 */
object Json {
    fun parse(text: String): Any? {
        val r = Reader(text)
        val v = r.value()
        r.space()
        require(r.i == text.length) { "trailing" }
        return v
    }

    /** 字を JSON の字の形にする（前後の引用符を含む）。 */
    fun quote(s: String): String {
        val b = StringBuilder("\"")
        for (ch in s) {
            when {
                ch == '"' -> b.append("\\\"")
                ch == '\\' -> b.append("\\\\")
                ch == '\n' -> b.append("\\n")
                ch == '\r' -> b.append("\\r")
                ch == '\t' -> b.append("\\t")
                ch < ' ' -> b.append("\\u%04x".format(ch.code))
                else -> b.append(ch)
            }
        }
        return b.append('"').toString()
    }

    private const val DEPTH_MAX = 32  // 口の字の入れ子は 2 段まで。深い字で落ちないように

    private class Reader(val s: String) {
        var i = 0
        private var depth = 0

        fun space() {
            while (i < s.length && s[i] in " \t\r\n") i++
        }

        fun value(): Any? {
            space()
            require(i < s.length) { "end" }
            return when (s[i]) {
                '{', '[' -> {
                    require(++depth <= DEPTH_MAX) { "depth" }
                    (if (s[i] == '{') obj() else arr()).also { depth-- }
                }
                '"' -> str()
                't' -> word("true", true)
                'f' -> word("false", false)
                'n' -> word("null", null)
                else -> num()
            }
        }

        private fun word(w: String, v: Any?): Any? {
            require(s.startsWith(w, i)) { "word" }
            i += w.length
            return v
        }

        private fun obj(): Map<String, Any?> {
            val m = LinkedHashMap<String, Any?>()
            i++
            space()
            if (s.getOrNull(i) == '}') { i++; return m }
            while (true) {
                space()
                require(s.getOrNull(i) == '"') { "key" }
                val k = str()
                space()
                require(s.getOrNull(i) == ':') { "colon" }
                i++
                m[k] = value()
                space()
                when (s.getOrNull(i)) {
                    ',' -> i++
                    '}' -> { i++; return m }
                    else -> throw IllegalArgumentException("object")
                }
            }
        }

        private fun arr(): List<Any?> {
            val l = ArrayList<Any?>()
            i++
            space()
            if (s.getOrNull(i) == ']') { i++; return l }
            while (true) {
                l.add(value())
                space()
                when (s.getOrNull(i)) {
                    ',' -> i++
                    ']' -> { i++; return l }
                    else -> throw IllegalArgumentException("array")
                }
            }
        }

        private fun str(): String {
            val b = StringBuilder()
            i++
            while (true) {
                require(i < s.length) { "string" }
                val ch = s[i++]
                when {
                    ch == '"' -> return b.toString()
                    ch == '\\' -> {
                        require(i < s.length) { "escape" }
                        when (val e = s[i++]) {
                            'n' -> b.append('\n')
                            'r' -> b.append('\r')
                            't' -> b.append('\t')
                            'b' -> b.append('\b')
                            'f' -> b.append('\u000c')
                            'u' -> {
                                val hex = s.substring(i, minOf(i + 4, s.length))
                                require(hex.length == 4 && hex.all { it in "0123456789abcdefABCDEF" }) { "unicode" }
                                b.append(hex.toInt(16).toChar())
                                i += 4
                            }
                            '"', '\\', '/' -> b.append(e)
                            else -> throw IllegalArgumentException("escape")
                        }
                    }
                    else -> b.append(ch)
                }
            }
        }

        private fun num(): Double {
            val start = i
            while (i < s.length && s[i] in "+-0123456789.eE") i++
            return s.substring(start, i).toDoubleOrNull() ?: throw IllegalArgumentException("number")
        }
    }
}
