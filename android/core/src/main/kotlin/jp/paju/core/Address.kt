package jp.paju.core

import java.net.URLEncoder

/** 設定の機体の番地から、端末の口へのつなぎ先を作る（課題9 §6.6）。 */
object Address {
    const val PORT = 8765 // 機体の DeviceLinkConfig の既定と同じ

    /**
     * `番地`（`100.64.0.1`・`ws://名前:ポート` など）と端末の名前から、`ws://番地:ポート/device?name=…` を作る。
     * 番地が空か ws・wss でない、または端末の名前が空か 32 字を超えるなら null（機体の口が断る）。
     */
    fun url(address: String, device: String): String? {
        var a = address.trim().trimEnd('/')
        val name = device.trim()
        if (a.isEmpty() || name.isEmpty() || name.codePointCount(0, name.length) > Wire.NAME_MAX) return null
        val scheme = when {
            a.startsWith("ws://") -> "ws"
            a.startsWith("wss://") -> "wss"
            "://" in a -> return null
            else -> "ws"
        }
        a = a.substringAfter("://")
        val host = hostAndPort(a) ?: return null
        return "$scheme://$host/device?name=${URLEncoder.encode(name, "UTF-8")}"
    }

    private val NAMED = Regex("""([A-Za-z0-9.-]+|\[[0-9A-Fa-f:.]+])(?::(\d{1,5}))?""")
    private val BARE_V6 = Regex("""[0-9A-Fa-f:.]*:[0-9A-Fa-f:.]*:[0-9A-Fa-f:.]*""")

    /** 名前か IPv4 か IPv6（括弧は無くてもよい）に、ポート（無ければ既定）を付ける。それ以外（空白・道・おかしいポート）は null。 */
    private fun hostAndPort(a: String): String? {
        if (BARE_V6.matches(a)) return "[$a]:$PORT"
        val m = NAMED.matchEntire(a) ?: return null
        val port = m.groupValues[2].ifEmpty { "$PORT" }.toInt()
        if (port !in 1..65535) return null
        return "${m.groupValues[1]}:$port"
    }
}
