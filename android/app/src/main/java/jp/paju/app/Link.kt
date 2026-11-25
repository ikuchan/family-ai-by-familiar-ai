package jp.paju.app

import android.os.Handler
import android.os.Looper
import jp.paju.core.FromMachine
import jp.paju.core.Wire
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.TimeUnit

/**
 * 機体の端末の口（課題9 §6.6）へのつながり。届いた字は読んで、画面の糸（main）で渡す。読めない字は捨てる。
 * 切れたら（機体が閉じたときも）、閉じるまで 3 秒ごとにつなぎ直す。
 */
class Link(private val onMessage: (FromMachine) -> Unit, private val onDown: () -> Unit) {
    private val main = Handler(Looper.getMainLooper())
    private val client = OkHttpClient.Builder().pingInterval(20, TimeUnit.SECONDS).build()
    private var url: String? = null
    private var ws: WebSocket? = null

    /** つなぐ。つなぎ先が読めなければ偽（つながない）。 */
    fun open(url: String): Boolean {
        close()
        val request = try {
            Request.Builder().url(url).build()
        } catch (e: IllegalArgumentException) {
            return false
        }
        this.url = url
        connect(request)
        return true
    }

    fun close() {
        url = null
        main.removeCallbacksAndMessages(null)
        ws?.close(1000, null)
        ws = null
    }

    /** 送れたら真（つながっていなければ偽）。 */
    fun send(raw: String): Boolean = ws?.send(raw) ?: false

    private fun connect(request: Request) {
        ws = client.newWebSocket(request, object : WebSocketListener() {
            override fun onMessage(webSocket: WebSocket, text: String) {
                val msg = Wire.decode(text) ?: return
                main.post { if (ws === webSocket) onMessage(msg) }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(1000, null) // 機体が閉じた（止まった・同じ名前の端末がつないだ）。すぐ切れたとして扱う
                down(webSocket)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) = down(webSocket)

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) = down(webSocket)
        })
    }

    private fun down(webSocket: WebSocket) {
        main.post {
            if (ws !== webSocket) return@post
            ws = null
            onDown()
            val u = url ?: return@post
            main.postDelayed({ if (ws == null && url == u) connect(Request.Builder().url(u).build()) }, RETRY_MS)
        }
    }

    private companion object {
        const val RETRY_MS = 3000L
    }
}
