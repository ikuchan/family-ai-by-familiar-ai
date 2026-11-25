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
class Link(
    private val onMessage: (FromMachine) -> Unit,
    private val onDown: () -> Unit,
    private val onDenied: () -> Unit, // 機体が合言葉で断った（つなぎ直さない）
) {
    private val main = Handler(Looper.getMainLooper())
    private val client = OkHttpClient.Builder().pingInterval(20, TimeUnit.SECONDS).build()
    private var request: Request? = null // つなぎ直しにも同じ頼み（合言葉ごと）を使う
    private var ws: WebSocket? = null

    /**
     * つなぐ。合言葉があれば `Authorization: Bearer 合言葉` を付ける（機体の外で待ち受ける口は、無いか違えば断る・§8.10）。
     * つなぎ先か合言葉が読めなければ偽（つながない）。
     */
    fun open(url: String, secret: String): Boolean {
        close()
        val request = try {
            Request.Builder().url(url).apply { if (secret.isNotEmpty()) header("Authorization", "Bearer $secret") }.build()
        } catch (e: IllegalArgumentException) {
            return false // 番地の形が違う、または合言葉に頭書きに書けない字（日本語など）がある
        }
        this.request = request
        connect(request)
        return true
    }

    fun close() {
        request = null
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

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                if (response?.code == 401) {
                    // 合言葉が無いか違う。つなぎ直しても同じなので、設定で直すまでつながない
                    main.post { if (ws === webSocket) { this@Link.request = null; onDenied() } }
                }
                down(webSocket)
            }
        })
    }

    private fun down(webSocket: WebSocket) {
        main.post {
            if (ws !== webSocket) return@post
            ws = null
            onDown()
            val r = request ?: return@post
            main.postDelayed({ if (ws == null && request === r) connect(r) }, RETRY_MS)
        }
    }

    private companion object {
        const val RETRY_MS = 3000L
    }
}
