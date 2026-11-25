package jp.paju.app

import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import jp.paju.core.Address
import jp.paju.core.FromMachine
import jp.paju.core.Screen
import jp.paju.core.Talk
import jp.paju.core.Wire

/**
 * 話す画面（課題9 端-c 段 c2）：話者のボタン、返事の字、字の入力。窓が家かほかの端末のものならグレーにする（§7.4）。
 *
 * スマホは 1 列（話者のボタンを上に横へ並べる）、タブレットは話者のボタンを左にいつも見える所へ縦に並べる（§4 の 7 の続き）。
 * 機体の番地・この端末の名前・いまの話者は、この端末にだけ残す（端末の設定）。話者のボタンに並べる名前は、つないだときに
 * 機体から受ける（アプリに家族の名前を書かない）。返事の声は端-e の鍵ができるまで出さない（字だけ・§8.3）。
 */
class MainActivity : Activity() {
    private val prefs by lazy { getSharedPreferences("paju", Context.MODE_PRIVATE) }
    private val main = Handler(Looper.getMainLooper())
    private val talk by lazy { Talk(speaker = prefs.getString(KEY_SPEAKER, "") ?: "") }
    private val link = Link(::onMessage, ::onDown)

    private lateinit var status: TextView
    private lateinit var log: TextView
    private lateinit var logScroll: ScrollView
    private lateinit var input: EditText
    private lateinit var send: Button
    private lateinit var speakerBox: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val tablet = resources.configuration.smallestScreenWidthDp >= 600
        speakerBox = LinearLayout(this).apply { orientation = if (tablet) LinearLayout.VERTICAL else LinearLayout.HORIZONTAL }
        status = TextView(this).apply { textSize = 16f; setPadding(dp(12), dp(8), dp(12), dp(8)) }
        val settings = Button(this).apply { text = "設定"; setOnClickListener { askSettings() } }
        val top = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            addView(status, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(settings)
        }
        log = TextView(this).apply { textSize = 18f; setPadding(dp(12), dp(8), dp(12), dp(8)) }
        logScroll = ScrollView(this).apply { addView(log) }
        input = EditText(this).apply { hint = "パジュに話す"; maxLines = 4 }
        send = Button(this).apply { text = "送る"; setOnClickListener { sendTyped() } }
        val bottom = LinearLayout(this).apply {
            addView(input, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(send)
        }
        val talkColumn = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(top)
            addView(logScroll, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
            addView(bottom)
        }
        val root = if (tablet) {
            LinearLayout(this).apply {
                addView(ScrollView(context).apply { addView(speakerBox) }, LinearLayout.LayoutParams(dp(180), MATCH_PARENT))
                addView(talkColumn, LinearLayout.LayoutParams(0, MATCH_PARENT, 1f))
            }
        } else {
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                addView(HorizontalScrollView(context).apply { addView(speakerBox) })
                addView(talkColumn, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
            }
        }
        root.fitsSystemWindows = true
        setContentView(root)
        render()
    }

    override fun onStart() {
        super.onStart()
        connect()
    }

    override fun onDestroy() {
        super.onDestroy()
        main.removeCallbacksAndMessages(null) // 時間切れの仕事が古い画面を掴まないように
    }

    override fun onStop() {
        super.onStop()
        link.close() // 画面を開いているあいだだけつなぐ（ずっと聞く設定は c3）
        talk.disconnected()
        render()
    }

    private fun url(): String? = Address.url(prefs.getString(KEY_ADDRESS, "") ?: "", deviceName())

    private fun connect() {
        val url = url()
        if (url == null || !link.open(url)) {
            link.close()
            talk.disconnected()
        }
        render()
    }

    private fun deviceName(): String =
        prefs.getString(KEY_DEVICE, null)?.takeIf { it.isNotBlank() } ?: Build.MODEL.take(Wire.NAME_MAX)

    private fun onMessage(msg: FromMachine) {
        talk.on(msg)
        when (msg) {
            is FromMachine.Hello -> showSpeakers(msg.speakers)
            is FromMachine.Say -> line("パジュ：${msg.display}")
            FromMachine.Refused -> line("（いまは話せません）")
            else -> Unit
        }
        render()
    }

    private fun onDown() {
        talk.disconnected()
        render()
    }

    private fun sendTyped() {
        val raw = Wire.encodeSay(input.text.toString(), talk.speaker)
        if (raw == null) {
            line("（送れない言葉です。空か、長すぎます）")
            return
        }
        if (!link.send(raw)) {
            line("（機体につながっていません）")
            return
        }
        line("${talk.speaker.ifEmpty { "わたし" }}：${input.text.toString().trim()}")
        input.setText("")
        val serial = talk.sent()
        main.postDelayed({ talk.timedOut(serial); render() }, WAIT_MS)
        render()
    }

    private fun showSpeakers(names: List<String>) {
        if (talk.speaker.isNotEmpty() && talk.speaker !in names) {
            talk.speaker = "" // 家族から外れた人は、いまの話者にしない
            prefs.edit().remove(KEY_SPEAKER).apply()
        }
        speakerBox.removeAllViews()
        for (name in names) {
            speakerBox.addView(Button(this).apply {
                text = name
                setOnClickListener {
                    talk.speaker = name
                    prefs.edit().putString(KEY_SPEAKER, name).apply()
                    render()
                }
            })
        }
    }

    private fun render() {
        val screen = talk.screen
        status.text = when (screen) {
            Screen.OFFLINE -> if (url() == null) "設定で機体の番地とこの端末の名前を入れてください" else "機体につながっていません"
            Screen.FREE -> "話せます"
            Screen.BUSY -> "いまはほかの人と話しています"
            Screen.WAITING -> "考えています…"
        } + (if (talk.speaker.isNotEmpty()) "（${talk.speaker}）" else "")
        val open = screen == Screen.FREE || screen == Screen.WAITING
        input.isEnabled = open
        send.isEnabled = open
        val alpha = if (open) 1f else 0.4f // 話せないときはグレー（§7.4）
        input.alpha = alpha
        send.alpha = alpha
        for (i in 0 until speakerBox.childCount) {
            val b = speakerBox.getChildAt(i) as Button
            b.setTextColor(if (b.text == talk.speaker) Color.rgb(0, 102, 204) else Color.DKGRAY)
        }
    }

    private fun line(text: String) {
        log.append(if (log.text.isEmpty()) text else "\n$text")
        logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun askSettings() {
        val address = EditText(this).apply { hint = "機体の番地（例 100.64.0.1）"; setText(prefs.getString(KEY_ADDRESS, "")) }
        val device = EditText(this).apply { hint = "この端末の名前"; setText(deviceName()) }
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), 0)
            addView(address)
            addView(device)
        }
        AlertDialog.Builder(this)
            .setTitle("設定")
            .setView(box)
            .setPositiveButton("つなぐ") { _, _ ->
                prefs.edit()
                    .putString(KEY_ADDRESS, address.text.toString().trim())
                    .putString(KEY_DEVICE, device.text.toString().trim())
                    .apply()
                connect()
            }
            .setNegativeButton("やめる", null)
            .show()
    }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    private companion object {
        const val KEY_ADDRESS = "address"
        const val KEY_DEVICE = "device"
        const val KEY_SPEAKER = "speaker"
        const val WAIT_MS = 180_000L // 返事を待つのは 3 分まで（仮・§8.5）
    }
}
