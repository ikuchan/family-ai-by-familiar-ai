package jp.paju.app

import android.app.Activity
import android.Manifest
import android.app.AlertDialog
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.AudioAttributes
import android.media.SoundPool
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import jp.paju.core.Address
import jp.paju.core.Beeps
import jp.paju.core.FromMachine
import jp.paju.core.Listen
import jp.paju.core.Screen
import jp.paju.core.Talk
import jp.paju.core.Wire

/**
 * 話す画面（課題9 端-c 段 c2）：話者のボタン、返事の字、字の入力。窓が家かほかの端末のものならグレーにする（§7.4）。
 *
 * スマホは 1 列（話者のボタンを上に横へ並べる）、タブレットは話者のボタンを左にいつも見える所へ縦に並べる（§4 の 7 の続き）。
 * 機体の番地・この端末の名前・いまの話者は、この端末にだけ残す（端末の設定）。話者のボタンに並べる名前は、つないだときに
 * 機体から受ける（アプリに家族の名前を書かない）。返事の声は端-e の鍵ができるまで出さない（字だけ・§8.3）。
 *
 * 声（c3）：画面を開いているあいだ、端末の中の音声認識で聞き、名前を呼ぶ合図のあとの言葉を送る（[Listen]）。「ずっと聞く」に
 * した端末（充電して置くタブレット）は、画面を消さずに開いたままにして聞き続ける。
 *
 * 考え中の合図（c4）：機体から `cue` が届いたら、その種類の電子音を、返事か閉じるまで 1 秒ごとに鳴らす（[Talk.beep]）。
 */
class MainActivity : Activity() {
    private val prefs by lazy { getSharedPreferences("paju", Context.MODE_PRIVATE) }
    private val main = Handler(Looper.getMainLooper())
    private val talk by lazy { Talk(speaker = prefs.getString(KEY_SPEAKER, "") ?: "") }
    private val link = Link(::onMessage, ::onDown)
    private val listen = Listen()
    private val ear by lazy { Ear(this, ::onHeard, ::onEarLost) }
    private var started = false
    private val pool = SoundPool.Builder()
        .setMaxStreams(1)
        .setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANT) // 声の助け手の音（c5 の返事の声と同じ流れにする）
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build(),
        )
        .build()
    private val sounds = mutableMapOf<String, Int>()
    private var ringing: String? = null // いま鳴らしている合図（Talk.beep を写す。変わったら鳴らし直す）
    private val beepTick = object : Runnable {
        override fun run() {
            val file = talk.beep ?: return
            sounds[file]?.let { pool.play(it, 1f, 1f, 1, 0, 1f) }
            main.postDelayed(this, Beeps.INTERVAL_MS)
        }
    }
    private var names: List<String> = emptyList()

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
        keepScreen()
        for (file in Beeps.FILES.values.toSet()) {
            runCatching { assets.openFd("beeps/$file").use { sounds[file] = pool.load(it, 1) } }
                .onFailure { Log.w("paju", "合図の音を読めなかった（$file）", it) }
        }
        render()
    }

    override fun onStart() {
        super.onStart()
        started = true
        connect()
        startEar()
    }

    override fun onDestroy() {
        super.onDestroy()
        main.removeCallbacksAndMessages(null) // 時間切れの仕事が古い画面を掴まないように
        pool.release()
    }

    override fun onStop() {
        super.onStop()
        started = false
        listen.cut()
        link.close() // 画面を開いているあいだだけつなぐ・聞く（ずっと聞く端末は画面を消さない）
        ear.stop()
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

    /**
     * 聞き始める。マイクの許しが無ければ 1 度だけ求める（端末に残す・断られても字は打てる）。聞けなくなった訳は、アプリを
     * 動かしているあいだ 1 度だけ出し、それからは聞き直さない。
     */
    private fun startEar() {
        if (earLost) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            if (!prefs.getBoolean(KEY_ASKED_MIC, false)) {
                prefs.edit().putBoolean(KEY_ASKED_MIC, true).apply()
                requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQ_MIC)
            }
            return
        }
        if (!ear.start()) onEarLost(Ear.CANNOT)
        render()
    }

    private fun onEarLost(why: String) {
        earLost = true
        line(why)
        render()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQ_MIC) return
        if (grantResults.firstOrNull() != PackageManager.PERMISSION_GRANTED) line("（マイクを使えないので、字で話してください）")
        else if (started) startEar() // 画面を閉じていれば、次に開いたときに聞き始める
    }

    private fun onHeard(text: String, startedAt: Long) {
        val words = listen.heard(text, names, talk.screen, startedAt) ?: return
        if (say(words)) listen.sent(SystemClock.elapsedRealtime())
    }

    /** ずっと聞く端末は、アプリを開いているあいだ画面を消さない（消えると聞けない）。 */
    private fun keepScreen() {
        if (prefs.getBoolean(KEY_ALWAYS, false)) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun deviceName(): String =
        prefs.getString(KEY_DEVICE, null)?.takeIf { it.isNotBlank() } ?: Build.MODEL.take(Wire.NAME_MAX)

    private fun onMessage(msg: FromMachine) {
        val orders = talk.on(msg) // 求めが閉じたら、預かった頼みが返る
        when (msg) {
            is FromMachine.Hello -> { names = msg.names; showSpeakers(msg.speakers) }
            is FromMachine.Say -> { line("パジュ：${msg.display}"); listen.replied(SystemClock.elapsedRealtime()) }
            FromMachine.Closed -> listen.replied(SystemClock.elapsedRealtime())
            is FromMachine.Window -> if (msg.busy) listen.cut()
            FromMachine.Refused -> line("（いまは話せません）")
            else -> Unit
        }
        render()
        for (o in orders) {
            when (o) {
                is FromMachine.Timer -> line(Clock.set(this, o))
                is FromMachine.Alarm -> line(Clock.set(this, o))
                is FromMachine.Music -> line(Clock.open(this, o))
                else -> Unit
            }
        }
    }

    private fun onDown() {
        listen.cut()
        talk.disconnected()
        render()
    }

    private fun sendTyped() {
        if (say(input.text.toString())) {
            input.setText("")
            listen.sent(SystemClock.elapsedRealtime()) // 打って送った言葉のあとも、声で続けられる
        }
    }

    /** 打った字か、聞いた言葉を送る。送れたら true。 */
    private fun say(text: String): Boolean {
        val raw = Wire.encodeSay(text, talk.speaker)
        if (raw == null) {
            line("（送れない言葉です。空か、長すぎます）")
            return false
        }
        if (!link.send(raw)) {
            line("（機体につながっていません）")
            return false
        }
        line("${talk.speaker.ifEmpty { "わたし" }}：${text.trim()}")
        val serial = talk.sent()
        main.postDelayed({ talk.timedOut(serial); render() }, WAIT_MS)
        render()
        return true
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
        } + (if (talk.speaker.isNotEmpty()) "（${talk.speaker}）" else "") + (if (ear.listening) "・聞いています" else "")
        if (talk.beep != ringing) {
            ringing = talk.beep
            main.removeCallbacks(beepTick)
            beepTick.run() // 新しい合図はすぐ鳴らす
        }
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
        val always = CheckBox(this).apply {
            text = "ずっと聞く（画面を消さない・充電して置く端末に）"
            isChecked = prefs.getBoolean(KEY_ALWAYS, false)
        }
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), 0)
            addView(address)
            addView(device)
            addView(always)
        }
        AlertDialog.Builder(this)
            .setTitle("設定")
            .setView(box)
            .setPositiveButton("つなぐ") { _, _ ->
                prefs.edit()
                    .putString(KEY_ADDRESS, address.text.toString().trim())
                    .putString(KEY_DEVICE, device.text.toString().trim())
                    .putBoolean(KEY_ALWAYS, always.isChecked)
                    .apply()
                keepScreen()
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
        const val KEY_ALWAYS = "always"
        const val KEY_ASKED_MIC = "asked_mic"
        const val REQ_MIC = 1

        /** 聞けなくなった（許しが無い・書き起こせない）。アプリを動かしているあいだ覚える。 */
        var earLost = false
        const val WAIT_MS = 180_000L // 返事を待つのは 3 分まで（仮・§8.5）
    }
}
