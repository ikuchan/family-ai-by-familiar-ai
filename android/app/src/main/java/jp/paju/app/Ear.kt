package jp.paju.app

import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer

/**
 * 端末の中の音声認識で聞き続ける耳（課題9 端-c c3・§4 の 4）。書き起こしを、話し始めた時刻と一緒に 1 つずつ渡し、聞き終えるたび
 * （話し終えた・何も聞こえなかった）に聞き直す。音は端末の外へ出さない：端末の中で書き起こせる認識（Android 12 から）だけを使い、
 * 無い端末では声で話せない（字は打てる）。認識は画面の糸でだけ動かす。認識の働き手が落ちたら作り直し、続けて
 * [FAILS_MAX] 回うまくいかなければ聞くのをやめる。
 */
class Ear(
    private val context: Context,
    private val onWords: (text: String, startedAt: Long) -> Unit,
    private val onLost: (String) -> Unit,
) {
    private val main = Handler(Looper.getMainLooper())
    private var recognizer: SpeechRecognizer? = null
    private var startedAt = 0L
    private var fails = 0

    val listening: Boolean get() = recognizer != null

    /** 聞き始める。端末の中で書き起こせなければ false。 */
    fun start(): Boolean {
        if (recognizer != null) return true
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S || !SpeechRecognizer.isOnDeviceRecognitionAvailable(context)) return false
        recognizer = SpeechRecognizer.createOnDeviceSpeechRecognizer(context).apply { setRecognitionListener(listener) }
        listen()
        return true
    }

    fun stop() {
        main.removeCallbacksAndMessages(null)
        recognizer?.destroy()
        recognizer = null
    }

    private fun listen() {
        startedAt = SystemClock.elapsedRealtime()
        recognizer?.startListening(
            Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                .putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ja-JP"),
        )
    }

    /** 聞き終えた（結果か誤り）あとに聞き直す。終えた回なので取り消しはしない。 */
    private fun again(delayMs: Long) {
        main.removeCallbacksAndMessages(null)
        main.postDelayed({ listen() }, delayMs)
    }

    /** 働き手が落ちた・おかしくなった：作り直して聞き直す。 */
    private fun rebuild() {
        stop()
        main.postDelayed({ if (!start()) onLost(CANNOT) }, RETRY_MS)
    }

    private val listener = object : RecognitionListener {
        override fun onResults(results: Bundle?) {
            fails = 0
            results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.let { onWords(it, startedAt) }
            again(RESTART_MS)
        }

        override fun onError(error: Int) {
            when (error) {
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> lose("（マイクを使えないので、字で話してください）")
                SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED, SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE ->
                    lose("（この端末の中に日本語の書き起こしが入っていません。字で話してください）")
                SpeechRecognizer.ERROR_NO_MATCH, SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> { fails = 0; again(RESTART_MS) }
                else -> {
                    // 認識が忙しい・音が取れない・働き手が落ちたなど
                    if (++fails >= FAILS_MAX) lose(CANNOT)
                    else if (error == SpeechRecognizer.ERROR_SERVER_DISCONNECTED || error == SpeechRecognizer.ERROR_CLIENT) rebuild()
                    else again(RETRY_MS)
                }
            }
        }

        override fun onBeginningOfSpeech() {
            startedAt = SystemClock.elapsedRealtime() // 会話が続くあいだは、言い終えた時刻でなく話し始めた時刻で見る
        }

        override fun onReadyForSpeech(params: Bundle?) = Unit
        override fun onRmsChanged(rmsdB: Float) = Unit
        override fun onBufferReceived(buffer: ByteArray?) = Unit
        override fun onEndOfSpeech() = Unit
        override fun onPartialResults(partialResults: Bundle?) = Unit
        override fun onEvent(eventType: Int, params: Bundle?) = Unit
    }

    private fun lose(why: String) {
        stop()
        fails = 0
        onLost(why)
    }

    companion object {
        private const val RESTART_MS = 200L
        private const val RETRY_MS = 1_000L
        private const val FAILS_MAX = 5
        const val CANNOT = "（この端末の中では声を書き起こせません。字で話してください）"
    }
}
