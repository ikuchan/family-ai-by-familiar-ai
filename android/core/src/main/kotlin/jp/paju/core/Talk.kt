package jp.paju.core

/** 話す画面の見え方（課題9 §7.4「端末での断り方」）。 */
enum class Screen {
    /** 機体につながっていない。 */
    OFFLINE,

    /** 話せる。 */
    FREE,

    /** 窓が家かほかの端末のもの。話す画面をグレーにする。 */
    BUSY,

    /** 送った言葉の返事を待っている（求めが閉じるまで）。 */
    WAITING,
}

/**
 * 端末の話す画面の状態。届いた字（[on]）と、送った・切れた、だけで変わる。
 *
 * いまの話者は、押して代えるまで、つなぎ直しても同じ人のまま（§4 の 5）。返事を待つのは、求めが閉じた・断られた・
 * 切れた・送った言葉の時間切れ（[timedOut]・画面が時計で呼ぶ）まで。機体の返事が来ないまま待ち続けない。
 *
 * いま鳴らす考え中の合図（[beep]・c4）は、`cue` が届いてから、返事・閉じた・断られた・切れた・時間切れまで。機体と同じく、
 * 答えを言うあいだは鳴らさない。知らない種類と、返事を待っていないときに届いた合図は鳴らさない。
 */
class Talk(var speaker: String = "") {
    private var connected = false
    private var busy = false
    private var waiting = false
    private var serial = 0

    /** いま鳴らす合図の音のファイル（[Beeps.FILES]）。無ければ鳴らさない。 */
    var beep: String? = null
        private set

    val screen: Screen
        get() = when {
            !connected -> Screen.OFFLINE
            busy -> Screen.BUSY
            waiting -> Screen.WAITING
            else -> Screen.FREE
        }

    fun on(msg: FromMachine) {
        when (msg) {
            is FromMachine.Hello -> { connected = true; busy = msg.busy; waiting = false; beep = null }
            is FromMachine.Window -> busy = msg.busy
            FromMachine.Closed, FromMachine.Refused -> { waiting = false; beep = null }
            is FromMachine.Say -> beep = null
            is FromMachine.Cue -> if (waiting) beep = Beeps.FILES[msg.kind] // 待ちを解いたあとの遅れた合図では鳴らさない
            is FromMachine.Timer, is FromMachine.Alarm -> Unit // 時計アプリへ渡すだけで、話す画面は変えない
        }
    }

    /** 言葉を送った。返りの番号を、その言葉の時間切れ（[timedOut]）に渡す。 */
    fun sent(): Int {
        waiting = true
        return ++serial
    }

    /** `sent` が返した番号の言葉の時間切れ。そのあとに送った言葉があれば、待ちは解かない。 */
    fun timedOut(sentSerial: Int) {
        if (sentSerial == serial) { waiting = false; beep = null }
    }

    fun disconnected() {
        connected = false
        waiting = false
        beep = null
    }
}
