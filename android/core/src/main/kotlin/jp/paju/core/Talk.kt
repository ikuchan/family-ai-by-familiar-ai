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
 */
class Talk(var speaker: String = "") {
    private var connected = false
    private var busy = false
    private var waiting = false
    private var serial = 0

    val screen: Screen
        get() = when {
            !connected -> Screen.OFFLINE
            busy -> Screen.BUSY
            waiting -> Screen.WAITING
            else -> Screen.FREE
        }

    fun on(msg: FromMachine) {
        when (msg) {
            is FromMachine.Hello -> { connected = true; busy = msg.busy; waiting = false }
            is FromMachine.Window -> busy = msg.busy
            FromMachine.Closed, FromMachine.Refused -> waiting = false
            is FromMachine.Say, is FromMachine.Cue -> Unit
        }
    }

    /** 言葉を送った。返りの番号を、その言葉の時間切れ（[timedOut]）に渡す。 */
    fun sent(): Int {
        waiting = true
        return ++serial
    }

    /** `sent` が返した番号の言葉の時間切れ。そのあとに送った言葉があれば、待ちは解かない。 */
    fun timedOut(sentSerial: Int) {
        if (sentSerial == serial) waiting = false
    }

    fun disconnected() {
        connected = false
        waiting = false
    }
}
