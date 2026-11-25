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
 *
 * 機体から届いた頼み（c6）は [on] が返す。時計（タイマー・アラーム）はすぐ返す（確かめの画面を出さないので前に出ず、遅らせると
 * 切れたときに失い、数え始めも遅れる）。Spotify は前に出てパジュの画面が止まり、つながりが切れるので、次に `closed` が届いたときに
 * 返す（それまで預かる）。機体は頼みを、その求めの `closed` より先に送る（送りは 1 本の鎖）ので、次の `closed` がその求めのもので
 * ある。返事を待つ時間切れと、別の言葉が断られた（`refused`）ときは捨てない（機体はもう「開く」と言っている）。切れたら捨てる。
 */
class Talk(var speaker: String = "") {
    private var connected = false
    private var busy = false
    private var waiting = false
    private var serial = 0
    private val held = mutableListOf<FromMachine>()

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

    /** 届いた字で状態を変える。返すのは、いま端末のアプリへ渡す頼み（求めが閉じたときだけ）。 */
    fun on(msg: FromMachine): List<FromMachine> {
        when (msg) {
            is FromMachine.Hello -> { connected = true; busy = msg.busy; end(); held.clear() }
            is FromMachine.Window -> busy = msg.busy
            FromMachine.Closed -> {
                val ready = held.toList()
                held.clear()
                end()
                return ready
            }
            FromMachine.Refused -> end()
            is FromMachine.Say -> beep = null
            is FromMachine.Cue -> if (waiting) beep = Beeps.FILES[msg.kind] // 待ちを解いたあとの遅れた合図では鳴らさない
            is FromMachine.Timer, is FromMachine.Alarm -> return listOf(msg)
            is FromMachine.Music -> held += msg
        }
        return emptyList()
    }

    /** 返事を待つのをやめる（合図も止める）。預かった頼みは残す。 */
    private fun end() {
        waiting = false
        beep = null
    }

    /** 言葉を送った。返りの番号を、その言葉の時間切れ（[timedOut]）に渡す。 */
    fun sent(): Int {
        waiting = true
        return ++serial
    }

    /** `sent` が返した番号の言葉の時間切れ。そのあとに送った言葉があれば、待ちは解かない。 */
    fun timedOut(sentSerial: Int) {
        if (sentSerial == serial) end()
    }

    fun disconnected() {
        connected = false
        end()
        held.clear()
    }
}
