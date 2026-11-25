package jp.paju.core

/**
 * 考え中の合図の音（課題9 端-c c4・b1）。機体は端末の求めのあいだ家で電子音を鳴らさず、考えているの子が変わったときに
 * その種類（`cue`）だけを送る。端末は同じ音を同じ間隔で鳴らす。音は機体の `sounds/` から組み立てのときに写す（1 か所が正本）。
 * 種類と音と間隔は、機体の `core/reaction_cue.BEEP_WAVS` と `wait_beep_interval_seconds` に見本で揃える。
 */
object Beeps {
    /** 種類 → 音のファイル（アプリの中の `beeps/` に入れる）。 */
    val FILES: Map<String, String> = mapOf(
        "recalling" to "react_beep_recalling.wav",
        "deciding" to "react_thinking.wav",
        "awaiting" to "react_beep_awaiting.wav",
        "checking" to "react_beep_checking.wav",
        "noting" to "react_beep_noting.wav",
    )

    /** 鳴らし始めから次の鳴らし始めまで（音はどれもこれより短い）。 */
    const val INTERVAL_MS = 1_000L
}
