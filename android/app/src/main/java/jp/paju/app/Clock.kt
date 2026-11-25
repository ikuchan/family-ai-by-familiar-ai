package jp.paju.app

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.provider.AlarmClock
import jp.paju.core.FromMachine

/**
 * 機体から届いた頼みを、端末のほかのアプリへ渡す（端-c c6・課題9 b2）。返りは画面に出す一行。
 * 渡すのは `Talk.on` が返したとき（時計はすぐ、Spotify は求めが閉じてから）。
 *
 * - タイマーとアラームは時計アプリへ（c6a）。鳴らすのも止めるのも時計アプリで、パジュのアプリは数えない（画面を閉じても鳴る）。
 * - 音楽は Spotify のアプリで開く（c6b 段 1）。URI が空なら Spotify を開くだけ。再生は本人が押す。
 */
object Clock {
    fun open(activity: Activity, msg: FromMachine.Music): String {
        val intent = if (msg.uri.isEmpty()) activity.packageManager.getLaunchIntentForPackage(SPOTIFY)
        else Intent(Intent.ACTION_VIEW, Uri.parse(msg.uri)).setPackage(SPOTIFY) // spotify: を受けるほかのアプリへ渡さない
        return try {
            activity.startActivity(intent ?: throw ActivityNotFoundException())
            "（Spotify を開きました）"
        } catch (e: ActivityNotFoundException) {
            "（この端末には Spotify がないので、音楽をかけられません）"
        } catch (e: SecurityException) {
            "（Spotify を開けませんでした）"
        }
    }

    /** Spotify のアプリ。名前の無い「音楽かけて」は、このアプリを開くだけ（見えるように AndroidManifest の queries に書く）。 */
    const val SPOTIFY = "com.spotify.music"

    fun set(activity: Activity, msg: FromMachine.Timer): String = start(
        activity,
        Intent(AlarmClock.ACTION_SET_TIMER)
            .putExtra(AlarmClock.EXTRA_LENGTH, msg.seconds)
            .putExtra(AlarmClock.EXTRA_MESSAGE, msg.label)
            .putExtra(AlarmClock.EXTRA_SKIP_UI, true),
        "「${msg.label}」のタイマー（${minutes(msg.seconds)}）",
    )

    fun set(activity: Activity, msg: FromMachine.Alarm): String = start(
        activity,
        Intent(AlarmClock.ACTION_SET_ALARM)
            .putExtra(AlarmClock.EXTRA_HOUR, msg.hour)
            .putExtra(AlarmClock.EXTRA_MINUTES, msg.minute)
            .putExtra(AlarmClock.EXTRA_MESSAGE, msg.label)
            .putExtra(AlarmClock.EXTRA_SKIP_UI, true),
        "「${msg.label}」のアラーム（${msg.hour}:${"%02d".format(msg.minute)}）",
    )

    /** 時計アプリが無い・断った端末では、掛からなかったと出す（機体は掛かったかを知らない）。 */
    private fun start(activity: Activity, intent: Intent, what: String): String = try {
        activity.startActivity(intent)
        "（時計に$what を頼みました）"
    } catch (e: ActivityNotFoundException) {
        "（この端末の時計では$what を掛けられません）"
    } catch (e: SecurityException) {
        "（この端末の時計では$what を掛けられません）"
    }

    private fun minutes(seconds: Int): String =
        if (seconds % 60 == 0) "${seconds / 60} 分" else "${seconds / 60} 分 ${seconds % 60} 秒"
}
