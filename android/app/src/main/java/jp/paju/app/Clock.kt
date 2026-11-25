package jp.paju.app

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.provider.AlarmClock
import jp.paju.core.FromMachine

/**
 * 機体から届いたタイマーとアラームを、端末の時計アプリへ渡す（端-c c6a・課題9 b2）。
 * 鳴らすのも止めるのも時計アプリで、パジュのアプリは数えない（画面を閉じても鳴る）。返りは画面に出す一行。
 */
object Clock {
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
