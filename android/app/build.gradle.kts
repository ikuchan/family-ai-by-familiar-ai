// パジュの端末のアプリ（課題9 端-c）。スマホとタブレットで 1 つ（§4 の 7 の続き）。
plugins { id("com.android.application") }

android {
    namespace = "jp.paju.app"
    compileSdk = 36
    defaultConfig { applicationId = "jp.paju.app"; minSdk = 26; targetSdk = 36; versionCode = 1; versionName = "0.1" }
}
dependencies {
    implementation(project(":core"))
    implementation("com.squareup.okhttp3:okhttp:4.12.0") // 機体の端末の口へ WebSocket でつなぐ（2026-11-25 本人の承認）。Origin は付けない
}
