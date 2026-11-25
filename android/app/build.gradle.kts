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

/**
 * 考え中の合図の音を、機体の `sounds/` から組み立てのときに写す（c4・写しをリポジトリに置かない）。写すのは口の形の見本の
 * `beeps` に書いた音だけで（アプリの `core/Beeps` と試験で揃える）、1 つでも無ければ組み立てを止める。
 */
abstract class CopyBeeps : DefaultTask() {
    @get:InputFiles
    abstract val wavs: ConfigurableFileCollection

    @get:OutputDirectory
    abstract val out: DirectoryProperty

    @TaskAction
    fun copy() {
        val dir = out.get().asFile.resolve("beeps")
        dir.deleteRecursively()
        dir.mkdirs()
        wavs.forEach {
            check(it.exists()) { "合図の音が機体の sounds/ に無い：${it.name}" }
            it.copyTo(dir.resolve(it.name))
        }
    }
}

val beeps = tasks.register<CopyBeeps>("copyBeeps") {
    val wire = rootProject.file("../tests/data/device_wire.json")
    inputs.file(wire)
    @Suppress("UNCHECKED_CAST")
    val names = ((groovy.json.JsonSlurper().parse(wire) as Map<String, Any>)["beeps"] as Map<String, String>).values.toSet()
    wavs.from(names.map { rootProject.file("../src/familiar_agent/sounds/$it") })
}
androidComponents { onVariants { it.sources.assets?.addGeneratedSourceDirectory(beeps, CopyBeeps::out) } }
