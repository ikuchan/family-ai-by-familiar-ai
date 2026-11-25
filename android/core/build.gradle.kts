// アプリの中の決まり（Android を読まない・JVM で試験が回る・課題9 §8.2 の c1）
plugins { id("org.jetbrains.kotlin.jvm") }
kotlin { jvmToolchain(21) }
dependencies { testImplementation(kotlin("test")) }
tasks.test {
    useJUnitPlatform()
    // 機体の試験と同じ口の形の見本（tests/test_device_wire_is_shared.py も読む）
    val wire = rootProject.file("../tests/data/device_wire.json")
    inputs.file(wire)  // 見本が変われば回し直す（入力に書かないと、済みと見て回さない）
    systemProperty("paju.wire", wire.absolutePath)
}
