// パジュの端末のアプリ（課題9 端-c・§8）。道具は機体の ~/dev-tools（`source env.sh`）。
pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositories { google(); mavenCentral() } }
rootProject.name = "paju-app"
include(":core", ":app")
