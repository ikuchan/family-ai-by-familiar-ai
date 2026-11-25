# パジュの端末のアプリのビルド用の環境。`source env.sh` で、このシェルだけに効く（~/.bashrc は触らない）。
# 道具は g-bowl と共有する（課題9 §8.1）。
export JAVA_HOME="$HOME/dev-tools/jdk"
export ANDROID_HOME="$HOME/dev-tools/android-sdk"
export PATH="$JAVA_HOME/bin:$HOME/dev-tools/gradle/gradle-9.8.2/bin:$ANDROID_HOME/platform-tools:$PATH"
