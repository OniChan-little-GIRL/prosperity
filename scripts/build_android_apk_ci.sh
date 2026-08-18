#!/usr/bin/env bash
set -euxo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ANDROID_API="${ANDROID_API:-35}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-29}"
ANDROID_BUILD_TOOLS="${ANDROID_BUILD_TOOLS:-35.0.0}"
ANDROID_NDK_VERSION="${ANDROID_NDK_VERSION:-26.1.10909125}"
GRADLE_VERSION="${GRADLE_VERSION:-8.10.2}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
ANDROID_HOME="${ANDROID_HOME:-$ROOT_DIR/.android-sdk}"
ANDROID_SDK_ROOT="$ANDROID_HOME"
GRADLE_HOME="$ROOT_DIR/.gradle-dist/gradle-$GRADLE_VERSION"
export ANDROID_HOME ANDROID_SDK_ROOT CMAKE_BUILD_PARALLEL_LEVEL PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$GRADLE_HOME/bin:$PATH"

install_host_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates \
      binutils \
      build-essential \
      clang \
      cmake \
      curl \
      file \
      git \
      jq \
      lld \
      make \
      nasm \
      ninja-build \
      openjdk-17-jdk \
      pkg-config \
      python3 \
      python3-pip \
      unzip \
      zip
  fi
}

install_android_sdk() {
  mkdir -p "$ANDROID_HOME/cmdline-tools"
  if [ ! -x "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" ]; then
    tmpdir="$(mktemp -d)"
    curl -fsSL "https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip" -o "$tmpdir/cmdline-tools.zip"
    unzip -q "$tmpdir/cmdline-tools.zip" -d "$tmpdir"
    rm -rf "$ANDROID_HOME/cmdline-tools/latest"
    mv "$tmpdir/cmdline-tools" "$ANDROID_HOME/cmdline-tools/latest"
    rm -rf "$tmpdir"
  fi

  yes | sdkmanager --licenses >/dev/null || true
  sdkmanager \
    "platform-tools" \
    "platforms;android-$ANDROID_API" \
    "build-tools;$ANDROID_BUILD_TOOLS" \
    "ndk;$ANDROID_NDK_VERSION"
}

install_gradle() {
  if [ ! -x "$GRADLE_HOME/bin/gradle" ]; then
    mkdir -p "$ROOT_DIR/.gradle-dist"
    tmpdir="$(mktemp -d)"
    curl -fsSL "https://services.gradle.org/distributions/gradle-$GRADLE_VERSION-bin.zip" -o "$tmpdir/gradle.zip"
    unzip -q "$tmpdir/gradle.zip" -d "$ROOT_DIR/.gradle-dist"
    rm -rf "$tmpdir"
  fi
}

init_submodules() {
  git submodule sync --recursive
  git config --file .gitmodules --get-regexp 'submodule\..*\.path' |
  while read -r _key path; do
    if [ "$path" = "vendor/equilibrium" ]; then
      continue
    fi

    git submodule update --init --recursive --force --jobs 4 "$path"
  done
}

restore_equilibrium() {
  expected_commit="$(git ls-tree HEAD vendor/equilibrium | awk '{print $3}')"
  rm -rf vendor/equilibrium
  git clone https://github.com/Force67/equilibrium.git vendor/equilibrium
  if ! git -C vendor/equilibrium checkout --detach "$expected_commit"; then
    echo "::warning::equilibrium commit $expected_commit is unavailable on the remote; using the remote default branch instead."
  fi
}

build_native_library() {
  ndk_home="$ANDROID_HOME/ndk/$ANDROID_NDK_VERSION"
  toolchain_file="$ndk_home/build/cmake/android.toolchain.cmake"
  if [ ! -f "$toolchain_file" ]; then
    echo "Android NDK toolchain file not found: $toolchain_file" >&2
    exit 1
  fi

  cmake -G Ninja -B build_android \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$toolchain_file" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DDELTA_ANDROID_APP=ON \
    -DDELTA_BUILD_TESTS=OFF
  cmake --build build_android --target ps4delta_app -- -v
}

sanitize_manifest() {
  mkdir -p build_android/apk_project/app/src/main
  python3 -c 'from pathlib import Path; import re; manifest = Path("android/AndroidManifest.xml").read_text(); manifest = re.sub(r"\s+package=\"[^\"]*\"", "", manifest, count=1); manifest = re.sub(r"\s+android:versionCode=\"[^\"]*\"", "", manifest, count=1); manifest = re.sub(r"\s+android:versionName=\"[^\"]*\"", "", manifest, count=1); Path("build_android/apk_project/app/src/main/AndroidManifest.xml").write_text(manifest)'
}

write_gradle_project() {
  apk_project=build_android/apk_project
  native_lib="$(find build_android -name libps4delta_app.so -print -quit)"
  if [ -z "$native_lib" ]; then
    echo "libps4delta_app.so was not produced by the native build" >&2
    exit 1
  fi

  mkdir -p "$apk_project/app/src/main/jniLibs/arm64-v8a"
  cp "$native_lib" "$apk_project/app/src/main/jniLibs/arm64-v8a/"
  sanitize_manifest

  cat > "$apk_project/settings.gradle" <<'GRADLE'
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement { repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS); repositories { google(); mavenCentral() } }
rootProject.name = 'ProsperityAndroid'
include ':app'
GRADLE

  cat > "$apk_project/build.gradle" <<'GRADLE'
plugins {
    id 'com.android.application' version '8.7.3' apply false
}
GRADLE

  mkdir -p "$apk_project/app"
  cat > "$apk_project/app/build.gradle" <<'GRADLE'
plugins {
    id 'com.android.application'
}

android {
    namespace 'com.prosperity.ps4'
    compileSdk 35

    defaultConfig {
        applicationId 'com.prosperity.ps4'
        minSdk 29
        targetSdk 35
        versionCode 1
        versionName '0.2'
    }

    sourceSets {
        main {
            manifest.srcFile 'src/main/AndroidManifest.xml'
            java.srcDirs = ['../../../android/java']
            res.srcDirs = ['../../../android/res']
            jniLibs.srcDirs = ['src/main/jniLibs']
        }
    }
}
GRADLE
}

package_apk() {
  write_gradle_project
  gradle -p build_android/apk_project :app:assembleDebug --no-daemon --stacktrace
  mkdir -p build_android/apk
  cp build_android/apk_project/app/build/outputs/apk/debug/app-debug.apk build_android/apk/prosperity-debug.apk
}

main() {
  install_host_packages
  install_android_sdk
  install_gradle
  init_submodules
  restore_equilibrium
  build_native_library
  package_apk
}

main "$@"
