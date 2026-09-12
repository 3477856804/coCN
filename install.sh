#!/bin/bash
# =============================================================
#  coCN 一键安装脚本
#    用法：
#      curl -fsSL https://gitee.com/COSMOnb666/coCN/raw/master/install.sh | bash
#      # 或
#      bash install.sh
#    可调环境变量：
#      COCN_PREFIX   安装目录（默认 /usr/local）→ 二进制进 <prefix>/bin/coCN
#      COCN_BRANCH   源码编译时的分支（默认 master）
#      COCN_FORCE_COMPILE  置 1 强制源码编译（跳过预编译二进制下载）
# =============================================================
set -e

VERSION="${COCN_VERSION:-0.0.1}"
PREFIX="${COCN_PREFIX:-/usr/local}"
BIN_DIR="$PREFIX/bin"
BIN_NAME="coCN"
EXE="$BIN_DIR/$BIN_NAME"

GH_RELEASES="https://github.com/3477856804/coCN/releases/download/v${VERSION}"
GE_RELEASES="https://gitee.com/COSMOnb666/coCN/releases/download/v${VERSION}"
SRC_TARBALL_GH="https://github.com/3477856804/coCN/archive/refs/heads/master.tar.gz"
SRC_TARBALL_GE="https://gitee.com/COSMOnb666/coCN/repository/archive/master.tar.gz"

say()  { printf '\033[34m[coCN]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[coCN!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[coCN!!]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 1. 平台探测 ----------
case "$(uname -s)" in
  Linux)  OS=linux ;;
  Darwin) OS=darwin ;;
  MINGW*|MSYS*|CYGWIN*) OS=windows ;;
  *) die "暂不支持的平台: $(uname -s)" ;;
esac

case "$(uname -m)" in
  x86_64|amd64)     ARCH=x86_64 ;;
  aarch64|arm64)    ARCH=aarch64 ;;
  *) warn "未收录架构 $(uname -m)，改为源码编译"; ARCH="" ;;
esac

say "coCN v${VERSION} 安装开始（${OS}/${ARCH:-自动}，目标 $EXE）"

# ---------- 2. 尝试预编译二进制 ----------
installed=0
if [ "$OS" != "windows" ] && [ -n "$ARCH" ]; then
  asset="coCN-${OS}-${ARCH}"
  for base in "$GH_RELEASES" "$GE_RELEASES"; do
    [ "$COCN_FORCE_COMPILE" = "1" ] && break
    tmpbin="$(mktemp)"
    if curl -fsSL -o "$tmpbin" "$base/$asset" 2>/dev/null; then
      chmod +x "$tmpbin"
      if "$tmpbin" --版本 >/dev/null 2>&1; then
        say "命中预编译二进制：$base/$asset"
        install -d "$BIN_DIR"
        install -m 0755 "$tmpbin" "$EXE"
        rm -f "$tmpbin"
        installed=1
        break
      fi
      rm -f "$tmpbin"
    fi
  done
fi

# ---------- 3. 回退：源码编译 ----------
if [ "$installed" != "1" ]; then
  say "未取到预编译二进制$( [ "$COCN_FORCE_COMPILE" = "1" ] && echo "（已强制源码编译）"),走源码编译（仅需 gcc/cc 与 make）"
  command -v curl >/dev/null || die "需要 curl 以拉取源码"
  command -v cc >/dev/null && CC=cc || command -v gcc >/dev/null && CC=gcc || die "未找到 cc/gcc"
  command -v make >/dev/null || die "未找到 make"
  command -v tar >/dev/null || die "未找到 tar"

  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  tarball="$tmp/coCN-src.tar.gz"

  fetched=0
  for src in "$SRC_TARBALL_GH" "$SRC_TARBALL_GE"; do
    if curl -fsSL -o "$tarball" "$src" 2>/dev/null; then
      fetched=1
      break
    fi
  done
  [ "$fetched" = "1" ] || die "源码下载失败，请检查网络后重试"

  cd "$tmp"
  tar -xzf "$tarball"
  srcdir="$(find "$tmp" -maxdepth 1 -type d -name 'coCN-*' | head -1)"
  [ -n "$srcdir" ] || srcdir="$tmp"
  cd "$srcdir"
  make 2>/dev/null || make CC="$CC" 2>/dev/null || die "源码编译失败：请确认已安装 gcc 与 make"
  [ -f ./co ] || die "编译产物缺失（未找到 ./co），请报告此问题"
  ./co --版本 || true

  install -d "$BIN_DIR"
  install -m 0755 ./co "$EXE"
  installed=1
fi

# ---------- 4. 完成 ----------
if [ "$installed" = "1" ]; then
  say "已就绪：$EXE"
  "$EXE" --版本
  say "运行方式：$BIN_NAME 程序.co    （--编译 可生成独立二进制）"
  say "升级：重新运行本脚本即可"
  exit 0
fi
die "安装失败"