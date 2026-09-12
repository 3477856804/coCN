# Homebrew formula：coCN 中文编程语言
# 用法（走源码编译，仅需 gcc/make）：
#   brew install https://raw.githubusercontent.com/3477856804/coCN/master/Formula/coCN.rb
# 生成独立命令名 coCN；brew 会自动处理 gcc/make 依赖。
class Cocn < Formula
  desc "coCN —— 让代码开口说中文的中文编程语言（自举里程碑 0.0.1）"
  homepage "https://github.com/3477856804/coCN"
  url "https://github.com/3477856804/coCN/archive/refs/tags/v0.0.1.tar.gz"
  sha256 "95523d0f60e1fc120ec2ede9eb0cb35d0bbb8365ae67967b0c868223c175a947"
  license "Apache-2.0"
  version "0.0.1"

  depends_on "gcc"

  def install
    # Makefile 需要 gcc、-lm、-lpthread；macOS 上 system gcc 即可
    system "make", "CC=gcc"
    bin.install "co" => "coCN"
  end

  test do
    assert_match "0.0.1", shell_output("#{bin}/coCN --版本")
  end
end