#!/bin/sh
# 原生编译通道的一致性门禁。
#
# 核心命题：同一份 .co 源码，走【解释器】和走【原生编译产物】，
#          标准输出必须逐字节相同、退出码必须相同。
# 只要这条不成立，原生后端就是不可信的——所以它必须进 make check。
#
# 用法: 测试/原生一致性.sh [co 可执行文件路径]
set -u

CO="${1:-./co}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

PASS=0
FAIL=0
RED=''
GRN=''
END=''
if [ -t 1 ]; then RED='\033[31m'; GRN='\033[32m'; END='\033[0m'; fi

say_fail() {
    printf "${RED}失败${END}: %s\n" "$1"
    FAIL=$((FAIL + 1))
}

# ---- 第一部分：正常用例的双通道一致性 ----
CASES="例子/原生编译示例.co 测试/原生/边界与断言.co 测试/原生/嵌套控制流.co 测试/原生/字符串边界.co 测试/原生/作用域与全局.co 测试/原生/错误流分离.co"

for f in $CASES; do
    printf '%-34s ' "$f"
    if [ ! -f "$f" ]; then say_fail "$f 不存在"; continue; fi

    base="$(basename "$f" .co)"
    io="$TMP/$base.interp.out"; ie="$TMP/$base.interp.err"
    no="$TMP/$base.native.out"; ne="$TMP/$base.native.err"
    bin="$TMP/$base.bin"

    "$CO" "$f" > "$io" 2> "$ie"; irc=$?
    if [ "$irc" -ne 0 ]; then
        say_fail "解释器执行失败（退出码 $irc）"; sed 's/^/    /' "$ie"; continue
    fi

    if ! "$CO" --编译 "$f" "$bin" > "$TMP/cc.log" 2>&1; then
        say_fail "原生编译失败"; sed 's/^/    /' "$TMP/cc.log"; continue
    fi
    # 生成的 C 必须自身零警告：机器生成的代码更没有理由带警告
    if grep -q 'warning:' "$TMP/cc.log"; then
        say_fail "生成的 C 代码存在编译警告"; grep 'warning:' "$TMP/cc.log" | sed 's/^/    /'; continue
    fi

    "$bin" > "$no" 2> "$ne"; nrc=$?
    if [ "$nrc" -ne "$irc" ]; then
        say_fail "退出码不一致（解释器 $irc / 原生 $nrc）"; sed 's/^/    /' "$ne"; continue
    fi
    if ! diff -u "$io" "$no" > "$TMP/diff.txt" 2>&1; then
        say_fail "标准输出不一致"; head -30 "$TMP/diff.txt" | sed 's/^/    /'; continue
    fi
    # 标准错误也必须一致：有了 `输出错误` 之后 stderr 是程序的正式输出通道之一，
    # 只对齐 stdout 是不够的。这些用例都成功退出，stderr 里不会有报错噪音。
    if ! diff -u "$ie" "$ne" > "$TMP/differr.txt" 2>&1; then
        say_fail "标准错误不一致"; head -30 "$TMP/differr.txt" | sed 's/^/    /'; continue
    fi

    printf "${GRN}一致${END}（出 %s 行 / 错 %s 行）\n" \
        "$(wc -l < "$io" | tr -d ' ')" "$(wc -l < "$ie" | tr -d ' ')"
    PASS=$((PASS + 1))
done

# ---- 第二部分：错误用例双通道都必须失败（不能一个报错一个静默通过）----
mk() { printf '%b' "$2" > "$TMP/$1.co"; echo "$TMP/$1.co"; }

ERRC="$(mk 除以零 '输出(1 / 0)\n')
$(mk 移位过大 '输出(1 << 64)\n')
$(mk 移位为负 '输出(1 << -1)\n')
$(mk 浮点位运算 '输出(1.5 & 1)\n')
$(mk 字符串减法 '输出("a" - "b")\n')
$(mk 字符串越界 '字符串 s = "ab"\n输出(s[5])\n')
$(mk 空替换 '输出(替换("aaa", "", "b"))\n')
$(mk 字节越界 '输出(字节("ab", 9))\n')
$(mk 步长为零 '对于 i 从 1 到 3 步长 0 循环\n    输出(i)\n结束\n')
$(mk 参数个数 '函数 加(x, y)\n    返回 x + y\n结束\n输出(加(1))\n')"

echo "---- 错误路径：两条通道都必须拒绝 ----"
for f in $ERRC; do
    base="$(basename "$f" .co)"
    printf '%-34s ' "错误用例/$base"
    "$CO" "$f" > /dev/null 2>&1; irc=$?
    if [ "$irc" -eq 0 ]; then say_fail "解释器竟然通过了（应当报错）"; continue; fi

    if ! "$CO" --编译 "$f" "$TMP/$base.bin" > "$TMP/cc.log" 2>&1; then
        # 编译期就拒绝也算正确（错误左移，比运行期更好）
        printf "${GRN}编译期拒绝${END}\n"; PASS=$((PASS + 1)); continue
    fi
    "$TMP/$base.bin" > /dev/null 2>&1; nrc=$?
    if [ "$nrc" -eq 0 ]; then say_fail "原生产物竟然通过了（应当报错）"; continue; fi
    printf "${GRN}运行期拒绝${END}\n"; PASS=$((PASS + 1))
done

# ---- 第三部分：--生成C 产出必须能被独立编译，且不依赖解释器 ----
echo "---- --生成C 产物独立可编译 ----"
printf '%-34s ' "生成C/独立编译"
# MinGW 的 cc 会自动给 -o 目标追加 .exe：产物落在哪就用哪个（POSIX 上无 .exe，不受影响）
SA="$TMP/standalone"
if [ ! -x "$SA" ] && [ -x "$SA.exe" ]; then SA="$SA.exe"; fi
if "$CO" --生成C 例子/原生编译示例.co "$TMP/standalone.c" > /dev/null 2>&1 \
   && cc -O2 -Wall -Wextra "$TMP/standalone.c" -o "$TMP/standalone" -lm 2> "$TMP/sa.log" \
   && [ ! -s "$TMP/sa.log" ] \
   && "$SA" > "$TMP/sa.out" 2>&1 \
   && diff -q "$TMP/原生编译示例.interp.out" "$TMP/sa.out" > /dev/null 2>&1; then
    printf "${GRN}通过${END}\n"; PASS=$((PASS + 1))
else
    say_fail "生成的 C 无法独立编译或输出不一致"; sed 's/^/    /' "$TMP/sa.log" 2>/dev/null
fi

echo "===================================="
if [ "$FAIL" -eq 0 ]; then
    printf " 原生通道一致性门禁通过：%d 项全部一致\n" "$PASS"
    echo "===================================="
    exit 0
fi
printf " 原生通道一致性门禁失败：%d 通过 / %d 失败\n" "$PASS" "$FAIL"
echo "===================================="
exit 1
