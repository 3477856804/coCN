#!/bin/sh
# 性能基准运行器：同一份 .co 走「解释器」与「原生编译」两条通道。
#
# 它同时是一道门禁，不只是跑分：
#   1. 两条通道的 stdout（计算结果）必须逐字节相同 —— 不同就直接失败。
#      「更快但算错」是最坏的结果，必须先钉死正确性再谈速度。
#   2. 耗时走 stderr，由本脚本抽取，算出每个负载的加速比。
#   3. 顺手校验原生产物是自包含的（只依赖 libc/libm，不依赖 co 解释器）。
#
# 用法: sh 测试/基准运行.sh [./co] [基准文件.co]
set -eu

CO=${1:-./co}
SRC=${2:-测试/基准/性能基准.co}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if [ ! -x "$CO" ]; then echo "找不到解释器: $CO" >&2; exit 1; fi
if [ ! -f "$SRC" ]; then echo "找不到基准文件: $SRC" >&2; exit 1; fi

echo "===================================="
echo " coCN 性能基准: $SRC"
echo "===================================="

# ---------- 通道一：解释器 ----------
printf '%s\n' "[1/2] 解释器通道 ..."
if ! "$CO" "$SRC" > "$TMP/interp.out" 2> "$TMP/interp.err"; then
    echo "解释器通道执行失败：" >&2; cat "$TMP/interp.err" >&2; exit 1
fi

# ---------- 通道二：原生编译 ----------
printf '%s\n' "[2/2] 原生编译通道 ..."
if ! "$CO" --编译 "$SRC" "$TMP/native" > "$TMP/build.log" 2>&1; then
    echo "原生编译失败：" >&2; cat "$TMP/build.log" >&2; exit 1
fi
if ! "$TMP/native" > "$TMP/native.out" 2> "$TMP/native.err"; then
    echo "原生二进制执行失败：" >&2; cat "$TMP/native.err" >&2; exit 1
fi

# ---------- 门禁 1：结果逐字节一致 ----------
if ! diff -u "$TMP/interp.out" "$TMP/native.out" > "$TMP/diff.txt" 2>&1; then
    echo ""
    echo "!! 致命：两条通道的计算结果不一致，原生后端不可信 !!" >&2
    cat "$TMP/diff.txt" >&2
    exit 1
fi

echo ""
echo "---- 计算结果（两条通道逐字节一致）----"
cat "$TMP/interp.out"

# ---------- 门禁 2：原生产物自包含 ----------
# 「自包含」按平台各说各话：
#   Linux  —— 只依赖 libc/libm/动态链接器（用户机器无需装任何运行时）。
#   Windows—— 禁止依赖 MinGW 运行时 DLL（libgcc/libstdc++/libwinpthread，
#             否则用户机器也得装 gcc 才能跑）；KERNEL32/ucrtbase/ntdll 属于
#             操作系统自带基线。沙箱/安全软件注入的 DLL（如本环境的
#             aiep_sbox.dll）出现在所有进程里，视为环境噪声，不算链接依赖。
echo ""
echo "---- 原生产物 ----"
SIZE=$(wc -c < "$TMP/native" | tr -d ' ')
printf '%-14s %s 字节\n' "二进制体积" "$SIZE"
if command -v ldd > /dev/null 2>&1; then
    ldd "$TMP/native" 2>/dev/null | awk '{print $1}' | grep -v '^$' > "$TMP/all_deps" || true
    if grep -qE '\.so' "$TMP/all_deps"; then
        # Linux：白名单之外即为失败
        grep -vE '^(linux-vdso|/lib64/ld-linux|ld-linux|libc\.so|libm\.so|libgcc_s\.so)' \
            "$TMP/all_deps" > "$TMP/extra_deps" || true
        WIN_MSG="仅 libc/libm（不依赖 co 解释器）"
    else
        # Windows：PE 产物禁止引入 MinGW 运行时 DLL
        grep -iE 'libgcc|libstdc\+\+|libwinpthread|libco' "$TMP/all_deps" > "$TMP/extra_deps" || true
        WIN_MSG="仅系统 DLL（不依赖 gcc 运行时与 co 解释器）"
    fi
    if [ -s "$TMP/extra_deps" ]; then
        printf '%-14s %s\n' "额外依赖" "$(tr '\n' ' ' < "$TMP/extra_deps")"
        echo "!! 原生产物出现预期外的动态依赖 !!" >&2
        exit 1
    fi
    printf '%-14s %s\n' "动态依赖" "$WIN_MSG"
fi

# ---------- 加速比 ----------
# 耗时行格式: "耗时/<名称> <秒>"，解释器与原生按行一一对应。
# 表格整体交给 awk 渲染：shell 的 printf 按【字节】补齐，而中文一个字 3 字节却只占
# 2 个显示列，直接 %-16s 会错位。awk 里按「非 ASCII 记 2 列」自己算填充。
echo ""
echo "---- 加速比（解释器耗时 / 原生耗时）----"

grep '^耗时/' "$TMP/interp.err" > "$TMP/t_interp" || true
grep '^耗时/' "$TMP/native.err" > "$TMP/t_native" || true

awk '
# 显示宽度计算。麻烦点在于 awk 的 length/substr 行为随 locale 变化：
#   C locale        -> 按【字节】切分，一个汉字是 3 个 substr 单位
#   *.UTF-8 locale  -> 按【字符】切分，一个汉字是 1 个单位
# 所以先在 BEGIN 探测模式，再分别处理，两种环境下都能对齐。
function dispw(s,   i, n, c, v, w) {
    w = 0; n = length(s);
    for (i = 1; i <= n; i++) {
        c = substr(s, i, 1); v = ORD[c] + 0;
        if (CHARMODE) {
            w += (v >= 32 && v <= 126) ? 1 : 2;     # 非 ASCII 字符按东亚全角记 2 列
        } else if (v >= 128 && v <= 191) {
            continue;                                # UTF-8 续字节，不占显示宽度
        } else if (v >= 192) {
            w += 2;                                  # 多字节序列首字节，记 2 列
        } else {
            w += 1;
        }
    }
    return w;
}
function pad(k,   s) { s = "                                "; return substr(s, 1, (k > 0) ? k : 0) }
function padr(s, width) { return s pad(width - dispw(s)) }
function padl(s, width) { return pad(width - dispw(s)) s }
BEGIN {
    for (i = 1; i < 256; i++) ORD[sprintf("%c", i)] = i;
    CHARMODE = (length("中") == 1);
    W1 = 14; W2 = 12; W3 = 12; W4 = 10;
}
NR == FNR { sub(/^耗时\//, "", $1); iname[FNR] = $1; itime[FNR] = $2; nrec = FNR; next }
{
    sub(/^耗时\//, "", $1);
    ntime[FNR] = $2;
}
END {
    print padr("负载", W1) " " padl("解释器(秒)", W2) " " padl("原生(秒)", W3) " " padl("加速比", W4);
    dash = "--------------------------------";
    print substr(dash,1,W1) " " substr(dash,1,W2) " " substr(dash,1,W3) " " substr(dash,1,W4);
    for (i = 1; i <= nrec; i++) {
        ti = itime[i] + 0; tn = ntime[i] + 0;
        cell = (tn > 0) ? sprintf("%.2fx", ti / tn) : "太快无法测";
        print padr(iname[i], W1) " " padl(sprintf("%.4f", ti), W2) " " padl(sprintf("%.4f", tn), W3) " " padl(cell, W4);
    }
}
' "$TMP/t_interp" "$TMP/t_native"

echo ""
echo "===================================="
echo " 基准完成：结果一致 + 原生产物自包含"
echo "===================================="
