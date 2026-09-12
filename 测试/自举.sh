#!/bin/sh
# 自举门禁：coCN 写的编译器（自举编译器.co）由当前 C 版编译器 co --编译 成 自举cc，
# 再用 自举cc 编译 自举示例.co 产出独立二进制，与解释器逐字节比对输出。
# 这验证「用我们的语言写编译器、再用现在 C 写的编译器编译它」的自举机制真实可用。
set -e
CO="${1:-./co}"
DEMO="${2:-自举示例.co}"
cd "$(dirname "$0")/.."

echo "---- 自举流水线 ----"
# 1) 用 C 版编译器编译「coCN 写的编译器」→ 原生编译器 自举cc
"$CO" --编译 自举编译器.co 自举cc >/dev/null 2>&1 || { echo "编译 自举编译器.co 失败"; exit 1; }
echo "① co --编译 自举编译器.co -> 自举cc   完成"

# 2) 把示例作为自举编译器的输入
cp "$DEMO" 自举输入.co

# 3) 用「coCN 写的编译器」编译示例 → 独立二进制 自举输出
./自举cc >/dev/null 2>&1 || { echo "自举cc 编译 自举示例.co 失败"; exit 1; }
echo "② 自举cc(由coCN编译器编译) 编译示例 -> 自举输出   完成"

# 4) 自举产物 vs 解释器：stdout 必须逐字节一致
"$CO" 自举输入.co > /tmp/自举期望.txt 2>&1
./自举输出 > /tmp/自举实际.txt 2>&1
if cmp -s /tmp/自举期望.txt /tmp/自举实际.txt; then
    echo "③ 自举产物与解释器输出逐字节一致   通过"
    cat /tmp/自举实际.txt
else
    echo "差异："
    diff /tmp/自举期望.txt /tmp/自举实际.txt
    exit 1
fi