CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
LDLIBS  = -lm -lpthread

# Windows / MSYS2 适配：
#  - 中文相对路径（测试/…、例子/…）不能被 MSYS 改写成 POSIX 前缀路径，
#    其余参数（/tmp 临时文件等）仍走正常转换；
#  - 原生后端调用的 cc 在 shell32 提供 CommandLineToArgvW 之后链接。
export MSYS2_ARG_CONV_EXCL = 测试;例子
ifeq ($(OS),Windows_NT)
LDLIBS += -lshell32
# Windows 主线程默认栈仅 2MB（Linux 默认 8MB），深递归用例会被栈守卫误伤：
# 提高到 16MB，与 Linux 默认值对齐并留出净化器构建的余量。
# -static：MSYS2 的 -lpthread 默认动态链接 libwinpthread-1.dll，该 DLL 不属于
# 系统基线，直接在 PowerShell/资源管理器运行会报「找不到 libwinpthread-1.dll」。
# 静态链入后 co.exe 只依赖系统 DLL，任何 Windows 环境开箱即用。
CFLAGS += -Wl,--stack,16777216 -static
endif

# 净化器构建参数（回归验证用，非发布产物）
SAN_CFLAGS  = -O1 -g -fno-omit-frame-pointer -Wall -Wextra
ASAN_FLAGS  = -fsanitize=address,undefined
TSAN_FLAGS  = -fsanitize=thread

# 平台事实：MSYS2 的 GCC 不附带任何 sanitizer 运行库（libasan/libubsan/libtsan
# 在全部仓库中都不存在，链接时报 cannot find -lasan）。ASan 在 Windows 上改用
# CLANG64 的 clang（compiler-rt 提供 libclang_rt.asan_dynamic）；TSan 运行库
# 在 Windows 上不存在任何实现（仅 POSIX），tsan 目标在 Windows 明确声明跳过。
# Windows 用法：在 CLANG64 环境（MSYSTEM=CLANG64）里执行 make asan。
ifeq ($(OS),Windows_NT)
SAN_CC = clang
else
SAN_CC = $(CC)
endif

# 测试/原生/ 里的用例双通道都要跑：
#   解释器侧跟其余套件一样进 test-suite / asan，
#   原生侧由 native 目标做「输出逐字节一致」校验。
TESTS       = $(wildcard 测试/*.co) $(wildcard 测试/原生/*.co)
ERRCASES    = $(wildcard 测试/错误用例/*.co)
DEMOS       = $(wildcard 例子/*.co)

co: co.c
	$(CC) $(CFLAGS) -o co co.c $(LDLIBS)

.PHONY: test test-suite test-error test-demo asan asan-error tsan native bench check clean

# 默认测试：功能套件 + 错误路径 + 全部例子
test: test-suite test-error test-demo
	@echo "===================================="
	@echo " 全部测试通过"
	@echo "===================================="

# 功能回归套件：每个文件必须以「XXX：全部通过」结尾
test-suite: co
	@echo "---- 功能套件 ----"
	@for f in $(TESTS); do \
		printf "%-34s " "$$f"; \
		if ./co "$$f" > /tmp/co_test.out 2>&1; then \
			tail -1 /tmp/co_test.out; \
		else \
			echo "失败"; cat /tmp/co_test.out; exit 1; \
		fi; \
	done

# 错误路径：每个用例都【必须】非零退出并给出中文诊断，否则视为回归
test-error: co
	@echo "---- 错误路径（应全部报错）----"
	@for f in $(ERRCASES); do \
		printf "%-34s " "$$f"; \
		if ./co "$$f" > /tmp/co_err.out 2>&1; then \
			echo "未报错，回归！"; cat /tmp/co_err.out; exit 1; \
		else \
			tail -1 /tmp/co_err.out; \
		fi; \
	done

# 例子：除 类型错误.co（故意演示类型报错）外都应成功
test-demo: co
	@echo "---- 例子 ----"
	@for f in $(DEMOS); do \
		case "$$f" in *类型错误.co) continue;; esac; \
		printf "%-34s " "$$f"; \
		if ./co "$$f" > /tmp/co_demo.out 2>&1; then echo "通过"; \
		else echo "失败"; cat /tmp/co_demo.out; exit 1; fi; \
	done
	@printf "%-34s " "例子/类型错误.co(应报错)"
	@if ./co 例子/类型错误.co > /dev/null 2>&1; then echo "未报错，回归！"; exit 1; else echo "通过"; fi

# 内存/未定义行为检查：零泄漏、零 UB 才算通过
asan: co.c
	$(SAN_CC) $(SAN_CFLAGS) $(ASAN_FLAGS) -o co_asan co.c $(LDLIBS)
	@echo "---- ASan + UBSan ----"
	@for f in $(TESTS) $(DEMOS); do \
		case "$$f" in *类型错误.co) continue;; esac; \
		printf "%-34s " "$$f"; \
		out=$$(ASAN_OPTIONS=detect_leaks=1 ./co_asan "$$f" 2>&1); \
		if echo "$$out" | grep -qE "ERROR: (AddressSanitizer|LeakSanitizer)|runtime error:"; then \
			echo "检出问题"; echo "$$out" | grep -E "SUMMARY|runtime error:" | head -3; exit 1; \
		else echo "干净"; fi; \
	done

# 错误路径的内存安全检查：错误路径会 exit(1)，进程退出时必然有未回收内存，
# 所以关闭泄漏检测，只查【越界读写 / 释放后使用 / 未定义行为】——
# 这类问题在错误路径上最容易藏（如张量索引越界曾直接读写堆外内存）。
asan-error: co.c
	@test -x co_asan || $(SAN_CC) $(SAN_CFLAGS) $(ASAN_FLAGS) -o co_asan co.c $(LDLIBS)
	@echo "---- ASan 错误路径（只查越界/UB，不查泄漏）----"
	@for f in $(ERRCASES); do \
		printf "%-34s " "$$f"; \
		out=$$(ASAN_OPTIONS=detect_leaks=0 ./co_asan "$$f" 2>&1); \
		if echo "$$out" | grep -qE "ERROR: AddressSanitizer|runtime error:"; then \
			echo "检出内存问题"; echo "$$out" | grep -E "SUMMARY|runtime error:" | head -3; exit 1; \
		else echo "干净"; fi; \
	done

# 原生编译通道：同一份源码走解释器与走原生二进制，
# 标准输出/标准错误必须逐字节相同、退出码必须一致。
# 这条不成立，原生后端就不可信，所以必须进 check。
native: co
	@sh 测试/原生一致性.sh ./co

# 性能基准：两条通道的计算结果必须一致（这部分是门禁），
# 顺带打印加速比与原生产物体积/依赖（这部分是报告）。
# 基准文件放在 测试/基准/ 而不是 测试/ 下，是为了不被 TESTS 的 wildcard 卷进
# ASan 轮次——ASan 会把两百万次循环拖慢十几倍，白等。
bench: co
	@sh 测试/基准运行.sh ./co 测试/基准/性能基准.co

# 自举门禁：coCN 写的编译器由 C 版编译器编译后，再编译示例并逐字节比对解释器。
# 这是「语言的编译器由语言自身编写」的自举机制验证。
selfhost: co
	@sh 测试/自举.sh ./co

# 数据竞争检查（针对并发用例）
# Windows 上不存在 TSan 运行库（libtsan 仅 POSIX：Linux/macOS/FreeBSD），
# 这是平台事实而非可修复缺陷——明确声明跳过，不假装检查；POSIX 照常执行。
ifeq ($(OS),Windows_NT)
tsan:
	@echo "---- TSan ----"
	@echo "跳过：TSan 运行库在 Windows 上不存在（仅 POSIX），并发用例已在功能套件中覆盖"
else
tsan: co.c
	$(CC) $(SAN_CFLAGS) $(TSAN_FLAGS) -o co_tsan co.c $(LDLIBS)
	@echo "---- TSan ----"
	@for f in 例子/并发演示.co 测试/模块并发测试.co; do \
		printf "%-34s " "$$f"; \
		out=$$(./co_tsan "$$f" 2>&1); \
		if echo "$$out" | grep -q "WARNING: ThreadSanitizer"; then \
			echo "检出数据竞争"; echo "$$out" | grep -A6 "WARNING: ThreadSanitizer" | head -12; exit 1; \
		else echo "干净"; fi; \
	done
endif

# 提交前的完整门禁
check: test native bench selfhost asan asan-error tsan
	@echo "===================================="
	@echo " 全部门禁通过：功能 + 错误路径 + 原生双通道一致 + 基准结果一致 + 自举流水线 + ASan/UBSan + 错误路径内存安全 + TSan"
	@echo "===================================="

clean:
	rm -f co co_asan co_tsan 自举cc 自举输出 自举产物.c 自举输入.co core.*
