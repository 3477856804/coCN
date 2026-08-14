CC      = gcc
CFLAGS  = -O2 -Wall
LDLIBS  = -lm -lpthread

# 净化器构建参数（回归验证用，非发布产物）
SAN_CFLAGS  = -O1 -g -fno-omit-frame-pointer -Wall
ASAN_FLAGS  = -fsanitize=address,undefined
TSAN_FLAGS  = -fsanitize=thread

TESTS    = $(wildcard 测试/*.co)
ERRCASES = $(wildcard 测试/错误用例/*.co)
DEMOS    = $(wildcard 例子/*.co)

co: co.c
	$(CC) $(CFLAGS) -o co co.c $(LDLIBS)

.PHONY: test test-suite test-error test-demo asan asan-error tsan check clean

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
	$(CC) $(SAN_CFLAGS) $(ASAN_FLAGS) -o co_asan co.c $(LDLIBS)
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
	@test -x co_asan || $(CC) $(SAN_CFLAGS) $(ASAN_FLAGS) -o co_asan co.c $(LDLIBS)
	@echo "---- ASan 错误路径（只查越界/UB，不查泄漏）----"
	@for f in $(ERRCASES); do \
		printf "%-34s " "$$f"; \
		out=$$(ASAN_OPTIONS=detect_leaks=0 ./co_asan "$$f" 2>&1); \
		if echo "$$out" | grep -qE "ERROR: AddressSanitizer|runtime error:"; then \
			echo "检出内存问题"; echo "$$out" | grep -E "SUMMARY|runtime error:" | head -3; exit 1; \
		else echo "干净"; fi; \
	done

# 数据竞争检查（针对并发用例）
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

# 提交前的完整门禁
check: test asan asan-error tsan
	@echo "===================================="
	@echo " 全部门禁通过：功能 + 错误路径 + ASan/UBSan + 错误路径内存安全 + TSan"
	@echo "===================================="

clean:
	rm -f co co_asan co_tsan core.*
