#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE            /* pthread_getattr_np：取本线程真实栈区间 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>
#include <setjmp.h>
#include <sys/resource.h>   /* getrlimit：探测真实栈容量 */
#include <errno.h>          /* strtoull 溢出检测：进制字面量必须拒绝越界而非静默回绕 */

/* ========== 基础定义 ========== */
#define MAX_ID_LEN     128
#define MAX_PARAMS     32
#define MAX_LOCALS     512
#define MAX_DICT_SIZE  256
/* 单次调用的实参上限：实参数组放在栈上，异常 longjmp 时不会泄漏 */
#define CO_MAX_ARGS    64
#define CO_MAX_DIM     8      /* 张量最大维数 */

/* 前向声明 */
struct Value;
struct Env;
struct Node;

/* 值类型 */
typedef enum {
    VAL_NULL, VAL_INT, VAL_FLOAT, VAL_STRING,
    VAL_LIST, VAL_TENSOR, VAL_FUNC, VAL_MAP, VAL_TASK,
    /* 布尔是独立的一等类型（不是整数别名）：打印为 真/假，
       比较与逻辑运算的结果都是布尔。追加在末尾以不扰动既有枚举序。 */
    VAL_BOOL
} ValType;

/* 张量结构（含自动求导所需字段） */
typedef struct Tensor {
    float *data;        /* 数值存储（行主序） */
    float *grad;        /* 梯度缓冲，按需分配 */
    int   *shape;       /* 各维度大小 */
    int    ndim;        /* 维度数 */
    int    size;        /* 元素总数 */
    int    requires_grad; /* 是否需要梯度 */
    int    tref;        /* 张量引用计数 */
    int    _tag;        /* 反向传播拓扑标记（临时） */
    struct Tensor **children; /* 计算图子节点 */
    int    nchildren;   /* 子节点数量 */
    void  (*backward)(struct Tensor*); /* 反向传播函数 */
} Tensor;

/* 前向声明：张量释放（带引用计数） */
static void tensor_free(Tensor *t);

/* 字典条目 */
typedef struct DictEntry {
    char *key;
    struct Value *val;
    int used;
} DictEntry;

/* 字典结构 */
typedef struct Map {
    DictEntry entries[MAX_DICT_SIZE];
    int count;
} Map;

/* 函数结构 */
typedef struct Func {
    char  *name;
    char **params;
    int    param_count;
    char **ptypes;   /* 参数类型标注（可空） */
    char  *ret_type; /* 返回类型标注（可空） */
    void  *body;
} Func;

/* 并发任务句柄：线程、待求值表达式、隔离环境、结果四者必须共存，
   因此独立成结构体——绝不可拆成 union 成员（会互相覆盖）。 */
typedef struct Task {
    pthread_t     tid;
    struct Node  *node;
    struct Env   *env;
    struct Value *result;   /* 由 worker 线程写入，join 后主线程读取 */
    int           started;  /* 线程是否成功创建 */
    int           joined;   /* 是否已 join，防止重复 join */
} Task;

/* 值对象 (带引用计数) */
typedef struct Value {
    ValType type;
    int     refcount;
    union {
        long      ival;
        double    fval;
        char     *sval;
        struct Value **lval;
        Tensor   *tval;
        Func     *fnval;
        Map      *mval;
        Task     *task;     /* 并发任务句柄（VAL_TASK） */
    };
    int list_len;
} Value;

/* 环境 (变量存储) */
typedef struct Env {
    char   *names[MAX_LOCALS];
    Value  *values[MAX_LOCALS];
    char   *types[MAX_LOCALS];   /* 变量类型标注（可空） */
    int     count;
    struct Env *parent;
    /* 是否可能被多个线程同时访问：创建并发任务时，其外层环境链会被标记为共享。
       只有共享环境的读写才加锁——任务自身的局部环境走无锁热路径。 */
    int     shared;
} Env;

/* 循环控制上下文 */
typedef struct LoopContext {
    int in_loop;
    int should_break;
    int should_continue;
    struct LoopContext *parent;
} LoopContext;

/* ========== 全局状态 ==========
   注意：解释器支持 并发/等待（真 pthread 并行），
   凡是"当前执行流"的状态都必须是【线程局部】的，
   否则多个任务会互相踩踏返回值与循环控制标志。 */
static __thread LoopContext *loop_ctx = NULL;

/* 全局根环境：高阶内置函数回调用户函数时作为父环境使用 */
static struct Env *g_root_env = NULL;

/* 已导入模块的 AST（其函数体被合并进调用方环境，故需保留至程序结束） */
static struct Node **g_imported = NULL;
static int g_imported_cnt = 0, g_imported_cap = 0;

/* 前向声明：值构造与真值判断（被定义位置更靠前的内置函数引用） */
static Value *val_int(long x);
static Value *val_flt(double x);
static Value *val_bool(int b);
static int    truthy(Value *v);
/* 前向声明：错误上报与「安全取数」。凡是要把一个 Value 当数字用的地方，
   必须走 num_of / int_of / dim_of，绝不允许裸读 v->ival / v->fval ——
   裸读会把字符串指针、列表指针当成整数算，产出「能跑但结果错」的答案。 */
static void   runtime_error(struct Node *n, const char *msg, ...);
static void   stack_guard_init(size_t total_bytes);
static void   stack_guard_check(struct Node *site);
static const char *val_type_name(Value *v);
static double num_of(Value *v, const char *who);
static long   int_of(Value *v, const char *who);
static int    dim_of(Value *v, const char *who);

/* ========== 异常机制的引用记账（RefLog） ==========
   语言的 尝试/捕获 基于 setjmp/longjmp 实现。longjmp 会直接丢弃中间
   C 栈帧，那些帧上"本该执行"的 val_free 全部被跳过 —— 若不管，每次
   捕获异常都会漏一批临时值（ASan 立刻报泄漏）。

   解决办法是在 try 期间登记每一次【无主的】引用增量，回滚时精确抵消：

     val_new          → +1   新值自带的初始引用
     val_retain       → +1   临时借用（表达式求值链上的中间引用）
     val_retain_root  →  —   值被写入容器/环境，引用归容器所有，不登记
     val_free 未归零  → -1   抵消掉一次已经正常发生的释放
     val_free 归零    → 抹除该指针在所有帧中的条目（防止回滚时悬空）

   于是回滚只释放"没有任何容器/环境持有的临时引用"，
   而 try 块内合法写入外层列表/字典/变量的值不会被误杀。 */
typedef struct RefLog { Value *v; int delta; } RefLog;

#define MAX_TRY_DEPTH 64
typedef struct TryFrame {
    jmp_buf  jb;
    RefLog  *log;
    int      nlog, caplog;
    /* longjmp 同样会跳过函数调用的 env_release，作用域本身（含其中变量的
       引用）也会泄漏。因此 try 期间创建的【函数局部环境】一并登记。 */
    struct Env **envs;
    int      nenv, capenv;
    /* 内建函数里的【裸 malloc 临时缓冲】同样会被 longjmp 跳过 free。
       例如 排序 的归并临时数组：比较两个不可比类型时会抛错，缓冲就漏了。
       统一登记，回滚时一并回收。 */
    void   **bufs;
    int      nbuf, capbuf;
    int      recording;    /* 回滚过程中置 0，避免自我干扰 */
} TryFrame;

/* 与循环控制标志同理：异常状态属于"当前执行流"，必须线程局部，
   否则并发任务之间会互相抢夺异常载荷。 */
static __thread TryFrame g_try[MAX_TRY_DEPTH];
static __thread int      g_try_depth = 0;
static __thread Value   *g_exc_value = NULL;   /* 正在传播的异常载荷 */

/* 登记一次引用增量。登记到【所有】活跃帧，这样内层 try 期间产生的
   临时引用在外层 try 捕获时同样可以被回滚（无需帧间合并）。 */
static void reflog_add(Value *v, int delta) {
    if (!v || g_try_depth == 0) return;
    for (int d = 0; d < g_try_depth; d++) {
        TryFrame *f = &g_try[d];
        if (!f->recording) continue;
        if (f->nlog >= f->caplog) {
            int nc = f->caplog ? f->caplog * 2 : 64;
            RefLog *nl = realloc(f->log, (size_t)nc * sizeof(RefLog));
            if (!nl) continue;         /* 内存不足则退化为不记账，绝不崩溃 */
            f->log = nl; f->caplog = nc;
        }
        f->log[f->nlog].v = v;
        f->log[f->nlog].delta = delta;
        f->nlog++;
    }
}

/* 该值即将被真正 free：把所有帧中指向它的条目打洞，防止回滚访问野指针。
   （打洞而非压缩数组，O(n) 且不打乱正在进行的遍历。） */
static void reflog_forget(Value *v) {
    if (!v || g_try_depth == 0) return;
    for (int d = 0; d < g_try_depth; d++) {
        TryFrame *f = &g_try[d];
        for (int i = 0; i < f->nlog; i++)
            if (f->log[i].v == v) f->log[i].v = NULL;
    }
}

/* 引用交接：把 C 栈上的临时引用【直接】移交给容器槽位
   （形如 list->lval[i] = eval_node(...)，没有额外 retain）。
   此后该引用由容器负责释放，不再属于 C 栈，故登记 -1 抵消。
   漏掉这一步的后果是回滚时把容器里的元素当成无主临时值释放掉 —— 悬空。 */
static Value *val_adopt(Value *v) {
    reflog_add(v, -1);
    return v;
}

/* 登记一个"其释放会被 longjmp 跳过"的函数局部作用域 */
static void reflog_track_env(struct Env *e) {
    if (!e || g_try_depth == 0) return;
    for (int d = 0; d < g_try_depth; d++) {
        TryFrame *f = &g_try[d];
        if (!f->recording) continue;
        if (f->nenv >= f->capenv) {
            int nc = f->capenv ? f->capenv * 2 : 16;
            struct Env **ne = realloc(f->envs, (size_t)nc * sizeof(struct Env*));
            if (!ne) continue;
            f->envs = ne; f->capenv = nc;
        }
        f->envs[f->nenv++] = e;
    }
}

/* 作用域已经正常释放：从所有帧摘除，避免回滚二次释放 */
static void reflog_untrack_env(struct Env *e) {
    if (!e || g_try_depth == 0) return;
    for (int d = 0; d < g_try_depth; d++) {
        TryFrame *f = &g_try[d];
        for (int i = 0; i < f->nenv; i++)
            if (f->envs[i] == e) f->envs[i] = NULL;
    }
}

/* ========== 值管理 ========== */
static Value *val_new(ValType type) {
    Value *v = calloc(1, sizeof(Value));
    v->type = type;
    v->refcount = 1;
    if (type == VAL_MAP) {
        v->mval = calloc(1, sizeof(Map));
    }
    reflog_add(v, +1);
    return v;
}

void val_free(Value *v) {
    if (!v) return;
    /* 原子递减：并发任务与主线程可能同时持有同一个值（如全局列表/函数），
       非原子的 refcount-- 会丢失更新，导致提前释放或永久泄漏。 */
    if (__atomic_sub_fetch(&v->refcount, 1, __ATOMIC_ACQ_REL) > 0) {
        reflog_add(v, -1);
        return;
    }
    reflog_forget(v);
    switch (v->type) {
        case VAL_STRING: free(v->sval); break;
        case VAL_LIST:
            for (int i = 0; i < v->list_len; i++) val_free(v->lval[i]);
            free(v->lval);
            break;
        case VAL_TENSOR:
            tensor_free(v->tval);
            break;
        case VAL_MAP: {
            for (int i = 0; i < MAX_DICT_SIZE; i++) {
                if (v->mval->entries[i].used) {
                    free(v->mval->entries[i].key);
                    val_free(v->mval->entries[i].val);
                }
            }
            free(v->mval);
            break;
        }
        case VAL_FUNC:
            if (v->fnval) {
                free(v->fnval->name);
                free(v->fnval);
            }
            break;
        case VAL_TASK:
            /* 句柄被丢弃时若任务仍在跑，必须 join 后再回收，避免线程访问已释放环境 */
            if (v->task) {
                if (v->task->started && !v->task->joined) {
                    pthread_join(v->task->tid, NULL);
                    v->task->joined = 1;
                }
                if (v->task->result) val_free(v->task->result);
                free(v->task);
            }
            break;
        default: break;
    }
    free(v);
}

static Value *val_retain(Value *v) {
    if (v) {
        __atomic_add_fetch(&v->refcount, 1, __ATOMIC_ACQ_REL);
        reflog_add(v, +1);
    }
    return v;
}

/* 生根 retain：值被写入容器（列表/字典）或环境（变量槽），
   该引用的生死由容器自己负责，因此【不参与】异常回滚。
   若误用普通 val_retain，try 块内 追加(外层列表, 值) 的元素
   会在捕获异常时被错误回收。 */
static Value *val_retain_root(Value *v) {
    if (v) __atomic_add_fetch(&v->refcount, 1, __ATOMIC_ACQ_REL);
    return v;
}

static void env_release(struct Env *e);   /* 前向声明：回滚残留作用域 */

/* ---------- 异常安全的临时缓冲 ----------
   规则：内建函数中「分配 → 可能报错 → free」的裸内存，必须走这对函数。
   co_tmp_alloc 会把指针登记到所有活跃 try 帧；正常路径用 co_tmp_free 释放
   （同时注销登记）；异常路径由 reflog_rollback 统一回收。 */
static void *co_tmp_alloc(size_t n) {
    void *p = malloc(n);
    if (!p) runtime_error(NULL, "内存不足（申请 %zu 字节）", n);
    for (int d = 0; d < g_try_depth; d++) {
        TryFrame *f = &g_try[d];
        if (!f->recording) continue;
        if (f->nbuf >= f->capbuf) {
            int nc = f->capbuf ? f->capbuf * 2 : 8;
            void **nb = realloc(f->bufs, (size_t)nc * sizeof(void*));
            if (!nb) { free(p); runtime_error(NULL, "内存不足（临时缓冲登记表扩容失败）"); }
            f->bufs = nb; f->capbuf = nc;
        }
        f->bufs[f->nbuf++] = p;
    }
    return p;
}
static void co_tmp_free(void *p) {
    if (!p) return;
    for (int d = 0; d < g_try_depth; d++) {
        TryFrame *f = &g_try[d];
        for (int i = 0; i < f->nbuf; i++)
            if (f->bufs[i] == p) f->bufs[i] = NULL;   /* 打洞，避免二次释放 */
    }
    free(p);
}

/* 逐指针聚合净增量并释放。调用前本帧必须已弹出帧栈且 recording=0。 */
static void reflog_rollback(TryFrame *f) {
    /* 顺序很关键：先回收无主的临时引用，再释放残留作用域。
       反过来会先让作用域里的值归零 free，导致随后的引用回滚访问野指针。
       （作用域持有的是生根引用，故此刻这些值的 refcount 必 > net，不会误 free。） */
    for (int i = 0; i < f->nlog; i++) {
        Value *v = f->log[i].v;
        if (!v) continue;
        int net = 0;
        for (int j = i; j < f->nlog; j++)
            if (f->log[j].v == v) { net += f->log[j].delta; f->log[j].v = NULL; }
        /* 真实 refcount ≥ net（可能还有生根引用），故释放 net 次
           最多恰好归零，不会提前 free 造成二次释放。 */
        while (net-- > 0) val_free(v);
    }
    free(f->log);
    f->log = NULL; f->nlog = f->caplog = 0;

    /* 被 longjmp 跳过的函数局部作用域，按创建顺序的逆序释放 */
    for (int i = f->nenv - 1; i >= 0; i--)
        if (f->envs[i]) env_release(f->envs[i]);
    free(f->envs);
    f->envs = NULL; f->nenv = f->capenv = 0;

    /* 被 longjmp 跳过的裸临时缓冲 */
    for (int i = 0; i < f->nbuf; i++) free(f->bufs[i]);
    free(f->bufs);
    f->bufs = NULL; f->nbuf = f->capbuf = 0;
}

/* 异常载荷及其内部子值必须免于回滚：它要活着交给 捕获 块 */
static void reflog_protect(Value *v) {
    if (!v) return;
    reflog_forget(v);
    if (v->type == VAL_LIST) {
        for (int i = 0; i < v->list_len; i++) reflog_protect(v->lval[i]);
    } else if (v->type == VAL_MAP && v->mval) {
        for (int i = 0; i < MAX_DICT_SIZE; i++)
            if (v->mval->entries[i].used) reflog_protect(v->mval->entries[i].val);
    }
}

/* 下标取值（兼容浮点下标，向零截断）。
   必须做类型检查：曾经这里裸读 union，导致 表["键"] 把字符串指针
   当下标用，越界读出垃圾值甚至崩溃。 */
static long idx_val(Value *idx) {
    if (!idx) runtime_error(NULL, "下标为空");
    switch (idx->type) {
        case VAL_INT:   return idx->ival;
        case VAL_BOOL:  return idx->ival ? 1 : 0;
        case VAL_FLOAT: return (long)idx->fval;
        default:
            runtime_error(NULL, "下标必须是数字");
            return 0;
    }
}

/* ========== UTF-8 字符层 ==========
   coCN 是中文编程语言，字符串的一切"长度/下标/截取/查找"
   都必须以【字符（码点）】为单位，而不是字节。
   否则 长度("苹果") 会得到 6，"苹果"[0] 会取到半个汉字。 */

/* 该起始字节所引导的 UTF-8 序列字节数（非法字节按 1 处理，保证不死循环） */
static int utf8_seq_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

/* 字符（码点）个数 */
static long utf8_strlen(const char *s) {
    long n = 0;
    for (const char *p = s; *p; ) { p += utf8_seq_len((unsigned char)*p); n++; }
    return n;
}

/* 第 ci 个字符的字节偏移；ci 等于总字符数时返回 strlen；越界返回 -1 */
static long utf8_byte_offset(const char *s, long ci) {
    if (ci < 0) return -1;
    long n = 0; const char *p = s;
    while (*p && n < ci) { p += utf8_seq_len((unsigned char)*p); n++; }
    if (n < ci) return -1;
    return (long)(p - s);
}

/* 把字节偏移换算成字符下标（供 查找 返回字符位置） */
static long utf8_char_index(const char *s, long byte_off) {
    long n = 0;
    for (const char *p = s; *p && (p - s) < byte_off; ) { p += utf8_seq_len((unsigned char)*p); n++; }
    return n;
}

/* 取第 ci 个字符，返回新分配的字符串；越界返回 NULL */
static char *utf8_char_at(const char *s, long ci) {
    long off = utf8_byte_offset(s, ci);
    if (off < 0 || !s[off]) return NULL;
    int len = utf8_seq_len((unsigned char)s[off]);
    char *r = malloc(len + 1);
    memcpy(r, s + off, len);
    r[len] = '\0';
    return r;
}

/* 按字符截取 [start, start+count)，自动裁剪到合法范围 */
static char *utf8_substr(const char *s, long start, long count) {
    long total = utf8_strlen(s);
    if (start < 0) start += total;            /* 支持负索引 */
    if (start < 0) start = 0;
    if (start > total) start = total;
    if (count < 0) count = 0;
    if (start + count > total) count = total - start;
    long b0 = utf8_byte_offset(s, start);
    long b1 = utf8_byte_offset(s, start + count);
    if (b0 < 0) b0 = 0;
    if (b1 < 0) b1 = (long)strlen(s);
    long nb = b1 - b0;
    char *r = malloc(nb + 1);
    memcpy(r, s + b0, nb);
    r[nb] = '\0';
    return r;
}

/* ---------- 动态字符串构建器：杜绝固定缓冲区溢出 ---------- */
typedef struct {
    char  *buf;
    size_t len;
    size_t cap;
} SBuf;

static void sb_init(SBuf *sb) {
    sb->cap = 64;
    sb->len = 0;
    sb->buf = malloc(sb->cap);
    if (!sb->buf) { fprintf(stderr, "内存不足: 字符串构建器\n"); exit(1); }
    sb->buf[0] = '\0';
}
static void sb_reserve(SBuf *sb, size_t extra) {
    if (sb->len + extra + 1 <= sb->cap) return;
    size_t nc = sb->cap ? sb->cap : 64;
    while (nc < sb->len + extra + 1) nc *= 2;
    char *nb = realloc(sb->buf, nc);
    if (!nb) { fprintf(stderr, "内存不足: 字符串构建器扩容\n"); exit(1); }
    sb->buf = nb;
    sb->cap = nc;
}
static void sb_append(SBuf *sb, const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    sb_reserve(sb, n);
    memcpy(sb->buf + sb->len, s, n);
    sb->len += n;
    sb->buf[sb->len] = '\0';
}
static void sb_appendf(SBuf *sb, const char *fmt, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof(tmp)) { sb_append(sb, tmp); return; }
    /* 超长：动态分配再写一次 */
    char *big = malloc((size_t)n + 1);
    if (!big) { fprintf(stderr, "内存不足: 格式化\n"); exit(1); }
    va_start(ap, fmt);
    vsnprintf(big, (size_t)n + 1, fmt, ap);
    va_end(ap);
    sb_append(sb, big);
    free(big);
}

/* 值转字符串（内部：写入构建器，支持任意深度/长度） */
static void val_to_sbuf(Value *v, SBuf *sb) {
    if (!v) { sb_append(sb, "空"); return; }
    switch (v->type) {
        case VAL_NULL:  sb_append(sb, "空"); break;
        case VAL_BOOL:  sb_append(sb, v->ival ? "真" : "假"); break;
        case VAL_INT:   sb_appendf(sb, "%ld", v->ival); break;
        case VAL_FLOAT: sb_appendf(sb, "%g", v->fval); break;
        case VAL_STRING: sb_append(sb, v->sval ? v->sval : ""); break;
        case VAL_LIST:
            sb_append(sb, "[");
            for (int i = 0; i < v->list_len; i++) {
                if (i) sb_append(sb, ", ");
                val_to_sbuf(v->lval[i], sb);
            }
            sb_append(sb, "]");
            break;
        case VAL_MAP: {
            sb_append(sb, "{");
            int first = 1;
            for (int i = 0; i < MAX_DICT_SIZE; i++) {
                if (!v->mval->entries[i].used) continue;
                if (!first) sb_append(sb, ", ");
                first = 0;
                sb_append(sb, "\"");
                sb_append(sb, v->mval->entries[i].key);
                sb_append(sb, "\": ");
                val_to_sbuf(v->mval->entries[i].val, sb);
            }
            sb_append(sb, "}");
            break;
        }
        case VAL_FUNC:
            sb_appendf(sb, "<函数 %s>", v->fnval && v->fnval->name ? v->fnval->name : "匿名");
            break;
        case VAL_TENSOR: {
            Tensor *t = v->tval;
            if (t->ndim == 1) {
                sb_append(sb, "[");
                for (int i = 0; i < t->size; i++) {
                    if (i) sb_append(sb, ", ");
                    sb_appendf(sb, "%g", t->data[i]);
                }
                sb_append(sb, "]");
            } else if (t->ndim == 2) {
                sb_append(sb, "[");
                for (int i = 0; i < t->shape[0]; i++) {
                    if (i) sb_append(sb, ", ");
                    sb_append(sb, "[");
                    for (int j = 0; j < t->shape[1]; j++) {
                        if (j) sb_append(sb, ", ");
                        sb_appendf(sb, "%g", t->data[i * t->shape[1] + j]);
                    }
                    sb_append(sb, "]");
                }
                sb_append(sb, "]");
            } else {
                sb_appendf(sb, "<张量 %d维 size=%d>", t->ndim, t->size);
            }
            break;
        }
        default: sb_append(sb, "<未知>"); break;
    }
}

/* 值转字符串（对外：返回堆上新串，调用方负责 free） */
static char *val_to_string(Value *v) {
    SBuf sb;
    sb_init(&sb);
    val_to_sbuf(v, &sb);
    return sb.buf;
}

/* ========== 字典操作 ========== */
static unsigned int dict_hash(const char *key) {
    unsigned int h = 0;
    while (*key) {
        h = h * 31 + *key++;
    }
    return h % MAX_DICT_SIZE;
}

static Value *dict_get(Map *m, const char *key) {
    unsigned int h = dict_hash(key);
    for (int i = 0; i < MAX_DICT_SIZE; i++) {
        unsigned int idx = (h + i) % MAX_DICT_SIZE;
        if (!m->entries[idx].used) return NULL;
        if (strcmp(m->entries[idx].key, key) == 0) {
            return m->entries[idx].val;
        }
    }
    return NULL;
}

static void dict_set(Map *m, const char *key, Value *val) {
    unsigned int h = dict_hash(key);
    for (int i = 0; i < MAX_DICT_SIZE; i++) {
        unsigned int idx = (h + i) % MAX_DICT_SIZE;
        if (m->entries[idx].used && strcmp(m->entries[idx].key, key) == 0) {
            /* 覆盖已有键：复用原 key，只换值（原实现会 strdup 新 key 并泄漏旧 key） */
            Value *old = m->entries[idx].val;
            m->entries[idx].val = val_retain_root(val);
            val_free(old);
            return;
        }
        if (!m->entries[idx].used) {
            m->entries[idx].key = strdup(key);
            m->entries[idx].val = val_retain_root(val);
            m->entries[idx].used = 1;
            m->count++;
            return;
        }
    }
    /* 原实现在字典写满时静默丢数据，这里明确报错 */
    fprintf(stderr, "运行时错误: 字典已满（上限 %d 个键），无法写入 \"%s\"\n", MAX_DICT_SIZE, key);
    exit(1);
}

/* 删除键：线性探测下用“向后回填”保持探测链完整，返回是否删除成功 */
static int dict_remove(Map *m, const char *key) {
    unsigned int h = dict_hash(key);
    int found = -1;
    for (int i = 0; i < MAX_DICT_SIZE; i++) {
        unsigned int idx = (h + i) % MAX_DICT_SIZE;
        if (!m->entries[idx].used) return 0;
        if (strcmp(m->entries[idx].key, key) == 0) { found = (int)idx; break; }
    }
    if (found < 0) return 0;
    free(m->entries[found].key);
    val_free(m->entries[found].val);
    m->entries[found].used = 0;
    m->entries[found].key = NULL;
    m->entries[found].val = NULL;
    m->count--;
    /* 回填：把后续同簇元素重新插入到正确位置，避免探测链断裂导致查不到 */
    int hole = found;
    for (int i = 1; i < MAX_DICT_SIZE; i++) {
        int idx = (found + i) % MAX_DICT_SIZE;
        if (!m->entries[idx].used) break;
        unsigned int ideal = dict_hash(m->entries[idx].key);
        /* 若该元素可以合法地放到 hole 上，则搬过去 */
        int dist_cur  = (idx - (int)ideal + MAX_DICT_SIZE) % MAX_DICT_SIZE;
        int dist_hole = (hole - (int)ideal + MAX_DICT_SIZE) % MAX_DICT_SIZE;
        if (dist_hole <= dist_cur) {
            m->entries[hole] = m->entries[idx];
            m->entries[idx].used = 0;
            m->entries[idx].key = NULL;
            m->entries[idx].val = NULL;
            hole = idx;
        }
    }
    return 1;
}

/* ========== 张量操作 ========== */
static Tensor *tensor_new(int ndim, int *shape) {
    Tensor *t = calloc(1, sizeof(Tensor));
    t->ndim = ndim;
    t->shape = malloc(ndim * sizeof(int));
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        size *= shape[i];
    }
    t->size = size;
    t->data = calloc(size, sizeof(float));
    t->grad = NULL;
    t->requires_grad = 0;
    t->tref = 1;
    t->_tag = 0;
    t->children = NULL;
    t->nchildren = 0;
    t->backward = NULL;
    return t;
}

/* 张量引用计数保留 */
static Tensor *tensor_retain(Tensor *t) {
    if (t) __atomic_add_fetch(&t->tref, 1, __ATOMIC_ACQ_REL);
    return t;
}

/* 张量引用计数释放（原子，支持并发任务共享张量） */
static void tensor_free(Tensor *t) {
    if (!t) return;
    if (__atomic_sub_fetch(&t->tref, 1, __ATOMIC_ACQ_REL) > 0) return;
    if (t->children) {
        for (int i = 0; i < t->nchildren; i++) tensor_free(t->children[i]);
        free(t->children);
    }
    free(t->data);
    free(t->grad);
    free(t->shape);
    free(t);
}

/* 确保梯度缓冲已分配并清零 */
static void ensure_grad(Tensor *t) {
    if (!t->grad) t->grad = calloc(t->size, sizeof(float));
}

/* 向父节点挂接子节点（保留引用） */
static void tensor_set_child(Tensor *parent, Tensor *child) {
    if (!child) return;
    parent->children = realloc(parent->children, (parent->nchildren + 1) * sizeof(Tensor*));
    parent->children[parent->nchildren++] = tensor_retain(child);
}

/* ========== 环境操作 ========== */
/* 必须用 calloc：names/values/types 三个指针数组若残留野指针，
   作用域销毁时的 free(types[i]) 会破坏堆元数据（历史致命 bug）。 */
static Env *env_new(Env *parent) {
    Env *e = calloc(1, sizeof(Env));
    if (!e) { fprintf(stderr, "错误：内存不足，无法创建作用域\n"); exit(1); }
    e->parent = parent;
    return e;
}

/* 共享环境的访问锁：递归锁，因为 env_get 会沿父链递归。
   仅当环境被标记 shared 时才使用，故单线程程序无锁开销。 */
static pthread_mutex_t g_env_lock;
static void env_lock_init(void) {
    pthread_mutexattr_t a;
    pthread_mutexattr_init(&a);
    pthread_mutexattr_settype(&a, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&g_env_lock, &a);
    pthread_mutexattr_destroy(&a);
}
#define ENV_IS_SHARED(e) (__atomic_load_n(&(e)->shared, __ATOMIC_ACQUIRE))
#define ENV_LOCK(e)   do { if (ENV_IS_SHARED(e)) pthread_mutex_lock(&g_env_lock);   } while (0)
#define ENV_UNLOCK(e) do { if (ENV_IS_SHARED(e)) pthread_mutex_unlock(&g_env_lock); } while (0)

/* 把某个环境及其所有祖先标记为跨线程共享（并发任务能看到它们） */
static void env_mark_shared(Env *e) {
    pthread_mutex_lock(&g_env_lock);
    for (Env *p = e; p; p = p->parent)
        __atomic_store_n(&p->shared, 1, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&g_env_lock);
}

/* 沿作用域链查找变量。
   必须用【迭代】而不是递归：作用域链长度等于调用深度，
   一旦程序递归几千层，递归版的 env_get 单次查找就会吃掉几千个栈帧，
   在栈守卫来不及介入的地方直接把栈撑爆（曾在 ASan 构建下复现段错误）。 */
static Value *env_get(Env *e, const char *name) {
    for (Env *cur = e; cur; cur = cur->parent) {
        ENV_LOCK(cur);
        for (int i = 0; i < cur->count; i++)
            if (strcmp(cur->names[i], name) == 0) {
                Value *v = cur->values[i];
                ENV_UNLOCK(cur);
                return v;
            }
        ENV_UNLOCK(cur);
    }
    return NULL;
}

static void env_set(Env *e, const char *name, Value *val) {
    ENV_LOCK(e);
    for (int i = 0; i < e->count; i++) {
        if (strcmp(e->names[i], name) == 0) {
            val_free(e->values[i]);
            e->values[i] = val_retain_root(val);
            ENV_UNLOCK(e);
            return;
        }
    }
    if (e->count < MAX_LOCALS) {
        e->names[e->count]  = strdup(name);
        e->values[e->count] = val_retain_root(val);
        e->types[e->count]  = NULL;   /* 必须显式置空，否则销毁时 free 野指针 */
        /* count 最后自增并使用 release 语义：确保其他线程看到 count 时，
           对应的 names/values 槽位内容已经写入完毕。 */
        __atomic_store_n(&e->count, e->count + 1, __ATOMIC_RELEASE);
        ENV_UNLOCK(e);
    } else {
        ENV_UNLOCK(e);
        fprintf(stderr, "错误：单个作用域变量数超过上限 %d（变量 %s）\n", MAX_LOCALS, name);
        exit(1);
    }
}

/* 记录/查询变量类型标注（用于静态类型检查） */
static void env_set_type(Env *e, const char *name, const char *type) {
    ENV_LOCK(e);
    for (int i = 0; i < e->count; i++) {
        if (strcmp(e->names[i], name) == 0) {
            if (e->types[i]) free(e->types[i]);
            e->types[i] = type ? strdup(type) : NULL;
            ENV_UNLOCK(e);
            return;
        }
    }
    if (e->count < MAX_LOCALS) {
        e->names[e->count]  = strdup(name);
        e->values[e->count] = NULL;
        e->types[e->count]  = type ? strdup(type) : NULL;
        __atomic_store_n(&e->count, e->count + 1, __ATOMIC_RELEASE);
        ENV_UNLOCK(e);
    } else {
        ENV_UNLOCK(e);
        fprintf(stderr, "错误：单个作用域变量数超过上限 %d（变量 %s）\n", MAX_LOCALS, name);
        exit(1);
    }
}

/* 同上：迭代遍历，不得递归 */
static const char *env_get_type(Env *e, const char *name) {
    for (Env *cur = e; cur; cur = cur->parent) {
        ENV_LOCK(cur);
        for (int i = 0; i < cur->count; i++)
            if (strcmp(cur->names[i], name) == 0) {
                const char *t = cur->types[i];
                ENV_UNLOCK(cur);
                return t;
            }
        ENV_UNLOCK(cur);
    }
    return NULL;
}

/* ========== 词法分析 ========== */
typedef enum {
    TOK_EOF, TOK_ID, TOK_INT, TOK_FLOAT, TOK_STRING, TOK_MULTILINE_STRING,
    TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH, TOK_DBLSLASH, TOK_MOD,
    TOK_EQ, TOK_NE, TOK_LT, TOK_LE, TOK_GT, TOK_GE,
    /* 位运算：系统编程基石（& | ^ ~ << >>，另有中文别名） */
    TOK_AMP, TOK_PIPE, TOK_CARET, TOK_TILDE, TOK_SHL, TOK_SHR,
    TOK_ASSIGN, TOK_LPAREN, TOK_RPAREN, TOK_LBRACK, TOK_RBRACK,
    TOK_LBRACE, TOK_RBRACE, TOK_COLON, TOK_COMMA, TOK_NEWLINE, TOK_DOT,
    /* 中文关键字 */
    TK_VAR, TK_CONST, TK_FUNC, TK_RETURN, TK_IF, TK_THEN,
    TK_ELSE, TK_ELIF, TK_END, TK_WHILE, TK_LOOP, TK_BREAK, TK_CONTINUE,
    TK_PRINT, TK_TENSOR, TK_FOR, TK_FROM, TK_TO, TK_STEP,
    /* 逻辑运算符 */
    TK_AND, TK_OR, TK_NOT,
    /* 强语言特性：布尔 / 模块 / 模式匹配 / 类型标注 / 推导式 */
    TK_TRUE, TK_FALSE, TK_IMPORT, TK_MATCH, TK_IS, TK_DEFAULT, TK_IN, TK_ARROW,
    TK_TYPE_INT, TK_TYPE_FLOAT, TK_TYPE_STRING, TK_TYPE_BOOL,
    TK_TYPE_LIST, TK_TYPE_MAP, TK_TYPE_NULL,
    TK_TYPE_NUM, TK_TYPE_FUNC, TK_TYPE_ANY,
    TK_CONCUR, TK_WAIT,  /* 并发原语：并发 / 等待 */
    TK_TRY, TK_CATCH, TK_FINALLY, TK_THROW  /* 异常：尝试 / 捕获 / 最终 / 抛出 */
} TokenType;

typedef struct {
    TokenType type;
    char      text[MAX_ID_LEN];
    int       line;
    int       col;
} Token;

static const char *src;
static int src_pos, src_line, src_col;
static Token cur_tok;

static void lex_error(const char *msg) {
    fprintf(stderr, "词法错误 (第%d行,%d列): %s\n", cur_tok.line, cur_tok.col, msg);
    exit(1);
}

/* 括号嵌套深度：>0 时换行被视为续行，支持多行列表/参数/字典字面量 */
static int g_bracket_depth = 0;

static void skip_whitespace() {
    while (src[src_pos]) {
        /* 跳过注释 */
        if (src[src_pos] == '#') { 
            while (src[src_pos] && src[src_pos] != '\n') {
                /* UTF-8多字节字符处理 */
                if ((src[src_pos] & 0x80) == 0) {
                    /* ASCII字符 */
                    src_pos++; src_col++;
                } else {
                    /* UTF-8字符，3-4字节 - 只对字符的起始字节增加列号 */
                    int bytes = (src[src_pos] & 0xE0) == 0xC0 ? 2 :
                                (src[src_pos] & 0xF0) == 0xE0 ? 3 :
                                (src[src_pos] & 0xF8) == 0xF0 ? 4 : 1;
                    for (int i = 0; i < bytes && src[src_pos]; i++) {
                        src_pos++;
                    }
                    src_col++; /* 每个字符只增加1列 */
                }
            }
        }
        /* 跳过空白 */
        if (src[src_pos] == ' ' || src[src_pos] == '\t') {
            src_pos++;
            src_col++;
            continue;
        }
        /* 换行符：括号内视为续行直接吞掉；顶层保留，由 advance 产出 TOK_NEWLINE
           以界定语句边界（否则解析器无法区分块与单行结构）。 */
        if (src[src_pos] == '\n') {
            if (g_bracket_depth > 0) { src_line++; src_col = 1; src_pos++; continue; }
            break;
        }
        /* 行尾反斜杠：显式续行 */
        if (src[src_pos] == '\\' && src[src_pos+1] == '\n') {
            src_pos += 2; src_line++; src_col = 1; continue;
        }
        /* 其他字符，停止跳过 */
        break;
    }
}

static void advance() {
    skip_whitespace();
    if (!src[src_pos]) {
        cur_tok.type = TOK_EOF;
        cur_tok.col = src_col;
        return;
    }
    cur_tok.line = src_line;
    cur_tok.col = src_col;

    /* 换行符 */
    if (src[src_pos] == '\n') {
        cur_tok.type = TOK_NEWLINE;
        cur_tok.text[0] = '\n';
        cur_tok.text[1] = 0;
        src_line++;
        src_col = 1;
        src_pos++;
        return;
    }
    
    /* 多行字符串 三引号 */
    if (src[src_pos] == '"' && src[src_pos+1] == '"' && src[src_pos+2] == '"') {
        src_pos += 3; src_col += 3;
        int st = src_pos;
        while (src[src_pos] && !(src[src_pos] == '"' && src[src_pos+1] == '"' && src[src_pos+2] == '"')) {
            if (src[src_pos] == '\n') { src_line++; src_col = 1; }
            else { src_col++; }
            src_pos++;
        }
        int len = src_pos - st;
        if (len >= MAX_ID_LEN) { lex_error("多行字符串过长"); }
        strncpy(cur_tok.text, src + st, len);
        cur_tok.text[len] = 0;
        if (src[src_pos]) { src_pos += 3; src_col += 3; }
        cur_tok.type = TOK_MULTILINE_STRING;
        return;
    }

    /* 字符串 */
    if (src[src_pos] == '"') {
        int st = ++src_pos; src_col++;
        while (src[src_pos] && src[src_pos] != '"') {
            if (src[src_pos] == '\\') { src_pos++; src_col++; }
            src_pos++; src_col++;
        }
        int len = src_pos - st;
        if (len >= MAX_ID_LEN) { lex_error("字符串过长"); }
        strncpy(cur_tok.text, src + st, len);
        cur_tok.text[len] = 0;
        if (src[src_pos] == '"') { src_pos++; src_col++; }
        cur_tok.type = TOK_STRING;
        return;
    }
    /* 数字：十进制 / 0x十六进制 / 0b二进制 / 0o八进制，均支持 _ 分隔符。
       系统编程离不开位模式字面量（0xFF、0b1010_0101），进制在词法期就折算成十进制文本，
       AST 与求值层无需感知进制；溢出一律报错，绝不静默回绕。 */
    if (isdigit((unsigned char)src[src_pos])) {
        char pfx = src[src_pos + 1];
        if (src[src_pos] == '0' && (pfx=='x'||pfx=='X'||pfx=='b'||pfx=='B'||pfx=='o'||pfx=='O')) {
            int base = (pfx=='x'||pfx=='X') ? 16 : (pfx=='b'||pfx=='B') ? 2 : 8;
            src_pos += 2; src_col += 2;
            char digits[MAX_ID_LEN]; int dn = 0, ndig = 0;
            while (src[src_pos]) {
                char ch = src[src_pos];
                if (ch == '_') { src_pos++; src_col++; continue; }   /* 分隔符仅作视觉分组 */
                int ok = (base == 16) ? isxdigit((unsigned char)ch)
                       : (base == 8)  ? (ch >= '0' && ch <= '7')
                                      : (ch == '0' || ch == '1');
                if (!ok) break;
                if (dn < (int)sizeof(digits) - 1) digits[dn++] = ch;
                ndig++; src_pos++; src_col++;
            }
            digits[dn] = 0;
            if (ndig == 0) lex_error("进制前缀后缺少有效数字");
            /* 紧跟字母/数字说明数字写错了进制（如 0b12、0xG、0o8） */
            if (isalnum((unsigned char)src[src_pos]))
                lex_error("进制字面量中出现该进制不允许的数字");
            if (ndig >= (int)sizeof(digits) - 1) lex_error("整数字面量超出范围");
            errno = 0;
            unsigned long long uv = strtoull(digits, NULL, base);
            if (errno == ERANGE || uv > 9223372036854775807ULL) lex_error("整数字面量超出范围");
            snprintf(cur_tok.text, MAX_ID_LEN, "%lld", (long long)uv);
            cur_tok.type = TOK_INT;
            return;
        }
        char buf[MAX_ID_LEN]; int bn = 0;
        while (isdigit((unsigned char)src[src_pos]) || src[src_pos] == '_') {
            if (src[src_pos] != '_' && bn < (int)sizeof(buf) - 1) buf[bn++] = src[src_pos];
            src_pos++; src_col++;
        }
        if (src[src_pos] == '.') {
            if (bn < (int)sizeof(buf) - 1) buf[bn++] = '.';
            src_pos++; src_col++;
            while (isdigit((unsigned char)src[src_pos]) || src[src_pos] == '_') {
                if (src[src_pos] != '_' && bn < (int)sizeof(buf) - 1) buf[bn++] = src[src_pos];
                src_pos++; src_col++;
            }
            buf[bn] = 0;
            memcpy(cur_tok.text, buf, bn + 1);
            cur_tok.type = TOK_FLOAT;
        } else {
            buf[bn] = 0;
            /* 十进制整数同样要拒绝越界：atol 溢出是未定义行为 */
            errno = 0;
            char *endp = NULL;
            unsigned long long uv = strtoull(buf, &endp, 10);
            if (errno == ERANGE || uv > 9223372036854775807ULL) lex_error("整数字面量超出范围");
            memcpy(cur_tok.text, buf, bn + 1);
            cur_tok.type = TOK_INT;
        }
        return;
    }
    /* 标识符 / 关键字 */
    if (isalpha(src[src_pos]) || (src[src_pos] & 0x80)) {
        int st = src_pos;
        while (src[src_pos]) {
            /* UTF-8字符起始字节检测 */
            int is_utf8_start = (src[src_pos] & 0x80);
            int bytes = 0;
            if (!is_utf8_start) {
                /* ASCII字符 */
                if (isalnum(src[src_pos]) || src[src_pos] == '_') {
                    src_pos++; src_col++;
                } else {
                    break; /* 不是有效的标识符字符 */
                }
            } else {
                /* UTF-8字符起始字节，计算字符宽度 */
                bytes = (src[src_pos] & 0xE0) == 0xC0 ? 2 :
                        (src[src_pos] & 0xF0) == 0xE0 ? 3 :
                        (src[src_pos] & 0xF8) == 0xF0 ? 4 : 1;
                for (int i = 0; i < bytes && src[src_pos]; i++) {
                    src_pos++;
                }
                src_col++; /* 每个字符只增加1列 */
            }
        }
        int len = src_pos - st;
        if (len >= MAX_ID_LEN) len = MAX_ID_LEN - 1;
        strncpy(cur_tok.text, src + st, len);
        cur_tok.text[len] = 0;
        /* 关键字匹配 */
        if (strcmp(cur_tok.text, "让") == 0 || strcmp(cur_tok.text, "变量") == 0) cur_tok.type = TK_VAR;
        else if (strcmp(cur_tok.text, "常量") == 0) cur_tok.type = TK_CONST;
        else if (strcmp(cur_tok.text, "函数") == 0) cur_tok.type = TK_FUNC;
        else if (strcmp(cur_tok.text, "返回") == 0) cur_tok.type = TK_RETURN;
        else if (strcmp(cur_tok.text, "如果") == 0) cur_tok.type = TK_IF;
        else if (strcmp(cur_tok.text, "则") == 0) cur_tok.type = TK_THEN;
        else if (strcmp(cur_tok.text, "那么") == 0) cur_tok.type = TK_THEN;  /* 则 的口语别名 */
        else if (strcmp(cur_tok.text, "否则") == 0) cur_tok.type = TK_ELSE;
        else if (strcmp(cur_tok.text, "否则如果") == 0) cur_tok.type = TK_ELIF;
        else if (strcmp(cur_tok.text, "否则若") == 0) cur_tok.type = TK_ELIF;  /* 与 若 配套 */
        else if (strcmp(cur_tok.text, "结束") == 0) cur_tok.type = TK_END;
        else if (strcmp(cur_tok.text, "当") == 0) cur_tok.type = TK_WHILE;
        else if (strcmp(cur_tok.text, "循环") == 0) cur_tok.type = TK_LOOP;
        else if (strcmp(cur_tok.text, "跳出") == 0) cur_tok.type = TK_BREAK;
        else if (strcmp(cur_tok.text, "继续") == 0) cur_tok.type = TK_CONTINUE;
        else if (strcmp(cur_tok.text, "输出") == 0) cur_tok.type = TK_PRINT;
        else if (strcmp(cur_tok.text, "张量") == 0) cur_tok.type = TK_TENSOR;
        else if (strcmp(cur_tok.text, "对于") == 0) cur_tok.type = TK_FOR;
        else if (strcmp(cur_tok.text, "从") == 0) cur_tok.type = TK_FROM;
        else if (strcmp(cur_tok.text, "到") == 0) cur_tok.type = TK_TO;
        else if (strcmp(cur_tok.text, "步长") == 0) cur_tok.type = TK_STEP;
        /* 逻辑运算符 */
        else if (strcmp(cur_tok.text, "且") == 0) cur_tok.type = TK_AND;
        else if (strcmp(cur_tok.text, "或") == 0) cur_tok.type = TK_OR;
        else if (strcmp(cur_tok.text, "非") == 0) cur_tok.type = TK_NOT;
        /* 位运算中文别名：AST 里统一存 ASCII 符号（op 字段仅 8 字节，装不下 3 个汉字） */
        else if (strcmp(cur_tok.text, "按位与") == 0)   { cur_tok.type = TOK_AMP;   strcpy(cur_tok.text, "&");  }
        else if (strcmp(cur_tok.text, "按位或") == 0)   { cur_tok.type = TOK_PIPE;  strcpy(cur_tok.text, "|");  }
        else if (strcmp(cur_tok.text, "按位异或") == 0) { cur_tok.type = TOK_CARET; strcpy(cur_tok.text, "^");  }
        else if (strcmp(cur_tok.text, "按位取反") == 0) { cur_tok.type = TOK_TILDE; strcpy(cur_tok.text, "~");  }
        else if (strcmp(cur_tok.text, "左移") == 0)     { cur_tok.type = TOK_SHL;   strcpy(cur_tok.text, "<<"); }
        else if (strcmp(cur_tok.text, "右移") == 0)     { cur_tok.type = TOK_SHR;   strcpy(cur_tok.text, ">>"); }
        /* 强语言特性关键字 */
        else if (strcmp(cur_tok.text, "真") == 0) cur_tok.type = TK_TRUE;
        else if (strcmp(cur_tok.text, "假") == 0) cur_tok.type = TK_FALSE;
        else if (strcmp(cur_tok.text, "导入") == 0) cur_tok.type = TK_IMPORT;
        else if (strcmp(cur_tok.text, "匹配") == 0) cur_tok.type = TK_MATCH;
        else if (strcmp(cur_tok.text, "是") == 0) cur_tok.type = TK_IS;
        else if (strcmp(cur_tok.text, "情况") == 0) cur_tok.type = TK_IS;  /* 是 的别名，读起来更像 case */
        else if (strcmp(cur_tok.text, "默认") == 0) cur_tok.type = TK_DEFAULT;
        /* 推导式里的"属于"：于 与 在 互为别名，后者更贴合其余语法的口语习惯 */
        else if (strcmp(cur_tok.text, "于") == 0) cur_tok.type = TK_IN;
        else if (strcmp(cur_tok.text, "在") == 0) cur_tok.type = TK_IN;
        else if (strcmp(cur_tok.text, "若") == 0) cur_tok.type = TK_IF; /* 若 作为 如果 的简写别名 */
        else if (strcmp(cur_tok.text, "整数") == 0) cur_tok.type = TK_TYPE_INT;
        else if (strcmp(cur_tok.text, "浮点") == 0) cur_tok.type = TK_TYPE_FLOAT;
        else if (strcmp(cur_tok.text, "字符串") == 0) cur_tok.type = TK_TYPE_STRING;
        else if (strcmp(cur_tok.text, "布尔") == 0) cur_tok.type = TK_TYPE_BOOL;
        else if (strcmp(cur_tok.text, "列表") == 0) cur_tok.type = TK_TYPE_LIST;
        else if (strcmp(cur_tok.text, "字典") == 0) cur_tok.type = TK_TYPE_MAP;
        else if (strcmp(cur_tok.text, "空") == 0) cur_tok.type = TK_TYPE_NULL;
        /* 新增类型标注：数字（整数或浮点）、函数（可调用）、任意（关闭检查） */
        else if (strcmp(cur_tok.text, "数字") == 0) cur_tok.type = TK_TYPE_NUM;
        else if (strcmp(cur_tok.text, "函数值") == 0) cur_tok.type = TK_TYPE_FUNC;
        else if (strcmp(cur_tok.text, "任意") == 0) cur_tok.type = TK_TYPE_ANY;
        else if (strcmp(cur_tok.text, "并发") == 0) cur_tok.type = TK_CONCUR;
        else if (strcmp(cur_tok.text, "等待") == 0) cur_tok.type = TK_WAIT;
        else if (strcmp(cur_tok.text, "尝试") == 0) cur_tok.type = TK_TRY;
        else if (strcmp(cur_tok.text, "捕获") == 0) cur_tok.type = TK_CATCH;
        else if (strcmp(cur_tok.text, "最终") == 0) cur_tok.type = TK_FINALLY;
        else if (strcmp(cur_tok.text, "抛出") == 0) cur_tok.type = TK_THROW;
        else cur_tok.type = TOK_ID;
        return;
    }
    /* 单字符符号 */
    char c = src[src_pos++]; src_col++;
    switch (c) {
        case '+': cur_tok.type = TOK_PLUS; break;
        case '-':
            /* 多字符符号必须 return：函数尾部会把 text 覆写成单字符 */
            if (src[src_pos] == '>') { src_pos++; src_col++; cur_tok.type = TK_ARROW; strcpy(cur_tok.text, "->"); return; }
            cur_tok.type = TOK_MINUS; break;
        case '*': cur_tok.type = TOK_STAR; break;
        case '/':
            if (src[src_pos] == '/') {   /* // 整除 */
                src_pos++; src_col++;
                cur_tok.type = TOK_DBLSLASH;
                cur_tok.text[0]='/'; cur_tok.text[1]='/'; cur_tok.text[2]=0;
                return;
            }
            cur_tok.type = TOK_SLASH; break;
        case '%': cur_tok.type = TOK_MOD; break;
        case '(': cur_tok.type = TOK_LPAREN; g_bracket_depth++; break;
        case ')': cur_tok.type = TOK_RPAREN; if (g_bracket_depth > 0) g_bracket_depth--; break;
        case '[': cur_tok.type = TOK_LBRACK; g_bracket_depth++; break;
        case ']': cur_tok.type = TOK_RBRACK; if (g_bracket_depth > 0) g_bracket_depth--; break;
        case '{': cur_tok.type = TOK_LBRACE; g_bracket_depth++; break;
        case '}': cur_tok.type = TOK_RBRACE; if (g_bracket_depth > 0) g_bracket_depth--; break;
        case ':': cur_tok.type = TOK_COLON; break;
        case ',': cur_tok.type = TOK_COMMA; break;
        case '.': cur_tok.type = TOK_DOT; break;
        case '=':
            if (src[src_pos] == '=') { src_pos++; src_col++; cur_tok.type = TOK_EQ; strcpy(cur_tok.text, "=="); }
            else { cur_tok.type = TOK_ASSIGN; cur_tok.text[0] = '='; cur_tok.text[1] = 0; }
            return;
        case '<':
            if (src[src_pos] == '=') { src_pos++; src_col++; cur_tok.type = TOK_LE;  strcpy(cur_tok.text, "<="); return; }
            if (src[src_pos] == '<') { src_pos++; src_col++; cur_tok.type = TOK_SHL; strcpy(cur_tok.text, "<<"); return; }
            cur_tok.type = TOK_LT; break;
        case '>':
            if (src[src_pos] == '=') { src_pos++; src_col++; cur_tok.type = TOK_GE;  strcpy(cur_tok.text, ">="); return; }
            if (src[src_pos] == '>') { src_pos++; src_col++; cur_tok.type = TOK_SHR; strcpy(cur_tok.text, ">>"); return; }
            cur_tok.type = TOK_GT; break;
        case '!':
            if (src[src_pos] == '=') { src_pos++; src_col++; cur_tok.type = TOK_NE; strcpy(cur_tok.text, "!="); return; }
            lex_error("未知符号 '!'，期望 '!='");
            break;
        /* 位运算符：&& / || 明确报错并指引到中文逻辑运算符，避免与「按位」语义混淆 */
        case '&':
            if (src[src_pos] == '&') lex_error("不支持 '&&'，逻辑与请用 '且'，按位与请用单个 '&' 或 '按位与'");
            cur_tok.type = TOK_AMP; break;
        case '|':
            if (src[src_pos] == '|') lex_error("不支持 '||'，逻辑或请用 '或'，按位或请用单个 '|' 或 '按位或'");
            cur_tok.type = TOK_PIPE; break;
        case '^': cur_tok.type = TOK_CARET; break;
        case '~': cur_tok.type = TOK_TILDE; break;
        default: 
            fprintf(stderr, "未识别的字符: 0x%02X at line=%d col=%d pos=%d\n", 
                    (unsigned char)c, src_line, src_col, src_pos);
            lex_error("未识别的字符");
    }
    cur_tok.text[0] = c; cur_tok.text[1] = 0;
}

/* 处理转义字符 */
static void process_escape(char *dest, const char *src, int len) {
    int j = 0;
    for (int i = 0; i < len; i++) {
        if (src[i] == '\\' && i + 1 < len) {
            i++;
            switch (src[i]) {
                case 'n': dest[j++] = '\n'; break;
                case 't': dest[j++] = '\t'; break;
                case 'r': dest[j++] = '\r'; break;
                case '\\': dest[j++] = '\\'; break;
                case '"': dest[j++] = '"'; break;
                case '\'': dest[j++] = '\''; break;
                default: dest[j++] = src[i]; break;
            }
        } else {
            dest[j++] = src[i];
        }
    }
    dest[j] = '\0';
}


static void expect(TokenType t) {
    if (cur_tok.type != t) {
        fprintf(stderr, "语法错误 (第%d行,%d列): 期望 %d, 得到 '%s'\n",
                cur_tok.line, cur_tok.col, t, cur_tok.text);
        exit(1);
    }
    advance();
}

/* ========== AST 节点定义 ========== */
typedef enum {
    ND_PROGRAM, ND_VAR_DECL, ND_ASSIGN, ND_IDENT,
    ND_FUNC_DEF, ND_FUNC_CALL, ND_RETURN,
    ND_IF, ND_WHILE, ND_FOR,
    ND_BINARY, ND_UNARY, ND_LITERAL,
    ND_LIST_LIT, ND_MAP_LIT, ND_BREAK, ND_CONTINUE,
    ND_IMPORT, ND_MATCH,  /* 强语言特性：模块导入 / 模式匹配 */
    ND_CONCUR, ND_WAIT,   /* 并发原语：并发 / 等待 */
    ND_TRY, ND_THROW      /* 异常处理：尝试/捕获/最终 与 抛出 */
} NodeType;

/* else分支结构 */
typedef struct ElseBranch {
    struct Node *cond;
    struct Node **body;
    int bcnt;
    int is_else;
} ElseBranch;

/* 模式匹配分支 */
typedef struct MatchArm {
    struct Node *pattern;     /* 字面量/范围/标识符(通配) */
    struct Node **body;
    int bcnt;
    int is_default;           /* 默认分支 */
} MatchArm;

typedef struct Node {
    NodeType type;
    int line;
    int col;
    union {
        struct { struct Node **stmts; int cnt; } prog;
        struct { char *name; struct Node *init; int is_const; char *type_annot; } var_decl;
        struct { struct Node *tgt, *val; } assign;
        struct { char *name; } ident;
        struct { char *name; char **pnames; int pcnt; char **ptypes; char *ret_type; struct Node *body; } func_def;
        struct { struct Node *callee; struct Node **args; int acnt; } func_call;
        struct { struct Node *val; } ret_stmt;
        struct { struct Node *cond; struct Node **then_body; int tcnt; 
                 ElseBranch *else_branches; int br_cnt; } if_stmt;
        struct { struct Node *cond; struct Node **body; int bcnt; } while_stmt;
        struct { char *var; struct Node *start, *end, *step; 
                 struct Node **body; int bcnt; } for_stmt;
        struct { char op[8]; struct Node *left, *right; } binary;
        struct { char op[8]; struct Node *operand; } unary;
        struct { char *value; ValType lit_type; } literal;
        struct { struct Node **elements; int ecnt; int is_comprehension; } list_lit;
        struct { struct Node **keys; struct Node **vals; int kcnt; int is_comprehension; } map_lit;
        struct { char *path; } import_stmt;
        struct { struct Node *expr; MatchArm *arms; int arm_cnt; } match_stmt;
        struct { struct Node *expr; } concur_stmt;  /* 并发 expr */
        struct { struct Node *expr; } wait_stmt;    /* 等待 expr */
        /* 尝试 ... 捕获 [变量] ... 最终 ... 结束
           has_catch/has_finally 区分「只捕获」「只清理」「两者都有」 */
        struct { struct Node **body;    int bcnt;
                 struct Node **cbody;   int ccnt;
                 struct Node **fbody;   int fcnt;
                 char *var; int has_catch; int has_finally; } try_stmt;
        struct { struct Node *expr; } throw_stmt;   /* 抛出 expr */
    };
} Node;

/* 释放整棵 AST（用于程序结束清理，避免退出时内存泄漏） */
static void node_free(Node *n) {
    if (!n) return;
    switch (n->type) {
        case ND_PROGRAM:
            for (int i = 0; i < n->prog.cnt; i++) node_free(n->prog.stmts[i]);
            free(n->prog.stmts); break;
        case ND_VAR_DECL:
            free(n->var_decl.name);
            free(n->var_decl.type_annot);
            if (n->var_decl.init) node_free(n->var_decl.init);
            break;
        case ND_ASSIGN:
            node_free(n->assign.tgt); node_free(n->assign.val); break;
        case ND_IDENT:
            free(n->ident.name); break;
        case ND_FUNC_DEF:
            free(n->func_def.name);
            free(n->func_def.ret_type);
            for (int i = 0; i < n->func_def.pcnt; i++) { free(n->func_def.pnames[i]); free(n->func_def.ptypes[i]); }
            free(n->func_def.pnames);
            free(n->func_def.ptypes);
            node_free(n->func_def.body);
            break;
        case ND_FUNC_CALL:
            node_free(n->func_call.callee);
            for (int i = 0; i < n->func_call.acnt; i++) node_free(n->func_call.args[i]);
            free(n->func_call.args);
            break;
        case ND_RETURN:
            if (n->ret_stmt.val) node_free(n->ret_stmt.val);
            break;
        case ND_IF:
            node_free(n->if_stmt.cond);
            for (int i = 0; i < n->if_stmt.tcnt; i++) node_free(n->if_stmt.then_body[i]);
            free(n->if_stmt.then_body);
            for (int b = 0; b < n->if_stmt.br_cnt; b++) {
                ElseBranch *eb = &n->if_stmt.else_branches[b];
                if (eb->cond) node_free(eb->cond);
                for (int i = 0; i < eb->bcnt; i++) node_free(eb->body[i]);
                free(eb->body);
            }
            free(n->if_stmt.else_branches);
            break;
        case ND_WHILE:
            node_free(n->while_stmt.cond);
            for (int i = 0; i < n->while_stmt.bcnt; i++) node_free(n->while_stmt.body[i]);
            free(n->while_stmt.body); break;
        case ND_FOR:
            free(n->for_stmt.var);
            if (n->for_stmt.start) node_free(n->for_stmt.start);
            if (n->for_stmt.end) node_free(n->for_stmt.end);
            if (n->for_stmt.step) node_free(n->for_stmt.step);
            for (int i = 0; i < n->for_stmt.bcnt; i++) node_free(n->for_stmt.body[i]);
            free(n->for_stmt.body); break;
        case ND_BINARY:
            node_free(n->binary.left); node_free(n->binary.right); break;
        case ND_UNARY:
            node_free(n->unary.operand); break;
        case ND_LITERAL:
            free(n->literal.value); break;
        case ND_LIST_LIT:
            for (int i = 0; i < n->list_lit.ecnt; i++) if (n->list_lit.elements[i]) node_free(n->list_lit.elements[i]);
            free(n->list_lit.elements); break;
        case ND_MAP_LIT:
            if (n->map_lit.is_comprehension) {
                /* 推导式：keys[2]==vals[2]、keys[3]==vals[3] 为同一节点，只释放一次 */
                if (n->map_lit.keys[0]) node_free(n->map_lit.keys[0]);
                if (n->map_lit.vals[0]) node_free(n->map_lit.vals[0]);
                if (n->map_lit.keys[1]) node_free(n->map_lit.keys[1]);
                if (n->map_lit.vals[1]) node_free(n->map_lit.vals[1]);
                if (n->map_lit.keys[2]) node_free(n->map_lit.keys[2]);
                if (n->map_lit.keys[3]) node_free(n->map_lit.keys[3]);
                free(n->map_lit.keys); free(n->map_lit.vals);
                break;
            }
            for (int i = 0; i < n->map_lit.kcnt; i++) { if (n->map_lit.keys[i]) node_free(n->map_lit.keys[i]); if (n->map_lit.vals[i]) node_free(n->map_lit.vals[i]); }
            free(n->map_lit.keys); free(n->map_lit.vals); break;
        case ND_IMPORT:
            free(n->import_stmt.path); break;
        case ND_CONCUR:
            if (n->concur_stmt.expr) node_free(n->concur_stmt.expr);
            break;
        case ND_WAIT:
            if (n->wait_stmt.expr) node_free(n->wait_stmt.expr);
            break;
        case ND_TRY:
            for (int i = 0; i < n->try_stmt.bcnt; i++) node_free(n->try_stmt.body[i]);
            free(n->try_stmt.body);
            for (int i = 0; i < n->try_stmt.ccnt; i++) node_free(n->try_stmt.cbody[i]);
            free(n->try_stmt.cbody);
            for (int i = 0; i < n->try_stmt.fcnt; i++) node_free(n->try_stmt.fbody[i]);
            free(n->try_stmt.fbody);
            free(n->try_stmt.var);
            break;
        case ND_THROW:
            if (n->throw_stmt.expr) node_free(n->throw_stmt.expr);
            break;
        case ND_MATCH: {
            if (n->match_stmt.expr) node_free(n->match_stmt.expr);
            for (int a = 0; a < n->match_stmt.arm_cnt; a++) {
                MatchArm *arm = &n->match_stmt.arms[a];
                if (!arm->is_default && arm->pattern) node_free(arm->pattern);
                for (int i = 0; i < arm->bcnt; i++) node_free(arm->body[i]);
                free(arm->body);
            }
            free(n->match_stmt.arms);
            break;
        }
        default: break;
    }
    free(n);
}

/* ========== 语法分析器 ========== */
static Node *parse_expression();
static Node *parse_statement();

static Node *make_node(NodeType t) {
    Node *n = calloc(1, sizeof(Node));
    n->type = t;
    n->line = cur_tok.line;
    n->col  = cur_tok.col;
    return n;
}

static Node *parse_primary() {
    if (cur_tok.type == TK_TRUE || cur_tok.type == TK_FALSE) {
        Node *n = make_node(ND_LITERAL);
        n->literal.lit_type = VAL_BOOL;   /* 布尔是一等类型，不再退化为整数 0/1 */
        n->literal.value = strdup(cur_tok.type == TK_TRUE ? "1" : "0");
        advance();
        return n;
    }
    if (cur_tok.type == TOK_INT || cur_tok.type == TOK_FLOAT) {
        Node *n = make_node(ND_LITERAL);
        n->literal.lit_type = (cur_tok.type == TOK_INT) ? VAL_INT : VAL_FLOAT;
        n->literal.value = strdup(cur_tok.text);
        advance();
        return n;
    } else if (cur_tok.type == TOK_STRING) {
        Node *n = make_node(ND_LITERAL);
        n->literal.lit_type = VAL_STRING;
        char buf[MAX_ID_LEN * 2];
        process_escape(buf, cur_tok.text, strlen(cur_tok.text));
        n->literal.value = strdup(buf);
        advance();
        return n;
    } else if (cur_tok.type == TOK_MULTILINE_STRING) {
        Node *n = make_node(ND_LITERAL);
        n->literal.lit_type = VAL_STRING;
        n->literal.value = strdup(cur_tok.text);
        advance();
        return n;
    } else if (cur_tok.type == TOK_ID) {
        Node *n = make_node(ND_IDENT);
        n->ident.name = strdup(cur_tok.text);
        advance();
        return n;
    } else if (cur_tok.type == TK_TENSOR) {
        /* 张量(...) 当作名为 "张量" 的内置函数调用 */
        Node *n = make_node(ND_IDENT);
        n->ident.name = strdup("张量");
        advance();
        return n;
    } else if (cur_tok.type == TOK_LPAREN) {
        advance();
        Node *n = parse_expression();
        expect(TOK_RPAREN);
        return n;
    } else if (cur_tok.type == TOK_LBRACK) {
        Node *n = make_node(ND_LIST_LIT);
        advance();
        n->list_lit.elements = NULL; int cnt = 0, cap = 0;
        n->list_lit.is_comprehension = 0;
        if (cur_tok.type != TOK_RBRACK) {
            Node *first = parse_expression();
            if (cur_tok.type == TK_FOR) {
                /* 列表推导式：[body 对于 x 于 src 若 filter] */
                advance(); /* 跳过 对于 */
                if (cur_tok.type != TOK_ID) { fprintf(stderr, "语法错误 (第%d行): 推导式需要迭代变量\n", cur_tok.line); exit(1); }
                Node *iter = make_node(ND_IDENT); iter->ident.name = strdup(cur_tok.text); advance();
                expect(TK_IN);
                Node *src = parse_expression();
                Node *filter = NULL;
                if (cur_tok.type == TK_IF) { advance(); filter = parse_expression(); }
                expect(TOK_RBRACK);
                n->list_lit.elements = malloc(4 * sizeof(Node*));
                n->list_lit.elements[0] = first;
                n->list_lit.elements[1] = iter;
                n->list_lit.elements[2] = src;
                n->list_lit.elements[3] = filter; /* 可能为 NULL */
                n->list_lit.ecnt = 4;
                n->list_lit.is_comprehension = 1;
                return n;
            }
            if (cnt >= cap) { cap = cap ? cap*2 : 4; n->list_lit.elements = realloc(n->list_lit.elements, cap * sizeof(Node*)); }
            n->list_lit.elements[cnt++] = first;
            while (cur_tok.type == TOK_COMMA && (advance(), 1)) {
                Node *elem = parse_expression();
                if (cnt >= cap) { cap = cap ? cap*2 : 4; n->list_lit.elements = realloc(n->list_lit.elements, cap * sizeof(Node*)); }
                n->list_lit.elements[cnt++] = elem;
            }
        }
        n->list_lit.ecnt = cnt;
        expect(TOK_RBRACK);
        return n;
    } else if (cur_tok.type == TOK_LBRACE) {
        Node *n = make_node(ND_MAP_LIT);
        advance();
        n->map_lit.keys = NULL;
        n->map_lit.vals = NULL;
        int cnt = 0, cap = 0;
        n->map_lit.is_comprehension = 0;
        if (cur_tok.type != TOK_RBRACE) {
            /* 解析第一个 key:val（用于判断是否为推导式）
               关键：裸标识符键有二义性——
                 普通字典 {苹果: 1}   → 键是字符串 "苹果"（语法糖）
                 推导式   {w: 长度(w) 对于 w,v 于 ...} → 键是变量 w 的值
               因此先按表达式解析，等看到（或看不到）"对于" 再决定语义。 */
            Node *k0, *v0;
            int k0_bare_id = 0;
            if (cur_tok.type == TOK_STRING || cur_tok.type == TOK_MULTILINE_STRING) {
                char buf[MAX_ID_LEN * 2];
                process_escape(buf, cur_tok.text, strlen(cur_tok.text));
                k0 = make_node(ND_LITERAL); k0->literal.lit_type = VAL_STRING;
                k0->literal.value = strdup(buf); advance();
            } else {
                k0 = parse_expression();
                k0_bare_id = (k0->type == ND_IDENT);   /* 单个裸标识符，语义待定 */
            }
            expect(TOK_COLON);
            v0 = parse_expression();
            if (cur_tok.type != TK_FOR && k0_bare_id) {
                /* 不是推导式 → 裸标识符键退化为字符串字面量（向后兼容语法糖） */
                Node *lit = make_node(ND_LITERAL);
                lit->literal.lit_type = VAL_STRING;
                lit->literal.value = strdup(k0->ident.name);
                node_free(k0);
                k0 = lit;
            }
            if (cur_tok.type == TK_FOR) {
                /* 字典推导式：{k:v 对于 kk, vv 于 src 若 filter} */
                advance(); /* 对于 */
                if (cur_tok.type != TOK_ID) { fprintf(stderr, "语法错误 (第%d行): 推导式需要迭代变量\n", cur_tok.line); exit(1); }
                Node *ik = make_node(ND_IDENT); ik->ident.name = strdup(cur_tok.text); advance();
                /* 第二个迭代变量可选：
                     {k: v 对于 键, 值 在 字典}  → 双变量，分别绑定键与值
                     {w: 长度(w) 对于 w 在 列表} → 单变量，直接绑定元素（字典源则绑定键） */
                Node *iv = NULL;
                if (cur_tok.type == TOK_COMMA) {
                    advance();
                    if (cur_tok.type != TOK_ID) { fprintf(stderr, "语法错误 (第%d行): 推导式需要第二个迭代变量\n", cur_tok.line); exit(1); }
                    iv = make_node(ND_IDENT); iv->ident.name = strdup(cur_tok.text); advance();
                }
                expect(TK_IN);
                Node *src = parse_expression();
                Node *filter = NULL;
                if (cur_tok.type == TK_IF) { advance(); filter = parse_expression(); }
                expect(TOK_RBRACE);
                n->map_lit.keys = malloc(4 * sizeof(Node*));
                n->map_lit.vals = malloc(4 * sizeof(Node*));
                n->map_lit.keys[0] = k0; n->map_lit.vals[0] = v0;
                n->map_lit.keys[1] = ik; n->map_lit.vals[1] = iv;
                n->map_lit.keys[2] = src; n->map_lit.vals[2] = src;
                n->map_lit.keys[3] = filter; n->map_lit.vals[3] = filter;
                n->map_lit.kcnt = 4;
                n->map_lit.is_comprehension = 1;
                return n;
            }
            if (cnt >= cap) { cap = cap ? cap*2 : 4;
                n->map_lit.keys = realloc(n->map_lit.keys, cap * sizeof(Node*));
                n->map_lit.vals = realloc(n->map_lit.vals, cap * sizeof(Node*)); }
            n->map_lit.keys[cnt] = k0; n->map_lit.vals[cnt] = v0; cnt++;
            while (cur_tok.type == TOK_COMMA && (advance(), 1)) {
                Node *key;
                if (cur_tok.type == TOK_STRING || cur_tok.type == TOK_MULTILINE_STRING) {
                    char buf[MAX_ID_LEN * 2];
                    process_escape(buf, cur_tok.text, strlen(cur_tok.text));
                    key = make_node(ND_LITERAL); key->literal.lit_type = VAL_STRING;
                    key->literal.value = strdup(buf); advance();
                } else if (cur_tok.type == TOK_ID) {
                    key = make_node(ND_LITERAL); key->literal.lit_type = VAL_STRING;
                    key->literal.value = strdup(cur_tok.text); advance();
                } else {
                    key = parse_expression();
                }
                expect(TOK_COLON);
                Node *val = parse_expression();
                if (cnt >= cap) { cap = cap ? cap*2 : 4;
                    n->map_lit.keys = realloc(n->map_lit.keys, cap * sizeof(Node*));
                    n->map_lit.vals = realloc(n->map_lit.vals, cap * sizeof(Node*)); }
                n->map_lit.keys[cnt] = key; n->map_lit.vals[cnt] = val; cnt++;
            }
        }
        n->map_lit.kcnt = cnt;
        expect(TOK_RBRACE);
        return n;
    } else {
        /* 多余的 结束 是最常见的手误，直接给出可操作的提示，而不是干巴巴报个符号 */
        if (cur_tok.type == TK_END) {
            fprintf(stderr, "语法错误 (第%d行,%d列): 多余的 '结束'"
                            "（单行的「如果 … 则 …」若要写 结束，必须写在同一行）\n",
                    cur_tok.line, cur_tok.col);
            exit(1);
        }
        fprintf(stderr, "语法错误 (第%d行,%d列): 意外的符号 '%s'\n", cur_tok.line, cur_tok.col, cur_tok.text);
        exit(1);
    }
}

static Node *parse_postfix() {
    Node *n = parse_primary();
    while (1) {
        if (cur_tok.type == TOK_LPAREN) {
            advance();
            Node *fc = make_node(ND_FUNC_CALL);
            fc->func_call.callee = n;
            fc->func_call.args = NULL; int cnt = 0, cap = 0;
            if (cur_tok.type != TOK_RPAREN) {
                do {
                    Node *arg = parse_expression();
                    if (cnt >= cap) { cap = cap ? cap*2 : 4; fc->func_call.args = realloc(fc->func_call.args, cap * sizeof(Node*)); }
                    fc->func_call.args[cnt++] = arg;
                } while (cur_tok.type == TOK_COMMA && (advance(), 1));
            }
            fc->func_call.acnt = cnt;
            expect(TOK_RPAREN);
            n = fc;
        } else if (cur_tok.type == TOK_LBRACK) {
            advance();
            Node *idx = parse_expression();
            expect(TOK_RBRACK);
            Node *acc = make_node(ND_BINARY);
            acc->binary.left = n;
            acc->binary.right = idx;
            strcpy(acc->binary.op, "[]");
            n = acc;
        } else break;
    }
    return n;
}

static Node *parse_unary() {
    /* 并发/等待 属于一元前缀运算符，优先级高于所有二元运算符。
       这样 `等待 甲 + 等待 乙` 才会解析成 (等待 甲) + (等待 乙)；
       若要并发整个表达式，请显式加括号：并发 (甲 + 乙)。 */
    if (cur_tok.type == TK_CONCUR) {
        advance();
        Node *n = make_node(ND_CONCUR);
        n->concur_stmt.expr = parse_unary();
        return n;
    }
    if (cur_tok.type == TK_WAIT) {
        advance();
        Node *n = make_node(ND_WAIT);
        n->wait_stmt.expr = parse_unary();
        return n;
    }
    if (cur_tok.type == TOK_MINUS || cur_tok.type == TOK_PLUS) {
        Token op = cur_tok;
        advance();
        Node *n = make_node(ND_UNARY);
        strcpy(n->unary.op, op.text);
        n->unary.operand = parse_unary();
        return n;
    }
    if (cur_tok.type == TK_NOT) {
        advance();
        Node *n = make_node(ND_UNARY);
        strcpy(n->unary.op, "not");
        n->unary.operand = parse_unary();
        return n;
    }
    if (cur_tok.type == TOK_TILDE) {   /* ~x / 按位取反 x */
        advance();
        Node *n = make_node(ND_UNARY);
        strcpy(n->unary.op, "~");
        n->unary.operand = parse_unary();
        return n;
    }
    return parse_postfix();
}

static Node *parse_mul() {
    Node *left = parse_unary();
    while (cur_tok.type == TOK_STAR || cur_tok.type == TOK_SLASH ||
           cur_tok.type == TOK_DBLSLASH || cur_tok.type == TOK_MOD) {
        Token op = cur_tok; advance();
        Node *right = parse_unary();
        Node *n = make_node(ND_BINARY);
        n->binary.left = left;
        n->binary.right = right;
        strcpy(n->binary.op, op.text);
        left = n;
    }
    return left;
}

static Node *parse_add() {
    Node *left = parse_mul();
    while (cur_tok.type == TOK_PLUS || cur_tok.type == TOK_MINUS) {
        Token op = cur_tok; advance();
        Node *right = parse_mul();
        Node *n = make_node(ND_BINARY);
        n->binary.left = left;
        n->binary.right = right;
        strcpy(n->binary.op, op.text);
        left = n;
    }
    return left;
}

/* 移位：优先级高于比较、低于加减（与 C 一致，避免 1 << 2 + 3 被误读） */
static Node *parse_shift() {
    Node *left = parse_add();
    while (cur_tok.type == TOK_SHL || cur_tok.type == TOK_SHR) {
        Token op = cur_tok; advance();
        Node *right = parse_add();
        Node *n = make_node(ND_BINARY);
        n->binary.left = left;
        n->binary.right = right;
        strcpy(n->binary.op, op.text);
        left = n;
    }
    return left;
}

static Node *parse_cmp() {
    Node *left = parse_shift();
    while (cur_tok.type == TOK_LT || cur_tok.type == TOK_LE || cur_tok.type == TOK_GT || cur_tok.type == TOK_GE) {
        Token op = cur_tok; advance();
        Node *right = parse_shift();
        Node *n = make_node(ND_BINARY);
        n->binary.left = left;
        n->binary.right = right;
        strcpy(n->binary.op, op.text);
        left = n;
    }
    return left;
}

static Node *parse_eq() {
    Node *left = parse_cmp();
    while (cur_tok.type == TOK_EQ || cur_tok.type == TOK_NE) {
        Token op = cur_tok; advance();
        Node *right = parse_cmp();
        Node *n = make_node(ND_BINARY);
        n->binary.left = left;
        n->binary.right = right;
        strcpy(n->binary.op, op.text);
        left = n;
    }
    return left;
}

/* 位运算三层：& 高于 ^ 高于 |，整体位于「相等比较」与「逻辑与」之间。
   与 C 完全一致的优先级，熟悉系统编程的人不必重新背表。 */
static Node *parse_bitand() {
    Node *left = parse_eq();
    while (cur_tok.type == TOK_AMP) {
        advance();
        Node *right = parse_eq();
        Node *n = make_node(ND_BINARY);
        n->binary.left = left;
        n->binary.right = right;
        strcpy(n->binary.op, "&");
        left = n;
    }
    return left;
}

static Node *parse_bitxor() {
    Node *left = parse_bitand();
    while (cur_tok.type == TOK_CARET) {
        advance();
        Node *right = parse_bitand();
        Node *n = make_node(ND_BINARY);
        n->binary.left = left;
        n->binary.right = right;
        strcpy(n->binary.op, "^");
        left = n;
    }
    return left;
}

static Node *parse_bitor() {
    Node *left = parse_bitxor();
    while (cur_tok.type == TOK_PIPE) {
        advance();
        Node *right = parse_bitxor();
        Node *n = make_node(ND_BINARY);
        n->binary.left = left;
        n->binary.right = right;
        strcpy(n->binary.op, "|");
        left = n;
    }
    return left;
}

static Node *parse_and() {
    Node *left = parse_bitor();
    while (cur_tok.type == TK_AND) {
        advance();
        Node *right = parse_bitor();
        Node *n = make_node(ND_BINARY);
        n->binary.left = left;
        n->binary.right = right;
        strcpy(n->binary.op, "且");
        left = n;
    }
    return left;
}

static Node *parse_or() {
    Node *left = parse_and();
    while (cur_tok.type == TK_OR) {
        advance();
        Node *right = parse_and();
        Node *n = make_node(ND_BINARY);
        n->binary.left = left;
        n->binary.right = right;
        strcpy(n->binary.op, "或");
        left = n;
    }
    return left;
}

static Node *parse_expression() {
    return parse_or();
}

/* 将类型 token 转换为类型标注字符串（用于类型系统） */
static const char *type_token_name(TokenType t) {
    switch (t) {
        case TK_TYPE_INT:    return "整数";
        case TK_TYPE_FLOAT:  return "浮点";
        case TK_TYPE_STRING: return "字符串";
        case TK_TYPE_BOOL:   return "布尔";
        case TK_TYPE_LIST:   return "列表";
        case TK_TYPE_MAP:    return "字典";
        case TK_TYPE_NULL:   return "空";
        case TK_TYPE_NUM:    return "数字";
        case TK_TYPE_FUNC:   return "函数";
        case TK_TYPE_ANY:    return "任意";
        case TK_TENSOR:      return "张量";
        default: return NULL;
    }
}

static Node *parse_var_decl() {
    int is_const = 0;
    /* 可选 让/常量 关键字 */
    if (cur_tok.type == TK_VAR || cur_tok.type == TK_CONST) {
        is_const = (strcmp(cur_tok.text, "常量") == 0);
        advance();
    }
    /* 可选类型标注：让 整数 x = 5  或  整数 x = 5 */
    const char *annot = NULL;
    const char *tn = type_token_name(cur_tok.type);
    /* type_token_name 返回静态常量，无需 strdup（否则与下方 strdup 重复分配而泄漏） */
    if (tn) { annot = tn; advance(); }
    if (cur_tok.type != TOK_ID) {
        fprintf(stderr, "语法错误 (第%d行,第%d列): 变量声明需要标识符, 得到 '%s'\n",
                cur_tok.line, cur_tok.col, cur_tok.text);
        exit(1);
    }
    char *name = strdup(cur_tok.text);
    advance();
    Node *init = NULL;
    if (cur_tok.type == TOK_ASSIGN) {
        advance();
        init = parse_expression();
    }
    Node *n = make_node(ND_VAR_DECL);
    n->var_decl.name = name;
    n->var_decl.init = init;
    n->var_decl.is_const = is_const;
    n->var_decl.type_annot = annot ? strdup(annot) : NULL;
    return n;
}

static Node *parse_print() {
    advance();
    Node **args = NULL; int cnt = 0, cap = 0;
    if (cur_tok.type != TOK_NEWLINE && cur_tok.type != TOK_EOF) {
        do {
            Node *arg = parse_expression();
            if (cnt >= cap) { cap = cap ? cap*2 : 4; args = realloc(args, cap * sizeof(Node*)); }
            args[cnt++] = arg;
        } while (cur_tok.type == TOK_COMMA && (advance(), 1));
    }
    Node *n = make_node(ND_FUNC_CALL);
    Node *id = make_node(ND_IDENT);
    id->ident.name = strdup("内置输出");
    n->func_call.callee = id;
    n->func_call.args = args;
    n->func_call.acnt = cnt;
    return n;
}

static Node *parse_if() {
    advance();
    Node *cond = parse_expression();
    /* 块形式由【换行】决定，而不是由 则 决定：
         若 n <= 1 返回 1          → 同行有语句，单行形式
         如果 x > 5 则 y = 1       → 同行有语句，单行形式
         如果 x > 5 [则] ⏎ ... 结束 → 换行，块形式
       这样多行分支不必强制写 则，且完全兼容旧写法。 */
    if (cur_tok.type == TK_THEN) advance();   /* 则 始终可选 */
    int block_form = (cur_tok.type == TOK_NEWLINE);

    Node **then_body = NULL; int tcnt = 0, tcap = 0;
    if (block_form) {
        while (cur_tok.type != TOK_EOF) {
            if (cur_tok.type == TOK_NEWLINE) { advance(); continue; }
            if (cur_tok.type == TK_ELSE || cur_tok.type == TK_ELIF || cur_tok.type == TK_END) break;
            Node *stmt = parse_statement();
            if (stmt) {
                if (tcnt >= tcap) { tcap = tcap ? tcap*2 : 4; then_body = realloc(then_body, tcap * sizeof(Node*)); }
                then_body[tcnt++] = stmt;
            }
        }
    } else {
        /* 单行 if：仅取紧跟的一条语句，且不消费外层 结束 */
        while (cur_tok.type == TOK_NEWLINE) advance();
        if (cur_tok.type != TK_ELSE && cur_tok.type != TK_ELIF && cur_tok.type != TK_END) {
            Node *stmt = parse_statement();
            if (stmt) { then_body = malloc(sizeof(Node*)); then_body[0] = stmt; tcnt = 1; }
        }
    }

    ElseBranch *branches = NULL; int br_cnt = 0, br_cap = 0;
    while (cur_tok.type == TK_ELIF || cur_tok.type == TK_ELSE) {
        ElseBranch br;
        if (cur_tok.type == TK_ELIF) {
            advance();
            br.cond = parse_expression();
            br.is_else = 0;
        } else {
            br.cond = NULL;
            br.is_else = 1;
            advance(); /* 跳过'否则' */
        }
        if (cur_tok.type == TK_THEN) advance();          /* 则 可选 */
        int br_block = (cur_tok.type == TOK_NEWLINE);    /* 与 如果 采用同一判定规则 */
        br.body = NULL; br.bcnt = 0;
        if (br_block) {
            while (cur_tok.type != TOK_EOF) {
                if (cur_tok.type == TOK_NEWLINE) { advance(); continue; }
                if (cur_tok.type == TK_ELSE || cur_tok.type == TK_ELIF || cur_tok.type == TK_END) break;
                Node *stmt = parse_statement();
                if (stmt) { br.body = realloc(br.body, (br.bcnt + 1) * sizeof(Node*)); br.body[br.bcnt++] = stmt; }
            }
        } else {
            while (cur_tok.type == TOK_NEWLINE) advance();
            if (cur_tok.type != TK_ELSE && cur_tok.type != TK_ELIF && cur_tok.type != TK_END) {
                Node *stmt = parse_statement();
                if (stmt) { br.body = malloc(sizeof(Node*)); br.body[0] = stmt; br.bcnt = 1; }
            }
        }
        if (br_cnt >= br_cap) { br_cap = br_cap ? br_cap*2 : 4; branches = realloc(branches, br_cap * sizeof(ElseBranch)); }
        branches[br_cnt++] = br;
    }

    /* 单行形式允许把 结束 写在同一行：如果 a 则 b 结束
       这是最自然的一行写法，可此前单行 if 不消费 结束，
       于是这个 结束 被外层的 当/函数/对于 抢走，报出「少一个 结束」的
       莫名其妙的语法错误 —— 典型的「写着对、解析错」。
       判定依据：parse_statement 不吞换行，所以此刻还能看到 TK_END
       就说明它与 如果 在同一行；跨行的 结束 属于外层块，绝不抢。 */
    if (block_form) expect(TK_END);
    else if (cur_tok.type == TK_END) advance();
    Node *n = make_node(ND_IF);
    n->if_stmt.cond = cond;
    n->if_stmt.then_body = then_body;
    n->if_stmt.tcnt = tcnt;
    n->if_stmt.else_branches = branches;
    n->if_stmt.br_cnt = br_cnt;
    return n;
}

static Node *parse_while() {
    advance();
    Node *cond = parse_expression();
    expect(TK_LOOP);
    Node **body = NULL; int bcnt = 0, bcap = 0;
    while (cur_tok.type != TK_END && cur_tok.type != TOK_EOF) {
        if (cur_tok.type == TOK_NEWLINE) { advance(); continue; }
        Node *stmt = parse_statement();
        if (stmt) {
            if (bcnt >= bcap) { bcap = bcap ? bcap*2 : 4; body = realloc(body, bcap * sizeof(Node*)); }
            body[bcnt++] = stmt;
        }
    }
    expect(TK_END);
    Node *n = make_node(ND_WHILE);
    n->while_stmt.cond = cond;
    n->while_stmt.body = body;
    n->while_stmt.bcnt = bcnt;
    return n;
}

/* 收集语句直到遇到给定的三个终止 token 之一（不消费终止符） */
static Node **parse_block_until(TokenType a, TokenType b, TokenType c, int *out_cnt) {
    Node **body = NULL; int bcnt = 0, bcap = 0;
    while (cur_tok.type != a && cur_tok.type != b && cur_tok.type != c &&
           cur_tok.type != TOK_EOF) {
        if (cur_tok.type == TOK_NEWLINE) { advance(); continue; }
        Node *stmt = parse_statement();
        if (stmt) {
            if (bcnt >= bcap) { bcap = bcap ? bcap*2 : 4; body = realloc(body, bcap * sizeof(Node*)); }
            body[bcnt++] = stmt;
        }
    }
    *out_cnt = bcnt;
    return body;
}

/* 尝试
       可能出错的语句
   捕获 错误            （变量名可省略）
       补救语句
   最终                  （可省略）
       无论如何都要执行的清理
   结束                                                     */
static Node *parse_try() {
    Node *n = make_node(ND_TRY);
    advance();   /* 吃掉 尝试 */
    n->try_stmt.body = parse_block_until(TK_CATCH, TK_FINALLY, TK_END, &n->try_stmt.bcnt);
    n->try_stmt.cbody = NULL; n->try_stmt.ccnt = 0;
    n->try_stmt.fbody = NULL; n->try_stmt.fcnt = 0;
    n->try_stmt.var = NULL;
    n->try_stmt.has_catch = 0; n->try_stmt.has_finally = 0;

    if (cur_tok.type == TK_CATCH) {
        advance();
        n->try_stmt.has_catch = 1;
        if (cur_tok.type == TOK_ID) { n->try_stmt.var = strdup(cur_tok.text); advance(); }
        n->try_stmt.cbody = parse_block_until(TK_FINALLY, TK_END, TK_END, &n->try_stmt.ccnt);
    }
    if (cur_tok.type == TK_FINALLY) {
        advance();
        n->try_stmt.has_finally = 1;
        n->try_stmt.fbody = parse_block_until(TK_END, TK_END, TK_END, &n->try_stmt.fcnt);
    }
    if (!n->try_stmt.has_catch && !n->try_stmt.has_finally) {
        fprintf(stderr, "语法错误 (第%d行,%d列): 尝试 块必须带 捕获 或 最终\n",
                n->line, n->col);
        exit(1);
    }
    expect(TK_END);
    return n;
}

static Node *parse_for() {
    advance();
    if (type_token_name(cur_tok.type)) advance(); /* 可选类型标注：对于 整数 i 从 ... */
    if (cur_tok.type != TOK_ID) {
        fprintf(stderr, "语法错误 (第%d行,第%d列): 循环变量需要标识符, 得到 '%s'\n",
                cur_tok.line, cur_tok.col, cur_tok.text);
        exit(1);
    }
    char *var = strdup(cur_tok.text);
    advance();
    expect(TK_FROM);
    Node *start = parse_expression();
    expect(TK_TO);
    Node *end = parse_expression();
    Node *step = NULL;
    if (cur_tok.type == TK_STEP) {
        advance();
        step = parse_expression();
    }
    expect(TK_LOOP);
    
    Node **body = NULL; int bcnt = 0, bcap = 0;
    while (cur_tok.type != TK_END && cur_tok.type != TOK_EOF) {
        if (cur_tok.type == TOK_NEWLINE) { advance(); continue; }
        Node *stmt = parse_statement();
        if (stmt) {
            if (bcnt >= bcap) { bcap = bcap ? bcap*2 : 4; body = realloc(body, bcap * sizeof(Node*)); }
            body[bcnt++] = stmt;
        }
    }
    expect(TK_END);
    
    Node *n = make_node(ND_FOR);
    n->for_stmt.var = var;
    n->for_stmt.start = start;
    n->for_stmt.end = end;
    n->for_stmt.step = step;
    n->for_stmt.body = body;
    n->for_stmt.bcnt = bcnt;
    return n;
}

static Node *parse_func_def() {
    advance();
    if (cur_tok.type != TOK_ID) {
        fprintf(stderr, "语法错误 (第%d行,第%d列): 函数声明需要标识符, 得到 '%s'\n",
                cur_tok.line, cur_tok.col, cur_tok.text);
        exit(1);
    }
    char *name = strdup(cur_tok.text);
    advance();
    expect(TOK_LPAREN);
    char **params = NULL, **ptypes = NULL; int pcnt = 0, pcap = 0;
    if (cur_tok.type != TOK_RPAREN) {
        do {
            const char *pt = NULL;
            const char *tn = type_token_name(cur_tok.type);
            if (tn) { pt = tn; advance(); }   /* 静态常量，见 parse_var_decl 同样说明 */
            if (cur_tok.type != TOK_ID) {
                fprintf(stderr, "语法错误 (第%d行,第%d列): 参数需要标识符, 得到 '%s'\n",
                        cur_tok.line, cur_tok.col, cur_tok.text);
                exit(1);
            }
            if (pcnt >= pcap) { pcap = pcap ? pcap*2 : 4; params = realloc(params, pcap * sizeof(char*)); ptypes = realloc(ptypes, pcap * sizeof(char*)); }
            params[pcnt] = strdup(cur_tok.text);
            ptypes[pcnt] = pt ? strdup(pt) : NULL;
            pcnt++;
            advance();
        } while (cur_tok.type == TOK_COMMA && (advance(), 1));
    }
    expect(TOK_RPAREN);
    /* 可选返回类型：函数 f(...) -> 整数 { ... } */
    char *ret_type = NULL;
    if (cur_tok.type == TK_ARROW) {
        advance();
        const char *tn = type_token_name(cur_tok.type);
        if (!tn) {
            fprintf(stderr, "语法错误 (第%d行,第%d列): 返回类型标注无效, 得到 '%s'\n",
                    cur_tok.line, cur_tok.col, cur_tok.text);
            exit(1);
        }
        ret_type = strdup(tn);
        advance();
    }
    Node *body = make_node(ND_PROGRAM);
    body->prog.stmts = NULL; int bcnt = 0, bcap = 0;
    while (cur_tok.type != TK_END && cur_tok.type != TOK_EOF) {
        if (cur_tok.type == TOK_NEWLINE) { advance(); continue; }
        Node *stmt = parse_statement();
        if (stmt) {
            if (bcnt >= bcap) { bcap = bcap ? bcap*2 : 4; body->prog.stmts = realloc(body->prog.stmts, bcap * sizeof(Node*)); }
            body->prog.stmts[bcnt++] = stmt;
        }
    }
    body->prog.cnt = bcnt;
    expect(TK_END);
    Node *n = make_node(ND_FUNC_DEF);
    n->func_def.name = name;
    n->func_def.pnames = params;
    n->func_def.ptypes = ptypes;
    n->func_def.pcnt = pcnt;
    n->func_def.ret_type = ret_type;
    n->func_def.body = body;
    return n;
}

/* 模块导入：导入 "路径.co" */
static Node *parse_import() {
    advance();
    char *path = NULL;
    if (cur_tok.type == TOK_STRING || cur_tok.type == TOK_MULTILINE_STRING) {
        path = strdup(cur_tok.text);
        advance();
    } else {
        fprintf(stderr, "语法错误 (第%d行,第%d列): 导入需要字符串路径, 得到 '%s'\n",
                cur_tok.line, cur_tok.col, cur_tok.text);
        exit(1);
    }
    Node *n = make_node(ND_IMPORT);
    n->import_stmt.path = path;
    return n;
}

/* 模式匹配：匹配 expr { 是 模式: ...; 默认: ... } */
static Node *parse_match() {
    advance();
    Node *expr = parse_expression();
    MatchArm *arms = NULL; int arm_cnt = 0, arm_cap = 0;
    /* 支持大括号包裹或缩进块；这里采用与 if/循环一致的换行块，
       形如：匹配 x 是 1: ... 是 2: ... 默认: ... 结束 */
    while (cur_tok.type != TK_END && cur_tok.type != TOK_EOF) {
        if (cur_tok.type == TOK_NEWLINE) { advance(); continue; }
        if (cur_tok.type != TK_IS && cur_tok.type != TK_DEFAULT) break;
        MatchArm arm;
        int is_default = (cur_tok.type == TK_DEFAULT);
        advance();
        arm.is_default = is_default;
        arm.pattern = is_default ? NULL : parse_expression();
        if (cur_tok.type == TOK_COLON) advance();
        arm.body = NULL; arm.bcnt = 0;
        /* 读取该分支语句，直到下一个 是/默认/结束 */
        while (cur_tok.type != TK_END && cur_tok.type != TOK_EOF
               && cur_tok.type != TK_IS && cur_tok.type != TK_DEFAULT) {
            if (cur_tok.type == TOK_NEWLINE) { advance(); continue; }
            Node *stmt = parse_statement();
            if (stmt) {
                arm.body = realloc(arm.body, (arm.bcnt + 1) * sizeof(Node*));
                arm.body[arm.bcnt++] = stmt;
            }
        }
        if (arm_cnt >= arm_cap) { arm_cap = arm_cap ? arm_cap*2 : 4; arms = realloc(arms, arm_cap * sizeof(MatchArm)); }
        arms[arm_cnt++] = arm;
    }
    expect(TK_END);
    Node *n = make_node(ND_MATCH);
    n->match_stmt.expr = expr;
    n->match_stmt.arms = arms;
    n->match_stmt.arm_cnt = arm_cnt;
    return n;
}

static Node *parse_statement() {
    if (cur_tok.type == TOK_NEWLINE) { advance(); return NULL; }
    if (cur_tok.type == TK_VAR || cur_tok.type == TK_CONST || type_token_name(cur_tok.type)) return parse_var_decl();
    if (cur_tok.type == TK_PRINT) return parse_print();
    if (cur_tok.type == TK_IF) return parse_if();
    if (cur_tok.type == TK_WHILE) return parse_while();
    if (cur_tok.type == TK_FOR) return parse_for();
    if (cur_tok.type == TK_FUNC) return parse_func_def();
    if (cur_tok.type == TK_IMPORT) return parse_import();
    if (cur_tok.type == TK_MATCH) return parse_match();
    if (cur_tok.type == TK_TRY) return parse_try();
    if (cur_tok.type == TK_THROW) {
        Node *n = make_node(ND_THROW);
        advance();
        n->throw_stmt.expr = parse_expression();
        return n;
    }
    if (cur_tok.type == TK_RETURN) {
        advance();
        Node *n = make_node(ND_RETURN);
        if (cur_tok.type != TOK_NEWLINE && cur_tok.type != TOK_EOF)
            n->ret_stmt.val = parse_expression();
        return n;
    }
    if (cur_tok.type == TK_BREAK) { advance(); return make_node(ND_BREAK); }
    if (cur_tok.type == TK_CONTINUE) { advance(); return make_node(ND_CONTINUE); }
    Node *expr = parse_expression();
    if (cur_tok.type == TOK_ASSIGN) {
        advance();
        Node *val = parse_expression();
        Node *n = make_node(ND_ASSIGN);
        n->assign.tgt = expr;
        n->assign.val = val;
        return n;
    }
    return expr;
}

static Node *parse_program() {
    Node *prog = make_node(ND_PROGRAM);
    prog->prog.stmts = NULL; int cnt = 0, cap = 0;
    while (cur_tok.type != TOK_EOF) {
        Node *stmt = parse_statement();
        if (stmt) {
            if (cnt >= cap) { cap = cap ? cap*2 : 8; prog->prog.stmts = realloc(prog->prog.stmts, cap * sizeof(Node*)); }
            prog->prog.stmts[cnt++] = stmt;
        }
    }
    prog->prog.cnt = cnt;
    return prog;
}

/* ========== 异常抛出 ==========
   错误载荷统一是一个字典 {类型, 消息, 行, 列}，
   于是 捕获 块里不论捕到哪种错误，都能用 错误["消息"] 拿到说明。 */
static Value *make_error_value(const char *kind, const char *msg, int line, int col) {
    Value *e = val_new(VAL_MAP);
    Value *k = val_new(VAL_STRING); k->sval = strdup(kind);
    Value *m = val_new(VAL_STRING); m->sval = strdup(msg);
    Value *l = val_int(line);
    Value *c = val_int(col);
    dict_set(e->mval, "类型", k);
    dict_set(e->mval, "消息", m);
    dict_set(e->mval, "行",   l);
    dict_set(e->mval, "列",   c);
    val_free(k); val_free(m); val_free(l); val_free(c);
    return e;
}

/* 取错误字典里的字段（缺失时给出兜底），用于无人捕获时打印诊断 */
static const char *err_field_str(Value *e, const char *key, const char *dflt) {
    if (!e || e->type != VAL_MAP) return dflt;
    Value *v = dict_get(e->mval, key);
    return (v && v->type == VAL_STRING && v->sval) ? v->sval : dflt;
}
static long err_field_int(Value *e, const char *key) {
    if (!e || e->type != VAL_MAP) return 0;
    Value *v = dict_get(e->mval, key);
    return (v && v->type == VAL_INT) ? v->ival : 0;
}

/* 无人捕获：打印诊断并退出（保持与原来完全一致的错误输出格式） */
static void die_uncaught(Value *err) {
    const char *kind = err_field_str(err, "类型", "运行时错误");
    const char *msg  = err_field_str(err, "消息", "未知错误");
    long line = err_field_int(err, "行"), col = err_field_int(err, "列");
    if (line > 0) fprintf(stderr, "%s (第%ld行,%ld列): %s\n", kind, line, col, msg);
    else          fprintf(stderr, "%s: %s\n", kind, msg);
    exit(1);
}

/* 跳转到最近的 尝试 帧：先保住载荷，再回滚该帧期间的临时引用与作用域 */
static void throw_to_frame(Value *err) {
    reflog_protect(err);          /* 载荷及其子值免于回滚 */
    g_exc_value = err;
    g_try_depth--;
    TryFrame *f = &g_try[g_try_depth];
    f->recording = 0;
    reflog_rollback(f);
    longjmp(f->jb, 1);
}

/* ========== 解释器 ========== */
/* 运行时错误：若身处 尝试 块内则转为可捕获的异常，否则按老规矩致命退出。
   如此一来全部 200 余处 runtime_error 调用点无需任何改动，
   语言里每一种运行时错误都自动变得可以被 捕获。 */
static void runtime_error(Node *n, const char *msg, ...) {
    char buf[1024];
    va_list args;
    va_start(args, msg);
    vsnprintf(buf, sizeof(buf), msg, args);
    va_end(args);

    if (g_try_depth > 0) {
        throw_to_frame(make_error_value("运行时错误", buf,
                                        n ? n->line : 0, n ? n->col : 0));
    }
    if (n) fprintf(stderr, "运行时错误 (第%d行,%d列): %s\n", n->line, n->col, buf);
    else   fprintf(stderr, "运行时错误: %s\n", buf);
    exit(1);
}

static Value *eval_node(Node *n, Env *env);
static void exec_node(Node *n, Env *env);

/* 控制流标志：用于实现返回/跳出/继续（线程局部，见上方说明） */
static __thread int g_return_flag = 0;
static __thread Value *g_return_value = NULL;

/* 内置函数 */
static Value *builtin_len(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "长度函数需要1个参数");
    Value *v = argv[0];
    Value *r = val_new(VAL_INT);
    switch (v->type) {
        case VAL_STRING: r->ival = utf8_strlen(v->sval); break;   /* 按字符计数，非字节 */
        case VAL_LIST: r->ival = v->list_len; break;
        case VAL_MAP: {
            int cnt = 0;
            for (int i = 0; i < MAX_DICT_SIZE; i++) if (v->mval->entries[i].used) cnt++;
            r->ival = cnt;
            break;
        }
        default: r->ival = 1; break;
    }
    return r;
}

static Value *builtin_type(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "类型函数需要1个参数");
    Value *v = argv[0];
    Value *r = val_new(VAL_STRING);
    switch (v->type) {
        case VAL_NULL: r->sval = strdup("空"); break;
        case VAL_BOOL: r->sval = strdup("布尔"); break;
        case VAL_INT: r->sval = strdup("整数"); break;
        /* 与类型标注关键字保持一致（浮点，而非浮点数），
           这样 类型(x) == "浮点" 与 浮点 x = ... 同名同义 */
        case VAL_FLOAT: r->sval = strdup("浮点"); break;
        case VAL_STRING: r->sval = strdup("字符串"); break;
        case VAL_LIST: r->sval = strdup("列表"); break;
        case VAL_MAP: r->sval = strdup("字典"); break;
        case VAL_FUNC: r->sval = strdup("函数"); break;
        case VAL_TENSOR: r->sval = strdup("张量"); break;
        case VAL_TASK: r->sval = strdup("任务"); break;
        default: r->sval = strdup("未知"); break;
    }
    return r;
}

static Value *builtin_int(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "转整数函数需要1个参数");
    Value *v = argv[0], *r = val_new(VAL_INT);
    switch (v->type) {
        case VAL_INT: r->ival = v->ival; break;
        case VAL_BOOL: r->ival = v->ival ? 1 : 0; break;
        case VAL_FLOAT: r->ival = (long)v->fval; break;
        case VAL_STRING: r->ival = atol(v->sval); break;
        default: r->ival = 0; break;
    }
    return r;
}

static Value *builtin_float(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "转浮点函数需要1个参数");
    Value *v = argv[0], *r = val_new(VAL_FLOAT);
    switch (v->type) {
        case VAL_INT: r->fval = v->ival; break;
        case VAL_BOOL: r->fval = v->ival ? 1.0 : 0.0; break;
        case VAL_FLOAT: r->fval = v->fval; break;
        case VAL_STRING: r->fval = atof(v->sval); break;
        default: r->fval = 0.0; break;
    }
    return r;
}

/* 转布尔(值)：按统一真值规则显式转换 */
static Value *builtin_to_bool(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "转布尔函数需要1个参数");
    return val_bool(truthy(argv[0]));
}

static Value *builtin_str(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "转字符串函数需要1个参数");
    Value *r = val_new(VAL_STRING);
    r->sval = val_to_string(argv[0]);   /* 真实转换：整数/浮点/列表/字典/张量均可 */
    return r;
}

static Value *builtin_input(int argc, Value **argv) {
    if (argc > 1) runtime_error(NULL, "输入函数最多需要1个参数");
    if (argc == 1 && argv[0]->type == VAL_STRING) {
        printf("%s", argv[0]->sval);
    }
    Value *r = val_new(VAL_STRING);
    char buf[1024];
    if (fgets(buf, sizeof(buf), stdin)) {
        size_t len = strlen(buf);
        if (len > 0 && buf[len-1] == '\n') buf[len-1] = '\0';
        r->sval = strdup(buf);
    } else {
        r->sval = strdup("");
    }
    return r;
}

/* 数学函数 */
static Value *builtin_abs(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "绝对值函数需要1个参数");
    Value *v = argv[0];
    if (v->type == VAL_INT) {
        Value *r = val_new(VAL_INT);
        r->ival = v->ival >= 0 ? v->ival : -v->ival;
        return r;
    } else if (v->type == VAL_FLOAT) {
        Value *r = val_new(VAL_FLOAT);
        r->fval = v->fval >= 0 ? v->fval : -v->fval;
        return r;
    }
    runtime_error(NULL, "绝对值函数需要数字参数");
    return NULL;
}

static Value *builtin_sqrt(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "平方根函数需要1个参数");
    Value *v = argv[0];
    Value *r = val_new(VAL_FLOAT);
    double x = num_of(v, "平方根函数");
    if (x < 0) runtime_error(NULL, "平方根函数不接受负数：%g", x);
    r->fval = sqrt(x);
    return r;
}

static Value *builtin_random(int argc, Value **argv) {
    (void)argc; (void)argv;
    Value *r = val_new(VAL_FLOAT);
    r->fval = (double)rand() / RAND_MAX;
    return r;
}

/* 墙钟时间（秒，double），用于性能基准 */
static Value *builtin_now(int argc, Value **argv) {
    (void)argc; (void)argv;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    Value *r = val_new(VAL_FLOAT);
    r->fval = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
    return r;
}

/* 数值取出（整数/浮点/布尔统一为 double），非数字报错 */
static double num_of(Value *v, const char *who) {
    if (!v) runtime_error(NULL, "%s 收到空值", who);
    if (v->type == VAL_INT)   return (double)v->ival;
    if (v->type == VAL_FLOAT) return v->fval;
    if (v->type == VAL_BOOL)  return v->ival ? 1.0 : 0.0;  /* 真=1 假=0，允许参与算术 */
    runtime_error(NULL, "%s 需要数字参数，收到 %s", who, val_type_name(v));
    return 0;
}

/* 取整数（向零截断）。用于下标、次数、维度之外的整数场合。 */
static long int_of(Value *v, const char *who) {
    double d = num_of(v, who);
    if (d != d) runtime_error(NULL, "%s 收到非法数字 (NaN)", who);
    if (d > 9.2e18 || d < -9.2e18) runtime_error(NULL, "%s 的整数值超出范围", who);
    return (long)d;
}

/* 取「维度」：必须是正整数且不得大到能撑爆内存。
   曾经这里直接 (int)v->ival，张量([0]) 会 calloc(0)、张量([-3]) 会
   calloc 一个巨大无符号数，属于典型的内存安全漏洞。 */
#define CO_MAX_DIM_SIZE 100000000L   /* 单维长度上限 1 亿，防止形状溢出 */
static int dim_of(Value *v, const char *who) {
    long d = int_of(v, who);
    if (d <= 0)          runtime_error(NULL, "%s 的维度必须是正整数，收到 %ld", who, d);
    if (d > CO_MAX_DIM_SIZE) runtime_error(NULL, "%s 的维度过大：%ld（上限 %ld）", who, d, CO_MAX_DIM_SIZE);
    return (int)d;
}

/* 真值判断（唯一权威实现，所有条件/逻辑运算都必须走这里）：
   假值 = 假 / 0 / 0.0 / 空 / "" / 空列表 / 空字典；其余为真。 */
static int truthy(Value *v) {
    if (!v) return 0;
    switch (v->type) {
        case VAL_BOOL:   return v->ival != 0;
        case VAL_INT:    return v->ival != 0;
        case VAL_FLOAT:  return v->fval != 0;
        case VAL_STRING: return v->sval && v->sval[0];
        case VAL_LIST:   return v->list_len > 0;
        case VAL_MAP:    return v->mval && v->mval->count > 0;
        case VAL_NULL:   return 0;
        default:         return 1;
    }
}
static Value *val_int(long x)   { Value *r = val_new(VAL_INT);   r->ival = x; return r; }
static Value *val_flt(double x) { Value *r = val_new(VAL_FLOAT); r->fval = x; return r; }
static Value *val_bool(int b)   { Value *r = val_new(VAL_BOOL);  r->ival = b ? 1 : 0; return r; }

/* 取整 = 四舍五入（正确处理负数：-3.7 -> -4） */
static Value *builtin_round(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "取整函数需要1个参数");
    if (argv[0]->type == VAL_INT) return val_int(argv[0]->ival);
    return val_int((long)llround(num_of(argv[0], "取整")));
}
static Value *builtin_floor(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "向下取整函数需要1个参数");
    if (argv[0]->type == VAL_INT) return val_int(argv[0]->ival);
    return val_int((long)floor(num_of(argv[0], "向下取整")));
}
static Value *builtin_ceil(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "向上取整函数需要1个参数");
    if (argv[0]->type == VAL_INT) return val_int(argv[0]->ival);
    return val_int((long)ceil(num_of(argv[0], "向上取整")));
}
static Value *builtin_trunc(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "截断函数需要1个参数");
    if (argv[0]->type == VAL_INT) return val_int(argv[0]->ival);
    return val_int((long)trunc(num_of(argv[0], "截断")));
}

/* 最大值/最小值：支持多参数，也支持单个列表；全整数则返回整数 */
static Value *builtin_minmax(int argc, Value **argv, int want_max) {
    const char *who = want_max ? "最大值" : "最小值";
    Value **items = argv;
    int n = argc;
    if (argc == 1 && argv[0]->type == VAL_LIST) {
        items = argv[0]->lval;
        n = argv[0]->list_len;
    }
    if (n < 1) runtime_error(NULL, "%s函数需要至少1个数字或一个非空列表", who);
    int all_int = 1;
    double m = num_of(items[0], who);
    if (items[0]->type != VAL_INT) all_int = 0;
    for (int i = 1; i < n; i++) {
        double v = num_of(items[i], who);
        if (items[i]->type != VAL_INT) all_int = 0;
        if (want_max ? (v > m) : (v < m)) m = v;
    }
    return all_int ? val_int((long)m) : val_flt(m);
}
static Value *builtin_max(int argc, Value **argv) { return builtin_minmax(argc, argv, 1); }
static Value *builtin_min(int argc, Value **argv) { return builtin_minmax(argc, argv, 0); }

/* 幂、三角、对数、常量、符号 */
static Value *builtin_pow(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "幂函数需要2个参数");
    double b = num_of(argv[0], "幂"), e = num_of(argv[1], "幂");
    double r = pow(b, e);
    /* 整数底 + 非负整数指数 且结果可精确表示 -> 返回整数 */
    if (argv[0]->type == VAL_INT && argv[1]->type == VAL_INT && argv[1]->ival >= 0 &&
        fabs(r) < 9.0e15) return val_int((long)llround(r));
    return val_flt(r);
}
static Value *builtin_sin(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "正弦函数需要1个参数");
    return val_flt(sin(num_of(argv[0], "正弦")));
}
static Value *builtin_cos(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "余弦函数需要1个参数");
    return val_flt(cos(num_of(argv[0], "余弦")));
}
static Value *builtin_tan(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "正切函数需要1个参数");
    return val_flt(tan(num_of(argv[0], "正切")));
}
static Value *builtin_ln(int argc, Value **argv) {
    if (argc < 1 || argc > 2) runtime_error(NULL, "自然对数函数需要1-2个参数");
    double x = num_of(argv[0], "自然对数");
    if (x <= 0) runtime_error(NULL, "自然对数的参数必须大于 0");
    if (argc == 2) {
        double b = num_of(argv[1], "自然对数");
        if (b <= 0 || fabs(b - 1.0) < 1e-12) runtime_error(NULL, "对数的底必须大于 0 且不等于 1");
        return val_flt(log(x) / log(b));
    }
    return val_flt(log(x));
}
static Value *builtin_pi(int argc, Value **argv) {
    (void)argc; (void)argv;
    return val_flt(3.14159265358979323846);
}
static Value *builtin_sign(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "符号函数需要1个参数");
    double x = num_of(argv[0], "符号");
    return val_int(x > 0 ? 1 : (x < 0 ? -1 : 0));
}

/* ===== 进制与位工具：系统编程的日常刚需 =====
   统一按 64 位补码位模式呈现，负数不再打印成「-0x...」这种没法直接对照寄存器的形式。
   可选第 2 参数为最小宽度（左侧补 0），便于对齐输出寄存器/掩码。 */
static Value *bit_format(int argc, Value **argv, int base, const char *who, const char *prefix) {
    if (argc < 1 || argc > 2) runtime_error(NULL, "%s函数需要1或2个参数（数值[, 最小宽度]）", who);
    if (!(argv[0]->type == VAL_INT || argv[0]->type == VAL_BOOL))
        runtime_error(NULL, "%s函数需要整数，收到 %s", who, val_type_name(argv[0]));
    unsigned long long u = (unsigned long long)argv[0]->ival;
    int width = 0;
    if (argc == 2) {
        long w = int_of(argv[1], who);
        if (w < 0 || w > 64) runtime_error(NULL, "%s函数的最小宽度须在 0~64 之间，收到 %ld", who, w);
        width = (int)w;
    }
    char digits[65]; int dn = 0;
    const char *tbl = "0123456789abcdef";
    if (u == 0) digits[dn++] = '0';
    while (u) { digits[dn++] = tbl[u % (unsigned)base]; u /= (unsigned)base; }
    while (dn < width) digits[dn++] = '0';
    char out[80]; int on = 0;
    for (const char *p = prefix; *p; p++) out[on++] = *p;
    for (int i = dn - 1; i >= 0; i--) out[on++] = digits[i];
    out[on] = 0;
    Value *r = val_new(VAL_STRING);
    r->sval = strdup(out);
    return r;
}
static Value *builtin_hex(int argc, Value **argv) { return bit_format(argc, argv, 16, "十六进制", "0x"); }
static Value *builtin_bin(int argc, Value **argv) { return bit_format(argc, argv,  2, "二进制",   "0b"); }
static Value *builtin_oct(int argc, Value **argv) { return bit_format(argc, argv,  8, "八进制",   "0o"); }

/* 位计数：统计 64 位补码里为 1 的位数（popcount） */
static Value *builtin_popcount(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "位计数函数需要1个参数");
    if (!(argv[0]->type == VAL_INT || argv[0]->type == VAL_BOOL))
        runtime_error(NULL, "位计数函数需要整数，收到 %s", val_type_name(argv[0]));
    unsigned long long u = (unsigned long long)argv[0]->ival;
    int c = 0;
    while (u) { u &= u - 1; c++; }
    return val_int(c);
}

/* 取位：取位(值, 第几位) → 0/1，位号从 0（最低位）开始 */
static Value *builtin_getbit(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "取位函数需要2个参数（值, 位号）");
    if (!(argv[0]->type == VAL_INT || argv[0]->type == VAL_BOOL))
        runtime_error(NULL, "取位函数需要整数，收到 %s", val_type_name(argv[0]));
    long b = int_of(argv[1], "取位");
    if (b < 0 || b > 63) runtime_error(NULL, "位号须在 0~63 之间，收到 %ld", b);
    unsigned long long u = (unsigned long long)argv[0]->ival;
    return val_int((long)((u >> b) & 1ULL));
}

/* 置位：置位(值, 位号, 0或1) → 新值（不改原值） */
static Value *builtin_setbit(int argc, Value **argv) {
    if (argc != 3) runtime_error(NULL, "置位函数需要3个参数（值, 位号, 0或1）");
    if (!(argv[0]->type == VAL_INT || argv[0]->type == VAL_BOOL))
        runtime_error(NULL, "置位函数需要整数，收到 %s", val_type_name(argv[0]));
    long b = int_of(argv[1], "置位");
    if (b < 0 || b > 63) runtime_error(NULL, "位号须在 0~63 之间，收到 %ld", b);
    long bit = int_of(argv[2], "置位");
    if (bit != 0 && bit != 1) runtime_error(NULL, "置位的目标值只能是 0 或 1，收到 %ld", bit);
    unsigned long long u = (unsigned long long)argv[0]->ival;
    u = bit ? (u | (1ULL << b)) : (u & ~(1ULL << b));
    return val_int((long)u);
}
/* 随机整数：随机整数(a, b) 返回 [a, b] 闭区间内均匀整数 */
static Value *builtin_rand_int(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "随机整数函数需要2个参数");
    long a = (long)num_of(argv[0], "随机整数"), b = (long)num_of(argv[1], "随机整数");
    if (a > b) { long t = a; a = b; b = t; }
    long span = b - a + 1;
    return val_int(a + (long)((double)rand() / ((double)RAND_MAX + 1.0) * (double)span));
}

/* 列表函数 */
static Value *builtin_append(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "追加函数需要2个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "追加函数的第一个参数必须是列表");
    Value *list = argv[0];
    Value *item = argv[1];
    Value **new_list = realloc(list->lval, (list->list_len + 1) * sizeof(Value*));
    list->lval = new_list;
    list->lval[list->list_len] = val_retain_root(item);
    list->list_len++;
    return val_retain(list);   /* 返回列表本身：既保留原地语义，又支持链式调用 */
}

static Value *builtin_remove(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "删除函数需要2个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "删除函数的第一个参数必须是列表");
    Value *list = argv[0];
    long i0 = (long)num_of(argv[1], "删除"), idx = i0;
    if (idx < 0) idx += list->list_len;   /* 负索引：-1 表示最后一个，与索引/切片/插入一致 */
    if (idx < 0 || idx >= list->list_len)
        runtime_error(NULL, "列表索引越界: %ld（列表长度 %d）", i0, list->list_len);
    val_free(list->lval[idx]);
    for (long i = idx; i < list->list_len - 1; i++) {
        list->lval[i] = list->lval[i + 1];
    }
    list->list_len--;
    return val_retain(list);
}

/* 通用值比较：数字按数值、字符串按字典序、其余按类型序，返回 <0 / 0 / >0 */
/* 是否可当作数值参与比较（布尔按 0/1 计入，便于 真 == 1 这类互操作） */
static int is_numeric_val(Value *v) {
    return v->type == VAL_INT || v->type == VAL_FLOAT || v->type == VAL_BOOL;
}
static double numeric_of(Value *v) {
    if (v->type == VAL_FLOAT) return v->fval;
    return (double)v->ival;   /* VAL_INT / VAL_BOOL 共用 ival */
}

static int val_cmp(Value *a, Value *b) {
    if (is_numeric_val(a) && is_numeric_val(b)) {
        double x = numeric_of(a), y = numeric_of(b);
        return (x < y) ? -1 : ((x > y) ? 1 : 0);
    }
    if (a->type == VAL_STRING && b->type == VAL_STRING)
        return strcmp(a->sval ? a->sval : "", b->sval ? b->sval : "");
    /* 同为列表：逐元素字典序比较，长度短者在前（前缀关系） */
    if (a->type == VAL_LIST && b->type == VAL_LIST) {
        int n = a->list_len < b->list_len ? a->list_len : b->list_len;
        for (int i = 0; i < n; i++) {
            int c = val_cmp(a->lval[i], b->lval[i]);
            if (c) return c;
        }
        return (a->list_len < b->list_len) ? -1 : ((a->list_len > b->list_len) ? 1 : 0);
    }
    /* 类型不同：按枚举序稳定排列，保证排序是全序，不会读错联合体成员 */
    return (a->type < b->type) ? -1 : ((a->type > b->type) ? 1 : 0);
}

/* 值相等判断（整数/浮点/布尔跨类型可比；容器按结构递归比较） */
static int val_equals(Value *a, Value *b) {
    if (!a || !b) return a == b;
    if (is_numeric_val(a) && is_numeric_val(b))
        return fabs(numeric_of(a) - numeric_of(b)) < 1e-12;
    if (a->type != b->type) return 0;
    switch (a->type) {
        case VAL_NULL:   return 1;
        case VAL_STRING: return strcmp(a->sval ? a->sval : "", b->sval ? b->sval : "") == 0;
        case VAL_LIST:
            if (a->list_len != b->list_len) return 0;
            for (int i = 0; i < a->list_len; i++)
                if (!val_equals(a->lval[i], b->lval[i])) return 0;
            return 1;
        case VAL_MAP: {
            /* 字典相等：键集合与对应值都相等（键顺序无关） */
            if (a->mval->count != b->mval->count) return 0;
            for (int i = 0; i < MAX_DICT_SIZE; i++) {
                if (!a->mval->entries[i].used) continue;
                Value *bv = dict_get(b->mval, a->mval->entries[i].key);
                if (!bv || !val_equals(a->mval->entries[i].val, bv)) return 0;
            }
            return 1;
        }
        case VAL_TENSOR: {
            Tensor *x = a->tval, *y = b->tval;
            if (x == y) return 1;
            if (!x || !y || x->size != y->size || x->ndim != y->ndim) return 0;
            for (int i = 0; i < x->ndim; i++) if (x->shape[i] != y->shape[i]) return 0;
            for (int i = 0; i < x->size; i++) if (fabsf(x->data[i] - y->data[i]) > 1e-6f) return 0;
            return 1;
        }
        default: return a == b;
    }
}

/* 归并排序：稳定 O(n log n)，取代原本只能比数字的冒泡 */
static void val_msort(Value **arr, Value **tmp, int lo, int hi, int desc) {
    if (hi - lo < 2) return;
    int mid = lo + (hi - lo) / 2;
    val_msort(arr, tmp, lo, mid, desc);
    val_msort(arr, tmp, mid, hi, desc);
    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        int c = val_cmp(arr[i], arr[j]);
        if (desc) c = -c;
        tmp[k++] = (c <= 0) ? arr[i++] : arr[j++];   /* <= 保证稳定 */
    }
    while (i < mid) tmp[k++] = arr[i++];
    while (j < hi)  tmp[k++] = arr[j++];
    for (int t = lo; t < hi; t++) arr[t] = tmp[t];
}

static Value *builtin_sort(int argc, Value **argv) {
    if (argc < 1 || argc > 2) runtime_error(NULL, "排序函数需要1-2个参数（列表[, 是否降序]）");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "排序函数的第一个参数必须是列表");
    Value *list = argv[0];
    int desc = 0;
    if (argc == 2) desc = truthy(argv[1]);
    if (list->list_len > 1) {
        /* 比较过程可能抛错（如列表里混有不可比类型），必须用异常安全缓冲 */
        Value **tmp = co_tmp_alloc((size_t)list->list_len * sizeof(Value*));
        val_msort(list->lval, tmp, 0, list->list_len, desc);
        co_tmp_free(tmp);
    }
    return val_retain(list);
}

/* ---------- 新增列表函数 ---------- */
static Value *list_new_empty(void) {
    Value *r = val_new(VAL_LIST);
    r->list_len = 0;
    r->lval = malloc(sizeof(Value*));
    if (!r->lval) runtime_error(NULL, "创建列表时内存不足");
    return r;
}
static void list_push(Value *list, Value *item /* 借用，内部 retain */) {
    Value **nl = realloc(list->lval, (list->list_len + 1) * sizeof(Value*));
    if (!nl) runtime_error(NULL, "列表扩容时内存不足");
    list->lval = nl;
    list->lval[list->list_len++] = val_retain_root(item);
}

/* 范围(n) -> [0..n-1]；范围(a,b) -> [a..b-1]；范围(a,b,步长) */
static Value *builtin_range(int argc, Value **argv) {
    if (argc < 1 || argc > 3) runtime_error(NULL, "范围函数需要1-3个参数");
    long a = 0, b = 0, s = 1;
    if (argc == 1) { b = (long)num_of(argv[0], "范围"); }
    else {
        a = (long)num_of(argv[0], "范围");
        b = (long)num_of(argv[1], "范围");
        if (argc == 3) s = (long)num_of(argv[2], "范围");
    }
    if (s == 0) runtime_error(NULL, "范围的步长不能为 0");
    Value *r = list_new_empty();
    if (s > 0) for (long i = a; i < b; i += s) { Value *v = val_int(i); list_push(r, v); val_free(v); }
    else       for (long i = a; i > b; i += s) { Value *v = val_int(i); list_push(r, v); val_free(v); }
    return r;
}

static Value *builtin_reverse(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "反转函数需要1个参数");
    if (argv[0]->type == VAL_STRING) {
        /* 按字符反转（UTF-8 安全） */
        const char *s = argv[0]->sval ? argv[0]->sval : "";
        long n = utf8_strlen(s);
        SBuf sb; sb_init(&sb);
        for (long i = n - 1; i >= 0; i--) {
            char *ch = utf8_char_at(s, i);
            sb_append(&sb, ch);
            free(ch);
        }
        Value *r = val_new(VAL_STRING);
        r->sval = sb.buf;
        return r;
    }
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "反转函数的参数必须是列表或字符串");
    Value *l = argv[0];
    for (int i = 0, j = l->list_len - 1; i < j; i++, j--) {
        Value *t = l->lval[i]; l->lval[i] = l->lval[j]; l->lval[j] = t;
    }
    return val_new(VAL_NULL);
}

/* 索引(列表, 元素) -> 首次出现的下标，找不到返回 -1 */
static Value *builtin_index_of(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "索引函数需要2个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "索引函数的第一个参数必须是列表");
    for (int i = 0; i < argv[0]->list_len; i++)
        if (val_equals(argv[0]->lval[i], argv[1])) return val_int(i);
    return val_int(-1);
}

/* 插入(列表, 下标, 元素)：下标支持负数与末尾追加 */
static Value *builtin_insert(int argc, Value **argv) {
    if (argc != 3) runtime_error(NULL, "插入函数需要3个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "插入函数的第一个参数必须是列表");
    Value *l = argv[0];
    long idx = (long)num_of(argv[1], "插入");
    if (idx < 0) idx += l->list_len;
    if (idx < 0) idx = 0;
    if (idx > l->list_len) idx = l->list_len;
    Value **nl = realloc(l->lval, (l->list_len + 1) * sizeof(Value*));
    if (!nl) runtime_error(NULL, "插入时内存不足");
    l->lval = nl;
    for (long i = l->list_len; i > idx; i--) l->lval[i] = l->lval[i-1];
    l->lval[idx] = val_retain_root(argv[2]);
    l->list_len++;
    return val_retain(l);
}

/* 弹出(列表[, 下标]) -> 被移除的元素，默认末尾 */
static Value *builtin_pop(int argc, Value **argv) {
    if (argc < 1 || argc > 2) runtime_error(NULL, "弹出函数需要1-2个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "弹出函数的第一个参数必须是列表");
    Value *l = argv[0];
    if (l->list_len == 0) runtime_error(NULL, "无法从空列表弹出元素");
    long idx = (argc == 2) ? (long)num_of(argv[1], "弹出") : l->list_len - 1;
    if (idx < 0) idx += l->list_len;
    if (idx < 0 || idx >= l->list_len) runtime_error(NULL, "弹出的下标越界");
    Value *out = l->lval[idx];              /* 引用移交给调用方，不再 retain */
    for (long i = idx; i < l->list_len - 1; i++) l->lval[i] = l->lval[i+1];
    l->list_len--;
    return out;
}

/* 唯一(列表) -> 新列表，保持首次出现顺序 */
static Value *builtin_unique(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "唯一函数需要1个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "唯一函数的参数必须是列表");
    Value *r = list_new_empty();
    for (int i = 0; i < argv[0]->list_len; i++) {
        int dup = 0;
        for (int j = 0; j < r->list_len; j++)
            if (val_equals(r->lval[j], argv[0]->lval[i])) { dup = 1; break; }
        if (!dup) list_push(r, argv[0]->lval[i]);
    }
    return r;
}

/* 计数(列表|字符串, 目标) -> 出现次数 */
static Value *builtin_count(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "计数函数需要2个参数");
    if (argv[0]->type == VAL_STRING) {
        if (argv[1]->type != VAL_STRING) runtime_error(NULL, "在字符串中计数时第二个参数必须是字符串");
        const char *s = argv[0]->sval ? argv[0]->sval : "";
        const char *t = argv[1]->sval ? argv[1]->sval : "";
        if (!*t) runtime_error(NULL, "计数的目标不能是空字符串");
        long c = 0;
        size_t tl = strlen(t);
        for (const char *p = s; (p = strstr(p, t)) != NULL; p += tl) c++;
        return val_int(c);
    }
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "计数函数的第一个参数必须是列表或字符串");
    long c = 0;
    for (int i = 0; i < argv[0]->list_len; i++)
        if (val_equals(argv[0]->lval[i], argv[1])) c++;
    return val_int(c);
}

/* 切片(列表|字符串, 起, 止) -> 新列表/新串，支持负索引，止为开区间 */
static Value *builtin_slice(int argc, Value **argv) {
    if (argc < 2 || argc > 3) runtime_error(NULL, "切片函数需要2-3个参数");
    if (argv[0]->type == VAL_STRING) {
        const char *s = argv[0]->sval ? argv[0]->sval : "";
        long n = utf8_strlen(s);
        long a = (long)num_of(argv[1], "切片");
        long b = (argc == 3) ? (long)num_of(argv[2], "切片") : n;
        if (a < 0) a += n;
        if (b < 0) b += n;
        if (a < 0) a = 0;
        if (b > n) b = n;
        Value *r = val_new(VAL_STRING);
        r->sval = (b > a) ? utf8_substr(s, a, b - a) : strdup("");
        return r;
    }
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "切片函数的第一个参数必须是列表或字符串");
    Value *l = argv[0];
    long n = l->list_len;
    long a = (long)num_of(argv[1], "切片");
    long b = (argc == 3) ? (long)num_of(argv[2], "切片") : n;
    if (a < 0) a += n;
    if (b < 0) b += n;
    if (a < 0) a = 0;
    if (b > n) b = n;
    Value *r = list_new_empty();
    for (long i = a; i < b; i++) list_push(r, l->lval[i]);
    return r;
}

/* 扁平(嵌套列表) -> 一层展开 */
static Value *builtin_flatten(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "扁平函数需要1个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "扁平函数的参数必须是列表");
    Value *r = list_new_empty();
    for (int i = 0; i < argv[0]->list_len; i++) {
        Value *e = argv[0]->lval[i];
        if (e->type == VAL_LIST) for (int j = 0; j < e->list_len; j++) list_push(r, e->lval[j]);
        else list_push(r, e);
    }
    return r;
}

/* 累加(列表) -> 数值和；全整数返回整数 */
static Value *builtin_total(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "累加函数需要1个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "累加函数的参数必须是列表");
    double s = 0; int all_int = 1;
    for (int i = 0; i < argv[0]->list_len; i++) {
        s += num_of(argv[0]->lval[i], "累加");
        if (argv[0]->lval[i]->type != VAL_INT) all_int = 0;
    }
    return all_int ? val_int((long)s) : val_flt(s);
}

/* ---------- 字典函数 ---------- */
static Value *builtin_keys(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "键函数需要1个参数");
    if (argv[0]->type != VAL_MAP) runtime_error(NULL, "键函数的参数必须是字典");
    Value *r = list_new_empty();
    for (int i = 0; i < MAX_DICT_SIZE; i++) {
        if (!argv[0]->mval->entries[i].used) continue;
        Value *k = val_new(VAL_STRING);
        k->sval = strdup(argv[0]->mval->entries[i].key);
        list_push(r, k);
        val_free(k);
    }
    return r;
}
static Value *builtin_values(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "值函数需要1个参数");
    if (argv[0]->type != VAL_MAP) runtime_error(NULL, "值函数的参数必须是字典");
    Value *r = list_new_empty();
    for (int i = 0; i < MAX_DICT_SIZE; i++) {
        if (!argv[0]->mval->entries[i].used) continue;
        list_push(r, argv[0]->mval->entries[i].val);
    }
    return r;
}
static Value *builtin_del_key(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "移除键函数需要2个参数");
    if (argv[0]->type != VAL_MAP) runtime_error(NULL, "移除键函数的第一个参数必须是字典");
    if (argv[1]->type != VAL_STRING) runtime_error(NULL, "字典键必须是字符串");
    return val_int(dict_remove(argv[0]->mval, argv[1]->sval));
}
/* 项(字典) -> [[键, 值], ...]，便于用 对于 遍历 */
static Value *builtin_items(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "项函数需要1个参数");
    if (argv[0]->type != VAL_MAP) runtime_error(NULL, "项函数的参数必须是字典");
    Value *r = list_new_empty();
    for (int i = 0; i < MAX_DICT_SIZE; i++) {
        if (!argv[0]->mval->entries[i].used) continue;
        Value *pair = list_new_empty();
        Value *k = val_new(VAL_STRING);
        k->sval = strdup(argv[0]->mval->entries[i].key);
        list_push(pair, k);
        val_free(k);
        list_push(pair, argv[0]->mval->entries[i].val);
        list_push(r, pair);
        val_free(pair);
    }
    return r;
}

/* ---------- 高阶函数：映射 / 过滤 / 归约 / 全部满足 / 任一满足 ----------
   由解释器在定义 eval 之后注入回调，使内置函数能调用用户自定义函数。 */
static Value *(*g_call_fn)(Value *fnval, Value **args, int argc) = NULL;

static Value *hof_call1(Value *fn, Value *a) {
    if (!g_call_fn) runtime_error(NULL, "内部错误: 函数调用回调未初始化");
    Value *args[1] = { a };
    return g_call_fn(fn, args, 1);
}
static Value *builtin_map_fn(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "映射函数需要2个参数（列表, 函数）");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "映射函数的第一个参数必须是列表");
    if (argv[1]->type != VAL_FUNC) runtime_error(NULL, "映射函数的第二个参数必须是函数");
    Value *r = list_new_empty();
    for (int i = 0; i < argv[0]->list_len; i++) {
        Value *out = hof_call1(argv[1], argv[0]->lval[i]);
        list_push(r, out);
        val_free(out);
    }
    return r;
}
static Value *builtin_filter_fn(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "过滤函数需要2个参数（列表, 函数）");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "过滤函数的第一个参数必须是列表");
    if (argv[1]->type != VAL_FUNC) runtime_error(NULL, "过滤函数的第二个参数必须是函数");
    Value *r = list_new_empty();
    for (int i = 0; i < argv[0]->list_len; i++) {
        Value *keep = hof_call1(argv[1], argv[0]->lval[i]);
        int ok = truthy(keep);
        val_free(keep);
        if (ok) list_push(r, argv[0]->lval[i]);
    }
    return r;
}
static Value *builtin_reduce_fn(int argc, Value **argv) {
    if (argc < 2 || argc > 3) runtime_error(NULL, "归约函数需要2-3个参数（列表, 函数[, 初值]）");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "归约函数的第一个参数必须是列表");
    if (argv[1]->type != VAL_FUNC) runtime_error(NULL, "归约函数的第二个参数必须是函数");
    if (!g_call_fn) runtime_error(NULL, "内部错误: 函数调用回调未初始化");
    Value *acc = NULL;
    int start = 0;
    if (argc == 3) acc = val_retain(argv[2]);
    else {
        if (argv[0]->list_len == 0) runtime_error(NULL, "对空列表归约时必须提供初值");
        acc = val_retain(argv[0]->lval[0]);
        start = 1;
    }
    for (int i = start; i < argv[0]->list_len; i++) {
        Value *args[2] = { acc, argv[0]->lval[i] };
        Value *next = g_call_fn(argv[1], args, 2);
        val_free(acc);
        acc = next;
    }
    return acc;
}
static Value *builtin_all_fn(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "全部满足函数需要2个参数（列表, 函数）");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "全部满足函数的第一个参数必须是列表");
    if (argv[1]->type != VAL_FUNC) runtime_error(NULL, "全部满足函数的第二个参数必须是函数");
    for (int i = 0; i < argv[0]->list_len; i++) {
        Value *v = hof_call1(argv[1], argv[0]->lval[i]);
        int ok = truthy(v);
        val_free(v);
        if (!ok) return val_bool(0);
    }
    return val_bool(1);
}
static Value *builtin_any_fn(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "任一满足函数需要2个参数（列表, 函数）");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "任一满足函数的第一个参数必须是列表");
    if (argv[1]->type != VAL_FUNC) runtime_error(NULL, "任一满足函数的第二个参数必须是函数");
    for (int i = 0; i < argv[0]->list_len; i++) {
        Value *v = hof_call1(argv[1], argv[0]->lval[i]);
        int ok = truthy(v);
        val_free(v);
        if (ok) return val_bool(1);
    }
    return val_bool(0);
}

static Value *builtin_contains(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "包含函数需要2个参数");
    /* 统一的成员判断：字符串查子串、字典查键、列表查元素 */
    if (argv[0]->type == VAL_STRING) {
        if (argv[1]->type != VAL_STRING) runtime_error(NULL, "在字符串中查找时第二个参数必须是字符串");
        return val_bool(strstr(argv[0]->sval, argv[1]->sval) != NULL);
    }
    if (argv[0]->type == VAL_MAP) {
        if (argv[1]->type != VAL_STRING) runtime_error(NULL, "字典键必须是字符串");
        return val_bool(dict_get(argv[0]->mval, argv[1]->sval) != NULL);
    }
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "包含函数的第一个参数必须是列表、字符串或字典");
    /* 元素判等统一走 val_equals：支持嵌套列表/字典/布尔，而非只比三种标量 */
    for (int i = 0; i < argv[0]->list_len; i++)
        if (val_equals(argv[0]->lval[i], argv[1])) return val_bool(1);
    return val_bool(0);
}

/* 字符串函数 */
static Value *builtin_split(int argc, Value **argv) {
    if (argc < 1 || argc > 2) runtime_error(NULL, "分割函数需要1-2个参数");
    if (argv[0]->type != VAL_STRING) runtime_error(NULL, "分割函数的第一个参数必须是字符串");
    Value *str = argv[0];
    const char *sep = (argc >= 2 && argv[1]->type == VAL_STRING) ? argv[1]->sval : " ";
    
    /* 按【完整子串】切分，而非 strtok 的“字符集合”语义
       —— 这对多字节分隔符（如 "、"）是必须的，否则会按字节乱切 */
    Value *result = list_new_empty();
    const char *s = str->sval ? str->sval : "";
    size_t sl = strlen(sep);
    if (sl == 0) {
        /* 空分隔符：逐字符拆分（UTF-8 安全） */
        long n = utf8_strlen(s);
        for (long i = 0; i < n; i++) {
            Value *e = val_new(VAL_STRING);
            e->sval = utf8_char_at(s, i);
            list_push(result, e);
            val_free(e);
        }
        return result;
    }
    const char *p = s;
    while (1) {
        const char *q = strstr(p, sep);
        size_t seg = q ? (size_t)(q - p) : strlen(p);
        Value *e = val_new(VAL_STRING);
        e->sval = malloc(seg + 1);
        memcpy(e->sval, p, seg);
        e->sval[seg] = '\0';
        list_push(result, e);
        val_free(e);
        if (!q) break;
        p = q + sl;
    }
    return result;
}

static Value *builtin_join(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "连接函数需要2个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "连接函数的第一个参数必须是列表");
    if (argv[1]->type != VAL_STRING) runtime_error(NULL, "连接函数的第二个参数必须是字符串");
    Value *list = argv[0];
    const char *sep = argv[1]->sval ? argv[1]->sval : "";
    /* 用动态缓冲：空列表不再 size_t 下溢，非字符串元素自动转字符串 */
    SBuf sb;
    sb_init(&sb);
    for (int i = 0; i < list->list_len; i++) {
        if (i) sb_append(&sb, sep);
        val_to_sbuf(list->lval[i], &sb);
    }
    Value *result = val_new(VAL_STRING);
    result->sval = sb.buf;
    return result;
}

static Value *builtin_replace(int argc, Value **argv) {
    if (argc != 3) runtime_error(NULL, "替换函数需要3个参数");
    if (argv[0]->type != VAL_STRING) runtime_error(NULL, "替换函数的第一个参数必须是字符串");
    if (argv[1]->type != VAL_STRING || argv[2]->type != VAL_STRING) runtime_error(NULL, "替换函数的第二、三个参数必须是字符串");
    
    const char *src = argv[0]->sval ? argv[0]->sval : "";
    const char *old = argv[1]->sval ? argv[1]->sval : "";
    const char *rep = argv[2]->sval ? argv[2]->sval : "";
    size_t old_len = strlen(old);
    /* 空的被替换串会造成死循环，明确报错而不是挂死 */
    if (old_len == 0) runtime_error(NULL, "替换函数的被替换串不能为空");

    /* 动态缓冲：替换串比原串长时不再溢出（原实现按 2 倍预估，会写越界） */
    SBuf sb;
    sb_init(&sb);
    const char *p = src;
    while (*p) {
        if (strncmp(p, old, old_len) == 0) {
            sb_append(&sb, rep);
            p += old_len;
        } else {
            char one[2] = { *p++, '\0' };
            sb_append(&sb, one);
        }
    }
    Value *result = val_new(VAL_STRING);
    result->sval = sb.buf;
    return result;
}

static Value *builtin_find(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "查找函数需要2个参数");
    if (argv[0]->type != VAL_STRING || argv[1]->type != VAL_STRING) runtime_error(NULL, "查找函数的参数必须是字符串");
    
    Value *result = val_new(VAL_INT);
    char *p = strstr(argv[0]->sval, argv[1]->sval);
    /* 返回字符下标而非字节偏移，与 长度/截取/下标 保持同一坐标系 */
    result->ival = p ? utf8_char_index(argv[0]->sval, (long)(p - argv[0]->sval)) : -1;
    return result;
}

static Value *builtin_substring(int argc, Value **argv) {
    if (argc != 3) runtime_error(NULL, "截取函数需要3个参数");
    if (argv[0]->type != VAL_STRING) runtime_error(NULL, "截取函数的第一个参数必须是字符串");
    if (argv[1]->type != VAL_INT || argv[2]->type != VAL_INT) runtime_error(NULL, "截取函数的后两个参数必须是整数");
    
    /* 按【字符】截取，支持负起点（-1 表示最后一个字符） */
    Value *result = val_new(VAL_STRING);
    result->sval = utf8_substr(argv[0]->sval, argv[1]->ival, argv[2]->ival);
    return result;
}

/* ---------- 新增字符串函数 ---------- */
static const char *str_arg(Value *v, const char *who) {
    if (v->type != VAL_STRING) runtime_error(NULL, "%s 需要字符串参数", who);
    return v->sval ? v->sval : "";
}

/* 去空白 / 去左空白 / 去右空白 */
static Value *str_trim_impl(const char *s, int left, int right) {
    const char *a = s;
    const char *b = s + strlen(s);
    if (left)  while (a < b && (unsigned char)*a <= ' ') a++;
    if (right) while (b > a && (unsigned char)*(b-1) <= ' ') b--;
    Value *r = val_new(VAL_STRING);
    size_t n = (size_t)(b - a);
    r->sval = malloc(n + 1);
    memcpy(r->sval, a, n);
    r->sval[n] = '\0';
    return r;
}
static Value *builtin_trim(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "去空白函数需要1个参数");
    return str_trim_impl(str_arg(argv[0], "去空白"), 1, 1);
}
static Value *builtin_ltrim(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "去左空白函数需要1个参数");
    return str_trim_impl(str_arg(argv[0], "去左空白"), 1, 0);
}
static Value *builtin_rtrim(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "去右空白函数需要1个参数");
    return str_trim_impl(str_arg(argv[0], "去右空白"), 0, 1);
}

/* 大写 / 小写：只改 ASCII 字母，多字节字符原样保留（不破坏 UTF-8） */
static Value *str_case_impl(const char *s, int up) {
    Value *r = val_new(VAL_STRING);
    r->sval = strdup(s);
    for (char *p = r->sval; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (c < 0x80) *p = up ? (char)toupper(c) : (char)tolower(c);
    }
    return r;
}
static Value *builtin_upper(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "大写函数需要1个参数");
    return str_case_impl(str_arg(argv[0], "大写"), 1);
}
static Value *builtin_lower(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "小写函数需要1个参数");
    return str_case_impl(str_arg(argv[0], "小写"), 0);
}

/* 重复(串, 次数) */
static Value *builtin_repeat(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "重复函数需要2个参数");
    const char *s = str_arg(argv[0], "重复");
    long n = (long)num_of(argv[1], "重复");
    if (n < 0) runtime_error(NULL, "重复次数不能为负数");
    SBuf sb; sb_init(&sb);
    for (long i = 0; i < n; i++) sb_append(&sb, s);
    Value *r = val_new(VAL_STRING);
    r->sval = sb.buf;
    return r;
}

/* 开头是(串, 前缀) / 结尾是(串, 后缀) */
static Value *builtin_startswith(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "开头是函数需要2个参数");
    const char *s = str_arg(argv[0], "开头是"), *p = str_arg(argv[1], "开头是");
    return val_bool(strncmp(s, p, strlen(p)) == 0);
}
static Value *builtin_endswith(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "结尾是函数需要2个参数");
    const char *s = str_arg(argv[0], "结尾是"), *p = str_arg(argv[1], "结尾是");
    size_t ls = strlen(s), lp = strlen(p);
    return val_bool(lp <= ls && strcmp(s + ls - lp, p) == 0);
}

/* 字符码(串[, 下标]) -> Unicode 码点；码转字符(码点) -> 串 */
static Value *builtin_char_code(int argc, Value **argv) {
    if (argc < 1 || argc > 2) runtime_error(NULL, "字符码函数需要1-2个参数");
    const char *s = str_arg(argv[0], "字符码");
    long idx = (argc == 2) ? (long)num_of(argv[1], "字符码") : 0;
    char *ch = utf8_char_at(s, idx);
    if (!ch) runtime_error(NULL, "字符码的下标越界");
    unsigned char c0 = (unsigned char)ch[0];
    long cp;
    int n = utf8_seq_len(c0);
    if (n == 1) cp = c0;
    else if (n == 2) cp = ((c0 & 0x1F) << 6) | ((unsigned char)ch[1] & 0x3F);
    else if (n == 3) cp = ((c0 & 0x0F) << 12) | (((unsigned char)ch[1] & 0x3F) << 6) | ((unsigned char)ch[2] & 0x3F);
    else cp = ((long)(c0 & 0x07) << 18) | (((unsigned char)ch[1] & 0x3F) << 12) |
              (((unsigned char)ch[2] & 0x3F) << 6) | ((unsigned char)ch[3] & 0x3F);
    free(ch);
    return val_int(cp);
}
static Value *builtin_from_code(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "码转字符函数需要1个参数");
    long cp = (long)num_of(argv[0], "码转字符");
    if (cp < 0 || cp > 0x10FFFF) runtime_error(NULL, "码点超出 Unicode 范围");
    char b[5]; int n;
    if (cp < 0x80)        { b[0] = (char)cp; n = 1; }
    else if (cp < 0x800)  { b[0] = (char)(0xC0 | (cp >> 6));  b[1] = (char)(0x80 | (cp & 0x3F)); n = 2; }
    else if (cp < 0x10000){ b[0] = (char)(0xE0 | (cp >> 12)); b[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            b[2] = (char)(0x80 | (cp & 0x3F)); n = 3; }
    else                  { b[0] = (char)(0xF0 | (cp >> 18)); b[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
                            b[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); b[3] = (char)(0x80 | (cp & 0x3F)); n = 4; }
    b[n] = '\0';
    Value *r = val_new(VAL_STRING);
    r->sval = strdup(b);
    return r;
}

/* ===== 字节级视图：系统编程必须能直接摸到字节 =====
   长度/下标 一律以【字符】为单位（不切碎汉字），这对文本处理是对的；
   但编码、散列、协议解析必须按字节走，否则根本写不了。
   因此另开一套按字节的入口，两套语义各自明确、互不污染。 */
static Value *builtin_byte_len(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "字节数函数需要1个参数");
    const char *s = str_arg(argv[0], "字节数");
    return val_int((long)strlen(s));
}

static Value *builtin_byte_at(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "字节函数需要2个参数（文本, 下标）");
    const char *s = str_arg(argv[0], "字节");
    long n = (long)strlen(s);
    long i = int_of(argv[1], "字节");
    if (i < 0) i += n;                       /* 负下标：-1 表示最后一个字节 */
    if (i < 0 || i >= n) runtime_error(NULL, "字节下标越界: %ld（共 %ld 字节）", i, n);
    return val_int((long)(unsigned char)s[i]);
}

static Value *builtin_bytes_of(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "字节列表函数需要1个参数");
    const char *s = str_arg(argv[0], "字节列表");
    long n = (long)strlen(s);
    Value *lst = list_new_empty();
    for (long i = 0; i < n; i++) {
        Value *b = val_int((long)(unsigned char)s[i]);
        list_push(lst, b);
        val_free(b);                          /* list_push 内部已 retain */
    }
    return lst;
}

static Value *builtin_bytes_to_str(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "字节转文本函数需要1个参数（字节列表）");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "字节转文本函数需要列表，收到 %s", val_type_name(argv[0]));
    int n = argv[0]->list_len;
    /* 用异常安全的临时缓冲：中途遇到非法字节会 longjmp，普通 malloc 会泄漏 */
    char *buf = (char *)co_tmp_alloc((size_t)n + 1);
    for (int i = 0; i < n; i++) {
        long b = int_of(argv[0]->lval[i], "字节转文本");
        if (b < 0 || b > 255) runtime_error(NULL, "字节值须在 0~255 之间，第 %d 项是 %ld", i, b);
        if (b == 0) runtime_error(NULL, "字节值不能是 0（字符串以 0 结尾，会被截断）");
        buf[i] = (char)(unsigned char)b;
    }
    buf[n] = '\0';
    Value *r = val_new(VAL_STRING);
    r->sval = strdup(buf);
    co_tmp_free(buf);
    return r;
}

/* ---------- 文件读写：通用语言的基本能力 ---------- */
static Value *builtin_read_file(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "读文件函数需要1个参数（路径）");
    const char *path = str_arg(argv[0], "读文件");
    FILE *f = fopen(path, "rb");
    if (!f) runtime_error(NULL, "无法打开文件: %s", path);
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); runtime_error(NULL, "无法定位文件末尾: %s", path); }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); runtime_error(NULL, "无法获取文件大小: %s", path); }
    rewind(f);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); runtime_error(NULL, "读取文件时内存不足"); }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[got] = '\0';
    Value *r = val_new(VAL_STRING);
    r->sval = buf;
    return r;
}
static Value *file_write_impl(int argc, Value **argv, const char *mode, const char *who) {
    if (argc != 2) runtime_error(NULL, "%s函数需要2个参数（路径, 内容）", who);
    const char *path = str_arg(argv[0], who);
    char *content = val_to_string(argv[1]);
    FILE *f = fopen(path, mode);
    if (!f) { free(content); runtime_error(NULL, "无法写入文件: %s", path); }
    size_t n = strlen(content);
    size_t w = fwrite(content, 1, n, f);
    int cerr = fclose(f);
    free(content);
    if (w != n || cerr != 0) runtime_error(NULL, "写入文件未完成: %s", path);
    return val_int((long)n);
}
static Value *builtin_write_file(int argc, Value **argv)  { return file_write_impl(argc, argv, "wb", "写文件"); }
static Value *builtin_append_file(int argc, Value **argv) { return file_write_impl(argc, argv, "ab", "追加文件"); }
static Value *builtin_file_exists(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "文件存在函数需要1个参数");
    FILE *f = fopen(str_arg(argv[0], "文件存在"), "rb");
    if (f) { fclose(f); return val_bool(1); }
    return val_bool(0);
}
static Value *builtin_delete_file(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "删除文件函数需要1个参数");
    return val_bool(remove(str_arg(argv[0], "删除文件")) == 0);
}
/* 读取行(路径) -> 行列表（去掉行尾换行） */
static Value *builtin_read_lines(int argc, Value **argv) {
    Value *whole = builtin_read_file(argc, argv);
    Value *r = list_new_empty();
    const char *p = whole->sval;
    while (*p) {
        const char *q = strchr(p, '\n');
        size_t n = q ? (size_t)(q - p) : strlen(p);
        if (n > 0 && p[n-1] == '\r') n--;          /* 兼容 CRLF */
        Value *e = val_new(VAL_STRING);
        e->sval = malloc(n + 1);
        memcpy(e->sval, p, n);
        e->sval[n] = '\0';
        list_push(r, e);
        val_free(e);
        if (!q) break;
        p = q + 1;
    }
    val_free(whole);
    return r;
}

/* ========== 自动求导 / 张量算子 ========== */

/* 广播形状计算（numpy 风格，末尾对齐）；失败返回 -1 */
static int bc_shape(const int *a, int na, const int *b, int nb, int *out) {
    int ondim = na > nb ? na : nb;
    for (int i = 0; i < ondim; i++) {
        int da = (i < ondim - na) ? 1 : a[i - (ondim - na)];
        int db = (i < ondim - nb) ? 1 : b[i - (ondim - nb)];
        if (da != db && da != 1 && db != 1) return -1;
        out[i] = da > db ? da : db;
    }
    return ondim;
}

/* 将输出扁平索引映射到某个子节点的扁平索引（处理广播） */
static int bc_child_index(int out_flat, const int *out_shape, int ondim,
                          const int *c_shape, int cndim) {
    int coords[CO_MAX_DIM];
    int rem = out_flat;
    for (int i = ondim - 1; i >= 0; i--) { coords[i] = rem % out_shape[i]; rem /= out_shape[i]; }
    int coff = ondim - cndim;
    int cidx = 0;
    for (int i = 0; i < cndim; i++) {
        int oi = coff + i;
        int cd = c_shape[i];
        int ci = (cd == 1) ? 0 : coords[oi];
        cidx = cidx * cd + ci;
    }
    return cidx;
}

/* ---------- 逐元素广播算子 ---------- */
static void t_add_bw(Tensor *c) {
    Tensor *a = c->children[0], *b = c->children[1];
    if (a->requires_grad) { ensure_grad(a);
        for (int oi = 0; oi < c->size; oi++) a->grad[bc_child_index(oi, c->shape, c->ndim, a->shape, a->ndim)] += c->grad[oi];
    }
    if (b->requires_grad) { ensure_grad(b);
        for (int oi = 0; oi < c->size; oi++) b->grad[bc_child_index(oi, c->shape, c->ndim, b->shape, b->ndim)] += c->grad[oi];
    }
}
static Tensor *t_add(Tensor *a, Tensor *b) {
    int out[CO_MAX_DIM]; int ondim = bc_shape(a->shape, a->ndim, b->shape, b->ndim, out);
    if (ondim < 0) return NULL;
    Tensor *c = tensor_new(ondim, out);
    for (int oi = 0; oi < c->size; oi++) {
        int ia = bc_child_index(oi, out, ondim, a->shape, a->ndim);
        int ib = bc_child_index(oi, out, ondim, b->shape, b->ndim);
        c->data[oi] = a->data[ia] + b->data[ib];
    }
    c->requires_grad = a->requires_grad || b->requires_grad;
    c->backward = t_add_bw; tensor_set_child(c, a); tensor_set_child(c, b);
    return c;
}

static void t_sub_bw(Tensor *c) {
    Tensor *a = c->children[0], *b = c->children[1];
    if (a->requires_grad) { ensure_grad(a);
        for (int oi = 0; oi < c->size; oi++) a->grad[bc_child_index(oi, c->shape, c->ndim, a->shape, a->ndim)] += c->grad[oi];
    }
    if (b->requires_grad) { ensure_grad(b);
        for (int oi = 0; oi < c->size; oi++) b->grad[bc_child_index(oi, c->shape, c->ndim, b->shape, b->ndim)] -= c->grad[oi];
    }
}
static Tensor *t_sub(Tensor *a, Tensor *b) {
    int out[CO_MAX_DIM]; int ondim = bc_shape(a->shape, a->ndim, b->shape, b->ndim, out);
    if (ondim < 0) return NULL;
    Tensor *c = tensor_new(ondim, out);
    for (int oi = 0; oi < c->size; oi++) {
        int ia = bc_child_index(oi, out, ondim, a->shape, a->ndim);
        int ib = bc_child_index(oi, out, ondim, b->shape, b->ndim);
        c->data[oi] = a->data[ia] - b->data[ib];
    }
    c->requires_grad = a->requires_grad || b->requires_grad;
    c->backward = t_sub_bw; tensor_set_child(c, a); tensor_set_child(c, b);
    return c;
}

static void t_mul_bw(Tensor *c) {
    Tensor *a = c->children[0], *b = c->children[1];
    if (a->requires_grad) { ensure_grad(a);
        for (int oi = 0; oi < c->size; oi++) {
            int ia = bc_child_index(oi, c->shape, c->ndim, a->shape, a->ndim);
            int ib = bc_child_index(oi, c->shape, c->ndim, b->shape, b->ndim);
            a->grad[ia] += c->grad[oi] * b->data[ib];
        }
    }
    if (b->requires_grad) { ensure_grad(b);
        for (int oi = 0; oi < c->size; oi++) {
            int ia = bc_child_index(oi, c->shape, c->ndim, a->shape, a->ndim);
            int ib = bc_child_index(oi, c->shape, c->ndim, b->shape, b->ndim);
            b->grad[ib] += c->grad[oi] * a->data[ia];
        }
    }
}
static Tensor *t_mul(Tensor *a, Tensor *b) {
    int out[CO_MAX_DIM]; int ondim = bc_shape(a->shape, a->ndim, b->shape, b->ndim, out);
    if (ondim < 0) return NULL;
    Tensor *c = tensor_new(ondim, out);
    for (int oi = 0; oi < c->size; oi++) {
        int ia = bc_child_index(oi, out, ondim, a->shape, a->ndim);
        int ib = bc_child_index(oi, out, ondim, b->shape, b->ndim);
        c->data[oi] = a->data[ia] * b->data[ib];
    }
    c->requires_grad = a->requires_grad || b->requires_grad;
    c->backward = t_mul_bw; tensor_set_child(c, a); tensor_set_child(c, b);
    return c;
}

static void t_div_bw(Tensor *c) {
    Tensor *a = c->children[0], *b = c->children[1];
    if (a->requires_grad) { ensure_grad(a);
        for (int oi = 0; oi < c->size; oi++) {
            int ia = bc_child_index(oi, c->shape, c->ndim, a->shape, a->ndim);
            int ib = bc_child_index(oi, c->shape, c->ndim, b->shape, b->ndim);
            a->grad[ia] += c->grad[oi] / (b->data[ib] + 1e-12f);
        }
    }
    if (b->requires_grad) { ensure_grad(b);
        for (int oi = 0; oi < c->size; oi++) {
            int ia = bc_child_index(oi, c->shape, c->ndim, a->shape, a->ndim);
            int ib = bc_child_index(oi, c->shape, c->ndim, b->shape, b->ndim);
            b->grad[ib] += -c->grad[oi] * a->data[ia] / ((b->data[ib] * b->data[ib]) + 1e-12f);
        }
    }
}
static Tensor *t_div(Tensor *a, Tensor *b) {
    int out[CO_MAX_DIM]; int ondim = bc_shape(a->shape, a->ndim, b->shape, b->ndim, out);
    if (ondim < 0) return NULL;
    Tensor *c = tensor_new(ondim, out);
    for (int oi = 0; oi < c->size; oi++) {
        int ia = bc_child_index(oi, out, ondim, a->shape, a->ndim);
        int ib = bc_child_index(oi, out, ondim, b->shape, b->ndim);
        c->data[oi] = a->data[ia] / (b->data[ib] + 1e-12f);
    }
    c->requires_grad = a->requires_grad || b->requires_grad;
    c->backward = t_div_bw; tensor_set_child(c, a); tensor_set_child(c, b);
    return c;
}

/* ---------- 矩阵乘法 ---------- */
static void t_matmul_bw(Tensor *c) {
    Tensor *a = c->children[0], *b = c->children[1];
    int m = a->shape[0], K = a->shape[1], n = b->shape[1];
    if (a->requires_grad) { ensure_grad(a);
        for (int i = 0; i < m; i++) for (int k = 0; k < K; k++) for (int j = 0; j < n; j++)
            a->grad[i * K + k] += c->grad[i * n + j] * b->data[k * n + j];
    }
    if (b->requires_grad) { ensure_grad(b);
        for (int i = 0; i < m; i++) for (int k = 0; k < K; k++) for (int j = 0; j < n; j++)
            b->grad[k * n + j] += a->data[i * K + k] * c->grad[i * n + j];
    }
}
static Tensor *t_matmul(Tensor *a, Tensor *b) {
    if (a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) return NULL;
    int m = a->shape[0], K = a->shape[1], n = b->shape[1];
    int sh[2] = { m, n };
    Tensor *c = tensor_new(2, sh);
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) {
        float s = 0; for (int k = 0; k < K; k++) s += a->data[i * K + k] * b->data[k * n + j];
        c->data[i * n + j] = s;
    }
    c->requires_grad = a->requires_grad || b->requires_grad;
    c->backward = t_matmul_bw; tensor_set_child(c, a); tensor_set_child(c, b);
    return c;
}

/* ---------- 转置 ---------- */
static void t_transpose_bw(Tensor *c) {
    Tensor *a = c->children[0];
    if (a->requires_grad) { ensure_grad(a);
        int m = a->shape[0], n = a->shape[1];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++)
            a->grad[i * n + j] += c->grad[j * m + i];
    }
}
static Tensor *t_transpose(Tensor *a) {
    if (a->ndim != 2) return NULL;
    int m = a->shape[0], n = a->shape[1];
    int sh[2] = { n, m };
    Tensor *c = tensor_new(2, sh);
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) c->data[j * m + i] = a->data[i * n + j];
    c->requires_grad = a->requires_grad; c->backward = t_transpose_bw; tensor_set_child(c, a);
    return c;
}

/* ---------- 求和 / 均值 ---------- */
static void t_sum_bw(Tensor *c) {
    Tensor *a = c->children[0];
    if (a->requires_grad) { ensure_grad(a); float g = c->grad[0];
        for (int i = 0; i < a->size; i++) a->grad[i] += g;
    }
}
static Tensor *t_sum(Tensor *a) {
    int sh[1] = { 1 }; Tensor *c = tensor_new(1, sh);
    float s = 0; for (int i = 0; i < a->size; i++) s += a->data[i];
    c->data[0] = s;
    c->requires_grad = a->requires_grad; c->backward = t_sum_bw; tensor_set_child(c, a);
    return c;
}
static void t_mean_bw(Tensor *c) {
    Tensor *a = c->children[0];
    if (a->requires_grad) { ensure_grad(a); float g = c->grad[0] / a->size;
        for (int i = 0; i < a->size; i++) a->grad[i] += g;
    }
}
static Tensor *t_mean(Tensor *a) {
    int sh[1] = { 1 }; Tensor *c = tensor_new(1, sh);
    float s = 0; for (int i = 0; i < a->size; i++) s += a->data[i];
    c->data[0] = s / a->size;
    c->requires_grad = a->requires_grad; c->backward = t_mean_bw; tensor_set_child(c, a);
    return c;
}

/* ---------- 逐元素激活 / 数学 ---------- */
static void t_sigmoid_bw(Tensor *c) {
    Tensor *a = c->children[0];
    if (a->requires_grad) { ensure_grad(a);
        for (int i = 0; i < a->size; i++) { float x = a->data[i]; float s = 1.0f / (1.0f + expf(-x)); a->grad[i] += s * (1 - s) * c->grad[i]; }
    }
}
static Tensor *t_sigmoid(Tensor *a) {
    Tensor *c = tensor_new(a->ndim, a->shape);
    for (int i = 0; i < a->size; i++) c->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
    c->requires_grad = a->requires_grad; c->backward = t_sigmoid_bw; tensor_set_child(c, a);
    return c;
}
static void t_tanh_bw(Tensor *c) {
    Tensor *a = c->children[0];
    if (a->requires_grad) { ensure_grad(a);
        for (int i = 0; i < a->size; i++) { float t = tanhf(a->data[i]); a->grad[i] += (1 - t * t) * c->grad[i]; }
    }
}
static Tensor *t_tanh(Tensor *a) {
    Tensor *c = tensor_new(a->ndim, a->shape);
    for (int i = 0; i < a->size; i++) c->data[i] = tanhf(a->data[i]);
    c->requires_grad = a->requires_grad; c->backward = t_tanh_bw; tensor_set_child(c, a);
    return c;
}
static void t_relu_bw(Tensor *c) {
    Tensor *a = c->children[0];
    if (a->requires_grad) { ensure_grad(a);
        for (int i = 0; i < a->size; i++) if (a->data[i] > 0) a->grad[i] += c->grad[i];
    }
}
static Tensor *t_relu(Tensor *a) {
    Tensor *c = tensor_new(a->ndim, a->shape);
    for (int i = 0; i < a->size; i++) c->data[i] = a->data[i] > 0 ? a->data[i] : 0;
    c->requires_grad = a->requires_grad; c->backward = t_relu_bw; tensor_set_child(c, a);
    return c;
}
static void t_exp_bw(Tensor *c) {
    Tensor *a = c->children[0];
    if (a->requires_grad) { ensure_grad(a);
        for (int i = 0; i < a->size; i++) a->grad[i] += c->data[i] * c->grad[i];
    }
}
static Tensor *t_exp(Tensor *a) {
    Tensor *c = tensor_new(a->ndim, a->shape);
    for (int i = 0; i < a->size; i++) c->data[i] = expf(a->data[i]);
    c->requires_grad = a->requires_grad; c->backward = t_exp_bw; tensor_set_child(c, a);
    return c;
}
static void t_log_bw(Tensor *c) {
    Tensor *a = c->children[0];
    if (a->requires_grad) { ensure_grad(a);
        for (int i = 0; i < a->size; i++) a->grad[i] += c->grad[i] / (a->data[i] + 1e-12f);
    }
}
static Tensor *t_log(Tensor *a) {
    Tensor *c = tensor_new(a->ndim, a->shape);
    for (int i = 0; i < a->size; i++) c->data[i] = logf(a->data[i] + 1e-12f);
    c->requires_grad = a->requires_grad; c->backward = t_log_bw; tensor_set_child(c, a);
    return c;
}
static void t_softmax_bw(Tensor *c) {
    Tensor *a = c->children[0];
    if (a->requires_grad) { ensure_grad(a);
        int rows = (a->ndim == 1) ? 1 : a->shape[0];
        int cols = (a->ndim == 1) ? a->size : a->shape[1];
        for (int r = 0; r < rows; r++) {
            float s = 0;
            for (int k = 0; k < cols; k++) { int idx = (a->ndim == 1) ? k : (r * cols + k); s += c->grad[idx] * c->data[idx]; }
            for (int k = 0; k < cols; k++) { int idx = (a->ndim == 1) ? k : (r * cols + k); a->grad[idx] += c->data[idx] * (c->grad[idx] - s); }
        }
    }
}
static Tensor *t_softmax(Tensor *a) {
    int rows = (a->ndim == 1) ? 1 : a->shape[0];
    int cols = (a->ndim == 1) ? a->size : a->shape[1];
    Tensor *c = tensor_new(a->ndim, a->shape);
    for (int r = 0; r < rows; r++) {
        float mx = -1e30f;
        for (int k = 0; k < cols; k++) { int idx = (a->ndim == 1) ? k : (r * cols + k); if (a->data[idx] > mx) mx = a->data[idx]; }
        float sum = 0;
        for (int k = 0; k < cols; k++) { int idx = (a->ndim == 1) ? k : (r * cols + k); float e = expf(a->data[idx] - mx); c->data[idx] = e; sum += e; }
        for (int k = 0; k < cols; k++) { int idx = (a->ndim == 1) ? k : (r * cols + k); c->data[idx] /= sum; }
    }
    c->requires_grad = a->requires_grad; c->backward = t_softmax_bw; tensor_set_child(c, a);
    return c;
}

/* ---------- 损失函数 ---------- */
static void t_mse_bw(Tensor *c) {
    Tensor *a = c->children[0], *b = c->children[1];
    int n = a->size; float inv = c->grad[0] / n;
    if (a->requires_grad) { ensure_grad(a);
        for (int i = 0; i < n; i++) a->grad[i] += 2.0f * (a->data[i] - b->data[i]) * inv;
    }
    if (b->requires_grad) { ensure_grad(b);
        for (int i = 0; i < n; i++) b->grad[i] += -2.0f * (a->data[i] - b->data[i]) * inv;
    }
}
static Tensor *t_mse(Tensor *a, Tensor *b) {
    if (a->size != b->size) return NULL;
    int sh[1] = { 1 }; Tensor *c = tensor_new(1, sh);
    float s = 0; for (int i = 0; i < a->size; i++) { float d = a->data[i] - b->data[i]; s += d * d; }
    c->data[0] = s / a->size;
    c->requires_grad = a->requires_grad || b->requires_grad; c->backward = t_mse_bw; tensor_set_child(c, a); tensor_set_child(c, b);
    return c;
}
static int ce_label(Tensor *lab, int r) {
    return (lab->ndim == 1) ? (int)lab->data[r] : (int)lab->data[r * lab->shape[1]];
}
static void t_ce_bw(Tensor *c) {
    Tensor *p = c->children[0], *lab = c->children[1];
    if (p->requires_grad) { ensure_grad(p);
        int rows = p->shape[0], cols = p->shape[1], N = rows;
        for (int r = 0; r < rows; r++) {
            int k = ce_label(lab, r);
            p->grad[r * cols + k] += -1.0f / (N * (p->data[r * cols + k] + 1e-12f)) * c->grad[0];
        }
    }
}
static Tensor *t_cross_entropy(Tensor *p, Tensor *lab) {
    if (p->ndim != 2) return NULL;
    int rows = p->shape[0], cols = p->shape[1], N = rows;
    int sh[1] = { 1 }; Tensor *c = tensor_new(1, sh);
    float s = 0;
    for (int r = 0; r < rows; r++) { int k = ce_label(lab, r); s += -logf(p->data[r * cols + k] + 1e-12f); }
    c->data[0] = s / N;
    c->requires_grad = p->requires_grad; c->backward = t_ce_bw; tensor_set_child(c, p); tensor_set_child(c, lab);
    return c;
}
static void t_cel_bw(Tensor *c) {
    Tensor *lg = c->children[0], *lab = c->children[1];
    if (lg->requires_grad) { ensure_grad(lg);
        int rows = lg->shape[0], cols = lg->shape[1], N = rows;
        float *prob = malloc(cols * sizeof(float));
        for (int r = 0; r < rows; r++) {
            float mx = -1e30f;
            for (int k = 0; k < cols; k++) { int idx = r * cols + k; if (lg->data[idx] > mx) mx = lg->data[idx]; }
            float sum = 0;
            for (int k = 0; k < cols; k++) { int idx = r * cols + k; float e = expf(lg->data[idx] - mx); prob[k] = e; sum += e; }
            int k = ce_label(lab, r);
            for (int kk = 0; kk < cols; kk++) { int idx = r * cols + kk; lg->grad[idx] += (prob[kk] / sum - (kk == k ? 1.0f : 0.0f)) / N * c->grad[0]; }
        }
        free(prob);
    }
}
static Tensor *t_cross_entropy_logits(Tensor *lg, Tensor *lab) {
    if (lg->ndim != 2) return NULL;
    int rows = lg->shape[0], cols = lg->shape[1], N = rows;
    int sh[1] = { 1 }; Tensor *c = tensor_new(1, sh);
    float s = 0;
    for (int r = 0; r < rows; r++) {
        float mx = -1e30f;
        for (int k = 0; k < cols; k++) { int idx = r * cols + k; if (lg->data[idx] > mx) mx = lg->data[idx]; }
        float sum = 0;
        for (int k = 0; k < cols; k++) { int idx = r * cols + k; sum += expf(lg->data[idx] - mx); }
        int k = ce_label(lab, r);
        s += -(lg->data[r * cols + k] - mx) + logf(sum);
    }
    c->data[0] = s / N;
    c->requires_grad = lg->requires_grad; c->backward = t_cel_bw; tensor_set_child(c, lg); tensor_set_child(c, lab);
    return c;
}

/* ---------- 反向传播引擎 ---------- */
static void topo_visit(Tensor *t, Tensor ***out, int *n, int *cap) {
    if (!t || t->_tag == 2) return;
    if (t->_tag == 1) return;
    t->_tag = 1;
    for (int i = 0; i < t->nchildren; i++) topo_visit(t->children[i], out, n, cap);
    t->_tag = 2;
    if (*n >= *cap) { *cap = *cap ? *cap * 2 : 16; *out = realloc(*out, (*cap) * sizeof(Tensor*)); }
    (*out)[(*n)++] = t;
}
static void tensor_backward(Tensor *root) {
    Tensor **order = NULL; int n = 0, cap = 0;
    topo_visit(root, &order, &n, &cap);
    for (int i = 0; i < n; i++) ensure_grad(order[i]);
    ensure_grad(root);
    for (int i = 0; i < root->size; i++) root->grad[i] = 1.0f;
    for (int i = n - 1; i >= 0; i--) if (order[i]->backward) order[i]->backward(order[i]);
    for (int i = 0; i < n; i++) order[i]->_tag = 0;
    root->_tag = 0;
    free(order);
}

/* 表达式层：两个张量的二元运算（自动求导） */
static Value *tensor_elem_binop(Node *n, const char *op, Value *l, Value *r) {
    Tensor *a = l->tval, *b = r->tval;
    Tensor *c = NULL;
    if (strcmp(op, "+") == 0) c = t_add(a, b);
    else if (strcmp(op, "-") == 0) c = t_sub(a, b);
    else if (strcmp(op, "*") == 0) c = t_mul(a, b);
    else if (strcmp(op, "/") == 0) c = t_div(a, b);
    else runtime_error(n, "无效的张量运算符: %s", op);
    if (!c) runtime_error(n, "张量形状无法广播（维度不匹配）");
    Value *tv = val_new(VAL_TENSOR); tv->tval = c;
    return tv;
}

/* 赋值到张量元素：t[i][j] = v 或 t[i] = v */
static int assign_tensor(Node *tgt, Value *val, Env *env) {
    if (tgt->type != ND_BINARY || strcmp(tgt->binary.op, "[]") != 0) return 0;
    Node *inner = tgt->binary.left;
    if (inner->type == ND_BINARY && strcmp(inner->binary.op, "[]") == 0 && inner->binary.left->type == ND_IDENT) {
        char *name = inner->binary.left->ident.name;
        Value *i1 = eval_node(inner->binary.right, env);
        Value *i2 = eval_node(tgt->binary.right, env);
        Value *base = env_get(env, name);
        if (base && base->type == VAL_TENSOR && base->tval->ndim == 2) {
            int r = (int)i1->ival, c = (int)i2->ival;
            int cols = base->tval->shape[1];
            if (r < 0 || r >= base->tval->shape[0] || c < 0 || c >= cols) runtime_error(tgt, "张量索引越界");
            double v = num_of(val, "张量元素赋值");
            base->tval->data[r * cols + c] = (float)v;
            val_free(i1); val_free(i2);
            return 1;
        }
        val_free(i1); val_free(i2);
        return 0;
    }
    if (inner->type == ND_IDENT) {
        Value *idx = eval_node(tgt->binary.right, env);
        Value *base = env_get(env, inner->ident.name);
        if (base && base->type == VAL_TENSOR && base->tval->ndim == 1) {
            int i = (int)idx_val(idx);
            if (i < 0 || i >= base->tval->size) runtime_error(tgt, "张量索引越界");
            double v = num_of(val, "张量元素赋值");
            base->tval->data[i] = (float)v;
            val_free(idx);
            return 1;
        }
        val_free(idx);
        return 0;
    }
    return 0;
}

/* ========== 张量 / ML 内置函数 ========== */
static double list_val(Value *v) {
    if (v->type == VAL_FLOAT) return v->fval;
    if (v->type == VAL_INT) return (double)v->ival;
    return 0.0;
}
static int *compute_shape(Value *v, int *ndim) {
    if (v->type != VAL_LIST) { *ndim = 0; return NULL; }
    int child_ndim = 0; int *child_shape = NULL;
    if (v->list_len > 0 && v->lval[0]->type == VAL_LIST)
        child_shape = compute_shape(v->lval[0], &child_ndim);
    *ndim = 1 + child_ndim;
    int *shape = malloc((*ndim) * sizeof(int));
    shape[0] = v->list_len;
    for (int i = 0; i < child_ndim; i++) shape[1 + i] = child_shape[i];
    free(child_shape);
    return shape;
}
static void fill_tensor(Tensor *t, Value *v, int *coord, int dim) {
    if (dim == t->ndim) {
        int flat = 0;
        for (int i = 0; i < t->ndim; i++) flat = flat * t->shape[i] + coord[i];
        t->data[flat] = (float)list_val(v);
        return;
    }
    for (int i = 0; i < v->list_len; i++) { coord[dim] = i; fill_tensor(t, v->lval[i], coord, dim + 1); }
}

/* 张量(嵌套列表) 构造；张量(一维数字列表) 构造 1D；张量(n) 构造 1D 零张量；张量(r,c) 构造 r×c 零张量 */
static Value *builtin_tensor(int argc, Value **argv) {
    if (argc == 1) {
        Value *a = argv[0];
        if (a->type == VAL_LIST) {
            if (a->list_len > 0 && a->lval[0]->type == VAL_LIST) {
                int ndim = 0; int *shape = compute_shape(a, &ndim);
                Tensor *t = tensor_new(ndim, shape);
                int *coord = calloc(ndim, sizeof(int));
                fill_tensor(t, a, coord, 0);
                free(coord); free(shape);
                Value *r = val_new(VAL_TENSOR); r->tval = t; return r;
            } else {
                int sh[1] = { a->list_len };
                Tensor *t = tensor_new(1, sh);
                for (int i = 0; i < a->list_len; i++) t->data[i] = (float)list_val(a->lval[i]);
                Value *r = val_new(VAL_TENSOR); r->tval = t; return r;
            }
        } else if (a->type == VAL_INT) {
            int sh[1] = { (int)a->ival };
            Tensor *t = tensor_new(1, sh);
            Value *r = val_new(VAL_TENSOR); r->tval = t; return r;
        }
    } else if (argc == 2 && argv[0]->type == VAL_INT && argv[1]->type == VAL_INT) {
        int sh[2] = { (int)argv[0]->ival, (int)argv[1]->ival };
        Tensor *t = tensor_new(2, sh);
        Value *r = val_new(VAL_TENSOR); r->tval = t; return r;
    }
    runtime_error(NULL, "张量构造需要: 嵌套列表 / 一维数字列表 / 单个整数 / 两个整数");
    return val_new(VAL_NULL);
}

static Value *builtin_zeros(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_LIST) runtime_error(NULL, "全零函数需要形状列表");
    int ndim = argv[0]->list_len;
    if (ndim < 1 || ndim > CO_MAX_DIM) runtime_error(NULL, "形状维数必须在 1..%d 之间，收到 %d", CO_MAX_DIM, ndim);
    int sh[CO_MAX_DIM]; int size = 1;   /* 栈数组：异常 longjmp 时不会漏 */
    for (int i = 0; i < ndim; i++) { sh[i] = dim_of(argv[0]->lval[i], "形状列表"); size *= sh[i]; }
    Tensor *c = tensor_new(ndim, sh);
    Value *r = val_new(VAL_TENSOR); r->tval = c; return r;
}
static Value *builtin_ones(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_LIST) runtime_error(NULL, "全一函数需要形状列表");
    int ndim = argv[0]->list_len;
    if (ndim < 1 || ndim > CO_MAX_DIM) runtime_error(NULL, "形状维数必须在 1..%d 之间，收到 %d", CO_MAX_DIM, ndim);
    int sh[CO_MAX_DIM]; int size = 1;   /* 栈数组：异常 longjmp 时不会漏 */
    for (int i = 0; i < ndim; i++) { sh[i] = dim_of(argv[0]->lval[i], "形状列表"); size *= sh[i]; }
    Tensor *c = tensor_new(ndim, sh);
    for (int i = 0; i < size; i++) c->data[i] = 1.0f;
    Value *r = val_new(VAL_TENSOR); r->tval = c; return r;
}
static Value *builtin_rand_tensor(int argc, Value **argv) {
    if (argc != 3 || argv[0]->type != VAL_LIST) runtime_error(NULL, "随机张量需要: 形状列表, 下限, 上限");
    int ndim = argv[0]->list_len;
    if (ndim < 1 || ndim > CO_MAX_DIM) runtime_error(NULL, "形状维数必须在 1..%d 之间，收到 %d", CO_MAX_DIM, ndim);
    int sh[CO_MAX_DIM]; int size = 1;   /* 栈数组：异常 longjmp 时不会漏 */
    for (int i = 0; i < ndim; i++) { sh[i] = dim_of(argv[0]->lval[i], "形状列表"); size *= sh[i]; }
    double lo = num_of(argv[1], "随机张量的下限");
    double hi = num_of(argv[2], "随机张量的上限");
    if (lo > hi) runtime_error(NULL, "随机张量的下限不能大于上限");
    Tensor *c = tensor_new(ndim, sh);
    for (int i = 0; i < size; i++) c->data[i] = (float)(lo + (double)rand() / RAND_MAX * (hi - lo));
    Value *r = val_new(VAL_TENSOR); r->tval = c; return r;
}
static Value *builtin_randn(int argc, Value **argv) {
    if (argc != 3 || argv[0]->type != VAL_LIST) runtime_error(NULL, "高斯随机需要: 形状列表, 均值, 标准差");
    int ndim = argv[0]->list_len;
    if (ndim < 1 || ndim > CO_MAX_DIM) runtime_error(NULL, "形状维数必须在 1..%d 之间，收到 %d", CO_MAX_DIM, ndim);
    int sh[CO_MAX_DIM]; int size = 1;   /* 栈数组：异常 longjmp 时不会漏 */
    for (int i = 0; i < ndim; i++) { sh[i] = dim_of(argv[0]->lval[i], "形状列表"); size *= sh[i]; }
    double mean = num_of(argv[1], "高斯随机的均值");
    double std  = num_of(argv[2], "高斯随机的标准差");
    if (std < 0) runtime_error(NULL, "高斯随机的标准差不能为负");
    Tensor *c = tensor_new(ndim, sh);
    for (int i = 0; i < size; i++) {
        double u1 = ((double)rand() + 1) / ((double)RAND_MAX + 1);
        double u2 = (double)rand() / RAND_MAX;
        double z = sqrt(-2 * log(u1)) * cos(2 * 3.14159265358979323846 * u2);
        c->data[i] = (float)(mean + std * z);
    }
    Value *r = val_new(VAL_TENSOR); r->tval = c; return r;
}
static Value *builtin_matmul(int argc, Value **argv) {
    if (argc != 2 || argv[0]->type != VAL_TENSOR || argv[1]->type != VAL_TENSOR) runtime_error(NULL, "矩阵乘需要两个张量");
    Tensor *c = t_matmul(argv[0]->tval, argv[1]->tval);
    if (!c) runtime_error(NULL, "矩阵乘形状不匹配");
    Value *r = val_new(VAL_TENSOR); r->tval = c; return r;
}
static Value *builtin_transpose(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "转置需要1个张量");
    Tensor *c = t_transpose(argv[0]->tval);
    if (!c) runtime_error(NULL, "转置仅支持二维张量");
    Value *r = val_new(VAL_TENSOR); r->tval = c; return r;
}
static Value *builtin_shape(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "形状函数需要1个张量");
    Tensor *t = argv[0]->tval;
    Value *r = val_new(VAL_LIST); r->list_len = t->ndim; r->lval = malloc(t->ndim * sizeof(Value*));
    for (int i = 0; i < t->ndim; i++) { Value *e = val_new(VAL_INT); e->ival = t->shape[i]; r->lval[i] = val_adopt(e); }
    return r;
}
static Value *builtin_reshape(int argc, Value **argv) {
    if (argc != 2 || argv[0]->type != VAL_TENSOR || argv[1]->type != VAL_LIST) runtime_error(NULL, "重塑函数需要: 张量, 形状列表");
    Tensor *t = argv[0]->tval;
    int ndim = argv[1]->list_len;
    if (ndim < 1 || ndim > CO_MAX_DIM) runtime_error(NULL, "形状维数必须在 1..%d 之间，收到 %d", CO_MAX_DIM, ndim);
    int sh[CO_MAX_DIM]; int size = 1;   /* 栈数组：异常 longjmp 时不会漏 */
    for (int i = 0; i < ndim; i++) { sh[i] = dim_of(argv[1]->lval[i], "形状列表"); size *= sh[i]; }
    if (size != t->size) runtime_error(NULL, "重塑前后元素总数不一致");
    Tensor *c = tensor_new(ndim, sh);
    memcpy(c->data, t->data, t->size * sizeof(float));
    Value *r = val_new(VAL_TENSOR); r->tval = c; return r;
}
/* 求和：张量 -> 标量张量；列表 -> 数值（多态，与 累加 等价） */
static Value *builtin_sum(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "求和需要1个参数（张量或列表）");
    if (argv[0]->type == VAL_LIST) return builtin_total(argc, argv);
    if (argv[0]->type != VAL_TENSOR) runtime_error(NULL, "求和需要1个张量或列表");
    Value *r = val_new(VAL_TENSOR); r->tval = t_sum(argv[0]->tval); return r;
}
/* 均值：张量 -> 标量张量；列表 -> 浮点平均值（多态） */
static Value *builtin_mean(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "均值需要1个参数（张量或列表）");
    if (argv[0]->type == VAL_LIST) {
        if (argv[0]->list_len == 0) runtime_error(NULL, "均值不能作用于空列表");
        double s = 0;
        for (int i = 0; i < argv[0]->list_len; i++) s += num_of(argv[0]->lval[i], "均值");
        return val_flt(s / argv[0]->list_len);
    }
    if (argv[0]->type != VAL_TENSOR) runtime_error(NULL, "均值需要1个张量或列表");
    Value *r = val_new(VAL_TENSOR); r->tval = t_mean(argv[0]->tval); return r;
}
static Value *builtin_exp(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "指数需要1个张量");
    Value *r = val_new(VAL_TENSOR); r->tval = t_exp(argv[0]->tval); return r;
}
static Value *builtin_log(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "对数需要1个张量");
    Value *r = val_new(VAL_TENSOR); r->tval = t_log(argv[0]->tval); return r;
}
static Value *builtin_sigmoid(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "西格莫德需要1个张量");
    Value *r = val_new(VAL_TENSOR); r->tval = t_sigmoid(argv[0]->tval); return r;
}
static Value *builtin_tanh(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "双曲正切需要1个张量");
    Value *r = val_new(VAL_TENSOR); r->tval = t_tanh(argv[0]->tval); return r;
}
static Value *builtin_relu(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "线性整流需要1个张量");
    Value *r = val_new(VAL_TENSOR); r->tval = t_relu(argv[0]->tval); return r;
}
static Value *builtin_softmax(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "柔性最大值需要1个张量");
    Value *r = val_new(VAL_TENSOR); r->tval = t_softmax(argv[0]->tval); return r;
}
static Value *builtin_mse(int argc, Value **argv) {
    if (argc != 2 || argv[0]->type != VAL_TENSOR || argv[1]->type != VAL_TENSOR) runtime_error(NULL, "均方误差需要两个张量");
    Tensor *c = t_mse(argv[0]->tval, argv[1]->tval);
    if (!c) runtime_error(NULL, "均方误差张量大小不一致");
    Value *r = val_new(VAL_TENSOR); r->tval = c; return r;
}
static Value *builtin_cross_entropy(int argc, Value **argv) {
    if (argc != 2 || argv[0]->type != VAL_TENSOR || argv[1]->type != VAL_TENSOR) runtime_error(NULL, "交叉熵需要: 概率张量, 标签张量");
    Tensor *c = t_cross_entropy(argv[0]->tval, argv[1]->tval);
    if (!c) runtime_error(NULL, "交叉熵概率张量需为二维");
    Value *r = val_new(VAL_TENSOR); r->tval = c; return r;
}
static Value *builtin_cross_entropy_logits(int argc, Value **argv) {
    if (argc != 2 || argv[0]->type != VAL_TENSOR || argv[1]->type != VAL_TENSOR) runtime_error(NULL, "对数交叉熵需要: 对数张量, 标签张量");
    Tensor *c = t_cross_entropy_logits(argv[0]->tval, argv[1]->tval);
    if (!c) runtime_error(NULL, "对数交叉熵对数张量需为二维");
    Value *r = val_new(VAL_TENSOR); r->tval = c; return r;
}
static Value *builtin_backward(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "反向函数需要1个损失张量");
    tensor_backward(argv[0]->tval);
    return val_new(VAL_NULL);
}
static Value *builtin_optim(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "优化函数需要2个参数: 学习率, 参数列表");
    double lr = num_of(argv[0], "优化函数的学习率");
    if (argv[1]->type != VAL_LIST) runtime_error(NULL, "优化函数第二个参数必须是参数列表");
    for (int i = 0; i < argv[1]->list_len; i++) {
        Value *p = argv[1]->lval[i];
        if (p->type == VAL_TENSOR && p->tval->requires_grad && p->tval->grad) {
            Tensor *t = p->tval;
            for (int j = 0; j < t->size; j++) t->data[j] -= (float)(lr * t->grad[j]);
            memset(t->grad, 0, t->size * sizeof(float));
        }
    }
    return val_new(VAL_NULL);
}
/* 张量索引 + 越界检查。
   原先 取/置 直接用 t->data[i] 而不检查 i，负索引或超界会
   越界读写堆内存——是可被利用的内存安全漏洞，不只是"结果错"。 */
static int tensor_linear(Tensor *t, Value *iv, const char *who) {
    long i = int_of(iv, who);
    if (i < 0) i += t->size;                       /* 支持负索引 */
    if (i < 0 || i >= t->size)
        runtime_error(NULL, "%s 的索引越界：%ld（共 %d 个元素）", who, i, t->size);
    return (int)i;
}
static int tensor_offset_2d(Tensor *t, Value *rv, Value *cv, const char *who) {
    long rows = t->shape[0], cols = t->shape[1];
    long i = int_of(rv, who);
    long j = cv ? int_of(cv, who) : 0;
    if (i < 0) i += rows;
    if (j < 0) j += cols;
    if (i < 0 || i >= rows) runtime_error(NULL, "%s 的行索引越界：%ld（共 %ld 行）", who, i, rows);
    if (j < 0 || j >= cols) runtime_error(NULL, "%s 的列索引越界：%ld（共 %ld 列）", who, j, cols);
    return (int)(i * cols + j);
}
static Value *builtin_get(int argc, Value **argv) {
    if (argc < 2 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "取函数需要: 张量, 索引...");
    Tensor *t = argv[0]->tval;
    int off = (t->ndim == 2) ? tensor_offset_2d(t, argv[1], (argc >= 3) ? argv[2] : NULL, "取函数")
                             : tensor_linear(t, argv[1], "取函数");
    Value *r = val_new(VAL_FLOAT); r->fval = t->data[off]; return r;
}
static Value *builtin_set(int argc, Value **argv) {
    if (argc < 3 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "置函数需要: 张量, 索引, 值");
    Tensor *t = argv[0]->tval;
    double v = num_of(argv[argc - 1], "置函数的值");
    /* 置(张量, 行, 列, 值) 走二维定位；置(张量, i, 值) 走线性偏移 */
    int off = (t->ndim == 2 && argc >= 4) ? tensor_offset_2d(t, argv[1], argv[2], "置函数")
                                          : tensor_linear(t, argv[1], "置函数");
    t->data[off] = (float)v;
    return val_new(VAL_NULL);
}
static Value *builtin_requires_grad(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "需梯度函数需要1个张量");
    argv[0]->tval->requires_grad = 1;
    Value *r = val_new(VAL_TENSOR); r->tval = tensor_retain(argv[0]->tval); return r;
}

/* 断言：条件为假立即报错退出，是自动化测试的基石 */
static Value *builtin_assert(int argc, Value **argv) {
    if (argc < 1) runtime_error(NULL, "断言函数需要至少1个参数");
    int ok = truthy(argv[0]);
    if (!ok) {
        const char *msg = (argc >= 2 && argv[1]->type == VAL_STRING) ? argv[1]->sval : "（无说明）";
        fprintf(stderr, "断言失败: %s\n", msg);
        exit(1);
    }
    return val_new(VAL_NULL);
}

/* 相等断言：不相等时打印期望值与实际值，便于定位 */
static Value *builtin_assert_eq(int argc, Value **argv) {
    if (argc < 2) runtime_error(NULL, "断言相等函数需要至少2个参数");
    char *a = val_to_string(argv[0]);
    char *b = val_to_string(argv[1]);
    int eq;
    if (is_numeric_val(argv[0]) && is_numeric_val(argv[1])) {
        eq = fabs(numeric_of(argv[0]) - numeric_of(argv[1])) < 1e-9;  /* 数值按容差比较，避免浮点抖动 */
    } else {
        eq = val_equals(argv[0], argv[1]);   /* 结构化比较：列表/字典/张量都能正确判等 */
    }
    if (!eq) {
        const char *msg = (argc >= 3 && argv[2]->type == VAL_STRING) ? argv[2]->sval : "（无说明）";
        fprintf(stderr, "断言失败: %s\n  期望: %s\n  实际: %s\n", msg, b, a);
        free(a); free(b);
        exit(1);
    }
    free(a); free(b);
    return val_new(VAL_NULL);
}

/* 调用内置函数 */
static Value *call_builtin(const char *name, int argc, Value **argv) {
    if (strcmp(name, "断言") == 0) return builtin_assert(argc, argv);
    if (strcmp(name, "断言相等") == 0) return builtin_assert_eq(argc, argv);
    if (strcmp(name, "长度") == 0) return builtin_len(argc, argv);
    if (strcmp(name, "类型") == 0) return builtin_type(argc, argv);
    if (strcmp(name, "转布尔") == 0) return builtin_to_bool(argc, argv);
    if (strcmp(name, "转整数") == 0) return builtin_int(argc, argv);
    if (strcmp(name, "转浮点") == 0) return builtin_float(argc, argv);
    if (strcmp(name, "转字符串") == 0) return builtin_str(argc, argv);
    if (strcmp(name, "输入") == 0) return builtin_input(argc, argv);
    /* 数学函数 */
    if (strcmp(name, "绝对值") == 0) return builtin_abs(argc, argv);
    if (strcmp(name, "平方根") == 0) return builtin_sqrt(argc, argv);
    if (strcmp(name, "随机数") == 0) return builtin_random(argc, argv);
    if (strcmp(name, "时钟") == 0) return builtin_now(argc, argv);
    if (strcmp(name, "取整") == 0) return builtin_round(argc, argv);
    if (strcmp(name, "向下取整") == 0) return builtin_floor(argc, argv);
    if (strcmp(name, "向上取整") == 0) return builtin_ceil(argc, argv);
    if (strcmp(name, "截断") == 0) return builtin_trunc(argc, argv);
    if (strcmp(name, "最大值") == 0) return builtin_max(argc, argv);
    if (strcmp(name, "最小值") == 0) return builtin_min(argc, argv);
    if (strcmp(name, "幂") == 0) return builtin_pow(argc, argv);
    if (strcmp(name, "正弦") == 0) return builtin_sin(argc, argv);
    if (strcmp(name, "余弦") == 0) return builtin_cos(argc, argv);
    if (strcmp(name, "正切") == 0) return builtin_tan(argc, argv);
    if (strcmp(name, "自然对数") == 0) return builtin_ln(argc, argv);
    if (strcmp(name, "圆周率") == 0) return builtin_pi(argc, argv);
    if (strcmp(name, "符号") == 0) return builtin_sign(argc, argv);
    if (strcmp(name, "随机整数") == 0) return builtin_rand_int(argc, argv);
    /* 进制与位工具 */
    if (strcmp(name, "十六进制") == 0) return builtin_hex(argc, argv);
    if (strcmp(name, "二进制") == 0) return builtin_bin(argc, argv);
    if (strcmp(name, "八进制") == 0) return builtin_oct(argc, argv);
    if (strcmp(name, "位计数") == 0) return builtin_popcount(argc, argv);
    if (strcmp(name, "取位") == 0) return builtin_getbit(argc, argv);
    if (strcmp(name, "置位") == 0) return builtin_setbit(argc, argv);
    /* 列表函数 */
    if (strcmp(name, "追加") == 0) return builtin_append(argc, argv);
    if (strcmp(name, "删除") == 0) return builtin_remove(argc, argv);
    if (strcmp(name, "排序") == 0) return builtin_sort(argc, argv);
    if (strcmp(name, "包含") == 0) return builtin_contains(argc, argv);
    if (strcmp(name, "范围") == 0) return builtin_range(argc, argv);
    if (strcmp(name, "反转") == 0) return builtin_reverse(argc, argv);
    if (strcmp(name, "索引") == 0) return builtin_index_of(argc, argv);
    if (strcmp(name, "插入") == 0) return builtin_insert(argc, argv);
    if (strcmp(name, "弹出") == 0) return builtin_pop(argc, argv);
    if (strcmp(name, "唯一") == 0) return builtin_unique(argc, argv);
    if (strcmp(name, "计数") == 0) return builtin_count(argc, argv);
    if (strcmp(name, "切片") == 0) return builtin_slice(argc, argv);
    if (strcmp(name, "扁平") == 0) return builtin_flatten(argc, argv);
    if (strcmp(name, "累加") == 0) return builtin_total(argc, argv);
    /* 字典函数 */
    if (strcmp(name, "键") == 0) return builtin_keys(argc, argv);
    if (strcmp(name, "值") == 0) return builtin_values(argc, argv);
    if (strcmp(name, "移除键") == 0) return builtin_del_key(argc, argv);
    if (strcmp(name, "项") == 0) return builtin_items(argc, argv);
    /* 高阶函数 */
    if (strcmp(name, "映射") == 0) return builtin_map_fn(argc, argv);
    if (strcmp(name, "过滤") == 0) return builtin_filter_fn(argc, argv);
    if (strcmp(name, "归约") == 0) return builtin_reduce_fn(argc, argv);
    if (strcmp(name, "全部满足") == 0) return builtin_all_fn(argc, argv);
    if (strcmp(name, "任一满足") == 0) return builtin_any_fn(argc, argv);
    /* 字符串函数 */
    if (strcmp(name, "分割") == 0) return builtin_split(argc, argv);
    if (strcmp(name, "连接") == 0) return builtin_join(argc, argv);
    if (strcmp(name, "替换") == 0) return builtin_replace(argc, argv);
    if (strcmp(name, "查找") == 0) return builtin_find(argc, argv);
    if (strcmp(name, "截取") == 0) return builtin_substring(argc, argv);
    if (strcmp(name, "去空白") == 0) return builtin_trim(argc, argv);
    if (strcmp(name, "去左空白") == 0) return builtin_ltrim(argc, argv);
    if (strcmp(name, "去右空白") == 0) return builtin_rtrim(argc, argv);
    if (strcmp(name, "大写") == 0) return builtin_upper(argc, argv);
    if (strcmp(name, "小写") == 0) return builtin_lower(argc, argv);
    if (strcmp(name, "重复") == 0) return builtin_repeat(argc, argv);
    if (strcmp(name, "开头是") == 0) return builtin_startswith(argc, argv);
    if (strcmp(name, "结尾是") == 0) return builtin_endswith(argc, argv);
    if (strcmp(name, "字符码") == 0) return builtin_char_code(argc, argv);
    if (strcmp(name, "码转字符") == 0) return builtin_from_code(argc, argv);
    /* 字节级视图（编码/散列/协议解析用） */
    if (strcmp(name, "字节数") == 0) return builtin_byte_len(argc, argv);
    if (strcmp(name, "字节") == 0) return builtin_byte_at(argc, argv);
    if (strcmp(name, "字节列表") == 0) return builtin_bytes_of(argc, argv);
    if (strcmp(name, "字节转文本") == 0) return builtin_bytes_to_str(argc, argv);
    /* 文件读写 */
    if (strcmp(name, "读文件") == 0) return builtin_read_file(argc, argv);
    if (strcmp(name, "写文件") == 0) return builtin_write_file(argc, argv);
    if (strcmp(name, "追加文件") == 0) return builtin_append_file(argc, argv);
    if (strcmp(name, "文件存在") == 0) return builtin_file_exists(argc, argv);
    if (strcmp(name, "删除文件") == 0) return builtin_delete_file(argc, argv);
    if (strcmp(name, "读取行") == 0) return builtin_read_lines(argc, argv);
    /* 张量 / 模型开发 */
    if (strcmp(name, "张量") == 0) return builtin_tensor(argc, argv);
    if (strcmp(name, "全零") == 0) return builtin_zeros(argc, argv);
    if (strcmp(name, "全一") == 0) return builtin_ones(argc, argv);
    if (strcmp(name, "随机张量") == 0) return builtin_rand_tensor(argc, argv);
    if (strcmp(name, "高斯随机") == 0) return builtin_randn(argc, argv);
    if (strcmp(name, "矩阵乘") == 0) return builtin_matmul(argc, argv);
    if (strcmp(name, "转置") == 0) return builtin_transpose(argc, argv);
    if (strcmp(name, "形状") == 0) return builtin_shape(argc, argv);
    if (strcmp(name, "重塑") == 0) return builtin_reshape(argc, argv);
    if (strcmp(name, "求和") == 0) return builtin_sum(argc, argv);
    if (strcmp(name, "均值") == 0) return builtin_mean(argc, argv);
    if (strcmp(name, "指数") == 0) return builtin_exp(argc, argv);
    if (strcmp(name, "对数") == 0) return builtin_log(argc, argv);
    if (strcmp(name, "西格莫德") == 0) return builtin_sigmoid(argc, argv);
    if (strcmp(name, "双曲正切") == 0) return builtin_tanh(argc, argv);
    if (strcmp(name, "线性整流") == 0) return builtin_relu(argc, argv);
    if (strcmp(name, "柔性最大值") == 0) return builtin_softmax(argc, argv);
    if (strcmp(name, "均方误差") == 0) return builtin_mse(argc, argv);
    if (strcmp(name, "交叉熵") == 0) return builtin_cross_entropy(argc, argv);
    if (strcmp(name, "对数交叉熵") == 0) return builtin_cross_entropy_logits(argc, argv);
    if (strcmp(name, "反向") == 0) return builtin_backward(argc, argv);
    if (strcmp(name, "优化") == 0) return builtin_optim(argc, argv);
    if (strcmp(name, "取") == 0) return builtin_get(argc, argv);
    if (strcmp(name, "置") == 0) return builtin_set(argc, argv);
    if (strcmp(name, "需梯度") == 0) return builtin_requires_grad(argc, argv);

    runtime_error(NULL, "未知内置函数: %s", name);
    return NULL;
}

/* ========== 类型系统（渐进式静态类型检查） ========== */
static const char *val_type_name(Value *v) {
    switch (v->type) {
        case VAL_NULL:   return "空";
        case VAL_BOOL:   return "布尔";
        case VAL_INT:    return "整数";
        case VAL_FLOAT:  return "浮点";
        case VAL_STRING: return "字符串";
        case VAL_LIST:   return "列表";
        case VAL_MAP:    return "字典";
        case VAL_FUNC:   return "函数";
        case VAL_TENSOR: return "张量";
        default:         return "未知";
    }
}

/* 值是否符合给定类型标注（标注为 NULL 表示动态，永远通过） */
static int type_matches(const char *annot, Value *v) {
    if (!annot) return 1;
    if (strcmp(annot, "整数") == 0)   return v->type == VAL_INT;
    /* 数值加宽：整数可赋给 浮点（无精度损失），反向不允许。
       这与 C/Java/Go 的隐式提升一致，避免写 安全比值(1, 2) 这类调用被无谓拒绝。 */
    if (strcmp(annot, "浮点") == 0)   return v->type == VAL_FLOAT || v->type == VAL_INT;
    if (strcmp(annot, "数字") == 0)   return v->type == VAL_FLOAT || v->type == VAL_INT;
    if (strcmp(annot, "字符串") == 0) return v->type == VAL_STRING;
    if (strcmp(annot, "布尔") == 0)   return v->type == VAL_BOOL;
    if (strcmp(annot, "列表") == 0)   return v->type == VAL_LIST;
    if (strcmp(annot, "字典") == 0)   return v->type == VAL_MAP;
    if (strcmp(annot, "张量") == 0)   return v->type == VAL_TENSOR;
    if (strcmp(annot, "函数") == 0)   return v->type == VAL_FUNC;
    if (strcmp(annot, "任意") == 0)   return 1;
    if (strcmp(annot, "空") == 0)     return v->type == VAL_NULL;
    return 1; /* 未知标注，放行 */
}

static void ensure_type(const char *annot, Value *v, Node *n, const char *where) {
    if (!annot) return;
    if (!type_matches(annot, v)) {
        runtime_error(n, "类型错误（%s）：期望 %s，得到 %s", where, annot, val_type_name(v));
    }
}

/* 当前函数返回类型（用于返回语句检查；线程局部） */
static __thread const char *g_cur_ret_type = NULL;

/* ========== 并发原语（并发 spawn / 等待 join） ========== */
static void *task_worker(void *p) {
    Task *t = (Task*)p;
    /* 栈守卫是线程局部的：每个任务线程必须自己立基准，
       否则 g_stack_base 为 NULL，该线程内的无限递归将直接爆栈。 */
    stack_guard_init(0);
    /* 控制流标志是线程局部的，此处天然从干净状态开始 */
    t->result = eval_node(t->node, t->env);
    /* 释放任务隔离环境（返回值已独立持有引用，不依赖环境） */
    for (int i = 0; i < t->env->count; i++) {
        free(t->env->names[i]);
        if (t->env->types[i]) free(t->env->types[i]);
        val_free(t->env->values[i]);
    }
    free(t->env);
    t->env = NULL;
    return NULL;
}

/* 函数提升（hoisting）：在执行一段语句序列之前，先把其中所有顶层
   `函数 ...` 定义登记进环境，这样调用顺序就与书写顺序无关
   —— 互相递归、"主流程写在最前面" 等写法都能自然成立。 */
static void hoist_functions(Node *prog, Env *env) {
    if (!prog || prog->type != ND_PROGRAM) return;
    for (int i = 0; i < prog->prog.cnt; i++) {
        Node *s = prog->prog.stmts[i];
        if (!s || s->type != ND_FUNC_DEF) continue;
        Func *fn = malloc(sizeof(Func));
        if (!fn) { fprintf(stderr, "内存不足: 函数提升\n"); exit(1); }
        fn->name        = strdup(s->func_def.name);
        fn->params      = s->func_def.pnames;
        fn->param_count = s->func_def.pcnt;
        fn->ptypes      = s->func_def.ptypes;
        fn->ret_type    = s->func_def.ret_type;
        fn->body        = s->func_def.body;
        Value *fnv = val_new(VAL_FUNC);
        fnv->fnval = fn;
        env_set(env, fn->name, fnv);
        val_free(fnv);
    }
}

/* 释放一个调用/任务环境（其中的名字、类型标注、值均由该环境持有） */
static void env_release(Env *e) {
    if (!e) return;
    reflog_untrack_env(e);   /* 正常释放：从异常回滚名单里摘除 */
    for (int i = 0; i < e->count; i++) {
        free(e->names[i]);
        if (e->types[i]) free(e->types[i]);
        val_free(e->values[i]);
    }
    free(e);
}

/* ========== 栈守卫 ==========
   原先仅用「固定 3000 层调用深度」防无限递归，这不安全：
   - 栈帧大小随编译选项浮动：净化器构建下帧大数倍，3000 层就已真的爆栈；
   - 并发任务线程的栈通常远小于主线程；
   - 深度嵌套的表达式同样吃栈，却完全不计入「调用层数」。

   实现要点（两次踩坑换来的）：
   1) 判定必须用 __builtin_frame_address(0)，不能用「局部变量取地址」——
      ASan 会把取地址的局部变量搬到堆上的 fake stack，量出来的"栈用量"
      会忽大忽小甚至倒退，守卫形同失效。
   2) 栈区间要问 pthread_getattr_np 拿真值，而不是拿 RLIMIT_STACK 猜；
      线程栈和进程栈上限常常并不一致。
   3) 检查点要覆盖全部递归热点：invoke_func / eval_node / exec_node。
   目标：无限递归得到一条中文诊断（还能被 尝试/捕获 接住），而不是 SIGSEGV。 */
static __thread char *g_stack_redline = NULL;   /* 帧地址触到这里即判定「快见底」 */

static void stack_guard_init(size_t hint) {
    void  *lo = NULL;
    size_t sz = 0;
    pthread_attr_t at;
    if (pthread_getattr_np(pthread_self(), &at) == 0) {
        if (pthread_attr_getstack(&at, &lo, &sz) != 0) { lo = NULL; sz = 0; }
        pthread_attr_destroy(&at);
    }
    if (lo && sz > (1u << 20)) {
        /* lo 是栈的最低可用地址，栈自 lo+sz 向低地址生长。
           预留 1/8（至少 1 MB）：报错路径本身也要栈。 */
        size_t reserve = sz / 8;
        if (reserve < (1u << 20)) reserve = 1u << 20;
        g_stack_redline = (char *)lo + reserve;
    } else {
        /* 退化路径：以当前帧为基准，按 RLIMIT_STACK 估算可用额度 */
        size_t total = hint ? hint : (8u << 20);
        struct rlimit rl;
        if (!hint && getrlimit(RLIMIT_STACK, &rl) == 0 &&
            rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur > (1u << 20))
            total = (size_t)rl.rlim_cur;
        g_stack_redline = (char *)__builtin_frame_address(0) - (total - total / 4);
    }
}

/* 递归深度上限：仅作为「明显失控」的第二道防线，真正把关的是栈红线。
   有了栈守卫，这个值可以放得很高，让合法的深递归（如深树遍历）跑得通。 */
#define CO_MAX_CALL_DEPTH 200000
static __thread int g_call_depth = 0;

static void stack_guard_check(struct Node *site) {
    if (g_stack_redline && (char *)__builtin_frame_address(0) <= g_stack_redline)
        runtime_error(site, "调用栈即将耗尽（剩余栈空间不足预留量），请检查是否存在无限递归");
}

/* 统一的用户函数调用入口
   —— 原先有两条重复实现，其中表达式调用那条不支持 g_return_flag，
      导致 `如果 ... 返回` 在函数值调用时失效；此处合并为一条并补齐：
      1) 参数个数/类型检查   2) 返回值语义统一
      3) 循环上下文隔离（函数体内的 跳出/继续 不再污染调用处的循环）
      4) 递归深度保护 */
static Value *invoke_func(Func *fn, Value **args, int argc, Node *site, Env *env) {
    if (fn->param_count != argc)
        runtime_error(site, "函数 %s 参数数量错误: 期望 %d, 得到 %d",
                      fn->name ? fn->name : "匿名", fn->param_count, argc);
    /* 双重保护：真实栈余量（主）+ 层数上限（兜底，防计数溢出） */
    stack_guard_check(site);
    if (++g_call_depth > CO_MAX_CALL_DEPTH) {
        g_call_depth--;
        runtime_error(site, "递归过深（超过 %d 层），请检查是否存在无限递归", CO_MAX_CALL_DEPTH);
    }
    for (int i = 0; i < argc; i++)
        if (fn->ptypes && fn->ptypes[i]) ensure_type(fn->ptypes[i], args[i], site, "函数参数");

    Env *call_env = env_new(env);
    /* 若外层有 尝试 块，函数体内抛出的异常会 longjmp 越过下面的
       env_release —— 登记后由回滚统一释放，杜绝作用域泄漏。 */
    reflog_track_env(call_env);
    for (int i = 0; i < argc; i++) {
        env_set(call_env, fn->params[i], args[i]);   /* env_set 内部 retain，实参仍由调用方释放 */
        if (fn->ptypes && fn->ptypes[i]) env_set_type(call_env, fn->params[i], fn->ptypes[i]);
    }

    LoopContext *saved_loop  = loop_ctx;
    const char  *saved_rtype = g_cur_ret_type;
    int          saved_flag  = g_return_flag;
    Value       *saved_rval  = g_return_value;
    loop_ctx       = NULL;          /* 函数体不属于调用处的循环 */
    g_cur_ret_type = fn->ret_type;
    g_return_flag  = 0;
    g_return_value = NULL;

    Node *body = (Node*)fn->body;
    for (int i = 0; i < body->prog.cnt; i++) {
        if (g_return_flag) break;
        exec_node(body->prog.stmts[i], call_env);
    }
    Value *ret = g_return_flag ? g_return_value : NULL;

    g_return_flag  = saved_flag;
    g_return_value = saved_rval;
    g_cur_ret_type = saved_rtype;
    loop_ctx       = saved_loop;
    g_call_depth--;

    env_release(call_env);
    return ret ? ret : val_new(VAL_NULL);
}

/* 供高阶内置函数（映射/过滤/归约…）回调用户函数 */
static Value *call_fn_value(Value *fnval, Value **args, int argc) {
    if (!fnval || fnval->type != VAL_FUNC) runtime_error(NULL, "传入的不是函数");
    return invoke_func(fnval->fnval, args, argc, NULL, g_root_env);
}

static Value *eval_node(Node *n, Env *env) {
    stack_guard_check(n);   /* 表达式递归同样吃栈，必须设卡 */
    if (!n) return val_new(VAL_NULL);
    switch (n->type) {
        case ND_LITERAL: {
            Value *v = val_new(n->literal.lit_type);
            switch (n->literal.lit_type) {
                case VAL_INT: v->ival = atol(n->literal.value); break;
                case VAL_BOOL: v->ival = (n->literal.value[0] == '1'); break;
                case VAL_FLOAT: v->fval = atof(n->literal.value); break;
                case VAL_STRING: v->sval = strdup(n->literal.value); break;
                default: break;
            }
            return v;
        }
        case ND_IDENT: {
            Value *v = env_get(env, n->ident.name);
            if (!v) runtime_error(n, "未定义变量: %s", n->ident.name);
            return val_retain(v);
        }
        case ND_BINARY: {
            Value *l = eval_node(n->binary.left, env);
            
            if (strcmp(n->binary.op, "[]") == 0) {
                Value *idx = eval_node(n->binary.right, env);
                if (l->type == VAL_LIST) {
                    int i0 = (int)idx_val(idx), i = i0;
                    if (i < 0) i += l->list_len;      /* 负索引：-1 表示最后一个元素 */
                    if (i < 0 || i >= l->list_len) runtime_error(n, "列表索引越界: %d (列表长度 %d)", i0, l->list_len);
                    /* 先 retain 元素再释放容器：顺序颠倒会把元素一起释放掉 */
                    Value *elem = val_retain(l->lval[i]);
                    val_free(idx);
                    val_free(l);
                    return elem;
                } else if (l->type == VAL_STRING) {
                    int i = (int)idx_val(idx);
                    /* 按【字符】取值：支持负索引，取出完整 UTF-8 序列（不会切碎汉字） */
                    long slen = utf8_strlen(l->sval);
                    long ci = i < 0 ? i + slen : i;
                    char *ch = (ci >= 0 && ci < slen) ? utf8_char_at(l->sval, ci) : NULL;
                    if (!ch) runtime_error(n, "字符串索引越界: %d（长度 %ld）", i, slen);
                    val_free(idx);
                    Value *r = val_new(VAL_STRING);
                    r->sval = ch;
                    val_free(l);
                    return r;
                } else if (l->type == VAL_MAP) {
                    if (idx->type != VAL_STRING) runtime_error(n, "字典键必须是字符串");
                    Value *v = dict_get(l->mval, idx->sval);
                    val_free(idx);
                    val_free(l);
                    if (!v) return val_new(VAL_NULL);
                    return val_retain(v);
                } else if (l->type == VAL_TENSOR) {
                    int i = (int)idx_val(idx);
                    Tensor *T = l->tval;
                    if (T->ndim == 1) {
                        if (i < 0 || i >= T->size) runtime_error(n, "张量索引越界");
                        Value *r = val_new(VAL_FLOAT); r->fval = T->data[i];
                        val_free(idx); val_free(l); return r;
                    } else if (T->ndim == 2) {
                        if (i < 0 || i >= T->shape[0]) runtime_error(n, "张量索引越界");
                        int cols = T->shape[1];
                        Tensor *row = tensor_new(1, &cols);
                        for (int c = 0; c < cols; c++) row->data[c] = T->data[i * cols + c];
                        Value *r = val_new(VAL_TENSOR); r->tval = row;
                        val_free(idx); val_free(l); return r;
                    }
                    val_free(idx); val_free(l);
                    runtime_error(n, "不支持的张量维度索引");
                }
                val_free(idx); val_free(l);
                runtime_error(n, "不支持的类型索引");
            }
            
            /* 逻辑运算符（短路求值，结果是布尔） */
            if (strcmp(n->binary.op, "且") == 0) {
                if (!truthy(l)) { val_free(l); return val_bool(0); }
                Value *r = eval_node(n->binary.right, env);
                int ri = truthy(r);
                val_free(l); val_free(r);
                return val_bool(ri);
            }
            if (strcmp(n->binary.op, "或") == 0) {
                if (truthy(l)) { val_free(l); return val_bool(1); }
                Value *r = eval_node(n->binary.right, env);
                int ri = truthy(r);
                val_free(l); val_free(r);
                return val_bool(ri);
            }
            
            Value *r = eval_node(n->binary.right, env);

            /* 比较运算符：必须在字符串拼接/数值分支之前处理。
               否则 "abc" == "xyz" 会被当成字符串拼接（结果非空 → 恒为真），
               列表/字典比较则会误读联合体。此处统一交给 val_equals / val_cmp，
               对任意类型都给出正确语义，结果一律为布尔。 */
            {
                const char *bop = n->binary.op;
                int is_eq  = (strcmp(bop, "==") == 0), is_ne = (strcmp(bop, "!=") == 0);
                int is_lt  = (strcmp(bop, "<")  == 0), is_gt = (strcmp(bop, ">")  == 0);
                int is_le  = (strcmp(bop, "<=") == 0), is_ge = (strcmp(bop, ">=") == 0);
                if (is_eq || is_ne) {
                    int eq = val_equals(l, r);
                    val_free(l); val_free(r);
                    return val_bool(is_eq ? eq : !eq);
                }
                if (is_lt || is_gt || is_le || is_ge) {
                    /* 大小比较要求同类可比：数值之间、字符串之间、列表之间 */
                    int ok = (is_numeric_val(l) && is_numeric_val(r)) ||
                             (l->type == VAL_STRING && r->type == VAL_STRING) ||
                             (l->type == VAL_LIST   && r->type == VAL_LIST);
                    if (!ok) {
                        const char *ln = val_type_name(l), *rn = val_type_name(r);
                        val_free(l); val_free(r);
                        runtime_error(n, "无法比较大小: %s %s %s", ln, bop, rn);
                    }
                    int c = val_cmp(l, r);
                    val_free(l); val_free(r);
                    return val_bool(is_lt ? c < 0 : is_gt ? c > 0 : is_le ? c <= 0 : c >= 0);
                }
            }
            /* 位运算：& | ^ << >>。
               只接受整数/布尔——浮点的位模式没有语言级意义，静默截断是「能跑但结果错」的温床，
               因此一律报错并指引用户显式 取整。移位位数越界会触发 C 的未定义行为，
               这里先拦截为中文错误；左移用无符号运算避免有符号溢出 UB，
               右移手写算术移位保证负数行为在所有平台一致（不依赖实现定义）。 */
            {
                const char *bop = n->binary.op;
                int is_band = (strcmp(bop, "&")  == 0), is_bor  = (strcmp(bop, "|")  == 0);
                int is_bxor = (strcmp(bop, "^")  == 0);
                int is_shl  = (strcmp(bop, "<<") == 0), is_shr  = (strcmp(bop, ">>") == 0);
                if (is_band || is_bor || is_bxor || is_shl || is_shr) {
                    int lok = (l->type == VAL_INT || l->type == VAL_BOOL);
                    int rok = (r->type == VAL_INT || r->type == VAL_BOOL);
                    if (!lok || !rok) {
                        const char *ln = val_type_name(l), *rn = val_type_name(r);
                        char opbuf[8]; snprintf(opbuf, sizeof(opbuf), "%s", bop);
                        val_free(l); val_free(r);
                        runtime_error(n, "位运算 %s 需要整数，收到 %s %s %s（浮点请先用 取整 转换）",
                                      opbuf, ln, opbuf, rn);
                    }
                    long a = l->ival, b = r->ival, res = 0;
                    val_free(l); val_free(r);
                    if (is_shl || is_shr) {
                        if (b < 0)  runtime_error(n, "移位位数不能为负: %ld", b);
                        if (b > 63) runtime_error(n, "移位位数超出范围: %ld（0~63）", b);
                        if (is_shl) {
                            res = (long)((unsigned long)a << b);
                        } else {
                            /* 算术右移的可移植写法：负数用补码取反两次实现符号填充 */
                            res = (a < 0) ? (long)~((~(unsigned long)a) >> b)
                                          : (long)((unsigned long)a >> b);
                        }
                    }
                    else if (is_band) res = a & b;
                    else if (is_bor)  res = a | b;
                    else              res = a ^ b;
                    return val_int(res);
                }
            }
            /* 字符串拼接：只有 + 才拼接。
               以前任何运算符碰到字符串都会拼接，导致 "a" - 1 静默产出 "a1" 这种荒谬结果。 */
            if (l->type == VAL_STRING || r->type == VAL_STRING) {
                if (strcmp(n->binary.op, "+") != 0) {
                    const char *ln = val_type_name(l), *rn = val_type_name(r);
                    char opbuf[8]; snprintf(opbuf, sizeof(opbuf), "%s", n->binary.op);
                    val_free(l); val_free(r);
                    runtime_error(n, "字符串只支持 + 拼接，不支持 %s（%s %s %s）", opbuf, ln, opbuf, rn);
                }
                /* 任意类型都能与字符串拼接（走统一的值→文本转换，布尔显示为 真/假） */
                SBuf sb; sb_init(&sb);
                val_to_sbuf(l, &sb);
                val_to_sbuf(r, &sb);
                val_free(l); val_free(r);
                Value *sv = val_new(VAL_STRING);
                sv->sval = sb.buf;
                return sv;
            }
            /* 张量运算（支持广播与自动求导） */
            if (l->type == VAL_TENSOR && r->type == VAL_TENSOR) {
                Value *tv = tensor_elem_binop(n, n->binary.op, l, r);
                val_free(l); val_free(r);
                return tv;
            }
            if (l->type == VAL_TENSOR || r->type == VAL_TENSOR) {
                runtime_error(n, "张量运算需要两个张量");
            }
            /* 普通数值算术（比较运算符已在上面处理完毕，此处只剩 + - * / // %） */
            const char *op = n->binary.op;
            if (!is_numeric_val(l) || !is_numeric_val(r)) {
                /* 列表 + 列表 = 拼接；其余非数值组合直接报错，绝不静默误算 */
                if (strcmp(op, "+") == 0 && l->type == VAL_LIST && r->type == VAL_LIST) {
                    Value *cat = list_new_empty();
                    /* list_push 内部已 retain，此处不可再 retain（会造成引用计数永不归零） */
                    for (int i = 0; i < l->list_len; i++) list_push(cat, l->lval[i]);
                    for (int i = 0; i < r->list_len; i++) list_push(cat, r->lval[i]);
                    val_free(l); val_free(r);
                    return cat;
                }
                const char *ln = val_type_name(l), *rn = val_type_name(r);
                char opbuf[8]; snprintf(opbuf, sizeof(opbuf), "%s", op);
                val_free(l); val_free(r);
                runtime_error(n, "不支持的运算: %s %s %s", ln, opbuf, rn);
            }
            double ld = numeric_of(l);
            double rd = numeric_of(r);
            double result = 0;
            /* force：1=强制浮点（真除法），0=随操作数 */
            int force = 0;
            if (strcmp(op, "+") == 0) result = ld + rd;
            else if (strcmp(op, "-") == 0) result = ld - rd;
            else if (strcmp(op, "*") == 0) result = ld * rd;
            else if (strcmp(op, "/") == 0) {
                /* 真除法：3 / 2 得 1.5 而非 1；除零必须报错而不是静默返回 0 */
                if (rd == 0) { val_free(l); val_free(r); runtime_error(n, "除以零"); }
                result = ld / rd; force = 1;
            }
            else if (strcmp(op, "//") == 0) {
                if (rd == 0) { val_free(l); val_free(r); runtime_error(n, "整除的除数为零"); }
                result = floor(ld / rd);        /* 向下取整，与取模符号一致 */
            }
            else if (strcmp(op, "%") == 0) {
                if (rd == 0) { val_free(l); val_free(r); runtime_error(n, "取模的除数为零"); }
                result = fmod(ld, rd);          /* 支持浮点取模 */
            }
            else { val_free(l); val_free(r); runtime_error(n, "无效的二元运算符: %s", op); }
            int is_float = (force == 1) ? 1 :
                           (l->type == VAL_FLOAT || r->type == VAL_FLOAT);
            val_free(l); val_free(r);
            if (is_float) {
                Value *fv = val_new(VAL_FLOAT);
                fv->fval = result;
                return fv;
            } else {
                Value *iv = val_new(VAL_INT);
                iv->ival = (long)result;
                return iv;
            }
        }
        case ND_UNARY: {
            Value *v = eval_node(n->unary.operand, env);
            if (strcmp(n->unary.op, "-") == 0) {
                /* 必须返回【新值】：操作数可能是变量或容器元素的共享引用
                   （eval 返回的是 retain 后的同一对象），原地取负会篡改源数据。 */
                if (v->type == VAL_INT)   { long x = v->ival; val_free(v); return val_int(-x); }
                if (v->type == VAL_FLOAT) { double x = v->fval; val_free(v); return val_flt(-x); }
                if (v->type == VAL_BOOL)  { long x = v->ival; val_free(v); return val_int(-x); }
                if (v->type == VAL_TENSOR) {
                    Tensor *t = v->tval;
                    Tensor *o = tensor_new(t->ndim, t->shape);
                    for (int i = 0; i < t->size; i++) o->data[i] = -t->data[i];
                    val_free(v);
                    Value *r = val_new(VAL_TENSOR); r->tval = o; return r;
                }
                const char *tn = val_type_name(v);
                val_free(v);
                runtime_error(n, "取负需要数字或张量，得到 %s", tn);
            } else if (strcmp(n->unary.op, "+") == 0) {
                if (is_numeric_val(v) || v->type == VAL_TENSOR) return v;
                const char *tn = val_type_name(v);
                val_free(v);
                runtime_error(n, "取正需要数字或张量，得到 %s", tn);
            } else if (strcmp(n->unary.op, "not") == 0) {
                int b = truthy(v);
                val_free(v);
                return val_bool(!b);
            } else if (strcmp(n->unary.op, "~") == 0) {
                /* 按位取反只对整数有意义；布尔按 0/1 参与，结果是整数（真 → -2） */
                if (v->type == VAL_INT || v->type == VAL_BOOL) {
                    long x = v->ival; val_free(v); return val_int(~x);
                }
                const char *tn = val_type_name(v);
                val_free(v);
                runtime_error(n, "按位取反需要整数，得到 %s（浮点请先用 取整 转换）", tn);
            }
            return v;
        }
        case ND_LIST_LIT: {
            if (n->list_lit.is_comprehension) {
                /* 列表推导式：[body 对于 x 于 src 若 filter] */
                const char *iname = n->list_lit.elements[1]->ident.name;
                Node *filter = n->list_lit.elements[3];
                Value *src = eval_node(n->list_lit.elements[2], env);
                Value *result = val_new(VAL_LIST);
                result->list_len = 0; result->lval = NULL;
                int take;
                if (src->type == VAL_LIST) {
                    for (int i = 0; i < src->list_len; i++) {
                        env_set(env, iname, src->lval[i]);   /* env_set 内部已 retain */
                        take = 1;
                        if (filter) {
                            Value *fv = eval_node(filter, env);
                            take = truthy(fv);
                            val_free(fv);
                        }
                        if (take) {
                            Value *item = eval_node(n->list_lit.elements[0], env);
                            result->lval = realloc(result->lval, (result->list_len+1)*sizeof(Value*));
                            result->lval[result->list_len++] = val_adopt(item);
                        }
                    }
                } else if (src->type == VAL_STRING) {
                    /* 按字符遍历，中文字符串推导式不会产生乱码 */
                    long slen = utf8_strlen(src->sval);
                    for (long i = 0; i < slen; i++) {
                        Value *ch = val_new(VAL_STRING); ch->sval = utf8_char_at(src->sval, i);
                        env_set(env, iname, ch); val_free(ch);
                        take = 1;
                        if (filter) {
                            Value *fv = eval_node(filter, env);
                            take = truthy(fv);
                            val_free(fv);
                        }
                        if (take) {
                            Value *item = eval_node(n->list_lit.elements[0], env);
                            result->lval = realloc(result->lval, (result->list_len+1)*sizeof(Value*));
                            result->lval[result->list_len++] = val_adopt(item);
                        }
                    }
                } else if (src->type == VAL_MAP) {
                    for (int i = 0; i < MAX_DICT_SIZE; i++) {
                        if (!src->mval->entries[i].used) continue;
                        env_set(env, iname, src->mval->entries[i].val);   /* env_set 内部已 retain */
                        take = 1;
                        if (filter) {
                            Value *fv = eval_node(filter, env);
                            take = truthy(fv);
                            val_free(fv);
                        }
                        if (take) {
                            Value *item = eval_node(n->list_lit.elements[0], env);
                            result->lval = realloc(result->lval, (result->list_len+1)*sizeof(Value*));
                            result->lval[result->list_len++] = val_adopt(item);
                        }
                    }
                } else {
                    runtime_error(n, "推导式数据源必须是列表/字符串/字典");
                }
                val_free(src);
                return result;
            }
            Value *list = val_new(VAL_LIST);
            list->list_len = n->list_lit.ecnt;
            list->lval = malloc(list->list_len * sizeof(Value*));
            for (int i = 0; i < n->list_lit.ecnt; i++) {
                list->lval[i] = val_adopt(eval_node(n->list_lit.elements[i], env));
            }
            return list;
        }
        case ND_MAP_LIT: {
            if (n->map_lit.is_comprehension) {
                /* 字典推导式：{k:v 对于 kk[, vv] 在 src 若 filter}
                   单变量时：列表源绑定元素、字典源绑定键 */
                const char *ik = n->map_lit.keys[1]->ident.name;
                const char *iv = n->map_lit.vals[1] ? n->map_lit.vals[1]->ident.name : NULL;
                Node *filter = n->map_lit.keys[3];
                Value *src = eval_node(n->map_lit.keys[2], env);
                Value *result = val_new(VAL_MAP);
                int take;
                if (src->type == VAL_MAP) {
                    for (int i = 0; i < MAX_DICT_SIZE; i++) {
                        if (!src->mval->entries[i].used) continue;
                        Value *kv = val_new(VAL_STRING); kv->sval = strdup(src->mval->entries[i].key);
                        env_set(env, ik, kv); val_free(kv);
                        if (iv) env_set(env, iv, src->mval->entries[i].val);   /* env_set 内部已 retain */
                        take = 1;
                        if (filter) {
                            Value *fv = eval_node(filter, env);
                            take = truthy(fv);
                            val_free(fv);
                        }
                        if (take) {
                            Value *k = eval_node(n->map_lit.keys[0], env);
                            Value *v = eval_node(n->map_lit.vals[0], env);
                            if (k->type == VAL_STRING) dict_set(result->mval, k->sval, v);
                            else { char buf[64]; if(k->type==VAL_INT) snprintf(buf,sizeof(buf),"%ld",k->ival); else snprintf(buf,sizeof(buf),"%g",k->fval); dict_set(result->mval, buf, v); }
                            val_free(k); val_free(v);
                        }
                    }
                } else if (src->type == VAL_LIST) {
                    for (int i = 0; i < src->list_len; i++) {
                        if (iv) {
                            /* 双变量：第一个是下标，第二个是元素 */
                            Value *idv = val_new(VAL_INT); idv->ival = i;
                            env_set(env, ik, idv); val_free(idv);
                            env_set(env, iv, src->lval[i]);   /* env_set 内部已 retain */
                        } else {
                            /* 单变量：直接绑定元素本身 */
                            env_set(env, ik, src->lval[i]);
                        }
                        take = 1;
                        if (filter) {
                            Value *fv = eval_node(filter, env);
                            take = truthy(fv);
                            val_free(fv);
                        }
                        if (take) {
                            Value *k = eval_node(n->map_lit.keys[0], env);
                            Value *v = eval_node(n->map_lit.vals[0], env);
                            if (k->type == VAL_STRING) dict_set(result->mval, k->sval, v);
                            else { char buf[64]; if(k->type==VAL_INT) snprintf(buf,sizeof(buf),"%ld",k->ival); else snprintf(buf,sizeof(buf),"%g",k->fval); dict_set(result->mval, buf, v); }
                            val_free(k); val_free(v);
                        }
                    }
                } else {
                    runtime_error(n, "字典推导式数据源必须是列表/字典");
                }
                val_free(src);
                return result;
            }
            Value *map = val_new(VAL_MAP);
            for (int i = 0; i < n->map_lit.kcnt; i++) {
                Value *k = eval_node(n->map_lit.keys[i], env);
                Value *v = eval_node(n->map_lit.vals[i], env);
                if (k->type == VAL_STRING) {
                    dict_set(map->mval, k->sval, v);
                } else {
                    char buf[64];
                    if (k->type == VAL_INT) snprintf(buf, sizeof(buf), "%ld", k->ival);
                    else snprintf(buf, sizeof(buf), "%g", k->fval);
                    dict_set(map->mval, buf, v);
                }
                val_free(k);
                val_free(v);
            }
            return map;
        }
        case ND_CONCUR: {
            /* 在隔离子环境中并发求值表达式，立即返回任务句柄 */
            /* 外层环境链将被子线程读取（查找函数/全局变量），标记为共享以启用加锁 */
            env_mark_shared(env);
            Task *t = calloc(1, sizeof(Task));
            t->node = n->concur_stmt.expr;
            t->env  = env_new(env);
            Value *h = val_new(VAL_TASK);
            h->task = t;
            if (pthread_create(&t->tid, NULL, task_worker, t) != 0) {
                val_free(h);
                runtime_error(n, "无法创建并发线程");
            }
            t->started = 1;
            return h;
        }
        case ND_WAIT: {
            /* join 任务句柄并取出结果（结果所有权转交调用方） */
            Value *h = eval_node(n->wait_stmt.expr, env);
            if (h->type != VAL_TASK || !h->task) runtime_error(n, "等待的对象不是并发任务句柄");
            Task *t = h->task;
            if (t->started && !t->joined) {
                pthread_join(t->tid, NULL);
                t->joined = 1;
            }
            Value *ret = t->result;
            t->result = NULL;                 /* 摘出结果，避免句柄析构时重复释放 */
            if (!ret) ret = val_new(VAL_NULL);
            val_free(h);
            return ret;
        }
        case ND_FUNC_CALL: {
            if (n->func_call.callee->type == ND_IDENT && 
                strcmp(n->func_call.callee->ident.name, "内置输出") == 0) {
                for (int i = 0; i < n->func_call.acnt; i++) {
                    Value *arg = eval_node(n->func_call.args[i], env);
                    char *s = val_to_string(arg);
                    printf("%s", s);
                    if (i < n->func_call.acnt - 1) printf(" ");
                    free(s);
                    val_free(arg);
                }
                printf("\n");
                return val_new(VAL_NULL);
            }
            if (n->func_call.callee->type == ND_IDENT) {
                /* 先检查是否是用户定义的函数 */
                Value *fnval = env_get(env, n->func_call.callee->ident.name);
                if (fnval && fnval->type == VAL_FUNC) {
                    /* fnval 来自 env_get，不额外 retain，不能 free */
                    /* 实参数组用【栈】数组：原先是 malloc，一旦函数体内抛出异常，
                       longjmp 会跳过 free(args) 造成泄漏。栈数组随栈自动消失。 */
                    if (n->func_call.acnt > CO_MAX_ARGS)
                        runtime_error(n, "实参个数超过上限 %d", CO_MAX_ARGS);
                    Value *args[CO_MAX_ARGS];
                    for (int i = 0; i < n->func_call.acnt; i++)
                        args[i] = eval_node(n->func_call.args[i], env);
                    Value *ret = invoke_func(fnval->fnval, args, n->func_call.acnt, n, env);
                    for (int i = 0; i < n->func_call.acnt; i++) val_free(args[i]);
                    return ret;
                }
                /* 不是用户函数，尝试内置函数 */
                if (n->func_call.acnt > CO_MAX_ARGS)
                    runtime_error(n, "实参个数超过上限 %d", CO_MAX_ARGS);
                Value *args[CO_MAX_ARGS];
                for (int i = 0; i < n->func_call.acnt; i++) {
                    args[i] = eval_node(n->func_call.args[i], env);
                }
                Value *result = call_builtin(n->func_call.callee->ident.name, n->func_call.acnt, args);
                for (int i = 0; i < n->func_call.acnt; i++) {
                    val_free(args[i]);
                }
                return result;
            }
            /* 被调用方是表达式（如列表/字典里取出的函数值） */
            Value *fnval = eval_node(n->func_call.callee, env);
            if (fnval->type != VAL_FUNC) { val_free(fnval); runtime_error(n, "值不是一个函数"); }
            if (n->func_call.acnt > CO_MAX_ARGS) {
                val_free(fnval);
                runtime_error(n, "实参个数超过上限 %d", CO_MAX_ARGS);
            }
            Value *cargs[CO_MAX_ARGS];
            for (int i = 0; i < n->func_call.acnt; i++)
                cargs[i] = eval_node(n->func_call.args[i], env);
            Value *ret = invoke_func(fnval->fnval, cargs, n->func_call.acnt, n, env);
            for (int i = 0; i < n->func_call.acnt; i++) val_free(cargs[i]);
            val_free(fnval);
            return ret;
        }
        default:
            runtime_error(n, "无法求值的节点类型: %d", n->type);
            return val_new(VAL_NULL);
    }
}

/* 执行语句 */
static void exec_node(Node *n, Env *env) {
    stack_guard_check(n);   /* 语句递归同样吃栈，必须设卡 */
    if (!n) return;
    if (g_return_flag) return;
    if (loop_ctx && loop_ctx->should_break) return;
    if (loop_ctx && loop_ctx->should_continue) return;
    
    switch (n->type) {
        case ND_VAR_DECL: {
            Value *init = n->var_decl.init ? eval_node(n->var_decl.init, env) : val_new(VAL_NULL);
            if (n->var_decl.is_const && env_get(env, n->var_decl.name)) {
                runtime_error(n, "常量不能重复声明");
            }
            if (n->var_decl.type_annot) {
                ensure_type(n->var_decl.type_annot, init, n, "变量声明");
                env_set_type(env, n->var_decl.name, n->var_decl.type_annot);
            }
            env_set(env, n->var_decl.name, init);
            val_free(init);
            break;
        }
        case ND_ASSIGN: {
            Value *val = eval_node(n->assign.val, env);
            if (n->assign.tgt->type == ND_IDENT) {
                Value *old = env_get(env, n->assign.tgt->ident.name);
                if (old && old->type == VAL_FUNC) runtime_error(n->assign.tgt, "不能给函数名赋值");
                const char *t = env_get_type(env, n->assign.tgt->ident.name);
                if (t) ensure_type(t, val, n->assign.tgt, "赋值");
                env_set(env, n->assign.tgt->ident.name, val);
            } else if (n->assign.tgt->type == ND_BINARY &&
                       strcmp(n->assign.tgt->binary.op, "[]") == 0) {
                if (assign_tensor(n->assign.tgt, val, env)) {
                    val_free(val);
                    break;
                }
                Value *obj = eval_node(n->assign.tgt->binary.left, env);
                Value *idx = eval_node(n->assign.tgt->binary.right, env);
                if (obj->type == VAL_LIST) {
                    int i = (int)idx_val(idx);
                    if (i < 0) runtime_error(n->assign.tgt, "索引不能为负数");
                    if (i >= obj->list_len) {
                        Value **new_list = realloc(obj->lval, (i+1) * sizeof(Value*));
                        for (int j = obj->list_len; j <= i; j++) new_list[j] = val_new(VAL_NULL);
                        obj->lval = new_list;
                        obj->list_len = i+1;
                    }
                    val_free(obj->lval[i]);
                    obj->lval[i] = val_retain_root(val);
                } else if (obj->type == VAL_MAP) {
                    if (idx->type != VAL_STRING) runtime_error(n->assign.tgt, "字典键必须是字符串");
                    dict_set(obj->mval, idx->sval, val);
                } else {
                    val_free(obj); val_free(idx); val_free(val);
                    runtime_error(n->assign.tgt, "只能对列表或字典进行索引赋值");
                    break;
                }
                val_free(obj); val_free(idx);
            } else {
                val_free(val);
                runtime_error(n->assign.tgt, "无效的赋值目标");
                break;
            }
            val_free(val);
            break;
        }
        case ND_IF: {
            Value *condv = eval_node(n->if_stmt.cond, env);
            int cond = truthy(condv);
            val_free(condv);
            if (cond) {
                for (int i = 0; i < n->if_stmt.tcnt; i++) exec_node(n->if_stmt.then_body[i], env);
            } else {
                int executed = 0;
                for (int b = 0; b < n->if_stmt.br_cnt && !executed; b++) {
                    ElseBranch *br = &n->if_stmt.else_branches[b];
                    if (br->is_else) {
                        for (int i = 0; i < br->bcnt; i++) exec_node(br->body[i], env);
                        executed = 1;
                    } else {
                        Value *elcond = eval_node(br->cond, env);
                        int elval = truthy(elcond);
                        val_free(elcond);
                        if (elval) {
                            for (int i = 0; i < br->bcnt; i++) exec_node(br->body[i], env);
                            executed = 1;
                        }
                    }
                }
            }
            break;
        }
        case ND_WHILE: {
            LoopContext ctx;
            ctx.in_loop = 1;
            ctx.should_break = 0;
            ctx.should_continue = 0;
            ctx.parent = loop_ctx;
            loop_ctx = &ctx;
            
            while (!ctx.should_break) {
                /* 每轮迭代开始清除 continue 标志：继续 只跳过当前这一轮 */
                ctx.should_continue = 0;
                Value *c = eval_node(n->while_stmt.cond, env);
                int cond = truthy(c);
                val_free(c);
                if (!cond) break;
                for (int i = 0; i < n->while_stmt.bcnt; i++) {
                    exec_node(n->while_stmt.body[i], env);
                    if (ctx.should_break || ctx.should_continue) break;
                }
                if (g_return_flag) break;
            }
            
            loop_ctx = ctx.parent;
            break;
        }
        case ND_FOR: {
            Value *start_v = eval_node(n->for_stmt.start, env);
            Value *end_v = eval_node(n->for_stmt.end, env);
            double start = num_of(start_v, "循环起点");
            double end   = num_of(end_v,   "循环终点");
            double step = 1.0;
            int step_is_float = 0;
            if (n->for_stmt.step) {
                Value *step_v = eval_node(n->for_stmt.step, env);
                step = num_of(step_v, "循环步长");
                step_is_float = (step_v->type == VAL_FLOAT);
                val_free(step_v);
            }
            /* 浮点判定基于求值后的实际值类型（起止可为任意表达式，而非仅字面量） */
            int is_float = (start_v->type == VAL_FLOAT || end_v->type == VAL_FLOAT || step_is_float);
            val_free(start_v);
            val_free(end_v);
            
            LoopContext ctx;
            ctx.in_loop = 1;
            ctx.should_break = 0;
            ctx.should_continue = 0;
            ctx.parent = loop_ctx;
            loop_ctx = &ctx;
            

            if (is_float) {
                double i = start;
                int direction = (step > 0) ? 1 : -1;
                while ((direction > 0 && i <= end + 0.0001) || (direction < 0 && i >= end - 0.0001)) {
                    Value *iv = val_new(VAL_FLOAT);
                    iv->fval = i;
                    env_set(env, n->for_stmt.var, iv);
                    val_free(iv);
                    
                    ctx.should_continue = 0;
                    for (int j = 0; j < n->for_stmt.bcnt; j++) {
                        exec_node(n->for_stmt.body[j], env);
                        if (ctx.should_break || ctx.should_continue) break;
                    }
                    if (ctx.should_break || g_return_flag) break;
                    i += step;
                }
            } else {
                long i = (long)start;
                long e = (long)end;
                long s = (long)step;
                int direction = (s > 0) ? 1 : -1;
                while ((direction > 0 && i <= e) || (direction < 0 && i >= e)) {
                    Value *iv = val_new(VAL_INT);
                    iv->ival = i;
                    env_set(env, n->for_stmt.var, iv);
                    val_free(iv);
                    
                    ctx.should_continue = 0;
                    for (int j = 0; j < n->for_stmt.bcnt; j++) {
                        exec_node(n->for_stmt.body[j], env);
                        if (ctx.should_break || ctx.should_continue) break;
                    }
                    if (ctx.should_break || g_return_flag) break;
                    i += s;
                }
            }
            
            loop_ctx = ctx.parent;
            break;
        }
        case ND_FUNC_DEF: {
            Func *fn = malloc(sizeof(Func));
            fn->name = strdup(n->func_def.name);
            fn->params = n->func_def.pnames;
            fn->param_count = n->func_def.pcnt;
            fn->ptypes = n->func_def.ptypes;
            fn->ret_type = n->func_def.ret_type;
            fn->body = n->func_def.body;
            Value *fnv = val_new(VAL_FUNC);
            fnv->fnval = fn;
            env_set(env, fn->name, fnv);
            val_free(fnv);
            break;
        }
        case ND_RETURN: {
            Value *rv = (n->ret_stmt.val) ? eval_node(n->ret_stmt.val, env) : val_new(VAL_NULL);
            if (g_cur_ret_type) ensure_type(g_cur_ret_type, rv, n, "返回语句");
            g_return_flag = 1;
            g_return_value = rv;
            break;
        }
        case ND_IMPORT: {
            /* 模块导入：读取文件 -> 解析 -> 在子环境执行 -> 合并导出符号 */
            char path[MAX_ID_LEN * 2];
            if (strstr(n->import_stmt.path, ".co") == NULL) {
                snprintf(path, sizeof(path), "%s.co", n->import_stmt.path);
            } else {
                snprintf(path, sizeof(path), "%s", n->import_stmt.path);
            }
            FILE *mf = fopen(path, "r");
            if (!mf) {
                fprintf(stderr, "导入错误: 无法打开模块 '%s'\n", path);
                exit(1);
            }
            if (fseek(mf, 0, SEEK_END) != 0) { fclose(mf); runtime_error(n, "无法读取模块 '%s'", path); }
            long msize = ftell(mf);
            if (msize < 0) { fclose(mf); runtime_error(n, "无法获取模块大小 '%s'", path); }
            rewind(mf);
            char *mbuf = malloc((size_t)msize + 1);
            if (!mbuf) { fclose(mf); runtime_error(n, "导入模块时内存不足"); }
            size_t mgot = fread(mbuf, 1, (size_t)msize, mf);
            fclose(mf);
            mbuf[mgot] = '\0';
            /* 保存词法状态 */
            char *save_src = (char*)src; int save_pos = src_pos, save_line = src_line, save_col = src_col;
            Token save_tok = cur_tok;
            src = mbuf; src_pos = 0; src_line = 1; src_col = 1;
            advance();
            Node *mprog = parse_program();
            Env *menv = env_new(env);
            hoist_functions(mprog, menv);     /* 模块内部同样享受函数提升 */
            for (int i = 0; i < mprog->prog.cnt; i++) exec_node(mprog->prog.stmts[i], menv);
            /* 将模块顶层符号合并到当前环境。
               注意：env_set 内部已 val_retain，此处不可再手动 retain，
               否则引用计数永久偏高（内存泄漏）。 */
            for (int i = 0; i < menv->count; i++) {
                if (menv->values[i]) {
                    env_set(env, menv->names[i], menv->values[i]);
                    if (menv->types[i]) env_set_type(env, menv->names[i], menv->types[i]);
                }
            }
            /* 释放模块执行环境（符号已被调用方环境 retain，释放安全） */
            env_release(menv);
            /* 保留模块 AST：其函数体已被合并进调用方环境，释放会导致悬垂指针。
               改为登记到全局列表，程序结束统一释放。 */
            if (g_imported_cnt >= g_imported_cap) {
                g_imported_cap = g_imported_cap ? g_imported_cap*2 : 4;
                g_imported = realloc(g_imported, g_imported_cap * sizeof(Node*));
            }
            g_imported[g_imported_cnt++] = mprog;
            free(mbuf);
            /* 恢复词法状态 */
            src = save_src; src_pos = save_pos; src_line = save_line; src_col = save_col;
            cur_tok = save_tok;
            break;
        }
        case ND_MATCH: {
            Value *ev = eval_node(n->match_stmt.expr, env);
            int matched = 0;
            for (int a = 0; a < n->match_stmt.arm_cnt && !matched; a++) {
                MatchArm *arm = &n->match_stmt.arms[a];
                if (arm->is_default) {
                    for (int i = 0; i < arm->bcnt; i++) exec_node(arm->body[i], env);
                    matched = 1;
                } else {
                    Value *pv = eval_node(arm->pattern, env);
                    /* 统一走 val_equals：布尔、列表、字典、张量也能作为匹配模式，
                       不再只支持整数/浮点/字符串这三种手写组合 */
                    int eq = val_equals(ev, pv);
                    val_free(pv);
                    if (eq) {
                        for (int i = 0; i < arm->bcnt; i++) exec_node(arm->body[i], env);
                        matched = 1;
                    }
                }
            }
            if (!matched) {
                /* 无默认分支且未匹配：静默跳过（与多数语言一致） */
            }
            val_free(ev);
            break;
        }
        case ND_THROW: {
            Value *v = eval_node(n->throw_stmt.expr, env);
            Value *err;
            if (v->type == VAL_MAP) {
                err = v;                 /* 已是错误对象：原样再抛出，保留原始位置 */
            } else {
                char *s = val_to_string(v);
                err = make_error_value("抛出", s ? s : "", n->line, n->col);
                free(s);
                val_free(v);
            }
            if (g_try_depth == 0) die_uncaught(err);
            throw_to_frame(err);
            break;
        }
        case ND_TRY: {
            if (g_try_depth >= MAX_TRY_DEPTH)
                runtime_error(n, "尝试 块嵌套过深（超过 %d 层）", MAX_TRY_DEPTH);

            TryFrame *f = &g_try[g_try_depth];
            f->log = NULL;  f->nlog = f->caplog = 0;
            f->envs = NULL; f->nenv = f->capenv = 0;
            f->recording = 1;

            /* longjmp 会跳过 invoke_func 里对这些执行状态的恢复，
               必须自己存档：否则捕获一次异常后调用深度只增不减、
               循环控制与返回标志也会残留，程序逻辑随之错乱。 */
            volatile int          sv_depth = g_call_depth;
            LoopContext * volatile sv_loop = loop_ctx;
            volatile int          sv_rflag = g_return_flag;
            Value       * volatile sv_rval = g_return_value;
            const char  * volatile sv_rtyp = g_cur_ret_type;
            volatile int caught = 0;

            g_try_depth++;
            if (setjmp(f->jb) == 0) {
                for (int i = 0; i < n->try_stmt.bcnt; i++) {
                    exec_node(n->try_stmt.body[i], env);
                    if (g_return_flag) break;
                    if (loop_ctx && (loop_ctx->should_break || loop_ctx->should_continue)) break;
                }
                /* 正常走完：弹帧但不回滚——该释放的都已经正常释放过了 */
                g_try_depth--;
                f->recording = 0;
                free(f->log);  f->log = NULL;  f->nlog = f->caplog = 0;
                free(f->envs); f->envs = NULL; f->nenv = f->capenv = 0;
            } else {
                /* 异常路径：帧已在 throw_to_frame 中弹出并回滚完毕 */
                caught = 1;
                g_call_depth   = sv_depth;
                loop_ctx       = sv_loop;
                g_return_flag  = sv_rflag;
                g_return_value = sv_rval;
                g_cur_ret_type = sv_rtyp;
            }

            if (caught) {
                Value *exc = g_exc_value;   /* 接管载荷所有权 */
                g_exc_value = NULL;
                /* 把这一份引用托管给【外层】帧：捕获块自身也可能抛异常，
                   那时下面的 val_free(exc) 会被跳过。登记后由外层回滚兜底；
                   正常走完时 val_free 内部会登记 -1 自动抵消，两条路径对称。 */
                reflog_add(exc, +1);
                if (n->try_stmt.has_catch) {
                    if (n->try_stmt.var) env_set(env, n->try_stmt.var, exc);
                    for (int i = 0; i < n->try_stmt.ccnt; i++) {
                        exec_node(n->try_stmt.cbody[i], env);
                        if (g_return_flag) break;
                        if (loop_ctx && (loop_ctx->should_break || loop_ctx->should_continue)) break;
                    }
                    val_free(exc);          /* 交还我们持有的那一份 */
                    /* 有 捕获：异常到此为止，最终 块在下方统一执行 */
                } else {
                    /* 只有 最终 没有 捕获：先做清理，再把异常继续往外抛 */
                    for (int i = 0; i < n->try_stmt.fcnt; i++)
                        exec_node(n->try_stmt.fbody[i], env);
                    if (g_try_depth == 0) die_uncaught(exc);
                    throw_to_frame(exc);
                }
            }

            /* 最终 块：正常结束、以及"捕获处理完"两种情况都要执行。
               注意 exec_node 开头就会因 g_return_flag / 跳出 / 继续 直接返回，
               所以必须先把这些"挂起中的控制流"摘下来，跑完清理再挂回去 ——
               否则 `尝试 ... 返回 x ... 最终 ...` 的清理代码会被静默跳过。 */
            if (n->try_stmt.has_finally && (!caught || n->try_stmt.has_catch)) {
                int    sf = g_return_flag;
                Value *sv = g_return_value;
                int    sb = loop_ctx ? loop_ctx->should_break    : 0;
                int    sc = loop_ctx ? loop_ctx->should_continue : 0;
                g_return_flag = 0; g_return_value = NULL;
                if (loop_ctx) { loop_ctx->should_break = 0; loop_ctx->should_continue = 0; }

                for (int i = 0; i < n->try_stmt.fcnt; i++)
                    exec_node(n->try_stmt.fbody[i], env);

                /* 清理块自己若也 返回，则以它为准（与主流语言一致），旧返回值丢弃 */
                if (!g_return_flag) { g_return_flag = sf; g_return_value = sv; }
                else if (sv) val_free(sv);
                if (loop_ctx) {
                    if (!loop_ctx->should_break)    loop_ctx->should_break    = sb;
                    if (!loop_ctx->should_continue) loop_ctx->should_continue = sc;
                }
            }
            break;
        }
        case ND_BREAK: {
            if (!loop_ctx || !loop_ctx->in_loop) {
                runtime_error(n, "跳出语句只能在循环内使用");
            }
            loop_ctx->should_break = 1;
            break;
        }
        case ND_CONTINUE: {
            if (!loop_ctx || !loop_ctx->in_loop) {
                runtime_error(n, "继续语句只能在循环内使用");
            }
            loop_ctx->should_continue = 1;
            break;
        }
        default: {
            Value *v = eval_node(n, env);
            val_free(v);
            break;
        }
    }
}

/* ========== 主程序 ========== */
int main(int argc, char **argv) {
    stack_guard_init(0);          /* 必须最先执行：以 main 的栈帧为基准 */
    srand((unsigned int)time(NULL));
    env_lock_init();
    
    if (argc < 2) {
        fprintf(stderr, "用法: co <脚本.co>\n");
        fprintf(stderr, "示例: ./co hello.co\n");
        return 1;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "错误: 无法打开文件 '%s'\n", argv[1]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size < 0) { perror("ftell"); fclose(f); return 1; }
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(size+1);
    if (!buf) { perror("malloc"); fclose(f); return 1; }
    if (fread(buf, 1, size, f) != (size_t)size) {
        fprintf(stderr, "错误: 文件读取不完整\n");
        free(buf); fclose(f);
        return 1;
    }
    buf[size] = '\0';
    fclose(f);

    src = buf;
    src_pos = 0;
    src_line = 1;
    src_col = 1;
    advance();

    Node *prog = parse_program();
    Env *global = env_new(NULL);
    g_root_env = global;              /* 高阶函数回调时的父环境 */
    g_call_fn  = call_fn_value;       /* 注入用户函数调用能力给内置高阶函数 */

    hoist_functions(prog, global);    /* 函数提升：定义顺序不再限制调用 */

    for (int i = 0; i < prog->prog.cnt; i++) {
        exec_node(prog->prog.stmts[i], global);
    }

    g_root_env = NULL;
    env_release(global);
    node_free(prog);
    /* 统一释放已导入模块的 AST（其函数体此前被合并进全局环境） */
    for (int i = 0; i < g_imported_cnt; i++) node_free(g_imported[i]);
    free(g_imported);
    free(buf);
    return 0;
}
