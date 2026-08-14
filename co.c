#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <time.h>

/* ========== 基础定义 ========== */
#define MAX_ID_LEN     128
#define MAX_PARAMS     32
#define MAX_LOCALS     512
#define MAX_DICT_SIZE  256

/* 前向声明 */
struct Value;
struct Env;
struct Node;

/* 值类型 */
typedef enum {
    VAL_NULL, VAL_INT, VAL_FLOAT, VAL_STRING,
    VAL_LIST, VAL_TENSOR, VAL_FUNC, VAL_MAP
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
    void  *body;
} Func;

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
    };
    int list_len;
} Value;

/* 环境 (变量存储) */
typedef struct Env {
    char   *names[MAX_LOCALS];
    Value  *values[MAX_LOCALS];
    int     count;
    struct Env *parent;
} Env;

/* 循环控制上下文 */
typedef struct LoopContext {
    int in_loop;
    int should_break;
    int should_continue;
    struct LoopContext *parent;
} LoopContext;

/* ========== 全局状态 ========== */
static LoopContext *loop_ctx = NULL;

/* ========== 值管理 ========== */
static Value *val_new(ValType type) {
    Value *v = calloc(1, sizeof(Value));
    v->type = type;
    v->refcount = 1;
    if (type == VAL_MAP) {
        v->mval = calloc(1, sizeof(Map));
    }
    return v;
}

void val_free(Value *v) {
    if (!v) return;
    v->refcount--;
    if (v->refcount > 0) return;
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
        default: break;
    }
    free(v);
}

static Value *val_retain(Value *v) {
    if (v) v->refcount++;
    return v;
}

/* 值转字符串 */
static char *val_to_string(Value *v) {
    char buf[256];
    switch (v->type) {
        case VAL_NULL: return strdup("空");
        case VAL_INT: sprintf(buf, "%ld", v->ival); return strdup(buf);
        case VAL_FLOAT: sprintf(buf, "%g", v->fval); return strdup(buf);
        case VAL_STRING: return strdup(v->sval);
        case VAL_LIST: {
            char *s = malloc(4096);
            strcpy(s, "[");
            for (int i = 0; i < v->list_len; i++) {
                char *item = val_to_string(v->lval[i]);
                strcat(s, item);
                if (i < v->list_len - 1) strcat(s, ", ");
                free(item);
            }
            strcat(s, "]");
            return s;
        }
        case VAL_MAP: {
            char *s = malloc(4096);
            strcpy(s, "{");
            int first = 1;
            for (int i = 0; i < MAX_DICT_SIZE; i++) {
                if (v->mval->entries[i].used) {
                    if (!first) strcat(s, ", ");
                    first = 0;
                    strcat(s, "\"");
                    strcat(s, v->mval->entries[i].key);
                    strcat(s, "\": ");
                    char *item = val_to_string(v->mval->entries[i].val);
                    strcat(s, item);
                    free(item);
                }
            }
            strcat(s, "}");
            return s;
        }
        case VAL_FUNC: sprintf(buf, "<函数 %s>", v->fnval->name); return strdup(buf);
        case VAL_TENSOR: {
            Tensor *t = v->tval;
            char *s = malloc(8192);
            s[0] = '\0';
            if (t->ndim == 1) {
                strcat(s, "[");
                for (int i = 0; i < t->size; i++) {
                    char b[32]; snprintf(b, sizeof(b), "%g", t->data[i]);
                    strcat(s, b);
                    if (i < t->size - 1) strcat(s, ", ");
                }
                strcat(s, "]");
            } else if (t->ndim == 2) {
                strcat(s, "[");
                for (int i = 0; i < t->shape[0]; i++) {
                    strcat(s, "[");
                    for (int j = 0; j < t->shape[1]; j++) {
                        char b[32]; snprintf(b, sizeof(b), "%g", t->data[i * t->shape[1] + j]);
                        strcat(s, b);
                        if (j < t->shape[1] - 1) strcat(s, ", ");
                    }
                    strcat(s, "]");
                    if (i < t->shape[0] - 1) strcat(s, ", ");
                }
                strcat(s, "]");
            } else {
                snprintf(s, 8192, "<张量 shape=%dx%d>", t->shape[0], t->ndim > 1 ? t->shape[1] : 1);
            }
            return s;
        }
        default: return strdup("<未知>");
    }
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
        if (!m->entries[idx].used || strcmp(m->entries[idx].key, key) == 0) {
            if (m->entries[idx].used) val_free(m->entries[idx].val);
            m->entries[idx].key = strdup(key);
            m->entries[idx].val = val_retain(val);
            m->entries[idx].used = 1;
            return;
        }
    }
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
static Tensor *tensor_retain(Tensor *t) { if (t) t->tref++; return t; }

/* 张量引用计数释放 */
static void tensor_free(Tensor *t) {
    if (!t) return;
    t->tref--;
    if (t->tref > 0) return;
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
static Env *env_new(Env *parent) {
    Env *e = malloc(sizeof(Env));
    e->count = 0;
    e->parent = parent;
    return e;
}

static Value *env_get(Env *e, const char *name) {
    for (int i = 0; i < e->count; i++)
        if (strcmp(e->names[i], name) == 0) return e->values[i];
    if (e->parent) return env_get(e->parent, name);
    return NULL;
}

static void env_set(Env *e, const char *name, Value *val) {
    for (int i = 0; i < e->count; i++) {
        if (strcmp(e->names[i], name) == 0) {
            val_free(e->values[i]);
            e->values[i] = val_retain(val);
            return;
        }
    }
    if (e->count < MAX_LOCALS) {
        e->names[e->count] = strdup(name);
        e->values[e->count] = val_retain(val);
        e->count++;
    }
}

/* ========== 词法分析 ========== */
typedef enum {
    TOK_EOF, TOK_ID, TOK_INT, TOK_FLOAT, TOK_STRING, TOK_MULTILINE_STRING,
    TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH, TOK_MOD,
    TOK_EQ, TOK_NE, TOK_LT, TOK_LE, TOK_GT, TOK_GE,
    TOK_ASSIGN, TOK_LPAREN, TOK_RPAREN, TOK_LBRACK, TOK_RBRACK,
    TOK_LBRACE, TOK_RBRACE, TOK_COLON, TOK_COMMA, TOK_NEWLINE, TOK_DOT,
    /* 中文关键字 */
    TK_VAR, TK_CONST, TK_FUNC, TK_RETURN, TK_IF, TK_THEN,
    TK_ELSE, TK_ELIF, TK_END, TK_WHILE, TK_LOOP, TK_BREAK, TK_CONTINUE,
    TK_PRINT, TK_TENSOR, TK_FOR, TK_FROM, TK_TO, TK_STEP,
    /* 逻辑运算符 */
    TK_AND, TK_OR, TK_NOT
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
        /* 换行符 - 只消费不设置token，由advance处理 */
        if (src[src_pos] == '\n') { 
            src_line++;
            src_col = 1;
            src_pos++;
            continue;
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
    /* 数字 */
    if (isdigit(src[src_pos])) {
        int st = src_pos;
        while (isdigit(src[src_pos])) { src_pos++; src_col++; }
        if (src[src_pos] == '.') {
            src_pos++; src_col++;
            while (isdigit(src[src_pos])) { src_pos++; src_col++; }
            int len = src_pos - st;
            strncpy(cur_tok.text, src + st, len); cur_tok.text[len] = 0;
            cur_tok.type = TOK_FLOAT;
        } else {
            int len = src_pos - st;
            strncpy(cur_tok.text, src + st, len); cur_tok.text[len] = 0;
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
        else if (strcmp(cur_tok.text, "否则") == 0) cur_tok.type = TK_ELSE;
        else if (strcmp(cur_tok.text, "否则如果") == 0) cur_tok.type = TK_ELIF;
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
        else cur_tok.type = TOK_ID;
        return;
    }
    /* 单字符符号 */
    char c = src[src_pos++]; src_col++;
    switch (c) {
        case '+': cur_tok.type = TOK_PLUS; break;
        case '-': cur_tok.type = TOK_MINUS; break;
        case '*': cur_tok.type = TOK_STAR; break;
        case '/': cur_tok.type = TOK_SLASH; break;
        case '%': cur_tok.type = TOK_MOD; break;
        case '(': cur_tok.type = TOK_LPAREN; break;
        case ')': cur_tok.type = TOK_RPAREN; break;
        case '[': cur_tok.type = TOK_LBRACK; break;
        case ']': cur_tok.type = TOK_RBRACK; break;
        case '{': cur_tok.type = TOK_LBRACE; break;
        case '}': cur_tok.type = TOK_RBRACE; break;
        case ':': cur_tok.type = TOK_COLON; break;
        case ',': cur_tok.type = TOK_COMMA; break;
        case '.': cur_tok.type = TOK_DOT; break;
        case '=':
            if (src[src_pos] == '=') { src_pos++; src_col++; cur_tok.type = TOK_EQ; strcpy(cur_tok.text, "=="); }
            else { cur_tok.type = TOK_ASSIGN; cur_tok.text[0] = '='; cur_tok.text[1] = 0; }
            return;
        case '<':
            if (src[src_pos] == '=') { src_pos++; src_col++; cur_tok.type = TOK_LE; strcpy(cur_tok.text, "<="); }
            else cur_tok.type = TOK_LT;
            break;
        case '>':
            if (src[src_pos] == '=') { src_pos++; src_col++; cur_tok.type = TOK_GE; strcpy(cur_tok.text, ">="); }
            else cur_tok.type = TOK_GT;
            break;
        case '!': 
            if (src[src_pos] == '=') { src_pos++; src_col++; cur_tok.type = TOK_NE; }
            else lex_error("未知符号 '!'，期望 '!='");
            break;
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
    ND_LIST_LIT, ND_MAP_LIT, ND_BREAK, ND_CONTINUE
} NodeType;

/* else分支结构 */
typedef struct ElseBranch {
    struct Node *cond;
    struct Node **body;
    int bcnt;
    int is_else;
} ElseBranch;

typedef struct Node {
    NodeType type;
    int line;
    int col;
    union {
        struct { struct Node **stmts; int cnt; } prog;
        struct { char *name; struct Node *init; int is_const; } var_decl;
        struct { struct Node *tgt, *val; } assign;
        struct { char *name; } ident;
        struct { char *name; char **pnames; int pcnt; struct Node *body; } func_def;
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
        struct { struct Node **elements; int ecnt; } list_lit;
        struct { struct Node **keys; struct Node **vals; int kcnt; } map_lit;
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
            if (n->var_decl.init) node_free(n->var_decl.init);
            break;
        case ND_ASSIGN:
            node_free(n->assign.tgt); node_free(n->assign.val); break;
        case ND_IDENT:
            free(n->ident.name); break;
        case ND_FUNC_DEF:
            free(n->func_def.name);
            for (int i = 0; i < n->func_def.pcnt; i++) free(n->func_def.pnames[i]);
            free(n->func_def.pnames);
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
            for (int i = 0; i < n->list_lit.ecnt; i++) node_free(n->list_lit.elements[i]);
            free(n->list_lit.elements); break;
        case ND_MAP_LIT:
            for (int i = 0; i < n->map_lit.kcnt; i++) { node_free(n->map_lit.keys[i]); node_free(n->map_lit.vals[i]); }
            free(n->map_lit.keys); free(n->map_lit.vals); break;
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
        if (cur_tok.type != TOK_RBRACK) {
            do {
                Node *elem = parse_expression();
                if (cnt >= cap) { cap = cap ? cap*2 : 4; n->list_lit.elements = realloc(n->list_lit.elements, cap * sizeof(Node*)); }
                n->list_lit.elements[cnt++] = elem;
            } while (cur_tok.type == TOK_COMMA && (advance(), 1));
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
        if (cur_tok.type != TOK_RBRACE) {
            do {
                if (cur_tok.type == TOK_STRING || cur_tok.type == TOK_MULTILINE_STRING) {
                    char buf[MAX_ID_LEN * 2];
                    process_escape(buf, cur_tok.text, strlen(cur_tok.text));
                    Node *key = make_node(ND_LITERAL);
                    key->literal.lit_type = VAL_STRING;
                    key->literal.value = strdup(buf);
                    if (cnt >= cap) { cap = cap ? cap*2 : 4; 
                        n->map_lit.keys = realloc(n->map_lit.keys, cap * sizeof(Node*));
                        n->map_lit.vals = realloc(n->map_lit.vals, cap * sizeof(Node*)); }
                    n->map_lit.keys[cnt] = key;
                    advance();
                } else if (cur_tok.type == TOK_ID) {
                    Node *key = make_node(ND_LITERAL);
                    key->literal.lit_type = VAL_STRING;
                    key->literal.value = strdup(cur_tok.text);
                    if (cnt >= cap) { cap = cap ? cap*2 : 4; 
                        n->map_lit.keys = realloc(n->map_lit.keys, cap * sizeof(Node*));
                        n->map_lit.vals = realloc(n->map_lit.vals, cap * sizeof(Node*)); }
                    n->map_lit.keys[cnt] = key;
                    advance();
                } else {
                    Node *key = parse_expression();
                    if (cnt >= cap) { cap = cap ? cap*2 : 4; 
                        n->map_lit.keys = realloc(n->map_lit.keys, cap * sizeof(Node*));
                        n->map_lit.vals = realloc(n->map_lit.vals, cap * sizeof(Node*)); }
                    n->map_lit.keys[cnt] = key;
                }
                expect(TOK_COLON);
                Node *val = parse_expression();
                n->map_lit.vals[cnt] = val;
                cnt++;
            } while (cur_tok.type == TOK_COMMA && (advance(), 1));
        }
        n->map_lit.kcnt = cnt;
        expect(TOK_RBRACE);
        return n;
    } else {
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
    return parse_postfix();
}

static Node *parse_mul() {
    Node *left = parse_unary();
    while (cur_tok.type == TOK_STAR || cur_tok.type == TOK_SLASH || cur_tok.type == TOK_MOD) {
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

static Node *parse_cmp() {
    Node *left = parse_add();
    while (cur_tok.type == TOK_LT || cur_tok.type == TOK_LE || cur_tok.type == TOK_GT || cur_tok.type == TOK_GE) {
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

static Node *parse_and() {
    Node *left = parse_eq();
    while (cur_tok.type == TK_AND) {
        advance();
        Node *right = parse_eq();
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

static Node *parse_var_decl() {
    Token kw = cur_tok; advance();
    int is_const = (strcmp(kw.text, "常量") == 0);
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
    expect(TK_THEN);
    
    Node **then_body = NULL; int tcnt = 0, tcap = 0;
    while (cur_tok.type != TOK_EOF) {
        if (cur_tok.type == TOK_NEWLINE) { advance(); continue; }
        if (cur_tok.type == TK_ELSE || cur_tok.type == TK_ELIF || cur_tok.type == TK_END) break;
        Node *stmt = parse_statement();
        if (stmt) {
            if (tcnt >= tcap) { tcap = tcap ? tcap*2 : 4; then_body = realloc(then_body, tcap * sizeof(Node*)); }
            then_body[tcnt++] = stmt;
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
            advance(); /* 跳过'否则'，不需要'则' */
        }
        if (cur_tok.type == TK_THEN) advance(); /* 否则如果有'则'也接受 */
        
        br.body = NULL; br.bcnt = 0;
        while (cur_tok.type != TOK_EOF) {
            if (cur_tok.type == TOK_NEWLINE) { advance(); continue; }
            if (cur_tok.type == TK_ELSE || cur_tok.type == TK_ELIF || cur_tok.type == TK_END) break;
            Node *stmt = parse_statement();
            if (stmt) {
                br.body = realloc(br.body, (br.bcnt + 1) * sizeof(Node*));
                br.body[br.bcnt++] = stmt;
            }
        }
        
        if (br_cnt >= br_cap) { br_cap = br_cap ? br_cap*2 : 4; branches = realloc(branches, br_cap * sizeof(ElseBranch)); }
        branches[br_cnt++] = br;
    }
    
    expect(TK_END);
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

static Node *parse_for() {
    advance();
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
    char **params = NULL; int pcnt = 0, pcap = 0;
    if (cur_tok.type != TOK_RPAREN) {
        do {
            if (cur_tok.type != TOK_ID) {
                fprintf(stderr, "语法错误 (第%d行,第%d列): 参数需要标识符, 得到 '%s'\n",
                        cur_tok.line, cur_tok.col, cur_tok.text);
                exit(1);
            }
            if (pcnt >= pcap) { pcap = pcap ? pcap*2 : 4; params = realloc(params, pcap * sizeof(char*)); }
            params[pcnt++] = strdup(cur_tok.text);
            advance();
        } while (cur_tok.type == TOK_COMMA && (advance(), 1));
    }
    expect(TOK_RPAREN);
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
    n->func_def.pcnt = pcnt;
    n->func_def.body = body;
    return n;
}

static Node *parse_statement() {
    if (cur_tok.type == TOK_NEWLINE) { advance(); return NULL; }
    if (cur_tok.type == TK_VAR || cur_tok.type == TK_CONST) return parse_var_decl();
    if (cur_tok.type == TK_PRINT) return parse_print();
    if (cur_tok.type == TK_IF) return parse_if();
    if (cur_tok.type == TK_WHILE) return parse_while();
    if (cur_tok.type == TK_FOR) return parse_for();
    if (cur_tok.type == TK_FUNC) return parse_func_def();
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

/* ========== 解释器 ========== */
static void runtime_error(Node *n, const char *msg, ...) {
    va_list args;
    va_start(args, msg);
    if (n) {
        fprintf(stderr, "运行时错误 (第%d行,%d列): ", n->line, n->col);
    } else {
        fprintf(stderr, "运行时错误: ");
    }
    vfprintf(stderr, msg, args);
    fprintf(stderr, "\n");
    va_end(args);
    exit(1);
}

static Value *eval_node(Node *n, Env *env);
static void exec_node(Node *n, Env *env);

/* 控制流标志：用于实现返回/跳出/继续 */
static int g_return_flag = 0;
static Value *g_return_value = NULL;

/* 内置函数 */
static Value *builtin_len(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "长度函数需要1个参数");
    Value *v = argv[0];
    Value *r = val_new(VAL_INT);
    switch (v->type) {
        case VAL_STRING: r->ival = strlen(v->sval); break;
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
        case VAL_INT: r->sval = strdup("整数"); break;
        case VAL_FLOAT: r->sval = strdup("浮点数"); break;
        case VAL_STRING: r->sval = strdup("字符串"); break;
        case VAL_LIST: r->sval = strdup("列表"); break;
        case VAL_MAP: r->sval = strdup("字典"); break;
        case VAL_FUNC: r->sval = strdup("函数"); break;
        case VAL_TENSOR: r->sval = strdup("张量"); break;
        default: r->sval = strdup("未知"); break;
    }
    return r;
}

static Value *builtin_int(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "转整数函数需要1个参数");
    Value *v = argv[0], *r = val_new(VAL_INT);
    switch (v->type) {
        case VAL_INT: r->ival = v->ival; break;
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
        case VAL_FLOAT: r->fval = v->fval; break;
        case VAL_STRING: r->fval = atof(v->sval); break;
        default: r->fval = 0.0; break;
    }
    return r;
}

static Value *builtin_str(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "转字符串函数需要1个参数");
    return val_new(VAL_STRING);
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
    r->fval = sqrt((v->type == VAL_FLOAT) ? v->fval : v->ival);
    return r;
}

static Value *builtin_random(int argc, Value **argv) {
    (void)argc; (void)argv;
    Value *r = val_new(VAL_FLOAT);
    r->fval = (double)rand() / RAND_MAX;
    return r;
}

static Value *builtin_round(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "取整函数需要1个参数");
    Value *v = argv[0], *r = val_new(VAL_INT);
    if (v->type == VAL_FLOAT) {
        r->ival = (long)(v->fval + 0.5);
    } else {
        r->ival = v->ival;
    }
    return r;
}

static Value *builtin_max(int argc, Value **argv) {
    if (argc < 2) runtime_error(NULL, "最大值函数需要至少2个参数");
    double m;
    if (argv[0]->type == VAL_FLOAT) m = argv[0]->fval;
    else m = argv[0]->ival;
    for (int i = 1; i < argc; i++) {
        double v = (argv[i]->type == VAL_FLOAT) ? argv[i]->fval : argv[i]->ival;
        if (v > m) m = v;
    }
    Value *r = val_new(VAL_FLOAT);
    r->fval = m;
    return r;
}

static Value *builtin_min(int argc, Value **argv) {
    if (argc < 2) runtime_error(NULL, "最小值函数需要至少2个参数");
    double m;
    if (argv[0]->type == VAL_FLOAT) m = argv[0]->fval;
    else m = argv[0]->ival;
    for (int i = 1; i < argc; i++) {
        double v = (argv[i]->type == VAL_FLOAT) ? argv[i]->fval : argv[i]->ival;
        if (v < m) m = v;
    }
    Value *r = val_new(VAL_FLOAT);
    r->fval = m;
    return r;
}

/* 列表函数 */
static Value *builtin_append(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "追加函数需要2个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "追加函数的第一个参数必须是列表");
    Value *list = argv[0];
    Value *item = argv[1];
    Value **new_list = realloc(list->lval, (list->list_len + 1) * sizeof(Value*));
    list->lval = new_list;
    list->lval[list->list_len] = val_retain(item);
    list->list_len++;
    return val_new(VAL_NULL);
}

static Value *builtin_remove(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "删除函数需要2个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "删除函数的第一个参数必须是列表");
    if (argv[1]->type != VAL_INT) runtime_error(NULL, "删除函数的第二个参数必须是整数索引");
    Value *list = argv[0];
    int idx = (int)argv[1]->ival;
    if (idx < 0 || idx >= list->list_len) runtime_error(NULL, "列表索引越界");
    val_free(list->lval[idx]);
    for (int i = idx; i < list->list_len - 1; i++) {
        list->lval[i] = list->lval[i + 1];
    }
    list->list_len--;
    return val_new(VAL_NULL);
}

static Value *builtin_sort(int argc, Value **argv) {
    if (argc != 1) runtime_error(NULL, "排序函数需要1个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "排序函数的参数必须是列表");
    Value *list = argv[0];
    for (int i = 0; i < list->list_len - 1; i++) {
        for (int j = 0; j < list->list_len - i - 1; j++) {
            double a = (list->lval[j]->type == VAL_FLOAT) ? list->lval[j]->fval : list->lval[j]->ival;
            double b = (list->lval[j+1]->type == VAL_FLOAT) ? list->lval[j+1]->fval : list->lval[j+1]->ival;
            if (a > b) {
                Value *tmp = list->lval[j];
                list->lval[j] = list->lval[j+1];
                list->lval[j+1] = tmp;
            }
        }
    }
    return val_new(VAL_NULL);
}

static Value *builtin_contains(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "包含函数需要2个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "包含函数的第一个参数必须是列表");
    Value *list = argv[0];
    Value *item = argv[1];
    Value *r = val_new(VAL_INT);
    r->ival = 0;
    for (int i = 0; i < list->list_len; i++) {
        int match = 0;
        if (list->lval[i]->type == item->type) {
            if (item->type == VAL_INT && list->lval[i]->ival == item->ival) match = 1;
            else if (item->type == VAL_FLOAT && list->lval[i]->fval == item->fval) match = 1;
            else if (item->type == VAL_STRING && strcmp(list->lval[i]->sval, item->sval) == 0) match = 1;
        }
        if (match) { r->ival = 1; break; }
    }
    return r;
}

/* 字符串函数 */
static Value *builtin_split(int argc, Value **argv) {
    if (argc < 1 || argc > 2) runtime_error(NULL, "分割函数需要1-2个参数");
    if (argv[0]->type != VAL_STRING) runtime_error(NULL, "分割函数的第一个参数必须是字符串");
    Value *str = argv[0];
    const char *sep = (argc >= 2 && argv[1]->type == VAL_STRING) ? argv[1]->sval : " ";
    
    Value *result = val_new(VAL_LIST);
    result->list_len = 0;
    result->lval = malloc(sizeof(Value*));
    
    char *copy = strdup(str->sval);
    char *token = strtok(copy, sep);
    while (token) {
        result->list_len++;
        result->lval = realloc(result->lval, result->list_len * sizeof(Value*));
        result->lval[result->list_len - 1] = val_new(VAL_STRING);
        result->lval[result->list_len - 1]->sval = strdup(token);
        token = strtok(NULL, sep);
    }
    free(copy);
    return result;
}

static Value *builtin_join(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "连接函数需要2个参数");
    if (argv[0]->type != VAL_LIST) runtime_error(NULL, "连接函数的第一个参数必须是列表");
    if (argv[1]->type != VAL_STRING) runtime_error(NULL, "连接函数的第二个参数必须是字符串");
    Value *list = argv[0];
    const char *sep = argv[1]->sval;
    
    size_t total_len = 1;
    for (int i = 0; i < list->list_len; i++) {
        if (list->lval[i]->type == VAL_STRING) {
            total_len += strlen(list->lval[i]->sval);
        }
    }
    total_len += (list->list_len - 1) * strlen(sep);
    
    Value *result = val_new(VAL_STRING);
    result->sval = malloc(total_len);
    result->sval[0] = '\0';
    
    for (int i = 0; i < list->list_len; i++) {
        if (list->lval[i]->type == VAL_STRING) {
            strcat(result->sval, list->lval[i]->sval);
        }
        if (i < list->list_len - 1) {
            strcat(result->sval, sep);
        }
    }
    return result;
}

static Value *builtin_replace(int argc, Value **argv) {
    if (argc != 3) runtime_error(NULL, "替换函数需要3个参数");
    if (argv[0]->type != VAL_STRING) runtime_error(NULL, "替换函数的第一个参数必须是字符串");
    if (argv[1]->type != VAL_STRING || argv[2]->type != VAL_STRING) runtime_error(NULL, "替换函数的第二、三个参数必须是字符串");
    
    Value *result = val_new(VAL_STRING);
    size_t len = strlen(argv[0]->sval) * 2 + 1;
    result->sval = malloc(len);
    
    const char *src = argv[0]->sval;
    const char *old = argv[1]->sval;
    const char *new = argv[2]->sval;
    size_t old_len = strlen(old);
    
    char *dest = result->sval;
    const char *p = src;
    while (*p) {
        if (strncmp(p, old, old_len) == 0) {
            strcpy(dest, new);
            dest += strlen(new);
            p += old_len;
        } else {
            *dest++ = *p++;
        }
    }
    *dest = '\0';
    return result;
}

static Value *builtin_find(int argc, Value **argv) {
    if (argc != 2) runtime_error(NULL, "查找函数需要2个参数");
    if (argv[0]->type != VAL_STRING || argv[1]->type != VAL_STRING) runtime_error(NULL, "查找函数的参数必须是字符串");
    
    Value *result = val_new(VAL_INT);
    char *p = strstr(argv[0]->sval, argv[1]->sval);
    result->ival = p ? (int)(p - argv[0]->sval) : -1;
    return result;
}

static Value *builtin_substring(int argc, Value **argv) {
    if (argc != 3) runtime_error(NULL, "截取函数需要3个参数");
    if (argv[0]->type != VAL_STRING) runtime_error(NULL, "截取函数的第一个参数必须是字符串");
    if (argv[1]->type != VAL_INT || argv[2]->type != VAL_INT) runtime_error(NULL, "截取函数的后两个参数必须是整数");
    
    int start = (int)argv[1]->ival;
    int len = (int)argv[2]->ival;
    const char *s = argv[0]->sval;
    int slen = strlen(s);
    
    if (start < 0) start = 0;
    if (start >= slen) start = slen - 1;
    if (len < 0) len = 0;
    if (start + len > slen) len = slen - start;
    
    Value *result = val_new(VAL_STRING);
    result->sval = malloc(len + 1);
    strncpy(result->sval, s + start, len);
    result->sval[len] = '\0';
    return result;
}

/* ========== 自动求导 / 张量算子 ========== */
#define CO_MAX_DIM 8

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
            double v = (val->type == VAL_FLOAT) ? val->fval : (val->type == VAL_INT ? val->ival : 0);
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
            int i = (int)idx->ival;
            if (i < 0 || i >= base->tval->size) runtime_error(tgt, "张量索引越界");
            double v = (val->type == VAL_FLOAT) ? val->fval : (val->type == VAL_INT ? val->ival : 0);
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
    int ndim = argv[0]->list_len; int *sh = malloc(ndim * sizeof(int)); int size = 1;
    for (int i = 0; i < ndim; i++) { sh[i] = (int)argv[0]->lval[i]->ival; size *= sh[i]; }
    Tensor *c = tensor_new(ndim, sh);
    Value *r = val_new(VAL_TENSOR); r->tval = c; free(sh); return r;
}
static Value *builtin_ones(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_LIST) runtime_error(NULL, "全一函数需要形状列表");
    int ndim = argv[0]->list_len; int *sh = malloc(ndim * sizeof(int)); int size = 1;
    for (int i = 0; i < ndim; i++) { sh[i] = (int)argv[0]->lval[i]->ival; size *= sh[i]; }
    Tensor *c = tensor_new(ndim, sh);
    for (int i = 0; i < size; i++) c->data[i] = 1.0f;
    Value *r = val_new(VAL_TENSOR); r->tval = c; free(sh); return r;
}
static Value *builtin_rand_tensor(int argc, Value **argv) {
    if (argc != 3 || argv[0]->type != VAL_LIST) runtime_error(NULL, "随机张量需要: 形状列表, 下限, 上限");
    int ndim = argv[0]->list_len; int *sh = malloc(ndim * sizeof(int)); int size = 1;
    for (int i = 0; i < ndim; i++) { sh[i] = (int)argv[0]->lval[i]->ival; size *= sh[i]; }
    double lo = (argv[1]->type == VAL_FLOAT) ? argv[1]->fval : (double)argv[1]->ival;
    double hi = (argv[2]->type == VAL_FLOAT) ? argv[2]->fval : (double)argv[2]->ival;
    Tensor *c = tensor_new(ndim, sh);
    for (int i = 0; i < size; i++) c->data[i] = (float)(lo + (double)rand() / RAND_MAX * (hi - lo));
    Value *r = val_new(VAL_TENSOR); r->tval = c; free(sh); return r;
}
static Value *builtin_randn(int argc, Value **argv) {
    if (argc != 3 || argv[0]->type != VAL_LIST) runtime_error(NULL, "高斯随机需要: 形状列表, 均值, 标准差");
    int ndim = argv[0]->list_len; int *sh = malloc(ndim * sizeof(int)); int size = 1;
    for (int i = 0; i < ndim; i++) { sh[i] = (int)argv[0]->lval[i]->ival; size *= sh[i]; }
    double mean = (argv[1]->type == VAL_FLOAT) ? argv[1]->fval : (double)argv[1]->ival;
    double std = (argv[2]->type == VAL_FLOAT) ? argv[2]->fval : (double)argv[2]->ival;
    Tensor *c = tensor_new(ndim, sh);
    for (int i = 0; i < size; i++) {
        double u1 = ((double)rand() + 1) / ((double)RAND_MAX + 1);
        double u2 = (double)rand() / RAND_MAX;
        double z = sqrt(-2 * log(u1)) * cos(2 * 3.14159265358979323846 * u2);
        c->data[i] = (float)(mean + std * z);
    }
    Value *r = val_new(VAL_TENSOR); r->tval = c; free(sh); return r;
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
    for (int i = 0; i < t->ndim; i++) { Value *e = val_new(VAL_INT); e->ival = t->shape[i]; r->lval[i] = e; }
    return r;
}
static Value *builtin_reshape(int argc, Value **argv) {
    if (argc != 2 || argv[0]->type != VAL_TENSOR || argv[1]->type != VAL_LIST) runtime_error(NULL, "重塑函数需要: 张量, 形状列表");
    Tensor *t = argv[0]->tval;
    int ndim = argv[1]->list_len; int *sh = malloc(ndim * sizeof(int)); int size = 1;
    for (int i = 0; i < ndim; i++) { sh[i] = (int)argv[1]->lval[i]->ival; size *= sh[i]; }
    if (size != t->size) runtime_error(NULL, "重塑前后元素总数不一致");
    Tensor *c = tensor_new(ndim, sh);
    memcpy(c->data, t->data, t->size * sizeof(float));
    Value *r = val_new(VAL_TENSOR); r->tval = c; free(sh); return r;
}
static Value *builtin_sum(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "求和需要1个张量");
    Value *r = val_new(VAL_TENSOR); r->tval = t_sum(argv[0]->tval); return r;
}
static Value *builtin_mean(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "均值需要1个张量");
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
    double lr = (argv[0]->type == VAL_FLOAT) ? argv[0]->fval : (double)argv[0]->ival;
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
static Value *builtin_get(int argc, Value **argv) {
    if (argc < 2 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "取函数需要: 张量, 索引...");
    Tensor *t = argv[0]->tval;
    int i = (int)argv[1]->ival;
    if (t->ndim == 2) {
        int j = (argc >= 3) ? (int)argv[2]->ival : 0;
        Value *r = val_new(VAL_FLOAT); r->fval = t->data[i * t->shape[1] + j]; return r;
    }
    Value *r = val_new(VAL_FLOAT); r->fval = t->data[i]; return r;
}
static Value *builtin_set(int argc, Value **argv) {
    if (argc < 3 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "置函数需要: 张量, 索引, 值");
    Tensor *t = argv[0]->tval;
    int i = (int)argv[1]->ival;
    double v = (argv[argc - 1]->type == VAL_FLOAT) ? argv[argc - 1]->fval : (double)argv[argc - 1]->ival;
    if (t->ndim == 2 && argc >= 4) { int j = (int)argv[2]->ival; t->data[i * t->shape[1] + j] = (float)v; }
    else t->data[i] = (float)v;
    return val_new(VAL_NULL);
}
static Value *builtin_requires_grad(int argc, Value **argv) {
    if (argc != 1 || argv[0]->type != VAL_TENSOR) runtime_error(NULL, "需梯度函数需要1个张量");
    argv[0]->tval->requires_grad = 1;
    Value *r = val_new(VAL_TENSOR); r->tval = tensor_retain(argv[0]->tval); return r;
}

/* 调用内置函数 */
static Value *call_builtin(const char *name, int argc, Value **argv) {
    if (strcmp(name, "长度") == 0) return builtin_len(argc, argv);
    if (strcmp(name, "类型") == 0) return builtin_type(argc, argv);
    if (strcmp(name, "转整数") == 0) return builtin_int(argc, argv);
    if (strcmp(name, "转浮点") == 0) return builtin_float(argc, argv);
    if (strcmp(name, "转字符串") == 0) return builtin_str(argc, argv);
    if (strcmp(name, "输入") == 0) return builtin_input(argc, argv);
    /* 数学函数 */
    if (strcmp(name, "绝对值") == 0) return builtin_abs(argc, argv);
    if (strcmp(name, "平方根") == 0) return builtin_sqrt(argc, argv);
    if (strcmp(name, "随机数") == 0) return builtin_random(argc, argv);
    if (strcmp(name, "取整") == 0) return builtin_round(argc, argv);
    if (strcmp(name, "最大值") == 0) return builtin_max(argc, argv);
    if (strcmp(name, "最小值") == 0) return builtin_min(argc, argv);
    /* 列表函数 */
    if (strcmp(name, "追加") == 0) return builtin_append(argc, argv);
    if (strcmp(name, "删除") == 0) return builtin_remove(argc, argv);
    if (strcmp(name, "排序") == 0) return builtin_sort(argc, argv);
    if (strcmp(name, "包含") == 0) return builtin_contains(argc, argv);
    /* 字符串函数 */
    if (strcmp(name, "分割") == 0) return builtin_split(argc, argv);
    if (strcmp(name, "连接") == 0) return builtin_join(argc, argv);
    if (strcmp(name, "替换") == 0) return builtin_replace(argc, argv);
    if (strcmp(name, "查找") == 0) return builtin_find(argc, argv);
    if (strcmp(name, "截取") == 0) return builtin_substring(argc, argv);
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

static Value *eval_node(Node *n, Env *env) {
    if (!n) return val_new(VAL_NULL);
    switch (n->type) {
        case ND_LITERAL: {
            Value *v = val_new(n->literal.lit_type);
            switch (n->literal.lit_type) {
                case VAL_INT: v->ival = atol(n->literal.value); break;
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
                    int i = (int)idx->ival;
                    if (i < 0 || i >= l->list_len) runtime_error(n, "列表索引越界: %d (列表长度 %d)", i, l->list_len);
                    val_free(idx);
                    return val_retain(l->lval[i]);
                } else if (l->type == VAL_STRING) {
                    int i = (int)idx->ival;
                    int slen = strlen(l->sval);
                    if (i < 0 || i >= slen) runtime_error(n, "字符串索引越界: %d", i);
                    char tmp[2] = { l->sval[i], 0 };
                    val_free(idx);
                    Value *r = val_new(VAL_STRING);
                    r->sval = strdup(tmp);
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
                    int i = (int)idx->ival;
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
            
            /* 逻辑运算符 */
            if (strcmp(n->binary.op, "且") == 0) {
                int li = (l->type == VAL_INT) ? (int)l->ival : (l->type == VAL_FLOAT ? l->fval != 0 : 0);
                if (!li) {
                    val_free(l);
                    Value *r = val_new(VAL_INT);
                    r->ival = 0;
                    return r;
                }
                Value *r = eval_node(n->binary.right, env);
                int ri = (r->type == VAL_INT) ? (int)r->ival : (r->type == VAL_FLOAT ? r->fval != 0 : 0);
                val_free(l); val_free(r);
                Value *result = val_new(VAL_INT);
                result->ival = ri ? 1 : 0;
                return result;
            }
            if (strcmp(n->binary.op, "或") == 0) {
                int li = (l->type == VAL_INT) ? (int)l->ival : (l->type == VAL_FLOAT ? l->fval != 0 : 0);
                if (li) {
                    val_free(l);
                    Value *r = val_new(VAL_INT);
                    r->ival = 1;
                    return r;
                }
                Value *r = eval_node(n->binary.right, env);
                int ri = (r->type == VAL_INT) ? (int)r->ival : (r->type == VAL_FLOAT ? r->fval != 0 : 0);
                val_free(l); val_free(r);
                Value *result = val_new(VAL_INT);
                result->ival = ri ? 1 : 0;
                return result;
            }
            
            Value *r = eval_node(n->binary.right, env);
            /* 字符串拼接 */
            if (l->type == VAL_STRING || r->type == VAL_STRING) {
                char buf[4096];
                buf[0] = '\0';
                if (l->type == VAL_STRING) {
                    strncpy(buf, l->sval, sizeof(buf) - 1);
                    buf[sizeof(buf) - 1] = '\0';
                } else if (l->type == VAL_INT) {
                    snprintf(buf, sizeof(buf), "%ld", l->ival);
                } else if (l->type == VAL_FLOAT) {
                    snprintf(buf, sizeof(buf), "%g", l->fval);
                }
                size_t buf_len = strlen(buf);
                if (r->type == VAL_STRING) {
                    if (buf_len + strlen(r->sval) < sizeof(buf)) {
                        strcat(buf, r->sval);
                    }
                } else if (r->type == VAL_INT) {
                    char tmp[64];
                    snprintf(tmp, sizeof(tmp), "%ld", r->ival);
                    if (buf_len + strlen(tmp) < sizeof(buf)) strcat(buf, tmp);
                } else if (r->type == VAL_FLOAT) {
                    char tmp[64];
                    snprintf(tmp, sizeof(tmp), "%g", r->fval);
                    if (buf_len + strlen(tmp) < sizeof(buf)) strcat(buf, tmp);
                }
                val_free(l); val_free(r);
                Value *sv = val_new(VAL_STRING);
                sv->sval = strdup(buf);
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
            /* 普通数值 */
            double ld = (l->type == VAL_FLOAT) ? l->fval : l->ival;
            double rd = (r->type == VAL_FLOAT) ? r->fval : r->ival;
            double result = 0;
            if (strcmp(n->binary.op, "+") == 0) result = ld + rd;
            else if (strcmp(n->binary.op, "-") == 0) result = ld - rd;
            else if (strcmp(n->binary.op, "*") == 0) result = ld * rd;
            else if (strcmp(n->binary.op, "/") == 0) result = rd != 0 ? ld / rd : 0;
            else if (strcmp(n->binary.op, "%") == 0) result = (long)ld % (long)rd;
            else if (strcmp(n->binary.op, "<") == 0) result = ld < rd;
            else if (strcmp(n->binary.op, ">") == 0) result = ld > rd;
            else if (strcmp(n->binary.op, "<=") == 0) result = ld <= rd;
            else if (strcmp(n->binary.op, ">=") == 0) result = ld >= rd;
            else if (strcmp(n->binary.op, "==") == 0) result = ld == rd;
            else if (strcmp(n->binary.op, "!=") == 0) result = ld != rd;
            else runtime_error(n, "无效的二元运算符: %s", n->binary.op);
            int is_float = (l->type == VAL_FLOAT || r->type == VAL_FLOAT);
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
                if (v->type == VAL_INT) {
                    v->ival = -v->ival;
                } else if (v->type == VAL_FLOAT) {
                    v->fval = -v->fval;
                }
                return v;
            } else if (strcmp(n->unary.op, "not") == 0) {
                int b = (v->type == VAL_INT) ? (int)v->ival : (v->type == VAL_FLOAT ? v->fval != 0 : 0);
                val_free(v);
                Value *r = val_new(VAL_INT);
                r->ival = b ? 0 : 1;
                return r;
            }
            return v;
        }
        case ND_LIST_LIT: {
            Value *list = val_new(VAL_LIST);
            list->list_len = n->list_lit.ecnt;
            list->lval = malloc(list->list_len * sizeof(Value*));
            for (int i = 0; i < n->list_lit.ecnt; i++) {
                list->lval[i] = eval_node(n->list_lit.elements[i], env);
            }
            return list;
        }
        case ND_MAP_LIT: {
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
                    Func *fn = fnval->fnval;
                    if (fn->param_count != n->func_call.acnt) runtime_error(n, "函数 %s 参数数量错误: 期望 %d, 得到 %d",
                        fn->name, fn->param_count, n->func_call.acnt);
                    Value **args = malloc(n->func_call.acnt * sizeof(Value*));
                    for (int i = 0; i < n->func_call.acnt; i++) {
                        args[i] = eval_node(n->func_call.args[i], env);
                    }
                    Env *call_env = env_new(env);
                    for (int i = 0; i < fn->param_count; i++) {
                        env_set(call_env, fn->params[i], args[i]);
                        val_free(args[i]);
                    }
                    free(args);
                    Node *body = (Node*)fn->body;
                    Value *ret = NULL;
                    g_return_flag = 0;
                    g_return_value = NULL;
                    for (int i = 0; i < body->prog.cnt; i++) {
                        if (g_return_flag) break;
                        exec_node(body->prog.stmts[i], call_env);
                    }
                    ret = g_return_flag ? g_return_value : NULL;
                    g_return_flag = 0;
                    g_return_value = NULL;
                    /* 清理调用环境 */
                    for (int i = 0; i < call_env->count; i++) {
                        free(call_env->names[i]);
                        val_free(call_env->values[i]);
                    }
                    free(call_env);
                    /* 注意: fnval 来自 env_get，不额外 retain，不能 free */
                    return ret ? ret : val_new(VAL_NULL);
                }
                /* fnval 来自 env_get，不额外 retain，不能 free */
                /* 不是用户函数，尝试内置函数 */
                Value **args = malloc(n->func_call.acnt * sizeof(Value*));
                for (int i = 0; i < n->func_call.acnt; i++) {
                    args[i] = eval_node(n->func_call.args[i], env);
                }
                Value *result = call_builtin(n->func_call.callee->ident.name, n->func_call.acnt, args);
                for (int i = 0; i < n->func_call.acnt; i++) {
                    val_free(args[i]);
                }
                free(args);
                return result;
            }
            Value *fnval = eval_node(n->func_call.callee, env);
            if (fnval->type != VAL_FUNC) runtime_error(n, "值不是一个函数");
            Func *fn = fnval->fnval;
            if (fn->param_count != n->func_call.acnt) runtime_error(n, "函数 %s 参数数量错误: 期望 %d, 得到 %d",
                fn->name, fn->param_count, n->func_call.acnt);
            Env *call_env = env_new(env);
            for (int i = 0; i < fn->param_count; i++) {
                Value *arg = eval_node(n->func_call.args[i], env);
                env_set(call_env, fn->params[i], arg);
                val_free(arg);
            }
            Node *body = (Node*)fn->body;
            Value *ret = NULL;
            for (int i = 0; i < body->prog.cnt; i++) {
                if (body->prog.stmts[i]->type == ND_RETURN) {
                    ret = eval_node(body->prog.stmts[i]->ret_stmt.val, call_env);
                    break;
                }
                exec_node(body->prog.stmts[i], call_env);
            }
            for (int i = 0; i < call_env->count; i++) {
                free(call_env->names[i]);
                val_free(call_env->values[i]);
            }
            free(call_env);
            val_free(fnval);
            return ret ? ret : val_new(VAL_NULL);
        }
        default:
            runtime_error(n, "无法求值的节点类型: %d", n->type);
            return val_new(VAL_NULL);
    }
}

/* 执行语句 */
static void exec_node(Node *n, Env *env) {
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
            env_set(env, n->var_decl.name, init);
            val_free(init);
            break;
        }
        case ND_ASSIGN: {
            Value *val = eval_node(n->assign.val, env);
            if (n->assign.tgt->type == ND_IDENT) {
                Value *old = env_get(env, n->assign.tgt->ident.name);
                if (old && old->type == VAL_FUNC) runtime_error(n->assign.tgt, "不能给函数名赋值");
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
                    int i = (int)idx->ival;
                    if (i < 0) runtime_error(n->assign.tgt, "索引不能为负数");
                    if (i >= obj->list_len) {
                        Value **new_list = realloc(obj->lval, (i+1) * sizeof(Value*));
                        for (int j = obj->list_len; j <= i; j++) new_list[j] = val_new(VAL_NULL);
                        obj->lval = new_list;
                        obj->list_len = i+1;
                    }
                    val_free(obj->lval[i]);
                    obj->lval[i] = val_retain(val);
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
            int cond = (condv->type == VAL_INT) ? (int)condv->ival : (condv->type == VAL_FLOAT ? condv->fval != 0 : 0);
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
                        int elval = (elcond->type == VAL_INT) ? (int)elcond->ival : (elcond->type == VAL_FLOAT ? elcond->fval != 0 : 0);
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
                Value *c = eval_node(n->while_stmt.cond, env);
                int cond = (c->type == VAL_INT) ? (int)c->ival : (c->type == VAL_FLOAT ? c->fval != 0 : 0);
                val_free(c);
                if (!cond) break;
                for (int i = 0; i < n->while_stmt.bcnt; i++) {
                    exec_node(n->while_stmt.body[i], env);
                    if (ctx.should_break) break;
                }
            }
            
            loop_ctx = ctx.parent;
            break;
        }
        case ND_FOR: {
            Value *start_v = eval_node(n->for_stmt.start, env);
            Value *end_v = eval_node(n->for_stmt.end, env);
            double start = (start_v->type == VAL_FLOAT) ? start_v->fval : start_v->ival;
            double end = (end_v->type == VAL_FLOAT) ? end_v->fval : end_v->ival;
            double step = 1.0;
            if (n->for_stmt.step) {
                Value *step_v = eval_node(n->for_stmt.step, env);
                step = (step_v->type == VAL_FLOAT) ? step_v->fval : step_v->ival;
                val_free(step_v);
            }
            val_free(start_v);
            val_free(end_v);
            
            LoopContext ctx;
            ctx.in_loop = 1;
            ctx.should_break = 0;
            ctx.should_continue = 0;
            ctx.parent = loop_ctx;
            loop_ctx = &ctx;
            
            int is_float = (n->for_stmt.start->literal.lit_type == VAL_FLOAT || 
                           n->for_stmt.end->literal.lit_type == VAL_FLOAT ||
                           n->for_stmt.step != NULL);
            
            if (is_float) {
                double i = start;
                int direction = (step > 0) ? 1 : -1;
                while ((direction > 0 && i <= end + 0.0001) || (direction < 0 && i >= end - 0.0001)) {
                    Value *iv = val_new(VAL_FLOAT);
                    iv->fval = i;
                    env_set(env, n->for_stmt.var, iv);
                    val_free(iv);
                    
                    for (int j = 0; j < n->for_stmt.bcnt; j++) {
                        exec_node(n->for_stmt.body[j], env);
                        if (ctx.should_break) break;
                    }
                    if (ctx.should_break) break;
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
                    
                    for (int j = 0; j < n->for_stmt.bcnt; j++) {
                        exec_node(n->for_stmt.body[j], env);
                        if (ctx.should_break) break;
                    }
                    if (ctx.should_break) break;
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
            fn->body = n->func_def.body;
            Value *fnv = val_new(VAL_FUNC);
            fnv->fnval = fn;
            env_set(env, fn->name, fnv);
            val_free(fnv);
            break;
        }
        case ND_RETURN: {
            Value *rv = (n->ret_stmt.val) ? eval_node(n->ret_stmt.val, env) : val_new(VAL_NULL);
            g_return_flag = 1;
            g_return_value = rv;
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
    srand((unsigned int)time(NULL));
    
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

    for (int i = 0; i < prog->prog.cnt; i++) {
        exec_node(prog->prog.stmts[i], global);
    }

    for (int i = 0; i < global->count; i++) {
        free(global->names[i]);
        val_free(global->values[i]);
    }
    free(global);
    node_free(prog);
    free(buf);
    return 0;
}
