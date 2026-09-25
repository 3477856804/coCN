#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>

/* 单行 if(...) return; return 0; 会让 -Wmisleading-indentation 误报（语义正确）。
   生成产物交给使用方编译，必须零告警，这里统一抑制该误导性缩进告警。 */
#if defined(__GNUC__)
# pragma GCC diagnostic ignored "-Wmisleading-indentation"
#endif

#ifndef CV_UNUSED
# if defined(__GNUC__) || defined(__clang__)
#  define CV_UNUSED __attribute__((unused))
# else
#  define CV_UNUSED
# endif
#endif

/* Windows：文件函数统一走宽字符 API，中文文件名/命令串与解释器侧行为对齐 */
#ifdef _WIN32
#include <windows.h>
static FILE *bo_fopen_utf8(const char *path, const char *mode){
    int wn = MultiByteToWideChar(CP_UTF8,0,path,-1,NULL,0);
    int wm = MultiByteToWideChar(CP_UTF8,0,mode,-1,NULL,0);
    if(wn<=0||wm<=0) return NULL;
    wchar_t *wp=(wchar_t*)malloc((size_t)wn*sizeof(wchar_t));
    wchar_t *wm2=(wchar_t*)malloc((size_t)wm*sizeof(wchar_t));
    if(!wp||!wm2){ free(wp); free(wm2); return NULL; }
    MultiByteToWideChar(CP_UTF8,0,path,-1,wp,wn);
    MultiByteToWideChar(CP_UTF8,0,mode,-1,wm2,wm);
    FILE *f=_wfopen(wp,wm2);
    free(wp); free(wm2);
    return f;
}
static int bo_remove_utf8(const char *path){
    int wn=MultiByteToWideChar(CP_UTF8,0,path,-1,NULL,0);
    if(wn<=0) return -1;
    wchar_t *wp=(wchar_t*)malloc((size_t)wn*sizeof(wchar_t));
    if(!wp) return -1;
    MultiByteToWideChar(CP_UTF8,0,path,-1,wp,wn);
    int r=_wremove(wp);
    free(wp);
    return r;
}
static int bo_system_utf8(const char *cmd8){
    int wn = MultiByteToWideChar(CP_UTF8,0,cmd8,-1,NULL,0);
    if(wn<=0) return -1;
    wchar_t *wcmd=(wchar_t*)malloc((size_t)wn*sizeof(wchar_t));
    if(!wcmd) return -1;
    MultiByteToWideChar(CP_UTF8,0,cmd8,-1,wcmd,wn);
    int r=_wsystem(wcmd);
    free(wcmd);
    return r;
}
#define fopen bo_fopen_utf8
#define remove bo_remove_utf8
#define system bo_system_utf8
#endif

/* 动态值：整数/浮点/布尔/字符串/空/列表，标签联合（自举子集产物用） */
typedef struct LList LList;
typedef struct { int t; long long i; double f; char *s; LList *l; } CoVal;
struct LList { CoVal *items; long long len; long long cap; };
#define K_NULL 0
#define K_INT  1
#define K_FLT  2
#define K_BOOL 3
#define K_STR  4
#define K_LIST 5

static CV_UNUSED void bo_die(const char *msg){ fprintf(stderr,"运行错误: %s\n",msg); exit(1); }
static CV_UNUSED CoVal cvi(long long x){ CoVal v; v.t=K_INT; v.i=x; v.f=0; v.s=0; v.l=0; return v; }
static CV_UNUSED CoVal cvf(double x){ CoVal v; v.t=K_FLT; v.i=0; v.f=x; v.s=0; v.l=0; return v; }
static CV_UNUSED CoVal cvb(int x){ CoVal v; v.t=K_BOOL; v.i=x?1:0; v.f=0; v.s=0; v.l=0; return v; }
static CV_UNUSED CoVal cvnull(void){ CoVal v; v.t=K_NULL; v.i=0; v.f=0; v.s=0; v.l=0; return v; }
static CV_UNUSED char *cvdup(const char *s){ size_t n=s?strlen(s):0; char *p=(char*)malloc(n+1); if(!p)bo_die("内存不足"); if(s)memcpy(p,s,n); p[n]=0; return p; }
static CV_UNUSED CoVal cvs(char *s){ CoVal v; v.t=K_STR; v.i=0; v.f=0; v.s=s; v.l=0; return v; }
static CV_UNUSED CoVal cvss(const char *s){ return cvs(cvdup(s)); }
static CV_UNUSED CoVal cvlist(void){ CoVal v; v.t=K_LIST; v.i=0; v.f=0; v.s=0; v.l=(LList*)calloc(1,sizeof(LList)); if(!v.l)bo_die("内存不足"); return v; }
static CV_UNUSED CoVal cvlistn(int n, ...){
    CoVal v=cvlist(); va_list ap; va_start(ap,n);
    for(int k=0;k<n;k++){ CoVal e=va_arg(ap,CoVal); if(v.l->len>=v.l->cap){ long long nc=v.l->cap?v.l->cap*2:4; CoVal *ni=(CoVal*)realloc(v.l->items,(size_t)nc*sizeof(CoVal)); if(!ni)bo_die("内存不足"); v.l->items=ni; v.l->cap=nc; } v.l->items[v.l->len]=e; v.l->len++; }
    va_end(ap); return v;
}
static CV_UNUSED int cvtruthy(CoVal v){ if(v.t==K_NULL)return 0; if(v.t==K_INT)return v.i!=0; if(v.t==K_FLT)return v.f!=0.0; if(v.t==K_BOOL)return v.i!=0; if(v.t==K_STR)return v.s&&v.s[0]!=0; if(v.t==K_LIST)return v.l&&v.l->len>0; return 0; }
static CV_UNUSED int cvisnum(CoVal v){ return v.t==K_INT||v.t==K_FLT||v.t==K_BOOL; }
static CV_UNUSED double cvnum(CoVal v){ if(v.t==K_INT||v.t==K_BOOL)return (double)v.i; if(v.t==K_FLT)return v.f; bo_die("需要数值，收到非数值"); return 0; }
static CV_UNUSED char *cvi2s(CoVal v){
    char b[64];
    switch(v.t){
        case K_INT: snprintf(b,64,"%lld",v.i); return cvdup(b);
        case K_FLT: snprintf(b,64,"%g",v.f); return cvdup(b);
        case K_BOOL: return cvdup(v.i?"真":"假");
        case K_STR: return v.s?cvdup(v.s):cvdup("");
        case K_NULL: return cvdup("空");
        case K_LIST: {
            /* [a, b, c] 形式；递归用 cvdup 字符串拼接 */
            char *out=cvdup("[");
            if(v.l) for(long long k=0;k<v.l->len;k++){
                char *e=cvi2s(v.l->items[k]);
                size_t n=strlen(out)+strlen(e)+3; char *o=(char*)malloc(n); if(!o)bo_die("内存不足");
                if(k==0) snprintf(o,n,"[%s",e); else snprintf(o,n,"%s, %s",out,e);
                free(out); free(e); out=o;
            }
            size_t n2=strlen(out)+2; char *o2=(char*)malloc(n2); if(!o2)bo_die("内存不足");
            snprintf(o2,n2,"%s]",out); free(out); return o2;
        }
    }
    return cvdup("?");
}
static CV_UNUSED int cvlen(CoVal v){ if(v.t==K_STR)return (int)(v.s?strlen(v.s):0); if(v.t==K_LIST)return (int)(v.l?v.l->len:0); bo_die("长度 需要字符串或列表"); return 0; }
static CV_UNUSED CoVal cvat(CoVal v, CoVal i){
    long long idx=(long long)cvnum(i);
    if(v.t==K_STR){ const char *s=v.s?v.s:""; long long n=(long long)strlen(s); long long k=idx; if(k<0)k+=n; if(k<0||k>=n){ char m[128]; snprintf(m,128,"字符串索引越界: %lld（长度 %lld）",idx,n); bo_die(m); } char b[2]; b[0]=s[k]; b[1]=0; return cvs(cvdup(b)); }
    if(v.t==K_LIST){ if(!v.l)bo_die("索引的目标不是列表"); long long n=v.l->len; long long k=idx; if(k<0)k+=n; if(k<0||k>=n){ char m[128]; snprintf(m,128,"列表索引越界: %lld（长度 %lld）",idx,n); bo_die(m); } return v.l->items[k]; }
    bo_die("索引的目标必须是字符串或列表"); return cvnull();
}
static CV_UNUSED CoVal cvlist_push(CoVal v, CoVal e){ if(v.t!=K_LIST||!v.l)bo_die("追加的目标不是列表"); if(v.l->len>=v.l->cap){ long long nc=v.l->cap?v.l->cap*2:4; CoVal *ni=(CoVal*)realloc(v.l->items,(size_t)nc*sizeof(CoVal)); if(!ni)bo_die("内存不足"); v.l->items=ni; v.l->cap=nc; } v.l->items[v.l->len]=e; v.l->len++; return v; }
static CV_UNUSED int cveq(CoVal a, CoVal b){
    if(a.t==b.t){
        if(a.t==K_INT||a.t==K_BOOL) return a.i==b.i;
        if(a.t==K_FLT) return a.f==b.f;
        if(a.t==K_STR) return (a.s&&b.s)?(strcmp(a.s,b.s)==0):(a.s==b.s);
        if(a.t==K_NULL) return 1;
        if(a.t==K_LIST){ if((a.l?a.l->len:0)!=(b.l?b.l->len:0))return 0; long long n=a.l?a.l->len:0; for(long long k=0;k<n;k++) if(!cveq(a.l->items[k],b.l->items[k])) return 0; return 1; }
    }
    if(cvisnum(a)&&cvisnum(b)) return cvnum(a)==cvnum(b); return 0;
}
static CV_UNUSED int cvcmp(CoVal a, CoVal b){ if(cvisnum(a)&&cvisnum(b)){ double x=cvnum(a),y=cvnum(b); return x<y?-1:(x>y?1:0); } if(a.t==K_STR&&b.t==K_STR) return strcmp(a.s?a.s:"",b.s?b.s:""); bo_die("无法比较大小"); return 0; }
static CV_UNUSED CoVal vneg(CoVal v){ if(v.t==K_INT||v.t==K_BOOL)return cvi(-v.i); if(v.t==K_FLT)return cvf(-v.f); bo_die("一元负号需要数值"); return cvnull(); }
static CV_UNUSED CoVal vbin(CoVal L, const char *op, CoVal R){
    if(strcmp(op,"==")==0)return cvb(cveq(L,R));
    if(strcmp(op,"!=")==0)return cvb(!cveq(L,R));
    if(strcmp(op,"<")==0)return cvb(cvcmp(L,R)<0);
    if(strcmp(op,">")==0)return cvb(cvcmp(L,R)>0);
    if(strcmp(op,"<=")==0)return cvb(cvcmp(L,R)<=0);
    if(strcmp(op,">=")==0)return cvb(cvcmp(L,R)>=0);
    if(L.t==K_STR||R.t==K_STR){
        if(strcmp(op,"+")!=0){ fprintf(stderr,"运行错误: 字符串只支持 + 拼接，不支持 %s\n",op); exit(1); }
        char *ls=cvi2s(L), *rs=cvi2s(R); size_t nl=strlen(ls)+strlen(rs)+1; char *o=(char*)malloc(nl); if(!o){fprintf(stderr,"内存不足\n");exit(1);} snprintf(o,nl,"%s%s",ls,rs); free(ls); free(rs); return cvs(o);
    }
    if(!cvisnum(L)||!cvisnum(R)){ fprintf(stderr,"运行错误: 不支持的运算：操作数类型不兼容\n"); exit(1); }
    double l=cvnum(L), r=cvnum(R), x=0; int isf=(L.t==K_FLT||R.t==K_FLT);
    if(strcmp(op,"+")==0)x=l+r; else if(strcmp(op,"-")==0)x=l-r; else if(strcmp(op,"*")==0)x=l*r;
    else if(strcmp(op,"/")==0){ if(r==0){fprintf(stderr,"运行错误: 除以零\n");exit(1);} x=l/r; isf=1; }
    else if(strcmp(op,"//")==0){ if(r==0){fprintf(stderr,"运行错误: 整除的除数为零\n");exit(1);} x=floor(l/r); }
    else if(strcmp(op,"%")==0){ if(r==0){fprintf(stderr,"运行错误: 取模的除数为零\n");exit(1);} x=fmod(l,r); }
    else { fprintf(stderr,"运行错误: 无效二元运算符 %s\n",op); exit(1); }
    return isf?cvf(x):cvi((long long)x);
}
static CV_UNUSED void cvprint(CoVal v){ char *s=cvi2s(v); printf("%s",s); free(s); }
static CV_UNUSED CoVal cveprint(CoVal v){ char *s=cvi2s(v); fprintf(stderr,"%s",s); free(s); return cvnull(); }
static CV_UNUSED CoVal cvstr(CoVal v){ return cvss(cvi2s(v)); }
static CV_UNUSED CoVal cvslice(CoVal s, CoVal st, CoVal ln){
    if(s.t!=K_STR)bo_die("截取 需要字符串");
    const char *src=s.s?s.s:""; long long n=(long long)strlen(src);
    long long a=(long long)cvnum(st), L=(long long)cvnum(ln);
    if(a<0)a=0; if(a>n)a=n; if(L<0)L=0; if(a+L>n)L=n-a;
    char *o=(char*)malloc((size_t)L+1); if(!o)bo_die("内存不足");
    memcpy(o,src+a,(size_t)L); o[L]=0; return cvs(o);
}
static CV_UNUSED CoVal cvreplace(CoVal s, CoVal o, CoVal nw){
    if(s.t!=K_STR||o.t!=K_STR||nw.t!=K_STR)bo_die("替换 需要三个字符串");
    const char *src=s.s?s.s:"", *old=o.s?o.s:"", *neu=nw.s?nw.s:"";
    if(!old[0])bo_die("替换函数的被替换串不能为空");
    long long cnt=0; const char *p=src; while((p=strstr(p,old))){ cnt++; p+=strlen(old); }
    size_t sz=strlen(src)+cnt*(strlen(neu)-strlen(old))+1;
    char *out=(char*)malloc(sz); if(!out)bo_die("内存不足");
    char *w=out; p=src; while((p=strstr(p,old))){ size_t k=p-src; memcpy(w,src,k); w+=k; memcpy(w,neu,strlen(neu)); w+=strlen(neu); p+=strlen(old); src=p; }
    strcpy(w,src); return cvs(out);
}
static CV_UNUSED CoVal cvreadfile(CoVal p){
    if(p.t!=K_STR)bo_die("读文件 需要路径字符串");
    FILE *f=fopen(p.s,"rb"); if(!f){ fprintf(stderr,"运行错误: 无法打开文件: %s\n",p.s); exit(1); }
    fseek(f,0,SEEK_END); long sz=ftell(f); rewind(f);
    char *b=(char*)malloc((size_t)sz+1); if(!b)bo_die("内存不足");
    size_t got=fread(b,1,(size_t)sz,f); fclose(f); b[got]=0; return cvs(b);
}
static CV_UNUSED CoVal cvwritefile(CoVal p, CoVal c){
    if(p.t!=K_STR)bo_die("写文件 需要路径字符串");
    char *cs=cvi2s(c); FILE *f=fopen(p.s,"wb"); if(!f){ fprintf(stderr,"运行错误: 无法写入文件: %s\n",p.s); free(cs); exit(1); }
    size_t n=strlen(cs); size_t w=fwrite(cs,1,n,f); int ce=fclose(f); free(cs);
    if(w!=n||ce!=0)bo_die("写入文件未完成");
    return cvi((long long)n);
}
static CV_UNUSED CoVal cvfileexists(CoVal p){
    if(p.t!=K_STR)bo_die("文件存在 需要路径字符串");
    FILE *f=fopen(p.s,"rb"); if(f){ fclose(f); return cvb(1); } return cvb(0);
}
static CV_UNUSED CoVal cvdeletefile(CoVal p){
    if(p.t!=K_STR)bo_die("删除文件 需要路径字符串");
    return cvb(remove(p.s)==0);
}
static CV_UNUSED CoVal cvsystem(CoVal p){
    if(p.t!=K_STR)bo_die("执行命令 需要命令字符串");
    return cvi((long long)system(p.s));
}
