#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* 最小动态值：整数/浮点/布尔/字符串/空，标签联合（0.0.1 自举子集产物用） */
typedef struct { int t; long long i; double f; char *s; } CoVal;
#define K_NULL 0
#define K_INT  1
#define K_FLT  2
#define K_BOOL 3
#define K_STR  4

static CV_UNUSED CoVal cvi(long long x){ CoVal v; v.t=K_INT; v.i=x; v.f=0; v.s=0; return v; }
static CV_UNUSED CoVal cvf(double x){ CoVal v; v.t=K_FLT; v.i=0; v.f=x; v.s=0; return v; }
static CV_UNUSED CoVal cvb(int x){ CoVal v; v.t=K_BOOL; v.i=x; v.f=0; v.s=0; return v; }
static CV_UNUSED CoVal cvnull(void){ CoVal v; v.t=K_NULL; v.i=0; v.f=0; v.s=0; return v; }
static CV_UNUSED char *cvdup(const char *s){ size_t n=s?strlen(s):0; char *p=(char*)malloc(n+1); if(!p){fprintf(stderr,"内存不足\n");exit(1);} if(s)memcpy(p,s,n); p[n]=0; return p; }
static CV_UNUSED CoVal cvs(char *s){ CoVal v; v.t=K_STR; v.i=0; v.f=0; v.s=s; return v; }
static CV_UNUSED CoVal cvss(const char *s){ return cvs(cvdup(s)); }
static CV_UNUSED int cvtruthy(CoVal v){ if(v.t==K_NULL)return 0; if(v.t==K_INT)return v.i!=0; if(v.t==K_FLT)return v.f!=0.0; if(v.t==K_BOOL)return v.i!=0; if(v.t==K_STR)return v.s&&v.s[0]!=0; return 0; }
static CV_UNUSED int cvisnum(CoVal v){ return v.t==K_INT||v.t==K_FLT||v.t==K_BOOL; }
static CV_UNUSED double cvnum(CoVal v){ if(v.t==K_INT||v.t==K_BOOL)return (double)v.i; if(v.t==K_FLT)return v.f; fprintf(stderr,"运行错误: 需要数值，收到非数值\n"); exit(1); return 0; }
static CV_UNUSED char *cvi2s(CoVal v){ char b[64]; switch(v.t){ case K_INT: snprintf(b,64,"%lld",v.i); return cvdup(b); case K_FLT: snprintf(b,64,"%g",v.f); return cvdup(b); case K_BOOL: return cvdup(v.i?"真":"假"); case K_STR: return v.s?cvdup(v.s):cvdup(""); case K_NULL: return cvdup("空"); } return cvdup("?"); }
static CV_UNUSED int cveq(CoVal a, CoVal b){
    if(a.t==b.t){ if(a.t==K_INT||a.t==K_BOOL) return a.i==b.i; if(a.t==K_FLT) return a.f==b.f; if(a.t==K_STR) return (a.s&&b.s)?(strcmp(a.s,b.s)==0):(a.s==b.s); if(a.t==K_NULL) return 1; }
    if(cvisnum(a)&&cvisnum(b)) return cvnum(a)==cvnum(b); return 0;
}
static CV_UNUSED int cvcmp(CoVal a, CoVal b){ if(cvisnum(a)&&cvisnum(b)){ double x=cvnum(a),y=cvnum(b); return x<y?-1:(x>y?1:0); } if(a.t==K_STR&&b.t==K_STR) return strcmp(a.s?a.s:"",b.s?b.s:""); fprintf(stderr,"运行错误: 无法比较大小\n"); exit(1); return 0; }
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