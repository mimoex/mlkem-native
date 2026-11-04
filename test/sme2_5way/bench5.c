#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <inttypes.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include "math_core.h"
#include "partial_scalar.h"
#include "direct_core.h"
#include "ntt_bridge.h"
#if defined(__aarch64__) && defined(__APPLE__)
#include <arm_sme.h>
#endif
#include "mlkem/src/common.h"
#include "mlkem/src/native/aarch64/sme2/mlkem_sme2_backend.h"

#if !defined(__aarch64__) || !defined(__APPLE__)
#error "5-way benchmark with SME2 requires an Apple Silicon macOS host"
#endif

#define B5_A 0
#define B5_B 1
#define B5_C 2
#define B5_D 3
#define B5_E 4
#define B5_MODES 5
static const char *b5_names[B5_MODES]={
    "A_direct_sme2", "B_ntt3_scalar", "C_ntt3_sme2",
    "D_native_full_ntt", "E_portable_c_full_ntt"
};

typedef struct {
    int16_t packed[3][128][32];
    int32_t raw[8][4][2][256];
} b5_direct_scratch;

typedef struct {
    bw_direct_ctx *a;
    bw_scalar_ctx *b;
    mlk_sme2_public_ctx *c;
    void *d;
    void *e;
    b5_direct_scratch *as;
    mlk_sme2_rolling_workspace *cs;
    int16_t (*pub)[3][256];
    int16_t (*sec)[3][256];
    int16_t (*out)[4][256];
    int16_t (*reference)[4][256];
    size_t byte_d,byte_e;
    uint64_t checksum;
} b5_state;

extern void bench5_direct_smopa(const int16_t *,const int16_t *,int32_t *)
    __arm_streaming __arm_inout("za");

static __attribute__((noinline)) void b5_direct_postgroup_ns(
    const b5_direct_scratch *scratch,int16_t output[16][4][256])
    __arm_preserves("za")
{
    bw_direct_scatter(scratch->raw,output);
}
__arm_new("za")
__arm_locally_streaming
static __attribute__((noinline)) void b5_direct_stream(
    const bw_direct_ctx *ctx, b5_direct_scratch *sc,
    const int16_t sec[][3][256],int16_t out[][4][256],unsigned n)
{
    /* Retain ZA across the entire batch, just as the production SME2 path
     * does.  One non-streaming reconstruction boundary per 16 requests. */
    for(unsigned off=0;off<n;off+=16){
        /* Pack locally in streaming mode: a non-streaming always_inline
         * helper cannot be inlined into __arm_locally_streaming code. */
        for (unsigned k = 0; k < 3u; k++)
            for (unsigned p = 0; p < 128u; p++)
                for (unsigned r = 0; r < 16u; r++) {
                    sc->packed[k][p][2u*r] = sec[off+r][k][2u*p];
                    sc->packed[k][p][2u*r+1u] = sec[off+r][k][2u*p+1u];
                }
        bench5_direct_smopa(&ctx->mat[0][0][0][0][0][0],
                            &sc->packed[0][0][0],
                            &sc->raw[0][0][0][0]);
        b5_direct_postgroup_ns(sc,&out[off]);
    }
}
static __attribute__((noinline)) int b5_run_direct(b5_state *s,unsigned n) {
    b5_direct_stream(s->a,s->as,(const int16_t (*)[3][256])s->sec,s->out,n);
    return 0;
}
static __attribute__((noinline)) int b5_run_sme2(b5_state *s,unsigned n) {
    unsigned r,k,d;
    for(r=0;r<n;r++)for(k=0;k<3;k++){
        mlk_polyvec *dst=&s->cs->arena.group[r/16].sec[r%16];
        memcpy(dst->vec[k].coeffs,s->sec[r][k],512);
    }
    if(mlk_sme2_arith_batch_rolling_preloaded(s->c,s->cs,n)!=0)return -1;
    for(r=0;r<n;r++){
        mlk_sme2_output *src=&s->cs->arena.group[r/16].out[r%16];
        for(d=0;d<3;d++)memcpy(s->out[r][d],src->b.vec[d].coeffs,512);
        memcpy(s->out[r][3],src->v.coeffs,512);
    }
    return 0;
}
static __attribute__((noinline)) int b5_eval(b5_state *s,int mode,unsigned n) {
    switch(mode){
        case B5_A:return b5_run_direct(s,n);
        case B5_B:return bw_scalar_eval(s->b,(const int16_t (*)[3][256])s->sec,s->out,n);
        case B5_C:return b5_run_sme2(s,n);
        case B5_D:return bench5_native_eval(s->d,(const int16_t (*)[3][256])s->sec,s->out,n);
        case B5_E:return bench5_portable_eval(s->e,(const int16_t (*)[3][256])s->sec,s->out,n);
    }
    return -1;
}
static void *b5_aligned_alloc(size_t n){
    void *p=NULL;
    if(posix_memalign(&p,64,n)!=0||!p){fprintf(stderr,"allocation failed: %zu B\n",n);exit(2);}
    memset(p,0,n);
    return p;
}
static void b5_state_init(b5_state *s){
    memset(s,0,sizeof(*s));
    s->a=b5_aligned_alloc(sizeof(*s->a));
    s->b=b5_aligned_alloc(sizeof(*s->b));
    s->c=b5_aligned_alloc(sizeof(*s->c));
    s->byte_d=bench5_native_ctx_bytes();s->d=b5_aligned_alloc(s->byte_d);
    s->byte_e=bench5_portable_ctx_bytes();s->e=b5_aligned_alloc(s->byte_e);
    s->as=b5_aligned_alloc(sizeof(*s->as));
    s->cs=b5_aligned_alloc(sizeof(*s->cs));
    s->pub=b5_aligned_alloc(sizeof(int16_t[4][3][256]));
    s->sec=b5_aligned_alloc(sizeof(int16_t[BW_MAX][3][256]));
    s->out=b5_aligned_alloc(sizeof(int16_t[BW_MAX][4][256]));
    s->reference=b5_aligned_alloc(sizeof(int16_t[BW_MAX][4][256]));
}
static void b5_state_free(b5_state *s){
    free(s->a);free(s->b);free(s->c);free(s->d);free(s->e);
    free(s->as);free(s->cs);free(s->pub);free(s->sec);free(s->out);free(s->reference);
}
static int b5_prepare(b5_state *s) {
    if(bw_direct_prepare(s->a,(const int16_t (*)[3][256])s->pub)!=0)return -1;
    if(bw_scalar_prepare(s->b,(const int16_t (*)[3][256])s->pub)!=0)return -1;
    if(mlk_sme2_prepare_public(s->c,(const int16_t (*)[3][256])s->pub)!=0)return -1;
    if(bench5_native_prepare(s->d,(const int16_t (*)[3][256])s->pub)!=0)return -1;
    if(bench5_portable_prepare(s->e,(const int16_t (*)[3][256])s->pub)!=0)return -1;
    return 0;
}
static uint64_t b5_time(void){
    struct timespec ts;
    if(clock_gettime(CLOCK_MONOTONIC,&ts)!=0){perror("clock_gettime");exit(2);}
    return (uint64_t)ts.tv_sec*1000000000ull+(uint64_t)ts.tv_nsec;
}
static void b5_oracle_sample(b5_state *s,unsigned n,unsigned trial){
    unsigned r,d,j;
    unsigned where[3]={0,n/2,n-1};
    for(unsigned x=0;x<3;x++){
        r=where[x];
        int16_t oracle[4][256];
        bw_oracle_one((const int16_t (*)[3][256])s->pub,s->sec[r],oracle);
        for(d=0;d<4;d++)for(j=0;j<256;j++){
            if(bw_mod(s->out[r][d][j])!=bw_mod(oracle[d][j])){
                fprintf(stderr,"FAIL oracle trial=%u N=%u req=%u d=%u i=%u got=%d expected=%d\n",
                        trial,n,r,d,j,s->out[r][d][j],oracle[d][j]);exit(1);
            }
        }
    }
}
static int b5_check(b5_state *s,unsigned n,unsigned trial) {
    unsigned m,r,d,j;
    if(b5_eval(s,B5_B,n))return -1;
    b5_oracle_sample(s,n,trial);
    memcpy(s->reference,s->out,(size_t)n*sizeof(s->reference[0]));
    for(m=0;m<B5_MODES;m++){
        if(m==B5_B)continue;
        if(b5_eval(s,(int)m,n))return -1;
        for(r=0;r<n;r++)for(d=0;d<4;d++)for(j=0;j<256;j++){
            if(bw_mod(s->out[r][d][j])!=bw_mod(s->reference[r][d][j])){
                fprintf(stderr,"FAIL mode=%s trial=%u n=%u req=%u d=%u coeff=%u actual=%d reference=%d\n",
                        b5_names[m],trial,n,r,d,j,s->out[r][d][j],s->reference[r][d][j]);
                return -1;
            }
        }
    }
    printf("PASS correctness trial=%u N=%u methods=A/B/C/D/E; 4 outputs/request\n",trial,n);
    fflush(stdout);
    return 0;
}
static void b5_print_mem(const b5_state *s){
    printf("prepared_bytes,A=%zu,B=%zu,C=%zu,D=%zu,E=%zu\n",
        sizeof(*s->a),sizeof(*s->b),sizeof(*s->c),s->byte_d,s->byte_e);
    printf("scratch_bytes,A=%zu,C=%zu\n",sizeof(*s->as),sizeof(*s->cs));
    puts("scope=4 polynomial outputs, 12 polynomial products per request; coeff-domain input/output");
    puts("timed_scope=arithmetic only, including request transforms/packing/GEMM/inverse/output, excluding public precomputation, SHAKE, FO decrypt, noise sampling, error add, compression, zeroize");
    puts("B=NEON forward partial-NTT on AArch64; signed scalar C Toeplitz multiplication and scalar inverse-NTT");
    puts("D/E=actual mlkem-native full NTT+mulcache+INTT: native AArch64 vs portable-C arithmetic backend");
}
static unsigned b5_batches(const char *arg,unsigned out[32]){
    char str[256],*p;
    unsigned n=0;
    if(strlen(arg)>=sizeof(str))return 0;
    strcpy(str,arg);
    p=strtok(str,",");
    while(p&&n<32){
        char *end=NULL; unsigned long v=strtoul(p,&end,10);
        if(!end||*end!='\0'||v<16||v>256||v%16)return 0;
        for(unsigned k=0;k<n;k++)if(out[k]==v)return 0;
        out[n++]=(unsigned)v;p=strtok(NULL,",");
    }
    return p?0:n;
}
static double b5_measure(b5_state *s,int mode,unsigned n,unsigned reps){
    uint64_t a=b5_time();
    for(unsigned i=0;i<reps;i++)if(b5_eval(s,mode,n)!=0){
        fprintf(stderr,"evaluation failed %s n=%u\n",b5_names[mode],n);exit(2);
    }
    uint64_t b=b5_time();
    s->checksum+=(uint16_t)s->out[(reps+n)%n][mode%4][(mode*53)%256];
    return (double)(b-a)/(double)(reps*n);
}
static unsigned b5_reps(b5_state *s,int mode,unsigned n,unsigned target_us){
    /* Estimated from a few warm runs, then use the same count for both passes. */
    double per=b5_measure(s,mode,n,2);
    double wanted=(double)target_us*1000.0/((double)n*per);
    if(wanted<1.0)return 1;
    if(wanted>5000.0)return 5000;
    return (unsigned)wanted+1u;
}
static void b5_shuffle_u(unsigned *a,unsigned n,uint64_t *state){
    for(unsigned k=n;k>1;k--){unsigned j=(unsigned)(bw_rand(state)%k);unsigned t=a[k-1];a[k-1]=a[j];a[j]=t;}
}
static void b5_shuffle_i(int *a,unsigned n,uint64_t *state){
    for(unsigned k=n;k>1;k--){unsigned j=(unsigned)(bw_rand(state)%k);int t=a[k-1];a[k-1]=a[j];a[j]=t;}
}
static int b5_bench(b5_state *s,const char *dest,unsigned samples,
                    const unsigned *batches,unsigned nb,unsigned target_us,uint64_t seed){
    FILE *csv=fopen(dest,"w");
    if(!csv){fprintf(stderr,"Cannot open %s: %s\n",dest,strerror(errno));return -1;}
    fprintf(csv,"block,n,method,pass,reps,ns_per_req\n");
    unsigned reps[32][5];
    for(unsigned b=0;b<nb;b++)for(int m=0;m<5;m++)reps[b][m]=b5_reps(s,m,batches[b],target_us);
    fprintf(stderr,"calibration complete: target=%u us/method/measurement\n",target_us);
    for(unsigned block=0;block<samples;block++){
        unsigned order_n[32];
        for(unsigned b=0;b<nb;b++)order_n[b]=b;
        b5_shuffle_u(order_n,nb,&seed);
        for(unsigned z=0;z<nb;z++){
            unsigned idx=order_n[z],n=batches[idx];
            int order_m[5]={0,1,2,3,4};
            b5_shuffle_i(order_m,5,&seed);
            for(unsigned pass=0;pass<2;pass++){
                for(unsigned j=0;j<5;j++){
                    int m=order_m[pass?4u-j:j];
                    double ns=b5_measure(s,m,n,reps[idx][m]);
                    fprintf(csv,"%u,%u,%s,%u,%u,%.6f\n",block,n,b5_names[m],pass,reps[idx][m],ns);
                }
            }
        }
        if((block+1)%10==0||block+1==samples){
            fflush(csv);
            fprintf(stderr,"benchmark progress: %u/%u blocks\n",block+1,samples);
        }
    }
    fclose(csv);
    printf("PASS benchmark blocks=%u batches=%u checksum=%" PRIu64 " raw=%s\n",samples,nb,s->checksum,dest);
    return 0;
}
static void b5_usage(const char *exe){
    fprintf(stderr,"Usage: %s --mode check|bench [--samples 31|301] [--batches 16,32,...] [--target-us 1000] [--csv output.csv] [--seed 1]\n",exe);
}


static int b5_check_za_overflow_case(b5_state *s)
{
    static const int16_t pub8[8] = {
        3328, 802, 3300, 0, 103, 0, 0, 0
    };
    static const int16_t sec8[8] = {
        2, 0, -2, 2, -1, -2, -2, -1
    };

    int16_t expected[4][256];

    for (unsigned d = 0; d < 4; d++)
        for (unsigned k = 0; k < 3; k++)
            for (unsigned i = 0; i < 256; i++)
                s->pub[d][k][i] = pub8[i / 32];

    for (unsigned r = 0; r < 16; r++)
        for (unsigned k = 0; k < 3; k++)
            for (unsigned i = 0; i < 256; i++)
                s->sec[r][k][i] = sec8[i / 32];

    if (mlk_sme2_prepare_public(
            s->c, (const int16_t (*)[3][256])s->pub))
        return -1;

    if (b5_run_sme2(s, 16))
        return -1;

    bw_oracle_one(
        (const int16_t (*)[3][256])s->pub,
        s->sec[0], expected);

    for (unsigned r = 0; r < 16; r++)
        for (unsigned d = 0; d < 4; d++)
            for (unsigned i = 0; i < 256; i++) {
                if (bw_mod(s->out[r][d][i]) !=
                    bw_mod(expected[d][i])) {
                    fprintf(stderr,
                        "FAIL ZA overflow regression "
                        "r=%u d=%u i=%u\n",
                        r, d, i);
                    return -1;
                }
            }

    puts("PASS ZA overflow regression");
    return 0;
}

static int b5_preflight(void)
{
    b5_state regression;
    int ret;

    /* The adversarial input must not replace the seeded measurement input. */
    b5_state_init(&regression);
    ret = b5_check_za_overflow_case(&regression);
    b5_state_free(&regression);
    return ret;
}


int main(int argc,char **argv){
    const char *mode="check",*csv="bench5_raw.csv";
    const char *bs="16,32,64,80,96,128,160,192,224,256";
    unsigned samples=31,target_us=1500,batches[32],nb;
    uint64_t seed=UINT64_C(0x5AE21234);
    b5_state s;
    for(int i=1;i<argc;i++){
        if(i+1>=argc){b5_usage(argv[0]);return 2;}
        if(strcmp(argv[i],"--mode")==0)mode=argv[++i];
        else if(strcmp(argv[i],"--samples")==0)samples=(unsigned)strtoul(argv[++i],NULL,10);
        else if(strcmp(argv[i],"--batches")==0)bs=argv[++i];
        else if(strcmp(argv[i],"--target-us")==0)target_us=(unsigned)strtoul(argv[++i],NULL,10);
        else if(strcmp(argv[i],"--csv")==0)csv=argv[++i];
        else if(strcmp(argv[i],"--seed")==0)seed=(uint64_t)strtoull(argv[++i],NULL,0);
        else {b5_usage(argv[0]);return 2;}
    }
    if(strcmp(mode,"check")&&strcmp(mode,"bench")){b5_usage(argv[0]);return 2;}
    nb=b5_batches(bs,batches);
    if(!nb||!samples||!target_us){b5_usage(argv[0]);return 2;}
    if(b5_preflight())return 1;
    b5_state_init(&s);
    bw_fill_input(s.pub,s.sec,BW_MAX,seed);

    if(b5_prepare(&s)){
        fprintf(stderr,"FAIL: public prepare\n");
        return 1;
    }

    b5_print_mem(&s);

    // 通常の正当性テスト
    if(b5_check(&s,16,1)||
       b5_check(&s,80,1)||
       b5_check(&s,256,1))
        return 1;

    // ベンチマーク
    if(!strcmp(mode,"bench")){
        if(b5_bench(&s,csv,samples,batches,nb,target_us,seed))
            return 1;
    }

    b5_state_free(&s);
    return 0;
}
