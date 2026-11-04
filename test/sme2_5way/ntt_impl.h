/* Included by native_ntt.c and portable_bridge.c with different prefixes.
 * This calls the real mlkem-native arithmetic routines for its build config. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "mlkem/src/common.h"
#include "mlkem/src/poly_k.h"
#include "mlkem/src/poly.h"
#include "ntt_bridge.h"
#if defined(BENCH_NTT_PORTABLE)
#define FN(n) bench5_portable_##n
#else
#define FN(n) bench5_native_##n
#endif

typedef struct {
    mlk_polyvec pub[4];
    mlk_polyvec_mulcache cache[4];
} b5_ntt_ctx;

size_t FN(ctx_bytes)(void) {return sizeof(b5_ntt_ctx);}
int FN(prepare)(void *vctx,const int16_t pub[4][3][256]) {
    b5_ntt_ctx *ctx=(b5_ntt_ctx*)vctx;
    unsigned d,k,i;
    if(!ctx||!pub)return -1;
    for(d=0;d<4;d++) {
        for(k=0;k<3;k++)for(i=0;i<256;i++) {
            int x=pub[d][k][i];
            if(x<0)x+=3329;
            ctx->pub[d].vec[k].coeffs[i]=(int16_t)x;
        }
        mlk_polyvec_ntt(&ctx->pub[d]);
        mlk_polyvec_reduce(&ctx->pub[d]);
        mlk_polyvec_mulcache_compute(&ctx->cache[d],&ctx->pub[d]);
    }
    return 0;
}
__attribute__((noinline))
int FN(eval)(const void *vctx, const int16_t sec[][3][256],int16_t out[][4][256],unsigned n) {
    const b5_ntt_ctx *ctx=(const b5_ntt_ctx*)vctx;
    unsigned r,d,k;
    mlk_polyvec tmp;
    mlk_poly h;
    if(!ctx||!sec||!out)return -1;
    for(r=0;r<n;r++) {
        for(k=0;k<3;k++)memcpy(tmp.vec[k].coeffs,sec[r][k],512);
        mlk_polyvec_ntt(&tmp);
        mlk_polyvec_reduce(&tmp);
        for(d=0;d<4;d++){
            mlk_polyvec_basemul_acc_montgomery_cached(&h,&tmp,&ctx->pub[d],&ctx->cache[d]);
            mlk_poly_invntt_tomont(&h);
            mlk_poly_reduce(&h);
            memcpy(out[r][d],h.coeffs,512);
        }
    }
    return 0;
}
#undef FN
