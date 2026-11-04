#ifndef SME2_5WAY_PARTIAL_SCALAR_H
#define SME2_5WAY_PARTIAL_SCALAR_H
#include "math_core.h"
/* Same 3-stage transform and same residual rings as production SME2,
 * with residual 32x32 Toeplitz multiplication in signed scalar C.
 * Not claimed to be a fully optimized Neon residual kernel. */
typedef struct {
    int16_t mat[4][8][3][32][32];
    int16_t gamma[8];
} bw_scalar_ctx;
static int bw_scalar_prepare(bw_scalar_ctx *c,const int16_t pub[4][3][256]) {
    unsigned d,k,b,i,j;
    if(!c||!pub)return -1;
    bw_gammas(c->gamma);
    for(d=0;d<4;d++)for(k=0;k<3;k++) {
        int16_t tmp[256];
        memcpy(tmp,pub[d][k],512);
        bw_ntt3(tmp);
        for(b=0;b<8;b++)for(i=0;i<32;i++)for(j=0;j<32;j++){
            const int rr=(int)(i-j);
            int v=(rr>=0)?(int)tmp[32*b+(unsigned)rr]:
                bw_mod((int64_t)c->gamma[b]*tmp[32*b+(unsigned)(rr+32)]);
            c->mat[d][b][k][i][j]=bw_center(v);
        }
    }
    return 0;
}
static __attribute__((noinline)) int bw_scalar_eval(const bw_scalar_ctx *ctx,
                  const int16_t sec[][3][256],int16_t out[][4][256],unsigned n) {
    unsigned r,d,k,b,i,j;
    if(!ctx||!sec||!out)return -1;
    for(r=0;r<n;r++) {
        int16_t tmpsec[3][256];
        for(k=0;k<3;k++){
            memcpy(tmpsec[k],sec[r][k],512);
            bw_ntt3(tmpsec[k]);
        }
        for(d=0;d<4;d++) {
            int16_t coeff[256];
            for(b=0;b<8;b++)for(i=0;i<32;i++){
                int64_t sum=0;
                for(k=0;k<3;k++)for(j=0;j<32;j++)
                    sum+=(int32_t)ctx->mat[d][b][k][i][j]*bw_center(tmpsec[k][32*b+j]);
                coeff[32*b+i]=(int16_t)bw_mod(sum);
            }
            bw_intt3(coeff);
            memcpy(out[r][d],coeff,512);
        }
    }
    return 0;
}
#endif
