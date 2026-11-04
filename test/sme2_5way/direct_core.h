#ifndef SME2_5WAY_DIRECT_CORE_H
#define SME2_5WAY_DIRECT_CORE_H
#include "math_core.h"
/* A: no NTT; 256x256 negacyclic Toeplitz via SME2 SMOPA.
 * 16 requests per group, 2 coefficients from each request in 32 halfword lanes.
 * Row blocks 8*32, 3 dot terms, 128 adjacent coefficient pairs. */
typedef struct {
    int16_t mat[2][8][3][128][4][32];
} bw_direct_ctx;
static __attribute__((unused)) int bw_direct_prepare(bw_direct_ctx *ctx,const int16_t pub[4][3][256]) {
    unsigned pair,b,k,p,rr;
    if(!ctx||!pub)return -1;
    for(pair=0;pair<2;pair++)for(b=0;b<8;b++)for(k=0;k<3;k++)
      for(p=0;p<128;p++)for(rr=0;rr<16;rr++){
        unsigned half,d;
        for(half=0;half<4;half++) {
            /* ZA0: d=2*pair, rows0..15; ZA1: rows16..31
             * ZA2: d=2*pair+1 rows0..15; ZA3: rows16..31 */
            unsigned row=b*32+(half%2)*16+rr;
            unsigned col0=2*p,col1=col0+1;
            d=2*pair+(half/2);
            int delta0=(int)row-(int)col0;
            int delta1=(int)row-(int)col1;
            int32_t a0=(delta0>=0)?pub[d][k][delta0]:-pub[d][k][256+delta0];
            int32_t a1=(delta1>=0)?pub[d][k][delta1]:-pub[d][k][256+delta1];
            ctx->mat[pair][b][k][p][half][2*rr]=bw_center(a0);
            ctx->mat[pair][b][k][p][half][2*rr+1]=bw_center(a1);
        }
      }
    return 0;
}
static __attribute__((always_inline,unused)) inline void bw_direct_pack(const int16_t sec[16][3][256],int16_t packed[3][128][32]) {
    unsigned k,p,r;
    for(k=0;k<3;k++)for(p=0;p<128;p++)for(r=0;r<16;r++){
        packed[k][p][2*r]=sec[r][k][2*p];
        packed[k][p][2*r+1]=sec[r][k][2*p+1];
    }
}
/* Portable test emulator of the exact SMOPA input and raw memory layout. */
static __attribute__((unused)) void bw_direct_emulate(const bw_direct_ctx *ctx,
            const int16_t pack[3][128][32],int32_t raw[8][4][2][256]) {
    unsigned b,d,r,i,k,p;
    for(b=0;b<8;b++)for(d=0;d<4;d++)for(i=0;i<32;i++)for(r=0;r<16;r++) {
        int64_t acc=0;
        for(k=0;k<3;k++)for(p=0;p<128;p++) {
            unsigned pair=d/2,tile=(d%2)*2+i/16;
            int16_t a0=ctx->mat[pair][b][k][p][tile][2*(i%16)];
            int16_t a1=ctx->mat[pair][b][k][p][tile][2*(i%16)+1];
            int16_t x0=pack[k][p][2*r],x1=pack[k][p][2*r+1];
            acc+=(int32_t)a0*x0+(int32_t)a1*x1;
        }
        raw[b][d][i/16][(i%16)*16+r]=(int32_t)acc;
    }
}
static __attribute__((always_inline,unused)) inline void bw_direct_scatter(const int32_t raw[8][4][2][256],
                              int16_t out[16][4][256]) {
    unsigned b,d,i,r;
    for(b=0;b<8;b++)for(d=0;d<4;d++)for(i=0;i<32;i++)for(r=0;r<16;r++)
        out[r][d][32*b+i]=(int16_t)bw_mod(raw[b][d][i/16][(i%16)*16+r]);
}
#endif
