#ifndef SME2_5WAY_MATH_CORE_H
#define SME2_5WAY_MATH_CORE_H
/* Reproducible, deterministic math for the 3-stage reference path.
 * General-purpose C, usable by the host-only mathematical self-test. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__aarch64__) && !defined(SME2_5WAY_FORCE_SCALAR_NTT)
#include <arm_neon.h>
#endif
#define BW_Q 3329
#define BW_N 256
#define BW_K 3
#define BW_D 4
#define BW_G 16
#define BW_MAX 256

static inline int bw_mod(int64_t a) {
    int v=(int)(a%BW_Q);
    return (v<0)?v+BW_Q:v;
}
static inline int16_t bw_center(int64_t a) {
    int x=bw_mod(a);
    return (int16_t)(x>1664?x-BW_Q:x);
}
static const int16_t bw_roots[8]={1,1729,2580,3289,2642,630,1897,848};
static const int16_t bw_roots_centered[8]={1,-1600,-749,-40,-687,630,-1432,848};
static const int16_t bw_roots_twisted[8]={10,-15749,-7373,-394,-6762,6201,-14095,8347};
static const int16_t bw_roots_inverse[8]={1,1600,40,749,2481,1432,2699,687};
static inline int bw_powmod(int a, int e) {
    int r=1;
    while(e){if(e&1)r=bw_mod((int64_t)r*a);a=bw_mod((int64_t)a*a);e>>=1;}
    return r;
}
static __attribute__((unused)) void bw_ntt3_scalar(int16_t a[256]) {
    unsigned l;
    for(l=1;l<=3;l++) {
        const unsigned len=256u>>l;
        unsigned k=1u<<(l-1u),start;
        for(start=0;start<256;start+=2u*len) {
            int z=bw_roots[k++];
            unsigned j;
            for(j=start;j<start+len;j++) {
                int x=bw_mod(a[j]);
                int t=bw_mod((int64_t)z*bw_mod(a[j+len]));
                a[j]=bw_center(x+t);
                a[j+len]=bw_center(x-t);
            }
        }
    }
}
static __attribute__((unused)) void bw_ntt3(int16_t a[256]) {
#if defined(__aarch64__) && !defined(SME2_5WAY_FORCE_SCALAR_NTT)
    unsigned layer;
    for(layer=1u;layer<=3u;layer++) {
        unsigned len=256u>>layer;
        unsigned start,k=1u<<(layer-1u);
        for(start=0u;start<256u;start+=2u*len) {
            const int16x8_t zv=vdupq_n_s16(bw_roots_centered[k]);
            const int16x8_t ztv=vdupq_n_s16(bw_roots_twisted[k]);
            const int16x8_t qv=vdupq_n_s16(BW_Q);
            unsigned j;
            k++;
            for(j=start;j<start+len;j+=8u) {
                int16x8_t x=vld1q_s16(a+j);
                int16x8_t y=vld1q_s16(a+j+len);
                int16x8_t qhat=vqrdmulhq_s16(y,ztv);
                int16x8_t t=vmlsq_s16(vmulq_s16(y,zv),qhat,qv);
                vst1q_s16(a+j,vaddq_s16(x,t));
                vst1q_s16(a+j+len,vsubq_s16(x,t));
            }
        }
    }
#else
    bw_ntt3_scalar(a);
#endif
}
static __attribute__((unused)) void bw_intt3(int16_t a[256]) {
    int layer;
    for(layer=3;layer>=1;layer--) {
        unsigned len=256u>>(unsigned)layer;
        unsigned k=1u<<((unsigned)layer-1u),start;
        for(start=0;start<256;start+=2u*len) {
            int iz=bw_roots_inverse[k++];
            unsigned j;
            for(j=start;j<start+len;j++) {
                int u=bw_mod(a[j]),v=bw_mod(a[j+len]);
                int x=bw_mod((int64_t)(u+v)*1665);
                int y=bw_mod((int64_t)bw_mod((int64_t)(u-v)*1665)*iz);
                a[j]=(int16_t)x;
                a[j+len]=(int16_t)y;
            }
        }
    }
}
static __attribute__((unused)) void bw_gammas(int16_t gamma[8]) {
    int16_t basis[256]={0};
    unsigned b;
    basis[32]=1;
    bw_ntt3_scalar(basis);
    for(b=0;b<8;b++)gamma[b]=bw_center(basis[32*b]);
}
static __attribute__((unused)) void bw_oracle_one(const int16_t pub[4][3][256],
                          const int16_t sec[3][256],int16_t out[4][256]) {
    int64_t accum[4][256];
    unsigned d,k,i,j;
    memset(accum,0,sizeof(accum));
    for(d=0;d<4;d++)for(k=0;k<3;k++)for(i=0;i<256;i++)for(j=0;j<256;j++){
        unsigned t=i+j;
        int64_t p=(int64_t)pub[d][k][i]*sec[k][j];
        accum[d][t&255u]+=t>=256u?-p:p;
    }
    for(d=0;d<4;d++)for(i=0;i<256;i++)out[d][i]=(int16_t)bw_mod(accum[d][i]);
}
static inline uint64_t bw_rand(uint64_t *state) {
    uint64_t x=*state;
    x^=x>>12;x^=x<<25;x^=x>>27;
    *state=x;
    return x*UINT64_C(2685821657736338717);
}
static __attribute__((unused)) void bw_fill_input(int16_t pub[4][3][256],
                          int16_t sec[][3][256],unsigned n,uint64_t seed) {
    unsigned d,k,j,r;
    uint64_t x=seed?seed:UINT64_C(0x9e3779b97f4a7c15);
    for(d=0;d<4;d++)for(k=0;k<3;k++)for(j=0;j<256;j++)
        pub[d][k][j]=bw_center((int64_t)(bw_rand(&x)%BW_Q));
    for(r=0;r<n;r++)for(k=0;k<3;k++)for(j=0;j<256;j++)
        sec[r][k][j]=(int16_t)((int)(bw_rand(&x)%5)-2);
}
#endif
