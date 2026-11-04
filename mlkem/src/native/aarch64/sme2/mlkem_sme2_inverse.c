/* INTTの処理
 * SMOPAで演算した結果は32bitである。
 * 1段目は32bit INTTを行い、2,3段目は16bit INTTを行う
 */

#include "../../../common.h"

#if defined(MLK_ARITH_BACKEND_AARCH64) && \
    defined(MLK_SYS_AARCH64_SME2) && MLKEM_K == 3

#include "mlkem_sme2_internal.h"

#include <arm_neon.h>

#define SME2_COMBINE64(A,B,WHICH) \
  vreinterpretq_s16_s64( \
      vcombine_s64( \
          vget_##WHICH##_s64(vreinterpretq_s64_s32((A))), \
          vget_##WHICH##_s64(vreinterpretq_s64_s32((B)))))

static inline int32x4_t sme2_inverse_reduce4(int32x4_t x);
static inline int32x4_t sme2_inverse_mulconst_reduce4(
    int32x4_t x,
    int32_t c);
static inline int32x4_t sme2_inverse_half4(int32x4_t x);
static inline int32_t *sme2_raw_ptr(
    mlk_sme2_workspace *ws,
    unsigned g,
    unsigned b,
    unsigned d,
    unsigned t
);
static inline void sme2_transpose8x8_s16(
    int16x8_t r0,
    int16x8_t r1,
    int16x8_t r2,
    int16x8_t r3,
    int16x8_t r4,
    int16x8_t r5,
    int16x8_t r6,
    int16x8_t r7,
    int16x8_t *o0,
    int16x8_t *o1,
    int16x8_t *o2,
    int16x8_t *o3,
    int16x8_t *o4,
    int16x8_t *o5,
    int16x8_t *o6,
    int16x8_t *o7);
static inline int16x8_t sme2_inverse_mulmod16(
    int16x8_t x,
    int16_t c,
    int16_t ctw);
static inline int16x8_t sme2_inverse_half16(int16x8_t x);
static void sme2_inverse_first_pair(
    mlk_sme2_workspace *ws,
    unsigned n,
    unsigned ba,
    unsigned bb,
    unsigned k
);
static MLK_SME2_NOINLINE void sme2_inverse_first32(
    mlk_sme2_workspace *ws,
    unsigned n
);
static void sme2_inverse_pair16(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    unsigned n,
    unsigned ba,
    unsigned bb,
    unsigned k
);
static MLK_SME2_NOINLINE void sme2_inverse_tail16(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    unsigned n
);
static MLK_SME2_NOINLINE void sme2_transpose_output(
    mlk_sme2_workspace *ws,
    unsigned n
);

/* ---- sme2_inverse_reduce4 ---- */
static inline int32x4_t sme2_inverse_reduce4(int32x4_t x)
{
  const int32x4_t c=vdupq_n_s32(645084);
  const int32x4_t q=vdupq_n_s32(3329);
  const int32x4_t qhat=vqrdmulhq_s32(x,c);
  return vmlsq_s32(x,qhat,q);
}

/* ---- sme2_inverse_mulconst_reduce4 ---- */
static inline int32x4_t sme2_inverse_mulconst_reduce4(
    int32x4_t x,
    int32_t c)
{
  return sme2_inverse_reduce4(vmulq_n_s32(x,c));
}

/* ---- sme2_inverse_half4 ---- */
static inline int32x4_t sme2_inverse_half4(int32x4_t x)
{
  const int32x4_t one=vdupq_n_s32(1);
  const int32x4_t odd=vandq_s32(x,one);
  x=vmlaq_n_s32(x,odd,3329);
  return vshrq_n_s32(x,1);
}

/* ---- sme2_raw_ptr ---- */
static inline int32_t *sme2_raw_ptr(
    mlk_sme2_workspace *ws,
    unsigned g,
    unsigned b,
    unsigned d,
    unsigned t
)
{
  return &ws->arena.raw[g][b][d][t>>4][16u*(t&15u)];
}

/* ---- sme2_transpose8x8_s16 ---- */
static inline void sme2_transpose8x8_s16(
    int16x8_t r0,
    int16x8_t r1,
    int16x8_t r2,
    int16x8_t r3,
    int16x8_t r4,
    int16x8_t r5,
    int16x8_t r6,
    int16x8_t r7,
    int16x8_t *o0,
    int16x8_t *o1,
    int16x8_t *o2,
    int16x8_t *o3,
    int16x8_t *o4,
    int16x8_t *o5,
    int16x8_t *o6,
    int16x8_t *o7)
{
  const int16x8x2_t a0=vtrnq_s16(r0,r1);
  const int16x8x2_t a1=vtrnq_s16(r2,r3);
  const int16x8x2_t a2=vtrnq_s16(r4,r5);
  const int16x8x2_t a3=vtrnq_s16(r6,r7);

  const int32x4x2_t b0=vtrnq_s32(
      vreinterpretq_s32_s16(a0.val[0]),
      vreinterpretq_s32_s16(a1.val[0]));

  const int32x4x2_t b1=vtrnq_s32(
      vreinterpretq_s32_s16(a0.val[1]),
      vreinterpretq_s32_s16(a1.val[1]));

  const int32x4x2_t b2=vtrnq_s32(
      vreinterpretq_s32_s16(a2.val[0]),
      vreinterpretq_s32_s16(a3.val[0]));

  const int32x4x2_t b3=vtrnq_s32(
      vreinterpretq_s32_s16(a2.val[1]),
      vreinterpretq_s32_s16(a3.val[1]));

#define SME2_COMBINE64(A,B,WHICH) \
  vreinterpretq_s16_s64( \
      vcombine_s64( \
          vget_##WHICH##_s64(vreinterpretq_s64_s32((A))), \
          vget_##WHICH##_s64(vreinterpretq_s64_s32((B)))))

  *o0=SME2_COMBINE64(b0.val[0],b2.val[0],low);
  *o4=SME2_COMBINE64(b0.val[0],b2.val[0],high);

  *o1=SME2_COMBINE64(b1.val[0],b3.val[0],low);
  *o5=SME2_COMBINE64(b1.val[0],b3.val[0],high);

  *o2=SME2_COMBINE64(b0.val[1],b2.val[1],low);
  *o6=SME2_COMBINE64(b0.val[1],b2.val[1],high);

  *o3=SME2_COMBINE64(b1.val[1],b3.val[1],low);
  *o7=SME2_COMBINE64(b1.val[1],b3.val[1],high);

#undef SME2_COMBINE64
}

/* ---- sme2_inverse_mulmod16 ---- */
static inline int16x8_t sme2_inverse_mulmod16(
    int16x8_t x,
    int16_t c,
    int16_t ctw)
{
  const int16x8_t vc=vdupq_n_s16(c);
  const int16x8_t vt=vdupq_n_s16(ctw);
  const int16x8_t vq=vdupq_n_s16(3329);

  const int16x8_t qhat=vqrdmulhq_s16(x,vt);
  int16x8_t y=vmulq_s16(x,vc);

  y=vmlsq_s16(y,qhat,vq);
  return y;
}

/* ---- sme2_inverse_half16 ---- */
static inline int16x8_t sme2_inverse_half16(int16x8_t x)
{
  const int16x8_t one=vdupq_n_s16(1);
  const int16x8_t odd=vandq_s16(x,one);

  x=vmlaq_n_s16(x,odd,3329);
  return vshrq_n_s16(x,1);
}

/* ---- sme2_inverse_first_pair ---- */
static void sme2_inverse_first_pair(
    mlk_sme2_workspace *ws,
    unsigned n,
    unsigned ba,
    unsigned bb,
    unsigned k
)
{
  const unsigned groups=n/16u;
  const int32_t ci=(int32_t)sme2_inv2_roots_centered[k];
  unsigned g,d,t;

  for (g=0u;g<groups;g++)
    for (d=0u;d<4u;d++)
      for (t=0u;t<32u;t++) {
        const int32_t *ap=sme2_raw_ptr(ws,g,ba,d,t);
        const int32_t *bp=sme2_raw_ptr(ws,g,bb,d,t);
        int16_t *ao=&ws->secondary.inverse_h[g][ba][d][t][0];
        int16_t *bo=&ws->secondary.inverse_h[g][bb][d][t][0];
        unsigned c;

        for (c=0u;c<16u;c+=8u) {
          int32x4_t u0=vld1q_s32(ap+c+0u);
          int32x4_t u1=vld1q_s32(ap+c+4u);
          int32x4_t v0=vld1q_s32(bp+c+0u);
          int32x4_t v1=vld1q_s32(bp+c+4u);

          u0=sme2_inverse_reduce4(u0);
          u1=sme2_inverse_reduce4(u1);
          v0=sme2_inverse_reduce4(v0);
          v1=sme2_inverse_reduce4(v1);

          {
            const int32x4_t x0=sme2_inverse_half4(vaddq_s32(u0,v0));
            const int32x4_t x1=sme2_inverse_half4(vaddq_s32(u1,v1));

            const int32x4_t y0=
                sme2_inverse_mulconst_reduce4(vsubq_s32(u0,v0),ci);

            const int32x4_t y1=
                sme2_inverse_mulconst_reduce4(vsubq_s32(u1,v1),ci);

            const int16x8_t hx=
                vcombine_s16(vmovn_s32(x0),vmovn_s32(x1));

            const int16x8_t hy=
                vcombine_s16(vmovn_s32(y0),vmovn_s32(y1));

            vst1q_s16(ao+c,hx);
            vst1q_s16(bo+c,hy);
          }
        }
      }
}

/* ---- sme2_inverse_first32 ---- */
static MLK_SME2_NOINLINE void sme2_inverse_first32(
    mlk_sme2_workspace *ws,
    unsigned n
)
{
  sme2_inverse_first_pair(ws,n,0u,1u,4u);
  sme2_inverse_first_pair(ws,n,2u,3u,5u);
  sme2_inverse_first_pair(ws,n,4u,5u,6u);
  sme2_inverse_first_pair(ws,n,6u,7u,7u);
}

/* ---- sme2_inverse_pair16 ---- */
static void sme2_inverse_pair16(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    unsigned n,
    unsigned ba,
    unsigned bb,
    unsigned k
)
{
  const unsigned groups=n/16u;
  const int16_t ci=ctx->inverse_c[k];
  const int16_t ctw=ctx->inverse_tw[k];
  unsigned g,d,t;

  for (g=0u;g<groups;g++)
    for (d=0u;d<4u;d++)
      for (t=0u;t<32u;t++) {
        int16_t *ap=&ws->secondary.inverse_h[g][ba][d][t][0];
        int16_t *bp=&ws->secondary.inverse_h[g][bb][d][t][0];
        unsigned c;

        for (c=0u;c<16u;c+=8u) {
          const int16x8_t u=vld1q_s16(ap+c);
          const int16x8_t v=vld1q_s16(bp+c);

          const int16x8_t sum=vaddq_s16(u,v);
          const int16x8_t dif=vsubq_s16(u,v);

          const int16x8_t x=sme2_inverse_half16(sum);
          const int16x8_t y=sme2_inverse_mulmod16(dif,ci,ctw);

          vst1q_s16(ap+c,x);
          vst1q_s16(bp+c,y);
        }
      }
}

/* ---- sme2_inverse_tail16 ---- */
static MLK_SME2_NOINLINE void sme2_inverse_tail16(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    unsigned n
)
{
  /* inverse layer2 */
  sme2_inverse_pair16(ctx,ws,n,0u,2u,2u);
  sme2_inverse_pair16(ctx,ws,n,1u,3u,2u);
  sme2_inverse_pair16(ctx,ws,n,4u,6u,3u);
  sme2_inverse_pair16(ctx,ws,n,5u,7u,3u);

  /* inverse layer1 */
  sme2_inverse_pair16(ctx,ws,n,0u,4u,1u);
  sme2_inverse_pair16(ctx,ws,n,1u,5u,1u);
  sme2_inverse_pair16(ctx,ws,n,2u,6u,1u);
  sme2_inverse_pair16(ctx,ws,n,3u,7u,1u);
}

/* ---- sme2_transpose_output ---- */

static MLK_SME2_NOINLINE void sme2_transpose_output(
    mlk_sme2_workspace *ws,
    unsigned n
)
{
  const unsigned groups=n/16u;
  unsigned g,d,b,cb,rb;

  for (g=0u;g<groups;g++)
    for (d=0u;d<4u;d++)
      for (b=0u;b<8u;b++)
        for (cb=0u;cb<32u;cb+=8u)
          for (rb=0u;rb<16u;rb+=8u) {
            int16x8_t o0,o1,o2,o3,o4,o5,o6,o7;

            const int16x8_t r0=
                vld1q_s16(&ws->secondary.inverse_h[g][b][d][cb+0u][rb]);
            const int16x8_t r1=
                vld1q_s16(&ws->secondary.inverse_h[g][b][d][cb+1u][rb]);
            const int16x8_t r2=
                vld1q_s16(&ws->secondary.inverse_h[g][b][d][cb+2u][rb]);
            const int16x8_t r3=
                vld1q_s16(&ws->secondary.inverse_h[g][b][d][cb+3u][rb]);
            const int16x8_t r4=
                vld1q_s16(&ws->secondary.inverse_h[g][b][d][cb+4u][rb]);
            const int16x8_t r5=
                vld1q_s16(&ws->secondary.inverse_h[g][b][d][cb+5u][rb]);
            const int16x8_t r6=
                vld1q_s16(&ws->secondary.inverse_h[g][b][d][cb+6u][rb]);
            const int16x8_t r7=
                vld1q_s16(&ws->secondary.inverse_h[g][b][d][cb+7u][rb]);

            sme2_transpose8x8_s16(
                r0,r1,r2,r3,r4,r5,r6,r7,
                &o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7);

            vst1q_s16(
                &(*sme2_output_poly(ws, 16u*g+rb+0u, d)).coeffs[32u*b+cb],o0);
            vst1q_s16(
                &(*sme2_output_poly(ws, 16u*g+rb+1u, d)).coeffs[32u*b+cb],o1);
            vst1q_s16(
                &(*sme2_output_poly(ws, 16u*g+rb+2u, d)).coeffs[32u*b+cb],o2);
            vst1q_s16(
                &(*sme2_output_poly(ws, 16u*g+rb+3u, d)).coeffs[32u*b+cb],o3);
            vst1q_s16(
                &(*sme2_output_poly(ws, 16u*g+rb+4u, d)).coeffs[32u*b+cb],o4);
            vst1q_s16(
                &(*sme2_output_poly(ws, 16u*g+rb+5u, d)).coeffs[32u*b+cb],o5);
            vst1q_s16(
                &(*sme2_output_poly(ws, 16u*g+rb+6u, d)).coeffs[32u*b+cb],o6);
            vst1q_s16(
                &(*sme2_output_poly(ws, 16u*g+rb+7u, d)).coeffs[32u*b+cb],o7);
          }
}

/* ---- このファイルの全体関数 ---- */
MLK_INTERNAL_API MLK_SME2_NOINLINE void sme2_inverse_reconstruct(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    unsigned n
)
{
  sme2_inverse_first32(ws,n);       // 32bit INTT 1段目
  sme2_inverse_tail16(ctx,ws,n);    // 16bit INTT 2,3段目
  sme2_transpose_output(ws,n);      // 要求ごとの多項式に再配置
}

#else

MLK_EMPTY_CU(aarch64_sme2_inverse)

#endif /* MLK_ARITH_BACKEND_AARCH64 && MLK_SYS_AARCH64_SME2 && MLKEM_K == 3 */
