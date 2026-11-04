#include "../../../common.h"

#if defined(MLK_ARITH_BACKEND_AARCH64) && \
    defined(MLK_SYS_AARCH64_SME2) && MLKEM_K == 3

#include "mlkem_sme2_internal.h"

#include <arm_neon.h>
#include <arm_sme.h>

MLK_INTERNAL_DATA_DEFINITION const int16_t sme2_inv2_roots_centered[128] = {
  -1664, 800, 20, -1290, -424, 716, -315, -1321,
  -873, 1380, -1630, -1393, 1266, 1568, 705, -531,
  1254, -987, 225, 468, -1441, 1397, -1047, -713,
  998, -1120, -28, -1523, -738, 995, 441, -148,
  -455, 1051, -1555, -1237, -324, 924, 356, 341,
  -1201, -767, -1434, -719, -226, 1261, 947, 505,
  707, -660, 1648, 232, -316, 408, 676, -325,
  637, 526, -1152, 1066, -878, 38, 1499, 1520,
  -1077, 1222, 197, -1055, 1150, -937, -606, -861,
  -443, 277, -1075, 1093, -513, 1463, -546, -1402,
  -1297, -1233, 1384, 615, 1286, 278, 1505, 1133,
  842, -1045, 390, 1477, -1610, 646, -1149, -792,
  1344, -134, 496, 1298, -1194, 446, -1154, 1195,
  321, 934, -476, 741, 770, 270, 839, 813,
  -1525, 157, -1078, -378, 1548, 24, -1331, 960,
  -550, -1144, 1303, 846, 340, 1373, 284, 1656,
};


/* ------------------------------------------------------------------------- */
/* ZA状態を保持したローリング算術処理経路 */
/* ------------------------------------------------------------------------- */

/*
 * 逆変換・出力処理への橋渡し部分だけが、通常の非ストリーミングAdvanced SIMD
 * コードを使用する。秘密値のパッキングは、ローカルストリーミング関数内で
 * スカラロード・ストアを行い、外部から渡されたバッチ全体を処理する間、
 * ZA状態を維持できるようにする。
 */
static __attribute__((always_inline)) inline int32x4_t sme2_roll_reduce4(int32x4_t x)
{
  const int32x4_t c = vdupq_n_s32(645084);
  const int32x4_t q = vdupq_n_s32(3329);
  const int32x4_t qhat = vqrdmulhq_s32(x, c);
  return vmlsq_s32(x, qhat, q);
}

static __attribute__((always_inline, unused)) inline int32x4_t sme2_roll_mulconst_reduce4(
    int32x4_t x,
    int32_t c)
{
  return sme2_roll_reduce4(vmulq_n_s32(x, c));
}

static __attribute__((always_inline)) inline int32x4_t sme2_roll_half4(int32x4_t x)
{
  const int32x4_t one = vdupq_n_s32(1);
  const int32x4_t odd = vandq_s32(x, one);
  x = vmlaq_n_s32(x, odd, 3329);
  return vshrq_n_s32(x, 1);
}

static __attribute__((always_inline)) inline int16x8_t sme2_roll_mulmod16(
    int16x8_t x,
    int16_t c,
    int16_t ctw)
{
  const int16x8_t vc = vdupq_n_s16(c);
  const int16x8_t vt = vdupq_n_s16(ctw);
  const int16x8_t vq = vdupq_n_s16(3329);
  const int16x8_t qhat = vqrdmulhq_s16(x, vt);
  int16x8_t y = vmulq_s16(x, vc);
  y = vmlsq_s16(y, qhat, vq);
  return y;
}

static __attribute__((always_inline)) inline int16x8_t sme2_roll_half16(int16x8_t x)
{
  const int16x8_t one = vdupq_n_s16(1);
  const int16x8_t odd = vandq_s16(x, one);
  x = vmlaq_n_s16(x, odd, 3329);
  return vshrq_n_s16(x, 1);
}

static __attribute__((always_inline)) inline int32_t *sme2_roll_raw_ptr(
    mlk_sme2_rolling_workspace *ws,
    unsigned b,
    unsigned d,
    unsigned t)
{
  return &ws->raw[0][b][d][t >> 4][16u * (t & 15u)];
}

/* BEGIN_EXPERIMENT_SME2_FIRST_MUL16 */
/*
 * 0: 比較基準。8要求レーンに対して、32ビットで回転因子を用いた剰余乗算を2回行う。
 * 1: 入力を32ビットで剰余還元した後、8レーンの16ビット回転因子付き剰余乗算を1回行う。
 *    生の累積値を直接16ビットへ縮小してはならない。
 */
#define SME2_ROLL_FIRST_MUL16 1
/* END_EXPERIMENT_SME2_FIRST_MUL16 */
static __attribute__((always_inline)) inline void sme2_roll_inverse_first_pair(
    mlk_sme2_rolling_workspace *ws,
    unsigned ba,
    unsigned bb,
    unsigned k)
{
  const int32_t ci = (int32_t)sme2_inv2_roots_centered[k];
#if SME2_ROLL_FIRST_MUL16
  /* round(sme2_inv2_roots_centered[k] * 32768 / 3329). */
  static const int16_t sme2_first_tw15[8] = {
    -16379, 7875, 197, -12698, -4174, 7048, -3101, -13003
  };
#endif
  unsigned d, t;

  for (d = 0u; d < 4u; d++)
    for (t = 0u; t < 32u; t++)
    {
      const int32_t *ap = sme2_roll_raw_ptr(ws, ba, d, t);
      const int32_t *bp = sme2_roll_raw_ptr(ws, bb, d, t);
      int16_t *ao = &ws->secondary.inverse_h[0][ba][d][t][0];
      int16_t *bo = &ws->secondary.inverse_h[0][bb][d][t][0];
      unsigned c;

      for (c = 0u; c < 16u; c += 8u)
      {
        int32x4_t u0 = vld1q_s32(ap + c + 0u);
        int32x4_t u1 = vld1q_s32(ap + c + 4u);
        int32x4_t v0 = vld1q_s32(bp + c + 0u);
        int32x4_t v1 = vld1q_s32(bp + c + 4u);
        int32x4_t x0, x1;
#if !SME2_ROLL_FIRST_MUL16
        int32x4_t y0, y1;
#endif
        int16x8_t hx, hy;

        u0 = sme2_roll_reduce4(u0);
        u1 = sme2_roll_reduce4(u1);
        v0 = sme2_roll_reduce4(v0);
        v1 = sme2_roll_reduce4(v1);

        x0 = sme2_roll_half4(vaddq_s32(u0, v0));
        x1 = sme2_roll_half4(vaddq_s32(u1, v1));
        hx = vcombine_s16(vmovn_s32(x0), vmovn_s32(x1));

#if SME2_ROLL_FIRST_MUL16
        {
          const int16x8_t delta = vcombine_s16(
              vmovn_s32(vsubq_s32(u0, v0)),
              vmovn_s32(vsubq_s32(u1, v1)));
          hy = sme2_roll_mulmod16(delta, (int16_t)ci, sme2_first_tw15[k]);
        }
#else
        y0 = sme2_roll_mulconst_reduce4(vsubq_s32(u0, v0), ci);
        y1 = sme2_roll_mulconst_reduce4(vsubq_s32(u1, v1), ci);
        hy = vcombine_s16(vmovn_s32(y0), vmovn_s32(y1));
#endif
        vst1q_s16(ao + c, hx);
        vst1q_s16(bo + c, hy);
      }
    }
}

static __attribute__((always_inline)) inline void sme2_roll_inverse_first32(
    mlk_sme2_rolling_workspace *ws)
{
  sme2_roll_inverse_first_pair(ws, 0u, 1u, 4u);
  sme2_roll_inverse_first_pair(ws, 2u, 3u, 5u);
  sme2_roll_inverse_first_pair(ws, 4u, 5u, 6u);
  sme2_roll_inverse_first_pair(ws, 6u, 7u, 7u);
}

static __attribute__((always_inline)) inline void sme2_roll_inverse_pair16(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_rolling_workspace *ws,
    unsigned ba,
    unsigned bb,
    unsigned k)
{
  const int16_t ci = ctx->inverse_c[k];
  const int16_t ctw = ctx->inverse_tw[k];
  unsigned d, t;

  for (d = 0u; d < 4u; d++)
    for (t = 0u; t < 32u; t++)
    {
      int16_t *ap = &ws->secondary.inverse_h[0][ba][d][t][0];
      int16_t *bp = &ws->secondary.inverse_h[0][bb][d][t][0];
      unsigned c;

      for (c = 0u; c < 16u; c += 8u)
      {
        const int16x8_t u = vld1q_s16(ap + c);
        const int16x8_t v = vld1q_s16(bp + c);
        const int16x8_t x = sme2_roll_half16(vaddq_s16(u, v));
        const int16x8_t y =
            sme2_roll_mulmod16(vsubq_s16(u, v), ci, ctw);
        vst1q_s16(ap + c, x);
        vst1q_s16(bp + c, y);
      }
    }
}

static __attribute__((always_inline)) inline void sme2_roll_inverse_tail16(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_rolling_workspace *ws)
{
  sme2_roll_inverse_pair16(ctx, ws, 0u, 2u, 2u);
  sme2_roll_inverse_pair16(ctx, ws, 1u, 3u, 2u);
  sme2_roll_inverse_pair16(ctx, ws, 4u, 6u, 3u);
  sme2_roll_inverse_pair16(ctx, ws, 5u, 7u, 3u);
  sme2_roll_inverse_pair16(ctx, ws, 0u, 4u, 1u);
  sme2_roll_inverse_pair16(ctx, ws, 1u, 5u, 1u);
  sme2_roll_inverse_pair16(ctx, ws, 2u, 6u, 1u);
  sme2_roll_inverse_pair16(ctx, ws, 3u, 7u, 1u);
}

static __attribute__((always_inline)) inline void sme2_roll_transpose8x8_s16(
    int16x8_t r0, int16x8_t r1, int16x8_t r2, int16x8_t r3,
    int16x8_t r4, int16x8_t r5, int16x8_t r6, int16x8_t r7,
    int16x8_t *o0, int16x8_t *o1, int16x8_t *o2, int16x8_t *o3,
    int16x8_t *o4, int16x8_t *o5, int16x8_t *o6, int16x8_t *o7)
{
  const int16x8x2_t a0 = vtrnq_s16(r0, r1);
  const int16x8x2_t a1 = vtrnq_s16(r2, r3);
  const int16x8x2_t a2 = vtrnq_s16(r4, r5);
  const int16x8x2_t a3 = vtrnq_s16(r6, r7);
  const int32x4x2_t b0 = vtrnq_s32(
      vreinterpretq_s32_s16(a0.val[0]), vreinterpretq_s32_s16(a1.val[0]));
  const int32x4x2_t b1 = vtrnq_s32(
      vreinterpretq_s32_s16(a0.val[1]), vreinterpretq_s32_s16(a1.val[1]));
  const int32x4x2_t b2 = vtrnq_s32(
      vreinterpretq_s32_s16(a2.val[0]), vreinterpretq_s32_s16(a3.val[0]));
  const int32x4x2_t b3 = vtrnq_s32(
      vreinterpretq_s32_s16(a2.val[1]), vreinterpretq_s32_s16(a3.val[1]));

#define SME2_ROLL_COMBINE64(A,B,WHICH) \
  vreinterpretq_s16_s64( \
      vcombine_s64( \
          vget_##WHICH##_s64(vreinterpretq_s64_s32((A))), \
          vget_##WHICH##_s64(vreinterpretq_s64_s32((B)))))

  *o0 = SME2_ROLL_COMBINE64(b0.val[0], b2.val[0], low);
  *o4 = SME2_ROLL_COMBINE64(b0.val[0], b2.val[0], high);
  *o1 = SME2_ROLL_COMBINE64(b1.val[0], b3.val[0], low);
  *o5 = SME2_ROLL_COMBINE64(b1.val[0], b3.val[0], high);
  *o2 = SME2_ROLL_COMBINE64(b0.val[1], b2.val[1], low);
  *o6 = SME2_ROLL_COMBINE64(b0.val[1], b2.val[1], high);
  *o3 = SME2_ROLL_COMBINE64(b1.val[1], b3.val[1], low);
  *o7 = SME2_ROLL_COMBINE64(b1.val[1], b3.val[1], high);

#undef SME2_ROLL_COMBINE64
}

static __attribute__((always_inline)) inline mlk_poly *sme2_roll_output_poly(
    mlk_sme2_rolling_workspace *ws,
    unsigned r,
    unsigned d)
{
  const unsigned g = r / MLK_SME2_GROUP;
  const unsigned lane = r % MLK_SME2_GROUP;
  if (d < MLKEM_K)
    return &ws->arena.group[g].out[lane].b.vec[d];
  return &ws->arena.group[g].out[lane].v;
}

static __attribute__((always_inline)) inline void sme2_roll_transpose_output_group(
    mlk_sme2_rolling_workspace *ws,
    unsigned out_base)
{
  unsigned d, b, cb, rb;

  for (d = 0u; d < 4u; d++)
    for (b = 0u; b < 8u; b++)
      for (cb = 0u; cb < 32u; cb += 8u)
        for (rb = 0u; rb < 16u; rb += 8u)
        {
          int16x8_t o0, o1, o2, o3, o4, o5, o6, o7;
          const int16x8_t r0 =
              vld1q_s16(&ws->secondary.inverse_h[0][b][d][cb + 0u][rb]);
          const int16x8_t r1 =
              vld1q_s16(&ws->secondary.inverse_h[0][b][d][cb + 1u][rb]);
          const int16x8_t r2 =
              vld1q_s16(&ws->secondary.inverse_h[0][b][d][cb + 2u][rb]);
          const int16x8_t r3 =
              vld1q_s16(&ws->secondary.inverse_h[0][b][d][cb + 3u][rb]);
          const int16x8_t r4 =
              vld1q_s16(&ws->secondary.inverse_h[0][b][d][cb + 4u][rb]);
          const int16x8_t r5 =
              vld1q_s16(&ws->secondary.inverse_h[0][b][d][cb + 5u][rb]);
          const int16x8_t r6 =
              vld1q_s16(&ws->secondary.inverse_h[0][b][d][cb + 6u][rb]);
          const int16x8_t r7 =
              vld1q_s16(&ws->secondary.inverse_h[0][b][d][cb + 7u][rb]);

          sme2_roll_transpose8x8_s16(
              r0, r1, r2, r3, r4, r5, r6, r7,
              &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7);

          vst1q_s16(&sme2_roll_output_poly(ws, out_base + rb + 0u, d)->coeffs[32u * b + cb], o0);
          vst1q_s16(&sme2_roll_output_poly(ws, out_base + rb + 1u, d)->coeffs[32u * b + cb], o1);
          vst1q_s16(&sme2_roll_output_poly(ws, out_base + rb + 2u, d)->coeffs[32u * b + cb], o2);
          vst1q_s16(&sme2_roll_output_poly(ws, out_base + rb + 3u, d)->coeffs[32u * b + cb], o3);
          vst1q_s16(&sme2_roll_output_poly(ws, out_base + rb + 4u, d)->coeffs[32u * b + cb], o4);
          vst1q_s16(&sme2_roll_output_poly(ws, out_base + rb + 5u, d)->coeffs[32u * b + cb], o5);
          vst1q_s16(&sme2_roll_output_poly(ws, out_base + rb + 6u, d)->coeffs[32u * b + cb], o6);
          vst1q_s16(&sme2_roll_output_poly(ws, out_base + rb + 7u, d)->coeffs[32u * b + cb], o7);
        }
}

/*
 * This is the only non-streaming shared-ZA bridge in the rolling group loop.
 * All Advanced-SIMD reconstruction helpers below this boundary are forced
 * inline so they do not create additional ZA-sharing ABI boundaries.
 */
static MLK_SME2_NOINLINE void sme2_roll_inverse_output_group_ns(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_rolling_workspace *ws,
    unsigned g) __arm_preserves("za")
{
  sme2_roll_inverse_first32(ws);
  sme2_roll_inverse_tail16(ctx, ws);
  sme2_roll_transpose_output_group(ws, g * MLK_SME2_GROUP);
}

/*
 * ZA状態を保持するローリングループ。
 *
 * スカラ方式のパッキング処理は、このローカルストリーミング関数内に直接記述する。
 * SMOPAは、共有されるストリーミング処理本体を直接呼び出す。
 * 結果の再構成では、グループごとに非ストリーミングの
 * __arm_preserves("za") 境界を1回だけ通過する。
 */

/* BEGIN EXPERIMENT_ZA_PACK_D3 */
/* 0: 従来のスカラ方式によるパッキング。1: ZAの行ロード・列ストアによる転置。 */
#define SME2_USE_ZA_PACK 1
extern void MLK_NAMESPACE(sme2_pack_secret_d3_body)(
    const int16_t *source, int16_t *destination)
    __arm_streaming __arm_inout("za");
/* END EXPERIMENT_ZA_PACK_D3 */

__arm_new("za")           // 新しいZA状態を使用する
__arm_locally_streaming // この関数内でのみストリーミングを使用する
static MLK_SME2_NOINLINE void sme2_roll_stream(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_rolling_workspace *ws,
    unsigned n)
{
  const unsigned groups = n / MLK_SME2_GROUP;
  unsigned g;

  for (g = 0u; g < groups; g++)
  {
    // 16要求分の秘密多項式を、ws->secondary.secret_gemmに代入
    {
      const int16_t *src =
          &ws->arena.group[g].sec[0].vec[0].coeffs[0];
      int16_t *dst = &ws->secondary.secret_gemm[0][0][0][0][0];
#if SME2_USE_ZA_PACK
      // 16要求分の秘密多項式をそれぞれ2係数ずつの並びにしたい
      MLK_NAMESPACE(sme2_pack_secret_d3_body)(src, dst);
#else
      unsigned b, k, p, c;

      for (b = 0u; b < 8u; b++)
        for (k = 0u; k < MLKEM_K; k++)
          for (p = 0u; p < 16u; p++)
            for (c = 0u; c < 16u; c++)
            {
              const unsigned base = 32u * b + 2u * p;
              const size_t sbase =
                  ((size_t)c * MLKEM_K + k) * MLK_SME2_N + base;
              const size_t dbase =
                  (((size_t)b * MLKEM_K + k) * 16u + p) * 32u + 2u * c;
              dst[dbase + 0u] = src[sbase + 0u];
              dst[dbase + 1u] = src[sbase + 1u];
            }
#endif
    }
    // 4個のZAタイルを使ったSMOPA積和
    MLK_NAMESPACE(sme2_cross_matrix_d3_body)(
        &ctx->public_gemm[0][0][0][0][0][0],
        &ws->secondary.secret_gemm[0][0][0][0][0],
        &ws->raw[0][0][0][0][0],
        1u);

    // NEONで剰余・逆NTT・出力転置
    sme2_roll_inverse_output_group_ns(ctx, ws, g);
  }
}

int mlk_sme2_arith_batch_rolling_preloaded(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_rolling_workspace *ws,
    unsigned n)
{
  unsigned r, k;

  // 引数チェック
  if (ctx == NULL || ws == NULL)
    return -1;
  if (n < MLK_SME2_GROUP ||
      n > MLK_SME2_MAX_BATCH ||
      (n % MLK_SME2_GROUP) != 0u)
    return -2;

  // CPUにSME2が搭載されているかチェック
  if (mlk_sme2_check_svl() != 0)
    return MLK_SME2_ERR_SVL;

  // 全要求に対して、3行の多項式ベクトルをNEONでNTT変換する
  for (r = 0u; r < n; r++)
    for (k = 0u; k < MLKEM_K; k++)
      sme2_partial_ntt3_neon(
          ws->arena.group[r / MLK_SME2_GROUP]
              .sec[r % MLK_SME2_GROUP].vec[k].coeffs);

  /*
   * 各グループのスカラ方式によるパッキングは、ZA状態を保持する
   * ストリーミングループ内で実行する。1グループ分の二次作業領域は、
   * SMOPAで使用した後、次のグループのパッキング処理までの間、
   * inverse_hとして再利用する。
   */
  sme2_roll_stream(ctx, ws, n);
  return 0;
}


#else

MLK_EMPTY_CU(aarch64_sme2_rolling)

#endif /* MLK_ARITH_BACKEND_AARCH64 && MLK_SYS_AARCH64_SME2 && MLKEM_K == 3 */
