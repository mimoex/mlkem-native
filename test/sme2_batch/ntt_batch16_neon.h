/*
 * Experimental inter-request SIMD (16-way) Kyber/ML-KEM-768 NTT baseline.
 *
 * Two 8-lane NEON vectors handle one coefficient of 16 independent
 * polynomials. Input/output transposition is included in the timed path.
 *
 * This is benchmark-only research code; do not use in production without
 * independent range proofs, constant-time review, and differential testing.
 *
 * The NTT scheduling and Montgomery constants follow ML-KEM's standard
 * 7-layer incomplete NTT, in standard bit-reversed NTT-domain order.
 */
#ifndef SME2_BENCH_NTT_BATCH16_NEON_H
#define SME2_BENCH_NTT_BATCH16_NEON_H

#include <arm_neon.h>
#include <stdint.h>

#if MLKEM_K != 3 || MLKEM_N != 256 || MLK_SME2_GROUP != 16
#error "This benchmark-only NEON NTT implementation needs ML-KEM-768 / 16 requests"
#endif

/* zetas[i] = signed_canonical(17^bitreverse7(i) * 2^16 mod 3329).
 * Index zero is unused by the forward/inverse butterflies. */
static const int16_t ntt16_zetas[128] = {
  -1044,  -758,  -359, -1517,  1493,  1422,   287,   202,
   -171,   622,  1577,   182,   962, -1202, -1474,  1468,
    573, -1325,   264,   383,  -829,  1458, -1602,  -130,
   -681,  1017,   732,   608, -1542,   411,  -205, -1571,
   1223,   652,  -552,  1015, -1293,  1491,  -282, -1544,
    516,    -8,  -320,  -666, -1618, -1162,   126,  1469,
   -853,   -90,  -271,   830,   107, -1421,  -247,  -951,
   -398,   961, -1508,  -725,   448, -1065,   677, -1275,
  -1103,   430,   555,   843, -1251,   871,  1550,   105,
    422,   587,   177,  -235,  -291,  -460,  1574,  1653,
   -246,   778,  1159,  -147,  -777,  1483,  -602,  1119,
  -1590,   644,  -872,   349,   418,   329,  -156,   -75,
    817,  1097,   603,   610,  1322, -1285, -1465,   384,
  -1215,  -136,  1218, -1335,  -874,   220, -1187, -1659,
  -1185, -1530, -1278,   794, -1510,  -854,  -870,   478,
   -108,  -308,   996,   991,   958, -1460,  1522,  1628
};

/* Montgomery reduction: t=(a*62209) mod 2^16, centered as int16_t;
 * result = (a - t*3329)/2^16.
 *
 * vmul_n_s16 and vmovn_s32 intentionally take the low 16 product bits.
 * The 32-bit intermediates remain within signed 32-bit range for NTT
 * inputs satisfying the ML-KEM NTT bounds and |zeta| <= 1664. */
static inline int16x8_t ntt16_montmul(int16x8_t a, int16_t zeta)
{
  const int32x4_t lo = vmull_n_s16(vget_low_s16(a), zeta);
  const int32x4_t hi = vmull_n_s16(vget_high_s16(a), zeta);
  const int16x4_t tlo = vmul_n_s16(vmovn_s32(lo), (int16_t)-3327);
  const int16x4_t thi = vmul_n_s16(vmovn_s32(hi), (int16_t)-3327);
  const int32x4_t xlo = vsubq_s32(lo, vmulq_n_s32(vmovl_s16(tlo), 3329));
  const int32x4_t xhi = vsubq_s32(hi, vmulq_n_s32(vmovl_s16(thi), 3329));
  return vcombine_s16(vshrn_n_s32(xlo, 16), vshrn_n_s32(xhi, 16));
}

/* Reference-compatible 26-bit Barrett reduction for signed int16 lanes. */
static inline int16x8_t ntt16_barrett(int16x8_t a)
{
  const int32x4_t lo = vmovl_s16(vget_low_s16(a));
  const int32x4_t hi = vmovl_s16(vget_high_s16(a));
  const int32x4_t rounded = vdupq_n_s32(INT32_C(1) << 25);
  const int32x4_t tlo = vshrq_n_s32(vmlaq_n_s32(rounded, lo, 20159), 26);
  const int32x4_t thi = vshrq_n_s32(vmlaq_n_s32(rounded, hi, 20159), 26);
  const int32x4_t rlo = vsubq_s32(lo, vmulq_n_s32(tlo, 3329));
  const int32x4_t rhi = vsubq_s32(hi, vmulq_n_s32(thi, 3329));
  return vcombine_s16(vmovn_s32(rlo), vmovn_s32(rhi));
}

/* The SoA scratch is [4 polynomial outputs][256 coefficients][16 requests].
 * Only [0..2] are active during the forward NTT. */
static void ntt16_forward(
    mlk_polyvec dst[16],
    const mlk_polyvec src[16],
    int16_t soa[4][256][16])
{
  unsigned poly, i, req;
  for (poly = 0; poly < 3; poly++)
  {
    unsigned layer;
    for (i = 0; i < 256; i++)
      for (req = 0; req < 16; req++)
        soa[poly][i][req] = src[req].vec[poly].coeffs[i];

    for (layer = 1; layer <= 7; layer++)
    {
      const unsigned len = 256u >> layer;
      unsigned k = 1u << (layer - 1u);
      unsigned start;
      for (start = 0; start < 256; start += 2u * len)
      {
        const int16_t zeta = ntt16_zetas[k++];
        unsigned j, lane;
        for (j = start; j < start + len; j++)
          for (lane = 0; lane < 16; lane += 8)
          {
            int16x8_t u = vld1q_s16(&soa[poly][j][lane]);
            int16x8_t v = vld1q_s16(&soa[poly][j + len][lane]);
            int16x8_t t = ntt16_montmul(v, zeta);
            vst1q_s16(&soa[poly][j][lane], vaddq_s16(u, t));
            vst1q_s16(&soa[poly][j + len][lane], vsubq_s16(u, t));
          }
      }
    }
    for (i = 0; i < 256; i++)
      for (req = 0; req < 16; req++)
        dst[req].vec[poly].coeffs[i] = soa[poly][i][req];
  }
}

/* Inverse order follows mlkem-native's portable ML-KEM algorithm:
 *  (1) scale input with f = 1441 (Montgomery factor + 128^-1)
 *  (2) inverse layers 7..1 with subtraction (v-u)*zeta
 *  (3) Barrett reduction on every butterfly sum.
 *
 * The four output polynomials are b.vec[0..2] and v.
 */
static void ntt16_inverse(
    mlk_polyvec b[16],
    mlk_poly v[16],
    int16_t soa[4][256][16])
{
  unsigned poly, i, req;
  for (poly = 0; poly < 4; poly++)
  {
    unsigned layer;
    for (i = 0; i < 256; i++)
      for (req = 0; req < 16; req++)
        soa[poly][i][req] =
            (poly < 3) ? b[req].vec[poly].coeffs[i] : v[req].coeffs[i];

    for (i = 0; i < 256; i++)
    {
      unsigned lane;
      for (lane = 0; lane < 16; lane += 8)
      {
        const int16x8_t x = vld1q_s16(&soa[poly][i][lane]);
        vst1q_s16(&soa[poly][i][lane], ntt16_montmul(x, 1441));
      }
    }

    for (layer = 7; layer > 0; layer--)
    {
      const unsigned len = 256u >> layer;
      unsigned k = (1u << layer) - 1u;
      unsigned start;
      for (start = 0; start < 256; start += 2u * len)
      {
        const int16_t zeta = ntt16_zetas[k--];
        unsigned j, lane;
        for (j = start; j < start + len; j++)
          for (lane = 0; lane < 16; lane += 8)
          {
            int16x8_t u = vld1q_s16(&soa[poly][j][lane]);
            int16x8_t w = vld1q_s16(&soa[poly][j + len][lane]);
            const int16x8_t sum = ntt16_barrett(vaddq_s16(u, w));
            const int16x8_t diff = ntt16_montmul(vsubq_s16(w, u), zeta);
            vst1q_s16(&soa[poly][j][lane], sum);
            vst1q_s16(&soa[poly][j + len][lane], diff);
          }
      }
    }
    for (i = 0; i < 256; i++)
      for (req = 0; req < 16; req++)
      {
        if (poly < 3)
          b[req].vec[poly].coeffs[i] = soa[poly][i][req];
        else
          v[req].coeffs[i] = soa[poly][i][req];
      }
  }
}

#endif /* SME2_BENCH_NTT_BATCH16_NEON_H */
