/* Depth-3 partial NTT implementations and forward transform constants. */

#include "../../../common.h"

#if defined(MLK_ARITH_BACKEND_AARCH64) && \
    defined(MLK_SYS_AARCH64_SME2) && MLKEM_K == 3

#include "mlkem_sme2_internal.h"

#include <arm_neon.h>

static const int16_t sme2_ntt_roots[128] = {
  1, 1729, 2580, 3289, 2642, 630, 1897, 848,
  1062, 1919, 193, 797, 2786, 3260, 569, 1746,
  296, 2447, 1339, 1476, 3046, 56, 2240, 1333,
  1426, 2094, 535, 2882, 2393, 2879, 1974, 821,
  289, 331, 3253, 1756, 1197, 2304, 2277, 2055,
  650, 1977, 2513, 632, 2865, 33, 1320, 1915,
  2319, 1435, 807, 452, 1438, 2868, 1534, 2402,
  2647, 2617, 1481, 648, 2474, 3110, 1227, 910,
  17, 2761, 583, 2649, 1637, 723, 2288, 1100,
  1409, 2662, 3281, 233, 756, 2156, 3015, 3050,
  1703, 1651, 2789, 1789, 1847, 952, 1461, 2687,
  939, 2308, 2437, 2388, 733, 2337, 268, 641,
  1584, 2298, 2037, 3220, 375, 2549, 2090, 1645,
  1063, 319, 2773, 757, 2099, 561, 2466, 2594,
  2804, 1092, 403, 1026, 1143, 2150, 2775, 886,
  1722, 1212, 1874, 1029, 2110, 2935, 885, 2154,
};

// -1664,1664に中心化した表現
static const int16_t sme2_ntt_roots_centered[128] = {
  1, -1600, -749, -40, -687, 630, -1432, 848,
  1062, -1410, 193, 797, -543, -69, 569, -1583,
  296, -882, 1339, 1476, -283, 56, -1089, 1333,
  1426, -1235, 535, -447, -936, -450, -1355, 821,
  289, 331, -76, -1573, 1197, -1025, -1052, -1274,
  650, -1352, -816, 632, -464, 33, 1320, -1414,
  -1010, 1435, 807, 452, 1438, -461, 1534, -927,
  -682, -712, 1481, 648, -855, -219, 1227, 910,
  17, -568, 583, -680, 1637, 723, -1041, 1100,
  1409, -667, -48, 233, 756, -1173, -314, -279,
  -1626, 1651, -540, -1540, -1482, 952, 1461, -642,
  939, -1021, -892, -941, 733, -992, 268, 641,
  1584, -1031, -1292, -109, 375, -780, -1239, 1645,
  1063, 319, -556, 757, -1230, 561, -863, -735,
  -525, 1092, 403, 1026, 1143, -1179, -554, 886,
  -1607, 1212, -1455, 1029, -1219, -394, 885, -1175,
};

// 16-bit SIMDの丸め付き乗算で剰余演算を効率化するための事前計算定数
static const int16_t sme2_ntt_roots_twisted[128] = {
  10, -15749, -7373, -394, -6762, 6201, -14095, 8347,
  10453, -13879, 1900, 7845, -5345, -679, 5601, -15582,
  2914, -8682, 13180, 14529, -2786, 551, -10719, 13121,
  14036, -12156, 5266, -4400, -9213, -4429, -13338, 8081,
  2845, 3258, -748, -15483, 11782, -10089, -10355, -12540,
  6398, -13308, -8032, 6221, -4567, 325, 12993, -13918,
  -9942, 14125, 7943, 4449, 14155, -4538, 15099, -9125,
  -6713, -7008, 14578, 6378, -8416, -2156, 12078, 8957,
  167, -5591, 5739, -6693, 16113, 7117, -10247, 10828,
  13869, -6565, -472, 2293, 7441, -11546, -3091, -2746,
  -16005, 16251, -5315, -15159, -14588, 9371, 14381, -6319,
  9243, -10050, -8780, -9262, 7215, -9764, 2638, 6309,
  15592, -10148, -12717, -1073, 3691, -7678, -12196, 16192,
  10463, 3140, -5473, 7451, -12107, 5522, -8495, -7235,
  -5168, 10749, 3967, 10099, 11251, -11605, -5453, 8721,
  -15818, 11930, -14322, 10129, -11999, -3878, 8711, -11566,
};

static int sme2_residual_center_q(int x);
static inline int16x8_t sme2_partial_ntt_mul_vec(
    int16x8_t x,
    int16_t z,
    int16_t ztw);
static void sme2_partial_ntt_neon(
    int16_t a[256],
    unsigned depth);

/* ---- 中央化 ---- */
static int sme2_residual_center_q(int x)
{
  x %= SME2_Q;
  if (x < 0)
    x += SME2_Q;
  if (x > SME2_Q/2)
    x -= SME2_Q;
  return x;
}

/* ---- sme2_partial_ntt_scalar ---- */
MLK_INTERNAL_API void sme2_partial_ntt_scalar(int16_t a[256],unsigned depth)
{
  unsigned layer;

  for (layer = 1u; layer <= depth; layer++) {
    const unsigned len = 256u >> layer;
    unsigned k = 1u << (layer-1u);
    unsigned start;

    for (start = 0u; start < 256u; start += 2u * len) {
      const int root = sme2_ntt_roots[k++];
      unsigned j;

      for (j = start; j < start + len; j++) {
        const int x = sme2_residual_mod_q(a[j]);
        const int y = sme2_residual_mod_q(a[j + len]);
        const int t = sme2_residual_mod_q((int64_t)y * root);

        a[j] = (int16_t)sme2_residual_center_q(x + t);
        a[j + len]=(int16_t)sme2_residual_center_q(x - t);
      }
    }
  }
}

/* ---- sme2_residual_mod_q ---- */
MLK_INTERNAL_API int sme2_residual_mod_q(int64_t x)
{
  int64_t r = x % SME2_Q;
  if (r < 0)
    r += SME2_Q;
  return (int)r;
}

/* ---- sme2_partial_ntt_mul_vec ---- */
static inline int16x8_t sme2_partial_ntt_mul_vec(
    int16x8_t x,
    int16_t z,
    int16_t ztw)
{
  const int16x8_t zv = vdupq_n_s16(z);
  const int16x8_t ztv = vdupq_n_s16(ztw);
  const int16x8_t qv = vdupq_n_s16(3329);

  int16x8_t qhat = vqrdmulhq_s16(x,ztv);
  int16x8_t r = vmulq_s16(x,zv);
  r = vmlsq_s16(r, qhat, qv);
  return r;
}

/* ---- sme2_partial_ntt_neon ---- */
static void sme2_partial_ntt_neon(
    int16_t a[256],
    unsigned depth)
{
  unsigned layer;

  for (layer = 1u; layer <= depth; layer++) {
    const unsigned len=256u >> layer;
    unsigned k = 1u << (layer - 1u);
    unsigned start;

    for (start = 0u; start < 256u; start += 2u * len) {
      const int16_t z = sme2_ntt_roots_centered[k];
      const int16_t ztw = sme2_ntt_roots_twisted[k];
      unsigned j;
      k++;

      for (j = start; j < start + len; j += 8u) {
        const int16x8_t x = vld1q_s16(a + j);
        const int16x8_t y = vld1q_s16(a + j + len);
        const int16x8_t t = sme2_partial_ntt_mul_vec(y, z, ztw);

        vst1q_s16(a + j, vaddq_s16(x, t));
        vst1q_s16(a + j + len, vsubq_s16(x, t));
      }
    }
  }
}

MLK_INTERNAL_API void sme2_partial_ntt3_neon(int16_t a[256])
{
  // int16_tを256個読み込んで3段NTTを実行する
  sme2_partial_ntt_neon(a,3u);
}

#else

MLK_EMPTY_CU(aarch64_sme2_ntt)

#endif /* MLK_ARITH_BACKEND_AARCH64 && MLK_SYS_AARCH64_SME2 && MLKEM_K == 3 */
