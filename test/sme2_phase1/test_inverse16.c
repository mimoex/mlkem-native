/*
 * Exhaustive range checks for the production rolling inverse NTT.
 *
 * Include the implementation so that the actual static NEON butterflies and
 * their actual constant tables are tested. The oracle uses int64_t arithmetic
 * in F_3329 and derives inverse twiddles from 17 and bit-reversed exponents.
 * No production arithmetic is reimplemented as the oracle.
 */
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mlkem/src/common.h"
#if !defined(MLK_ARITH_BACKEND_AARCH64) || !defined(MLK_SYS_AARCH64_SME2) || \
    MLKEM_K != 3
#error "This test requires the native AArch64 SME2 ML-KEM-768 configuration"
#endif
#include "mlkem/mlkem_native.c"

enum
{
  TEST_Q = 3329,
  TEST_PAIR_LANES = 4 * 32 * 16
};
static mlk_sme2_public_ctx inverse_ctx;
static mlk_sme2_rolling_workspace inverse_ws;
static uint64_t checked_multiplications;
static uint64_t checked_sums;

static int mod_q(int64_t x)
{
  const int r = (int)(x % TEST_Q);
  return r < 0 ? r + TEST_Q : r;
}

static int pow_q(int x, unsigned exponent)
{
  int64_t result = 1;
  for (; exponent != 0; exponent >>= 1)
  {
    if ((exponent & 1u) != 0)
    {
      result = result * x % TEST_Q;
    }
    x = (int)((int64_t)x * x % TEST_Q);
  }
  return (int)result;
}

static unsigned reverse7(unsigned k)
{
  unsigned result = 0, i;
  for (i = 0; i < 7; i++, k >>= 1)
  {
    result = 2u * result + (k & 1u);
  }
  return result;
}

static int abs_int(int x) { return x < 0 ? -x : x; }

static void fail(const char *what, unsigned stage, unsigned k, int input,
                 int actual, int expected)
{
  fprintf(stderr,
          "FAIL inverse16: %s stage=%u k=%u input=%d actual=%d expected=%d\n",
          what, stage, k, input, actual, expected);
  exit(1);
}

/* Exhaust the forward multipliers to derive the secret bound, starting with
 * eta2 samples in [-2,2]. Each butterfly is x +/- t with independent inputs
 * inside the preceding block's interval, so B_next = B + max(abs(t)) is
 * conservative. Keep separate bounds for blocks: using the global maximum for
 * every twiddle loses the tighter 3297 bound of the second half at stage 2.
 * Congruence and lane bounds are checked against an independent field oracle.
 */
static int derive_secret_bound(void)
{
  static const int expected_bounds[3] = {1602, 3306, 5104};
  int bounds[8] = {2}, maximum_bound = 2;
  unsigned stage;
  for (stage = 0; stage < 3; stage++)
  {
    int next_bounds[8] = {0};
    unsigned k;
    maximum_bound = 0;
    for (k = 1u << stage; k < (2u << stage); k++)
    {
      const int root = pow_q(17, reverse7(k));
      const unsigned block = k - (1u << stage);
      const int bound = bounds[block];
      int max_product = 0;
      int x;
      for (x = -bound; x <= bound; x += 8)
      {
        int16_t input[8], output[8];
        unsigned lane;
        for (lane = 0; lane < 8; lane++)
        {
          input[lane] =
              (int16_t)(x + (int)lane <= bound ? x + (int)lane : bound);
        }
        vst1q_s16(output, sme2_partial_ntt_mul_vec(vld1q_s16(input),
                                                   sme2_ntt_roots_centered[k],
                                                   sme2_ntt_roots_twisted[k]));
        for (lane = 0; lane < 8 && x + (int)lane <= bound; lane++)
        {
          const int expected = mod_q((int64_t)input[lane] * root);
          if (mod_q(output[lane]) != expected)
          {
            fail("forward congruence", stage, k, input[lane], output[lane],
                 expected);
          }
          if (abs_int(output[lane]) > max_product)
          {
            max_product = abs_int(output[lane]);
          }
        }
      }
      next_bounds[2u * block] = bound + max_product;
      next_bounds[2u * block + 1u] = bound + max_product;
      if (bound + max_product > maximum_bound)
      {
        maximum_bound = bound + max_product;
      }
    }
    for (k = 0; k < (2u << stage); k++)
    {
      bounds[k] = next_bounds[k];
    }
    if (maximum_bound != expected_bounds[stage] || maximum_bound > INT16_MAX)
    {
      fail("secret bound", stage, 0, 0, maximum_bound, expected_bounds[stage]);
    }
    printf("PASS forward stage %u: |secret| <= %d\n", stage + 1u,
           maximum_bound);
  }
  return maximum_bound;
}

/* The reciprocal is 645084/2^31 and 645084*3329 - 2^31 = 988.
 * For |raw| <= B, rounded multiplication gives
 * |reduce(raw)| <= 3329/2 + B*988/2^31. Flooring bounds the integer result.
 * Each ZA element accumulates K*32 = 96 products, not 16 requests together.
 */
static int derive_reduced_bound(int secret_bound, int64_t *raw_bound)
{
  const int64_t scale = INT64_C(1) << 31;
  const int64_t reciprocal_error = INT64_C(645084) * TEST_Q - scale;
  int bound;
  *raw_bound = INT64_C(3) * 32 * (TEST_Q / 2) * secret_bound;
  if (*raw_bound >= INT32_MAX || reciprocal_error != 988)
  {
    fail("raw bound", 0, 0, 0, (int)*raw_bound, 815333376);
  }
  bound =
      (int)(((int64_t)TEST_Q * (scale / 2) + *raw_bound * reciprocal_error) /
            scale);
  if (bound != 2039)
  {
    fail("reduced bound", 0, 0, 0, bound, 2039);
  }
  printf("PASS ZA bound: |raw| <= %" PRId64 ", |reduced| <= %d\n", *raw_bound,
         bound);
  return bound;
}

/* Construct a raw input with reduce(raw) == value. Near zero use the value
 * itself; outside the centered interval use the end of the proven raw range
 * with opposite sign. Check every constructed preimage with the real reducer.
 */
static int32_t raw_preimage(int value, int64_t raw_bound)
{
  int64_t raw = value;
  if (value > TEST_Q / 2)
  {
    raw -= ((raw_bound + value) / TEST_Q) * TEST_Q;
  }
  else if (value < -(TEST_Q / 2))
  {
    raw += ((raw_bound - value) / TEST_Q) * TEST_Q;
  }
  if (raw < -raw_bound || raw > raw_bound)
  {
    fail("preimage range", 0, 0, value, (int)raw, (int)raw_bound);
  }
  return (int32_t)raw;
}

static void check_preimages(int bound, int64_t raw_bound)
{
  int value;
  for (value = -bound; value <= bound; value++)
  {
    const int32_t raw = raw_preimage(value, raw_bound);
    const int actual = vgetq_lane_s32(sme2_roll_reduce4(vdupq_n_s32(raw)), 0);
    if (actual != value)
    {
      fail("raw preimage reduction", 0, 0, value, actual, value);
    }
  }
  printf("PASS raw preimages: all %d reduced inputs\n", 2 * bound + 1);
}

/* Select legal butterfly operands for every possible difference or sum in
 * [lo,hi]^2. Difference enumeration covers every multiplication input, while
 * sum enumeration also checks every parity and every half-add/narrow boundary.
 */
static void select_operands(int input, int sums, int lo, int hi, int *u, int *v)
{
  if (sums)
  {
    *u = input - lo > hi ? hi : input - lo;
    *v = input - *u;
  }
  else
  {
    *u = lo + (input > 0 ? input : 0);
    *v = lo + (input < 0 ? -input : 0);
  }
  if (*u < lo || *u > hi || *v < lo || *v > hi)
  {
    fail("operand range", 0, 0, input, *u, *v);
  }
}

static void check_pair_range(unsigned stage, unsigned k, int lo, int hi,
                             int output_lo, int output_hi, int64_t raw_bound,
                             int sums)
{
  const int first = sums ? 2 * lo : lo - hi;
  const int last = sums ? 2 * hi : hi - lo;
  const int inverse_twiddle =
      pow_q(2 * pow_q(17, reverse7(k)) % TEST_Q, TEST_Q - 2u);
  int base;
  for (base = first; base <= last; base += TEST_PAIR_LANES)
  {
    unsigned index;
    for (index = 0; index < TEST_PAIR_LANES; index++)
    {
      const int input = base + (int)index <= last ? base + (int)index : last;
      const unsigned d = index / (32u * 16u), t = (index / 16u) % 32u;
      const unsigned lane = index % 16u;
      int u, v;
      select_operands(input, sums, lo, hi, &u, &v);
      if (stage == 0)
      {
        sme2_roll_raw_ptr(&inverse_ws, 0, d, t)[lane] =
            raw_preimage(u, raw_bound);
        sme2_roll_raw_ptr(&inverse_ws, 1, d, t)[lane] =
            raw_preimage(v, raw_bound);
      }
      else
      {
        inverse_ws.secondary.inverse_h[0][0][d][t][lane] = (int16_t)u;
        inverse_ws.secondary.inverse_h[0][1][d][t][lane] = (int16_t)v;
      }
    }
    if (stage == 0)
    {
      sme2_roll_inverse_first_pair(&inverse_ws, 0, 1, k);
    }
    else
    {
      sme2_roll_inverse_pair16(&inverse_ctx, &inverse_ws, 0, 1, k);
    }
    for (index = 0; index < TEST_PAIR_LANES && base + (int)index <= last;
         index++)
    {
      const int input = base + (int)index;
      const unsigned d = index / (32u * 16u), t = (index / 16u) % 32u;
      const unsigned lane = index % 16u;
      const int x = inverse_ws.secondary.inverse_h[0][0][d][t][lane];
      const int y = inverse_ws.secondary.inverse_h[0][1][d][t][lane];
      int u, v, sum, adjusted_sum, expected_half, expected_product;
      select_operands(input, sums, lo, hi, &u, &v);
      sum = u + v;
      adjusted_sum = sum + (sum % 2 != 0 ? TEST_Q : 0);
      expected_half = adjusted_sum / 2;
      expected_product = mod_q((int64_t)(u - v) * inverse_twiddle);
      if (u - v < INT16_MIN || u - v > INT16_MAX || sum < INT16_MIN ||
          sum > INT16_MAX || adjusted_sum < INT16_MIN ||
          adjusted_sum > INT16_MAX)
      {
        fail("16-bit intermediate", stage, k, input, adjusted_sum, INT16_MAX);
      }
      if (x != expected_half)
      {
        fail("half/narrow", stage, k, input, x, expected_half);
      }
      if (mod_q(y) != expected_product)
      {
        fail("multiply congruence", stage, k, u - v, y, expected_product);
      }
      if (x < output_lo || x > output_hi || y < output_lo || y > output_hi)
      {
        fail("inverse output bound", stage, k, input, y, output_hi);
      }
      if (sums)
      {
        checked_sums++;
      }
      else
      {
        checked_multiplications++;
      }
    }
  }
}

int main(void)
{
  static const int expected_upper[3] = {3703, 5367, 7031};
  int64_t raw_bound;
  const int secret_bound = derive_secret_bound();
  const int reduced_bound = derive_reduced_bound(secret_bound, &raw_bound);
  int lo = -reduced_bound, hi = reduced_bound;
  unsigned stage;
  check_preimages(reduced_bound, raw_bound);
  sme2_init_inverse_constants(&inverse_ctx);
  for (stage = 0; stage < 3; stage++)
  {
    /* Maximum odd sum is 2*hi-1. Adding q and halving gives hi+1664. */
    const int next_hi = hi + TEST_Q / 2;
    const unsigned first_k = 4u >> stage, last_k = 8u >> stage;
    unsigned k;
    if (next_hi != expected_upper[stage])
    {
      fail("inverse interval derivation", stage, 0, hi, next_hi,
           expected_upper[stage]);
    }
    for (k = first_k; k < last_k; k++)
    {
      check_pair_range(stage, k, lo, hi, lo, next_hi, raw_bound, 0);
      check_pair_range(stage, k, lo, hi, lo, next_hi, raw_bound, 1);
    }
    printf(
        "PASS inverse stage %u: k=%u..%u, every delta=[%d,%d], "
        "every sum=[%d,%d], output=[%d,%d]\n",
        stage + 1u, first_k, last_k - 1u, lo - hi, hi - lo, 2 * lo, 2 * hi, lo,
        next_hi);
    hi = next_hi;
  }
  printf("PASS inverse16: %" PRIu64
         " exhaustive multiplication inputs; "
         "%" PRIu64 " exhaustive half-sum inputs\n",
         checked_multiplications, checked_sums);
  return 0;
}
