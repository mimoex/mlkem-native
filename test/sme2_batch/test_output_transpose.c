/* ZA layout/bounds checks and both inverse paths against a schoolbook oracle. */
#include <stdio.h>
#include <string.h>

#include "mlkem/src/native/aarch64/sme2/mlkem_sme2_internal.h"
#include "../sme2_5way/math_core.h"

#define CHECK(c) do { if (!(c)) { \
  fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
  return 1; } } while (0)

__arm_new("za") __arm_locally_streaming
static MLK_SME2_NOINLINE void transpose(
    const int16_t *source, int16_t *const ptrs[64])
{
  MLK_NAMESPACE(sme2_transpose_output_d3_body)(source, ptrs);
}

static int check_layout(void)
{
  static int16_t input[8][4][32][16];
  static int16_t saved[8][4][32][16];
  static struct {
    int16_t before[32];
    int16_t coeffs[256];
    int16_t after[32];
  } output[64];
  int16_t *ptrs[64];
  unsigned pass, b, d, t, r, i;

  /* Four disjoint patterns cover every int16_t value, including both limits.
   * Scramble destination addresses so no output stride can be assumed. */
  for (pass = 0; pass < 4; pass++)
  {
    for (b = 0; b < 8; b++)
      for (d = 0; d < 4; d++)
        for (t = 0; t < 32; t++)
          for (r = 0; r < 16; r++)
            input[b][d][t][r] = (int16_t)(
                (int)((((b * 4u + d) * 32u + t) * 16u + r)
                      + pass * 16384u) - 32768);
    memcpy(saved, input, sizeof(input));
    memset(output, 0x5a, sizeof(output));
    for (i = 0; i < 64; i++)
      ptrs[i] = output[(17u * i + 3u) % 64u].coeffs;
    transpose(&input[0][0][0][0], ptrs);
    CHECK(memcmp(input, saved, sizeof(input)) == 0);
    for (d = 0; d < 4; d++)
      for (r = 0; r < 16; r++)
        for (b = 0; b < 8; b++)
          for (t = 0; t < 32; t++)
            CHECK(ptrs[d * 16u + r][32u * b + t] == input[b][d][t][r]);
    for (i = 0; i < 64; i++)
      for (t = 0; t < 32; t++)
      {
        CHECK(output[i].before[t] == 0x5a5a);
        CHECK(output[i].after[t] == 0x5a5a);
      }
  }
  puts("PASS ZA transpose: all 65536 int16 values; shuffled pointers; input and output guards intact");
  return 0;
}

static int check_inverse_paths(void)
{
  static mlk_sme2_public_ctx ctx;
  static mlk_sme2_workspace full;
  static mlk_sme2_rolling_workspace rolling;
  static int16_t pub[4][3][256];
  static int16_t secret[256][3][256];
  static int16_t expected[256][4][256];
  unsigned step, n, r, d, k, i;

  bw_fill_input(pub, secret, 256, 1);
  for (r = 0; r < 256; r++)
    bw_oracle_one(pub, secret[r], expected[r]);
  CHECK(mlk_sme2_prepare_public(&ctx, pub) == 0);

  /* Alternating small/large batches exercise every group and scratch reuse. */
  for (step = 0; step < 16; step++)
  {
    n = 16u * ((step & 1u) ? 16u - step / 2u : step / 2u + 1u);
    memset(&full, 0x5a, sizeof(full));
    memset(&rolling, 0x5a, sizeof(rolling));
    for (r = 0; r < n; r++)
      for (k = 0; k < 3; k++)
      {
        memcpy(full.arena.sec_partial[r].vec[k].coeffs,
               secret[r][k], sizeof(secret[r][k]));
        memcpy(rolling.arena.group[r / 16u].sec[r % 16u].vec[k].coeffs,
               secret[r][k], sizeof(secret[r][k]));
      }
    CHECK(mlk_sme2_arith_batch_workspace_preloaded(&ctx, &full, n) == 0);
    CHECK(mlk_sme2_arith_batch_rolling_preloaded(&ctx, &rolling, n) == 0);
    for (r = 0; r < n; r++)
      for (d = 0; d < 4; d++)
      {
        const mlk_sme2_output *out = &rolling.arena.group[r / 16u].out[r % 16u];
        const int16_t *a = sme2_output_poly(&full, r, d)->coeffs;
        const int16_t *b = d < 3 ? out->b.vec[d].coeffs : out->v.coeffs;
        for (i = 0; i < 256; i++)
          if (bw_mod(a[i]) != bw_mod(expected[r][d][i]) ||
              bw_mod(b[i]) != bw_mod(expected[r][d][i]))
          {
            fprintf(stderr,
                    "FAIL inverse n=%u r=%u d=%u i=%u full=%d rolling=%d expected=%d\n",
                    n, r, d, i, a[i], b[i], expected[r][d][i]);
            return 1;
          }
      }
    /* Full raw storage occupies twice the output space, so its untouched
     * arena tail begins after the active raw groups, not after out[n]. */
    {
      size_t j;
      const size_t full_used = (n / 16u) * sizeof(full.arena.raw[0]);
      const size_t rolling_used = (n / 16u) * sizeof(rolling.arena.group[0]);
      const size_t secondary_used = (n / 16u) * sizeof(full.secondary.inverse_h[0]);
      for (j = full_used; j < sizeof(full.arena); j++)
        CHECK(((const unsigned char *)&full.arena)[j] == 0x5a);
      for (j = rolling_used; j < sizeof(rolling.arena); j++)
        CHECK(((const unsigned char *)&rolling.arena)[j] == 0x5a);
      for (j = secondary_used; j < sizeof(full.secondary); j++)
        CHECK(((const unsigned char *)&full.secondary)[j] == 0x5a);
    }
  }
  puts("PASS ordinary/rolling inverse: all 16 batch sizes against schoolbook convolution; unused tails intact");
  return 0;
}

int main(void)
{
  CHECK(mlk_sme2_check_svl() == 0);
  CHECK(check_layout() == 0);
  CHECK(check_inverse_paths() == 0);
  return 0;
}
