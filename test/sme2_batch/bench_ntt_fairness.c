
#include <mach/mach_time.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "prepared_ntt.h"
#include "common.h"
#include "indcpa.h"
#include "kem.h"
#include "params.h"
#include "poly.h"
#include "poly_k.h"
#include "sampling.h"
#include "symmetric.h"
#include "verify.h"
#include "native/aarch64/sme2/mlkem_sme2_decaps.h"
#include "ntt_batch16_neon.h"

#if !defined(MLK_SYS_AARCH64_SME2) || MLKEM_K != 3
#error "Apple-M4/SME2 ML-KEM-768 build required"
#endif

/* Experimental 16-request SIMD NTT scratch; front/finish is unchanged. */
typedef struct {
  ntt_fair_workspace core;
  MLK_ALIGN mlk_polyvec staged[MLK_SME2_GROUP];
  MLK_ALIGN int16_t soa[4][MLKEM_N][MLK_SME2_GROUP];
} ntt_neon16_workspace;

static mlk_sme2_decaps_key sme2_key;
static mlk_sme2_decaps_workspace sme2_ws;
static ntt_key nk;
static ntt_legacy_workspace legacy_ws;
static ntt_fair_workspace fair_ws;
static ntt_neon16_workspace neon16_ws;

static uint8_t pk[MLKEM_INDCCA_PUBLICKEYBYTES];
static uint8_t sk[MLKEM_INDCCA_SECRETKEYBYTES];
static uint8_t ct[MAX_N][MLKEM_INDCCA_CIPHERTEXTBYTES];
static uint8_t expected[MAX_N][MLKEM_SSBYTES];
static uint8_t got[MAX_N][MLKEM_SSBYTES];

static uint32_t xs32(uint32_t *s)
{
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return x;
}

static void fill(uint8_t *p, size_t n, uint32_t seed)
{
  size_t i;
  for (i = 0; i < n; i++)
    p[i] = (uint8_t)xs32(&seed);
}

/* BEGIN experimental NEON16-vs-SME2 comparator. */
static int ntt_neon16_batch(
    const ntt_key *key,
    ntt_neon16_workspace *ws,
    uint8_t ss[][MLKEM_SSBYTES],
    const uint8_t ct0[][MLKEM_INDCCA_CIPHERTEXTBYTES],
    unsigned n)
{
  ntt_fair_workspace *core = &ws->core;
  unsigned group, local, row, r;
  int ret = ntt_front(key, &core->sb, core->message, core->kr, ct0, n);
  if (ret != 0)
    goto cleanup;

  for (group = 0; group < n; group += MLK_SME2_GROUP)
  {
    /* Stage all 16 inputs before writes to the sp/b union. */
    ntt16_forward(ws->staged, &core->sb.sp[group], ws->soa);

    for (local = 0; local < MLK_SME2_GROUP; local++)
    {
      r = group + local;
      core->sp_ntt = ws->staged[local];
      mlk_polyvec_mulcache_compute(&core->cache, &core->sp_ntt);

      /* Keep the pre-existing assembly AArch64 basemul accumulator. */
      for (row = 0; row < MLKEM_K; row++)
        mlk_polyvec_basemul_acc_montgomery_cached(
            &core->sb.b[r].vec[row], &key->at_ntt.vec[row],
            &core->sp_ntt, &core->cache);

      mlk_polyvec_basemul_acc_montgomery_cached(
          &core->v[r], &key->t_ntt, &core->sp_ntt, &core->cache);
    }

    /* Four output polynomials x 16 requests, inverse NTT in NEON lanes. */
    ntt16_inverse(&core->sb.b[group], &core->v[group], ws->soa);
  }

  for (r = 0; r < n; r++)
    ntt_finish_one(key, &core->sb, core->v, core->message, core->kr,
                   ss[r], ct0[r], r);

cleanup:
  if (ret != 0)
    mlk_zeroize(ss, (size_t)n * MLKEM_SSBYTES);
  ntt_wipe_common(&core->sb, core->v, core->message, core->kr, n);
  mlk_zeroize(&core->sp_ntt, sizeof(core->sp_ntt));
  mlk_zeroize(&core->cache, sizeof(core->cache));
  mlk_zeroize(ws->staged, sizeof(ws->staged));
  mlk_zeroize(ws->soa, sizeof(ws->soa));
  return ret;
}
/* END experimental NEON16-vs-SME2 comparator. */

static uint64_t now_ns(void)
{
  static mach_timebase_info_data_t tb;
  uint64_t t = mach_continuous_time();
  if (tb.denom == 0)
    mach_timebase_info(&tb);
  return (uint64_t)(((__uint128_t)t * tb.numer) / tb.denom);
}

// run_variant(1, n)
//   └─ mlk_sme2_decaps_batch()
//        └─ mlk_sme2_arith_batch_rolling_preloaded()
//             ├─ sme2_partial_ntt_neon(..., 3u)
//             │    ↑ ここが3段打切りNTT
//             └─ sme2_roll_stream()
//                  ├─ Toeplitz用のデータ配置
//                  ├─ SMOPAによる多項式積和
//                  └─ 逆変換・係数復元

static int run_variant(unsigned variant, unsigned n)
{
  switch (variant)
  {
    case 1:
      return mlk_sme2_decaps_batch(&sme2_key, &sme2_ws, got, ct, n, NULL);
    case 2:
      return ntt_legacy_batch(&nk, &legacy_ws, got, ct, n);
    case 3:
      return ntt_fair_batch(&nk, &fair_ws, got, ct, n);
    case 4:
      return ntt_neon16_batch(&nk, &neon16_ws, got, ct, n);
    default:
      return -1;
  }
}

// 実行コマンド: ./a.out VARIANT N INNER WARMUP
// VARIANT: 1=SME2, 2=従来のNTT, 3=公平なNTT
// N: 要求数
// INNER: 内部ループ回数
// WARMUP: ウォームアップ回数
int main(int argc, char **argv)
{
  uint8_t key_coins[2 * MLKEM_SYMBYTES];
  uint8_t enc_coins[MLKEM_SYMBYTES];
  unsigned variant, n, inner, warmup, i, r;
  uint64_t t0, t1;
  int ret;

  if (argc != 5)
  {
    fprintf(stderr, "usage: %s VARIANT N INNER WARMUP\n", argv[0]);
    return 2;
  }

  variant = (unsigned)strtoul(argv[1], NULL, 10);
  n = (unsigned)strtoul(argv[2], NULL, 10);
  inner = (unsigned)strtoul(argv[3], NULL, 10);
  warmup = (unsigned)strtoul(argv[4], NULL, 10);

  if (variant < 1 || variant > 4 ||
      n < MLK_SME2_GROUP || n > MAX_N ||
      (n % MLK_SME2_GROUP) != 0 || inner == 0)
    return 2;

  fill(key_coins, sizeof(key_coins), UINT32_C(0x243f6a88));
  ret = mlk_kem_keypair_derand(pk, sk, key_coins, NULL);
  if (ret != 0)
    return 3;

  for (i = 0; i < n; i++)
  {
    fill(enc_coins, sizeof(enc_coins),
         UINT32_C(0x85a308d3) ^
         (UINT32_C(0x9e3779b9) * (uint32_t)(i + 1u)));
    ret = mlk_kem_enc_derand(ct[i], expected[i], pk, enc_coins, NULL);
    if (ret != 0)
      return 4;
  }

  ret = mlk_sme2_decaps_prepare(&sme2_key, sk, NULL);
  if (ret != 0)
    return 5;
  ret = prepare_ntt(&nk, sk);
  if (ret != 0)
    return 6;

  /* Correctness for all four paths before timing. */
  for (i = 1; i <= 4; i++)
  {
    ret = run_variant(i, n);
    if (ret != 0)
      return 7;
    for (r = 0; r < n; r++)
      if (memcmp(got[r], expected[r], MLKEM_SSBYTES) != 0)
        return 8;
  }

  for (r = 0; r < warmup; r++)
    if (run_variant(variant, n) != 0)
      return 9;

  t0 = now_ns();
  for (r = 0; r < inner; r++)
    if (run_variant(variant, n) != 0)
      return 10;
  t1 = now_ns();

  printf("variant=%u N=%u ns_per_req=%.6f "
         "sme2_key=%zu ntt_key=%zu "
         "sme2_ws=%zu ntt_legacy_ws=%zu ntt_fair_ws=%zu\n",
         variant, n,
         (double)(t1 - t0) / ((double)inner * (double)n),
         sizeof(sme2_key), sizeof(nk),
         sizeof(sme2_ws), sizeof(legacy_ws), sizeof(fair_ws));
  return 0;
}
