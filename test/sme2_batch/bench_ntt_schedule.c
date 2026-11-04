
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

#if !defined(MLK_SYS_AARCH64_SME2) || MLKEM_K != 3
#error "Apple-M4/SME2 ML-KEM-768 build required"
#endif

typedef struct {
  ntt_sb_arena sb;
  MLK_ALIGN mlk_poly v[MAX_N];
  uint8_t message[MAX_N][MLKEM_SYMBYTES];
  uint8_t kr[MAX_N][2 * MLKEM_SYMBYTES];
  MLK_ALIGN mlk_polyvec sp_ntt[4];
  MLK_ALIGN mlk_polyvec_mulcache cache[4];
} ntt_phase4_workspace;

typedef struct {
  ntt_sb_arena sb;
  MLK_ALIGN mlk_poly v[MAX_N];
  uint8_t message[MAX_N][MLKEM_SYMBYTES];
  uint8_t kr[MAX_N][2 * MLKEM_SYMBYTES];
  MLK_ALIGN mlk_polyvec sp_ntt[8];
  MLK_ALIGN mlk_polyvec_mulcache cache[8];
} ntt_phase8_workspace;

typedef struct {
  ntt_sb_arena sb;
  MLK_ALIGN mlk_poly v[MAX_N];
  uint8_t message[MAX_N][MLKEM_SYMBYTES];
  uint8_t kr[MAX_N][2 * MLKEM_SYMBYTES];
  MLK_ALIGN mlk_polyvec sp_ntt[16];
  MLK_ALIGN mlk_polyvec_mulcache cache[16];
} ntt_phase16_workspace;

_Static_assert(sizeof(ntt_legacy_workspace) == 548864u,
               "unexpected legacy prepared-NTT workspace");
_Static_assert(sizeof(ntt_fair_workspace) == 551168u,
               "unexpected fair prepared-NTT workspace");

_Static_assert(sizeof(ntt_phase4_workspace) == 558080u,
               "unexpected phase4 prepared-NTT workspace");
_Static_assert(sizeof(ntt_phase8_workspace) == 567296u,
               "unexpected phase8 prepared-NTT workspace");
_Static_assert(sizeof(ntt_phase16_workspace) == 585728u,
               "unexpected phase16 prepared-NTT workspace");

static mlk_sme2_decaps_key sme2_key;
static mlk_sme2_decaps_workspace sme2_ws;
static ntt_key nk;
static ntt_legacy_workspace legacy_ws;
static ntt_fair_workspace fair_ws;
static ntt_phase4_workspace phase4_ws;
static ntt_phase8_workspace phase8_ws;
static ntt_phase16_workspace phase16_ws;

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

static int ntt_phase_batch_raw(
    const ntt_key *key,
    ntt_sb_arena *sb,
    mlk_poly v[],
    uint8_t message[][MLKEM_SYMBYTES],
    uint8_t kr[][2 * MLKEM_SYMBYTES],
    mlk_polyvec sp_ntt[],
    mlk_polyvec_mulcache cache[],
    unsigned width,
    uint8_t ss[][MLKEM_SSBYTES],
    const uint8_t ct0[][MLKEM_INDCCA_CIPHERTEXTBYTES],
    unsigned n)
{
  unsigned base, cidx, row, r;
  int ret = ntt_front(key, sb, message, kr, ct0, n);
  if (ret != 0)
    goto cleanup;

  /*
   * Cross-request schedule-only batching.
   *
   * Arithmetic primitives are unchanged: the same mlkem-native NTT,
   * mulcache, cached basemul and inverse NTT functions are used.  Only their
   * order is changed within a microbatch.
   *
   *   phase 1: NTT + mulcache for W independent requests
   *   phase 2: public row 0 across W requests
   *   phase 3: public row 1 across W requests
   *   phase 4: public row 2 across W requests
   *   phase 5: t^T*s across W requests
   *   phase 6: inverse NTT across W requests
   *
   * This gives the NTT path the same kind of cross-request scheduling
   * opportunity as the SME2 path without introducing a new NTT kernel.
   */
  for (base = 0u; base < n; base += width)
  {
    for (cidx = 0u; cidx < width; cidx++)
    {
      sp_ntt[cidx] = sb->sp[base + cidx];
      mlk_polyvec_ntt(&sp_ntt[cidx]);
      mlk_polyvec_mulcache_compute(&cache[cidx], &sp_ntt[cidx]);
    }

    for (row = 0u; row < MLKEM_K; row++)
      for (cidx = 0u; cidx < width; cidx++)
        mlk_polyvec_basemul_acc_montgomery_cached(
            &sb->b[base + cidx].vec[row],
            &key->at_ntt.vec[row],
            &sp_ntt[cidx],
            &cache[cidx]);

    for (cidx = 0u; cidx < width; cidx++)
      mlk_polyvec_basemul_acc_montgomery_cached(
          &v[base + cidx],
          &key->t_ntt,
          &sp_ntt[cidx],
          &cache[cidx]);

    for (cidx = 0u; cidx < width; cidx++)
    {
      mlk_polyvec_invntt_tomont(&sb->b[base + cidx]);
      mlk_poly_invntt_tomont(&v[base + cidx]);
    }
  }

  for (r = 0u; r < n; r++)
    ntt_finish_one(key, sb, v, message, kr, ss[r], ct0[r], r);

cleanup:
  if (ret != 0)
    mlk_zeroize(ss, (size_t)n * MLKEM_SSBYTES);
  ntt_wipe_common(sb, v, message, kr, n);
  mlk_zeroize(sp_ntt, (size_t)width * sizeof(sp_ntt[0]));
  mlk_zeroize(cache, (size_t)width * sizeof(cache[0]));
  return ret;
}

static int ntt_phase4_batch(
    const ntt_key *key,
    ntt_phase4_workspace *ws,
    uint8_t ss[][MLKEM_SSBYTES],
    const uint8_t ct0[][MLKEM_INDCCA_CIPHERTEXTBYTES],
    unsigned n)
{
  return ntt_phase_batch_raw(
      key, &ws->sb, ws->v, ws->message, ws->kr,
      ws->sp_ntt, ws->cache, 4u, ss, ct0, n);
}

static int ntt_phase8_batch(
    const ntt_key *key,
    ntt_phase8_workspace *ws,
    uint8_t ss[][MLKEM_SSBYTES],
    const uint8_t ct0[][MLKEM_INDCCA_CIPHERTEXTBYTES],
    unsigned n)
{
  return ntt_phase_batch_raw(
      key, &ws->sb, ws->v, ws->message, ws->kr,
      ws->sp_ntt, ws->cache, 8u, ss, ct0, n);
}

static int ntt_phase16_batch(
    const ntt_key *key,
    ntt_phase16_workspace *ws,
    uint8_t ss[][MLKEM_SSBYTES],
    const uint8_t ct0[][MLKEM_INDCCA_CIPHERTEXTBYTES],
    unsigned n)
{
  return ntt_phase_batch_raw(
      key, &ws->sb, ws->v, ws->message, ws->kr,
      ws->sp_ntt, ws->cache, 16u, ss, ct0, n);
}

static uint64_t now_ns(void)
{
  static mach_timebase_info_data_t tb;
  uint64_t t = mach_continuous_time();
  if (tb.denom == 0)
    mach_timebase_info(&tb);
  return (uint64_t)(((__uint128_t)t * tb.numer) / tb.denom);
}

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
      return ntt_phase4_batch(&nk, &phase4_ws, got, ct, n);
    case 5:
      return ntt_phase8_batch(&nk, &phase8_ws, got, ct, n);
    case 6:
      return ntt_phase16_batch(&nk, &phase16_ws, got, ct, n);
    default:
      return -1;
  }
}

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

  if (variant < 1 || variant > 6 ||
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

  /* Correctness for all six paths before timing. */
  for (i = 1; i <= 6; i++)
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
         "sme2_ws=%zu ntt_legacy_ws=%zu ntt_fair_ws=%zu "
         "ntt_phase4_ws=%zu ntt_phase8_ws=%zu ntt_phase16_ws=%zu\n",
         variant, n,
         (double)(t1 - t0) / ((double)inner * (double)n),
         sizeof(sme2_key), sizeof(nk),
         sizeof(sme2_ws), sizeof(legacy_ws), sizeof(fair_ws),
         sizeof(phase4_ws), sizeof(phase8_ws), sizeof(phase16_ws));
  return 0;
}
