/* Keep the established benchmark implementation unchanged and share its
 * front/finish, deterministic inputs and clock with this experimental driver. */
#define main phase2_original_fairness_main
#include "bench_ntt_fairness.c"
#undef main

#include <errno.h>
#include <limits.h>
#include "prepared_ntt.h"

static ntt_public_key public_key;
static ntt_public_workspace public_ws;

static int parse_unsigned(const char *text, unsigned *value)
{
  char *end;
  unsigned long parsed;
  if (text[0] < '0' || text[0] > '9')
    return -1;
  errno = 0;
  parsed = strtoul(text, &end, 10);
  if (errno != 0 || *end != '\0' || parsed > UINT_MAX)
    return -1;
  *value = (unsigned)parsed;
  return 0;
}

static int run_public_variant(unsigned variant, unsigned n)
{
  if (variant == 1)
    return mlk_sme2_decaps_batch(&sme2_key, &sme2_ws, got, ct, n, NULL);
  if (variant == 5)
    return ntt_public_batch(&public_key, &public_ws, got, ct, n);
  return -1;
}

static void print_memory(void)
{
  printf("sme2_key=%zu ntt_key=%zu ntt_public_key=%zu "
         "sme2_ws=%zu ntt_legacy_ws=%zu ntt_fair_ws=%zu ntt_public_ws=%zu",
         sizeof(sme2_key), sizeof(nk), sizeof(public_key),
         sizeof(sme2_ws), sizeof(legacy_ws), sizeof(fair_ws), sizeof(public_ws));
}

static void wipe_driver_state(void)
{
  mlk_zeroize(&sme2_key, sizeof(sme2_key));
  mlk_zeroize(&nk, sizeof(nk));
  mlk_zeroize(&public_key, sizeof(public_key));
  mlk_zeroize(&sme2_ws, sizeof(sme2_ws));
  mlk_zeroize(&fair_ws, sizeof(fair_ws));
  mlk_zeroize(&public_ws, sizeof(public_ws));
  mlk_zeroize(sk, sizeof(sk));
  mlk_zeroize(expected, sizeof(expected));
  mlk_zeroize(got, sizeof(got));
}

/* Preparation includes the complete existing entry point. In particular,
 * SME2 still performs and prints its public gamma-structure validation.
 * stdout is flushed before the interval; any in-interval logging remains
 * part of the measurement. This is reported separately from Decaps. */
static int measure_prepare(unsigned inner, unsigned warmup)
{
  unsigned i;
  uint64_t t0, t1, sme2_ns, ntt_ns;
  for (i = 0; i < warmup; i++)
  {
    if (mlk_sme2_decaps_prepare(&sme2_key, sk, NULL) != 0 ||
        prepare_ntt_public(&public_key, sk) != 0)
      return -1;
  }

  (void)now_ns();
  fflush(stdout);
  t0 = now_ns();
  for (i = 0; i < inner; i++)
    if (mlk_sme2_decaps_prepare(&sme2_key, sk, NULL) != 0)
      return -1;
  t1 = now_ns();
  sme2_ns = t1 - t0;

  fflush(stdout);
  t0 = now_ns();
  for (i = 0; i < inner; i++)
    if (prepare_ntt_public(&public_key, sk) != 0)
      return -1;
  t1 = now_ns();
  ntt_ns = t1 - t0;

  printf("mode=prepare inner=%u warmup=%u sme2_ns_per_key=%.6f "
         "ntt_public_ns_per_key=%.6f sme2_prep_includes_diagnostic_printf=0 ",
         inner, warmup, (double)sme2_ns / inner, (double)ntt_ns / inner);
  print_memory();
  printf("\n");
  return 0;
}

int main(int argc, char **argv)
{
  uint8_t key_coins[2 * MLKEM_SYMBYTES];
  uint8_t enc_coins[MLKEM_SYMBYTES];
  unsigned variant = 0, n = 0, inner = 0, warmup = 0, i, r;
  uint64_t t0, t1;
  int prepare_mode = argc == 4 && strcmp(argv[1], "--prepare") == 0;
  int result = 0;

  if (argc == 2 && (strcmp(argv[1], "--metadata-only") == 0 ||
                    strcmp(argv[1], "--metadata") == 0))
  {
    print_memory();
    printf("\n");
    return 0;
  }
  if (prepare_mode)
  {
    if (parse_unsigned(argv[2], &inner) != 0 ||
        parse_unsigned(argv[3], &warmup) != 0 || inner == 0)
      return 2;
  }
  else if (argc != 5 || parse_unsigned(argv[1], &variant) != 0 ||
           parse_unsigned(argv[2], &n) != 0 ||
           parse_unsigned(argv[3], &inner) != 0 ||
           parse_unsigned(argv[4], &warmup) != 0 ||
           (variant != 1 && variant != 5) ||
           n < MLK_SME2_GROUP || n > MAX_N || n % MLK_SME2_GROUP != 0 ||
           inner == 0)
  {
    fprintf(stderr, "usage: %s {1|5} N INNER WARMUP\n"
                    "       %s --prepare INNER WARMUP\n"
                    "       %s --metadata-only\n", argv[0], argv[0], argv[0]);
    return 2;
  }

  fill(key_coins, sizeof(key_coins), UINT32_C(0x243f6a88));
  if (mlk_kem_keypair_derand(pk, sk, key_coins, NULL) != 0)
  {
    result = 3;
    goto cleanup;
  }
  if (prepare_mode)
  {
    result = measure_prepare(inner, warmup) == 0 ? 0 : 5;
    goto cleanup;
  }

  for (i = 0; i < n; i++)
  {
    fill(enc_coins, sizeof(enc_coins),
         UINT32_C(0x85a308d3) ^
         (UINT32_C(0x9e3779b9) * (uint32_t)(i + 1u)));
    if (mlk_kem_enc_derand(ct[i], expected[i], pk, enc_coins, NULL) != 0)
    {
      result = 4;
      goto cleanup;
    }
  }

  if (mlk_sme2_decaps_prepare(&sme2_key, sk, NULL) != 0 ||
      prepare_ntt(&nk, sk) != 0 ||
      prepare_ntt_public(&public_key, sk) != 0)
  {
    result = 5;
    goto cleanup;
  }

  /* Check all three paths before any timing; only variants 1 and 5 can be
   * timed. The original prepared NTT remains a correctness reference. */
  for (i = 0; i < 3; i++)
  {
    int ret = i == 1 ? ntt_fair_batch(&nk, &fair_ws, got, ct, n)
                     : run_public_variant(i == 0 ? 1u : 5u, n);
    if (ret != 0)
    {
      result = 7;
      goto cleanup;
    }
    for (r = 0; r < n; r++)
      if (memcmp(got[r], expected[r], MLKEM_SSBYTES) != 0)
      {
        result = 8;
        goto cleanup;
      }
  }

  for (r = 0; r < warmup; r++)
    if (run_public_variant(variant, n) != 0)
    {
      result = 9;
      goto cleanup;
    }

  t0 = now_ns();
  for (r = 0; r < inner; r++)
    if (run_public_variant(variant, n) != 0)
    {
      result = 10;
      goto cleanup;
    }
  t1 = now_ns();
  printf("variant=%u N=%u ns_per_req=%.6f ", variant, n,
         (double)(t1 - t0) / ((double)inner * n));
  print_memory();
  printf("\n");

cleanup:
  mlk_zeroize(enc_coins, sizeof(enc_coins));
  mlk_zeroize(key_coins, sizeof(key_coins));
  wipe_driver_state();
  if (result != 0)
    fprintf(stderr, "public-cache benchmark failed: %d\n", result);
  return result;
}
