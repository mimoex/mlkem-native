/* Phase 2 differential and erasure checks; never collect timing samples. */
#define main phase2_original_fairness_main
#include "bench_ntt_fairness.c"
#undef main
#include "prepared_ntt.h"

#define CHECK(c) do { if (!(c)) { \
  fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
  return 1; } } while (0)

#define GUARD_BYTES 64u
#define SENTINEL 0xa5

static ntt_public_key public_key;
static ntt_public_key public_key_copy;
static struct {
  uint8_t before[GUARD_BYTES];
  ntt_public_workspace ws;
  uint8_t after[GUARD_BYTES];
} public_guard;
static struct {
  uint8_t before[GUARD_BYTES];
  ntt_fair_workspace ws;
  uint8_t after[GUARD_BYTES];
} old_guard;
static struct {
  uint8_t before[GUARD_BYTES];
  mlk_sme2_decaps_workspace ws;
  uint8_t after[GUARD_BYTES];
} sme_guard;

static uint8_t valid_ct[MAX_N][MLKEM_INDCCA_CIPHERTEXTBYTES];
static uint8_t enc_ss[MAX_N][MLKEM_SSBYTES];
static uint64_t arithmetic_coefficients;
static uint64_t ntt_coefficients;
static uint64_t decaps_requests;
static uint64_t cases_checked;

static int bytes_are(const void *p, size_t n, unsigned char value)
{
  const unsigned char *b = p;
  size_t i;
  for (i = 0; i < n; i++)
    if (b[i] != value)
      return 0;
  return 1;
}

static int equal_mod_q(int16_t a, int16_t b)
{
  return ((int32_t)a - (int32_t)b) % MLKEM_Q == 0;
}

/* Check both sides of every live/unused boundary, not merely an initially
 * zero workspace. Secret-bearing scratch must be zero on return; untouched
 * capacity and guard bytes must retain the sentinel. */
static int common_wiped(const ntt_sb_arena *sb, const mlk_poly v[],
                        const uint8_t message[][MLKEM_SYMBYTES],
                        const uint8_t kr[][2 * MLKEM_SYMBYTES], unsigned n)
{
  CHECK(bytes_are(sb->sp, n * sizeof(sb->sp[0]), 0));
  CHECK(bytes_are(sb->sp + n, (MAX_N - n) * sizeof(sb->sp[0]), SENTINEL));
  CHECK(bytes_are(v, n * sizeof(v[0]), 0));
  CHECK(bytes_are(v + n, (MAX_N - n) * sizeof(v[0]), SENTINEL));
  CHECK(bytes_are(message, n * sizeof(message[0]), 0));
  CHECK(bytes_are(message + n, (MAX_N - n) * sizeof(message[0]), SENTINEL));
  CHECK(bytes_are(kr, n * sizeof(kr[0]), 0));
  CHECK(bytes_are(kr + n, (MAX_N - n) * sizeof(kr[0]), SENTINEL));
  return 0;
}

static int sme_wiped(unsigned n)
{
  const mlk_sme2_decaps_workspace *ws = &sme_guard.ws;
  unsigned groups = n / MLK_SME2_GROUP;
  CHECK(bytes_are(sme_guard.before, GUARD_BYTES, SENTINEL));
  CHECK(bytes_are(sme_guard.after, GUARD_BYTES, SENTINEL));
  CHECK(bytes_are(ws->arithmetic.arena.group,
                  groups * sizeof(ws->arithmetic.arena.group[0]), 0));
  CHECK(bytes_are(ws->arithmetic.arena.group + groups,
                  (MLK_SME2_MAX_GROUPS - groups) *
                      sizeof(ws->arithmetic.arena.group[0]), SENTINEL));
  CHECK(bytes_are(ws->arithmetic.raw, sizeof(ws->arithmetic.raw), 0));
  CHECK(bytes_are(&ws->arithmetic.secondary,
                  sizeof(ws->arithmetic.secondary), 0));
  CHECK(bytes_are(ws->message, n * sizeof(ws->message[0]), 0));
  CHECK(bytes_are(ws->message + n,
                  (MAX_N - n) * sizeof(ws->message[0]), SENTINEL));
  CHECK(bytes_are(ws->kr, n * sizeof(ws->kr[0]), 0));
  CHECK(bytes_are(ws->kr + n,
                  (MAX_N - n) * sizeof(ws->kr[0]), SENTINEL));
  if (n == MAX_N)
    CHECK(bytes_are(ws, sizeof(*ws), 0));
  return 0;
}

/* Reproduce the existing Prepared NTT arithmetic as the reference. In
 * addition to the final coefficient comparison, compare products before
 * inverse NTT so that a Montgomery scaling error cannot be hidden by it. */
static int old_arithmetic_and_swap(unsigned n)
{
  unsigned r, d, k, i;
  for (r = 0; r < n; r++)
  {
    mlk_polyvec canonical;
    old_guard.ws.sp_ntt = old_guard.ws.sb.sp[r];
    mlk_polyvec_ntt(&old_guard.ws.sp_ntt);
    mlk_polyvec_mulcache_compute(&old_guard.ws.cache, &old_guard.ws.sp_ntt);
    canonical = old_guard.ws.sp_ntt;
    mlk_polyvec_reduce(&canonical);
    for (k = 0; k < MLKEM_K; k++)
      for (i = 0; i < MLKEM_N; i++)
        CHECK(canonical.vec[k].coeffs[i] >= 0 &&
              canonical.vec[k].coeffs[i] < MLKEM_Q);

    for (d = 0; d < 4; d++)
    {
      const mlk_polyvec *pub = d < MLKEM_K ? &nk.at_ntt.vec[d] : &nk.t_ntt;
      const mlk_polyvec_mulcache *cache = d < MLKEM_K ?
          &public_key.at_cache[d] : &public_key.t_cache;
      mlk_poly *out = d < MLKEM_K ?
          &old_guard.ws.sb.b[r].vec[d] : &old_guard.ws.v[r];
      mlk_poly swapped;
      mlk_polyvec_basemul_acc_montgomery_cached(
          out, pub, &old_guard.ws.sp_ntt, &old_guard.ws.cache);
      mlk_polyvec_basemul_acc_montgomery_cached(
          &swapped, &canonical, pub, cache);
      for (i = 0; i < MLKEM_N; i++)
      {
        CHECK(equal_mod_q(out->coeffs[i], swapped.coeffs[i]));
        ntt_coefficients++;
      }
      mlk_zeroize(&swapped, sizeof(swapped));
    }
    mlk_polyvec_invntt_tomont(&old_guard.ws.sb.b[r]);
    mlk_poly_invntt_tomont(&old_guard.ws.v[r]);
    mlk_zeroize(&canonical, sizeof(canonical));
  }
  return 0;
}

static int compare_arithmetic(unsigned n)
{
  unsigned r, d, i;
  CHECK(ntt_front(&nk, &old_guard.ws.sb, old_guard.ws.message,
                  old_guard.ws.kr, ct, n) == 0);
  memcpy(public_guard.ws.sb.sp, old_guard.ws.sb.sp,
         n * sizeof(public_guard.ws.sb.sp[0]));
  for (r = 0; r < n; r++)
    sme_guard.ws.arithmetic.arena.group[r / MLK_SME2_GROUP]
        .sec[r % MLK_SME2_GROUP] = old_guard.ws.sb.sp[r];

  CHECK(old_arithmetic_and_swap(n) == 0);
  ntt_public_arith(&public_key, &public_guard.ws, n);
  CHECK(mlk_sme2_arith_batch_rolling_preloaded(
            &sme2_key.public_ctx, &sme_guard.ws.arithmetic, n) == 0);
  for (r = 0; r < n; r++)
  {
    const mlk_sme2_output *sme =
        &sme_guard.ws.arithmetic.arena.group[r / MLK_SME2_GROUP]
            .out[r % MLK_SME2_GROUP];
    for (d = 0; d < 4; d++)
    {
      const mlk_poly *old = d < MLKEM_K ?
          &old_guard.ws.sb.b[r].vec[d] : &old_guard.ws.v[r];
      const mlk_poly *improved = d < MLKEM_K ?
          &public_guard.ws.sb.b[r].vec[d] : &public_guard.ws.v[r];
      const mlk_poly *proposal = d < MLKEM_K ? &sme->b.vec[d] : &sme->v;
      for (i = 0; i < MLKEM_N; i++)
      {
        CHECK(equal_mod_q(old->coeffs[i], improved->coeffs[i]));
        CHECK(equal_mod_q(old->coeffs[i], proposal->coeffs[i]));
        arithmetic_coefficients++;
      }
    }
  }
  mlk_zeroize(&public_guard.ws, sizeof(public_guard.ws));
  mlk_zeroize(&old_guard.ws, sizeof(old_guard.ws));
  mlk_sme2_decaps_workspace_clear(&sme_guard.ws);
  return 0;
}

static int compare_decaps(unsigned n)
{
  unsigned variant;
  for (variant = 0; variant < 3; variant++)
  {
    memset(got, SENTINEL, sizeof(got));
    if (variant == 0)
    {
      memset(&old_guard, SENTINEL, sizeof(old_guard));
      CHECK(ntt_fair_batch(&nk, &old_guard.ws, got, ct, n) == 0);
      CHECK(common_wiped(&old_guard.ws.sb, old_guard.ws.v,
                         old_guard.ws.message, old_guard.ws.kr, n) == 0);
      CHECK(bytes_are(&old_guard.ws.sp_ntt, sizeof(old_guard.ws.sp_ntt), 0));
      CHECK(bytes_are(&old_guard.ws.cache, sizeof(old_guard.ws.cache), 0));
      CHECK(bytes_are(old_guard.before, GUARD_BYTES, SENTINEL));
      CHECK(bytes_are(old_guard.after, GUARD_BYTES, SENTINEL));
    }
    else if (variant == 1)
    {
      memset(&public_guard, SENTINEL, sizeof(public_guard));
      CHECK(ntt_public_batch(&public_key, &public_guard.ws, got, ct, n) == 0);
      CHECK(common_wiped(&public_guard.ws.sb, public_guard.ws.v,
                         public_guard.ws.message, public_guard.ws.kr, n) == 0);
      CHECK(bytes_are(&public_guard.ws.sp_ntt,
                      sizeof(public_guard.ws.sp_ntt), 0));
      CHECK(bytes_are(public_guard.before, GUARD_BYTES, SENTINEL));
      CHECK(bytes_are(public_guard.after, GUARD_BYTES, SENTINEL));
      CHECK(memcmp(&public_key, &public_key_copy, sizeof(public_key)) == 0);
    }
    else
    {
      memset(&sme_guard, SENTINEL, sizeof(sme_guard));
      CHECK(mlk_sme2_decaps_batch(&sme2_key, &sme_guard.ws,
                                 got, ct, n, NULL) == 0);
      CHECK(sme_wiped(n) == 0);
    }
    CHECK(memcmp(got, expected, n * MLKEM_SSBYTES) == 0);
    CHECK(bytes_are(got + n, (MAX_N - n) * MLKEM_SSBYTES, SENTINEL));
    decaps_requests += n;
  }
  return 0;
}

static int prepare_scenario(unsigned scenario)
{
  uint8_t reject_input[MLKEM_SYMBYTES + MLKEM_INDCCA_CIPHERTEXTBYTES];
  uint8_t rejected[MLKEM_SSBYTES];
  unsigned r;
  memcpy(ct, valid_ct, sizeof(ct));
  for (r = 0; r < MAX_N; r++)
  {
    const int invalid = scenario == 1 || (scenario == 2 && (r & 1u));
    if (invalid)
      ct[r][(37u * r + 11u) % MLKEM_INDCCA_CIPHERTEXTBYTES] ^=
          (uint8_t)(1u << (r % 8u));
    CHECK(mlk_kem_dec(expected[r], ct[r], sk, NULL) == 0);
    if (invalid)
    {
      memcpy(reject_input, nk.z, MLKEM_SYMBYTES);
      memcpy(reject_input + MLKEM_SYMBYTES, ct[r], sizeof(ct[r]));
      mlk_hash_j(rejected, reject_input, sizeof(reject_input));
      CHECK(memcmp(expected[r], rejected, sizeof(rejected)) == 0);
    }
    else
      CHECK(memcmp(expected[r], enc_ss[r], MLKEM_SSBYTES) == 0);
  }
  mlk_zeroize(reject_input, sizeof(reject_input));
  mlk_zeroize(rejected, sizeof(rejected));
  return 0;
}

int main(int argc, char **argv)
{
  static const uint32_t seeds[] = {
    UINT32_C(0x243f6a88), UINT32_C(0x13198a2e), UINT32_C(0xa4093822)
  };
  uint8_t key_coins[2 * MLKEM_SYMBYTES];
  uint8_t enc_coins[MLKEM_SYMBYTES];
  uint8_t bad_sk[MLKEM_INDCCA_SECRETKEYBYTES];
  unsigned seed, scenario, n, r;
  int corrupt_cache = argc == 2 && strcmp(argv[1], "--corrupt-cache") == 0;
  CHECK(argc == 1 || corrupt_cache);
  CHECK(mlk_sme2_check_svl() == 0);
  for (seed = 0; seed < sizeof(seeds) / sizeof(seeds[0]); seed++)
  {
    fill(key_coins, sizeof(key_coins), seeds[seed]);
    CHECK(mlk_kem_keypair_derand(pk, sk, key_coins, NULL) == 0);
    CHECK(prepare_ntt(&nk, sk) == 0);
    CHECK(prepare_ntt_public(&public_key, sk) == 0);
    CHECK(memcmp(&public_key.base, &nk, sizeof(nk)) == 0);
    CHECK(mlk_sme2_decaps_prepare(&sme2_key, sk, NULL) == 0);
    if (corrupt_cache)
      public_key.at_cache[0].vec[0].coeffs[0] ^= 1;
    public_key_copy = public_key;
    for (r = 0; r < MAX_N; r++)
    {
      fill(enc_coins, sizeof(enc_coins),
           seeds[seed] ^ (UINT32_C(0x9e3779b9) * (r + 1u)));
      CHECK(mlk_kem_enc_derand(valid_ct[r], enc_ss[r], pk,
                              enc_coins, NULL) == 0);
    }
    for (scenario = 0; scenario < 3; scenario++)
    {
      CHECK(prepare_scenario(scenario) == 0);
      for (n = 16; n <= MAX_N; n += 16)
      {
        CHECK(compare_arithmetic(n) == 0);
        CHECK(compare_decaps(n) == 0);
        cases_checked++;
      }
      printf("PASS seed=%u scenario=%s N=16..256 step=16: arithmetic, "
             "Montgomery swap, shared secrets, live wipes, canaries\n",
             seed, scenario == 0 ? "valid" :
                   scenario == 1 ? "tampered" : "mixed");
    }
    memcpy(bad_sk, sk, sizeof(bad_sk));
    bad_sk[MLKEM_INDCCA_SECRETKEYBYTES - 2u * MLKEM_SYMBYTES] ^= 1;
    memset(&public_key, SENTINEL, sizeof(public_key));
    CHECK(prepare_ntt_public(&public_key, bad_sk) != 0);
    CHECK(bytes_are(&public_key, sizeof(public_key), 0));
    CHECK(prepare_ntt_public(&public_key, sk) == 0);
    mlk_zeroize(&public_key, sizeof(public_key));
    CHECK(bytes_are(&public_key, sizeof(public_key), 0));
    mlk_zeroize(&public_key_copy, sizeof(public_key_copy));
    mlk_zeroize(&nk, sizeof(nk));
    mlk_sme2_decaps_key_clear(&sme2_key);
    CHECK(bytes_are(&sme2_key, sizeof(sme2_key), 0));
  }
  mlk_zeroize(sk, sizeof(sk));
  mlk_zeroize(bad_sk, sizeof(bad_sk));
  mlk_zeroize(key_coins, sizeof(key_coins));
  mlk_zeroize(enc_coins, sizeof(enc_coins));
  printf("PASS cases=%llu arithmetic_coefficients=%llu "
         "ntt_swap_coefficients=%llu decaps_requests=%llu "
         "key_prepare_error_wipes=3\n",
         (unsigned long long)cases_checked,
         (unsigned long long)arithmetic_coefficients,
         (unsigned long long)ntt_coefficients,
         (unsigned long long)decaps_requests);
  return 0;
}
