/* Verify the separately linked research baseline, including implicit rejection. */
#include <stdio.h>
#include <string.h>
#include "prepared_ntt.h"
#include "kem.h"
#include "verify.h"

#define CHECK(c) do { if (!(c)) { \
  fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
  return 1; } } while (0)

static ntt_key key, saved_key;
static ntt_fair_workspace ws, expected_ws;
static uint8_t pk[MLKEM_INDCCA_PUBLICKEYBYTES];
static uint8_t sk[MLKEM_INDCCA_SECRETKEYBYTES];
static uint8_t ct[MAX_N][MLKEM_INDCCA_CIPHERTEXTBYTES];
static uint8_t expected[MAX_N][MLKEM_SSBYTES];
static uint8_t got[MAX_N + 1][MLKEM_SSBYTES];

static int bytes_are(const void *p, size_t n, uint8_t value)
{
  const uint8_t *b = p;
  size_t i;
  for (i = 0; i < n; i++)
    if (b[i] != value) return 0;
  return 1;
}

int main(void)
{
  uint8_t coins[2 * MLKEM_SYMBYTES];
  uint8_t encapsulated[MLKEM_SSBYTES];
  unsigned seed, scenario, r, i, n;
  for (seed = 1; seed <= 2; seed++)
  {
    for (i = 0; i < sizeof(coins); i++) coins[i] = (uint8_t)(seed + 3u * i);
    CHECK(mlk_kem_keypair_derand(pk, sk, coins, NULL) == 0);
    sk[sizeof(sk) - 2u * MLKEM_SYMBYTES] ^= 1;
    memset(&key, 0xa5, sizeof(key));
    CHECK(prepare_ntt(&key, sk) == mlk_kem_check_sk(sk, NULL));
    CHECK(bytes_are(&key, sizeof(key), 0));
    sk[sizeof(sk) - 2u * MLKEM_SYMBYTES] ^= 1;
    CHECK(prepare_ntt(&key, sk) == 0);
    saved_key = key;
    for (scenario = 0; scenario < 3; scenario++)
    {
      for (r = 0; r < MAX_N; r++)
      {
        for (i = 0; i < MLKEM_SYMBYTES; i++) coins[i] = (uint8_t)(r + 7u * i + seed);
        CHECK(mlk_kem_enc_derand(ct[r], encapsulated, pk, coins, NULL) == 0);
        if (scenario == 2 || (scenario == 1 && (r & 1u)))
          ct[r][(17u * r) % sizeof(ct[r])] ^= 1;
        CHECK(mlk_kem_dec(expected[r], ct[r], sk, NULL) == 0);
        if (scenario == 0) CHECK(memcmp(expected[r], encapsulated, sizeof(encapsulated)) == 0);
      }
      memset(&ws, 0xa5, sizeof(ws));
      for (n = 16; n <= MAX_N; n += 16)
      {
        memset(got, 0xa5, sizeof(got));
        expected_ws = ws;
        memset(expected_ws.sb.sp, 0, n * sizeof(expected_ws.sb.sp[0]));
        memset(expected_ws.v, 0, n * sizeof(expected_ws.v[0]));
        memset(expected_ws.message, 0, n * sizeof(expected_ws.message[0]));
        memset(expected_ws.kr, 0, n * sizeof(expected_ws.kr[0]));
        memset(&expected_ws.sp_ntt, 0, sizeof(expected_ws.sp_ntt));
        memset(&expected_ws.cache, 0, sizeof(expected_ws.cache));
        CHECK(ntt_fair_batch(&key, &ws, got, ct, n) == 0);
        CHECK(memcmp(got, expected, n * MLKEM_SSBYTES) == 0);
        CHECK(bytes_are(got[n], (MAX_N + 1u - n) * MLKEM_SSBYTES, 0xa5));
        CHECK(memcmp(&ws, &expected_ws, sizeof(ws)) == 0);
        CHECK(memcmp(&key, &saved_key, sizeof(key)) == 0);
      }
    }
  }
  mlk_zeroize(&key, sizeof(key));
  mlk_zeroize(&saved_key, sizeof(saved_key));
  mlk_zeroize(&ws, sizeof(ws));
  mlk_zeroize(&expected_ws, sizeof(expected_ws));
  mlk_zeroize(sk, sizeof(sk));
  puts("PASS prepared NTT: 96 batches / 13056 secrets; valid/mixed/invalid ciphertexts; shared scratch wipe; untouched tails/key; invalid key rejection");
  return 0;
}
