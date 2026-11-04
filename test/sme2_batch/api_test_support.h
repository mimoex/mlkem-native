/* Shared fixtures for Phase 3 API behavior tests; no benchmark operations. */
#ifndef PHASE3_API_TEST_SUPPORT_H
#define PHASE3_API_TEST_SUPPORT_H

#include <stdio.h>
#include <string.h>
#include "mlkem/src/kem.h"
#include "mlkem/src/symmetric.h"
#include "mlkem/src/native/aarch64/sme2/mlkem_sme2_decaps.h"

#define CHECK(c) do { if (!(c)) { \
  fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
  return 1; } } while (0)

typedef struct {
  uint8_t before[64];
  mlk_sme2_decaps_key value;
  uint8_t after[64];
} phase3_guarded_key;

typedef struct {
  uint8_t before[64];
  mlk_sme2_decaps_workspace value;
  uint8_t after[64];
} phase3_guarded_workspace;

typedef struct {
  uint8_t before[64];
  uint8_t value[MLK_SME2_MAX_BATCH][MLKEM_SSBYTES];
  uint8_t after[64];
} phase3_guarded_output;

static phase3_guarded_key key;
static phase3_guarded_workspace workspace;
static phase3_guarded_output output;
static mlk_sme2_decaps_workspace expected_workspace;
static mlk_sme2_decaps_key expected_key;
static uint8_t pk[MLKEM_INDCCA_PUBLICKEYBYTES];
static uint8_t sk[MLKEM_INDCCA_SECRETKEYBYTES];
static uint8_t valid_ct[MLK_SME2_MAX_BATCH][MLKEM_INDCCA_CIPHERTEXTBYTES];
static uint8_t ct[MLK_SME2_MAX_BATCH][MLKEM_INDCCA_CIPHERTEXTBYTES];
static uint8_t expected_ss[MLK_SME2_MAX_BATCH][MLKEM_SSBYTES];
static uint8_t valid_ss[MLK_SME2_MAX_BATCH][MLKEM_SSBYTES];

static inline int all_bytes(const void *p, size_t n, uint8_t value)
{
  const uint8_t *b = p;
  size_t i;
  for (i = 0; i < n; i++)
    if (b[i] != value)
      return 0;
  return 1;
}

static inline int guards_intact(void)
{
  CHECK(all_bytes(key.before, sizeof(key.before), 0xa5));
  CHECK(all_bytes(key.after, sizeof(key.after), 0xa5));
  CHECK(all_bytes(workspace.before, sizeof(workspace.before), 0xa5));
  CHECK(all_bytes(workspace.after, sizeof(workspace.after), 0xa5));
  CHECK(all_bytes(output.before, sizeof(output.before), 0xa5));
  CHECK(all_bytes(output.after, sizeof(output.after), 0xa5));
  return 0;
}

static inline void reset_guards(void)
{
  memset(&key, 0xa5, sizeof(key));
  memset(&workspace, 0xa5, sizeof(workspace));
  memset(&output, 0xa5, sizeof(output));
}

/* Independently describe every documented live region. Preserve other bytes
 * from the pre-call snapshot, so an unintended tail write also fails. */
static inline void expected_cleanup(unsigned n)
{
  expected_workspace = workspace.value;
  if (n == MLK_SME2_MAX_BATCH)
  {
    memset(&expected_workspace, 0, sizeof(expected_workspace));
    return;
  }
  memset(expected_workspace.arithmetic.arena.group, 0,
         (n / MLK_SME2_GROUP) *
             sizeof(expected_workspace.arithmetic.arena.group[0]));
  memset(expected_workspace.arithmetic.raw, 0,
         sizeof(expected_workspace.arithmetic.raw));
  memset(&expected_workspace.arithmetic.secondary, 0,
         sizeof(expected_workspace.arithmetic.secondary));
  memset(expected_workspace.message, 0,
         n * sizeof(expected_workspace.message[0]));
  memset(expected_workspace.kr, 0, n * sizeof(expected_workspace.kr[0]));
}

static inline int cleanup_matches(unsigned n, int success)
{
  CHECK(memcmp(&workspace.value, &expected_workspace,
               sizeof(expected_workspace)) == 0);
  CHECK(memcmp(&key.value, &expected_key, sizeof(expected_key)) == 0);
  if (success)
    CHECK(memcmp(output.value, expected_ss, n * MLKEM_SSBYTES) == 0);
  else
    CHECK(all_bytes(output.value, n * MLKEM_SSBYTES, 0));
  CHECK(all_bytes(&output.value[n],
                  (MLK_SME2_MAX_BATCH - n) * MLKEM_SSBYTES, 0xa5));
  return guards_intact();
}

static inline int make_fixture(unsigned seed)
{
  uint8_t key_coins[2 * MLKEM_SYMBYTES];
  uint8_t coins[MLKEM_SYMBYTES];
  unsigned i, r;
  for (i = 0; i < sizeof(key_coins); i++)
    key_coins[i] = (uint8_t)(3u * i + seed);
  CHECK(mlk_kem_keypair_derand(pk, sk, key_coins, NULL) == 0);
  for (r = 0; r < MLK_SME2_MAX_BATCH; r++)
  {
    for (i = 0; i < sizeof(coins); i++)
      coins[i] = (uint8_t)(r + 7u * i + seed);
    CHECK(mlk_kem_enc_derand(valid_ct[r], valid_ss[r], pk, coins, NULL) == 0);
  }
  return 0;
}

/* scenario: 0 = valid, 1 = alternating valid/tampered, 2 = all tampered. */
static inline int select_scenario(unsigned scenario)
{
  uint8_t rejected[MLKEM_SSBYTES];
  uint8_t reject_input[MLKEM_SYMBYTES + MLKEM_INDCCA_CIPHERTEXTBYTES];
  unsigned r;
  memcpy(ct, valid_ct, sizeof(ct));
  for (r = 0; r < MLK_SME2_MAX_BATCH; r++)
  {
    const int invalid = scenario == 2u || (scenario == 1u && (r & 1u));
    if (invalid)
      ct[r][(17u * r) % MLKEM_INDCCA_CIPHERTEXTBYTES] ^= 1;
    CHECK(mlk_kem_dec(expected_ss[r], ct[r], sk, NULL) == 0);
    if (invalid)
    {
      memcpy(reject_input, sk + sizeof(sk) - MLKEM_SYMBYTES,
             MLKEM_SYMBYTES);
      memcpy(reject_input + MLKEM_SYMBYTES, ct[r], sizeof(ct[r]));
      mlk_hash_j(rejected, reject_input, sizeof(reject_input));
      CHECK(memcmp(expected_ss[r], rejected, sizeof(rejected)) == 0);
    }
    else
      CHECK(memcmp(expected_ss[r], valid_ss[r], MLKEM_SSBYTES) == 0);
  }
  return 0;
}

#endif
