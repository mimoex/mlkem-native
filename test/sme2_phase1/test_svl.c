/* VL rejection and valid/invalid ciphertext differential tests; no timing. */
#include <arm_sme.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "mlkem/src/kem.h"
#include "mlkem/src/native/aarch64/sme2/mlkem_sme2.h"
#include "mlkem/src/native/aarch64/sme2/mlkem_sme2_decaps.h"

#ifdef TEST_SVL_OVERRIDE
uint64_t phase1_svl_bytes;
#endif

static mlk_sme2_public_ctx public_ctx;
static mlk_sme2_workspace full_ws;
static mlk_sme2_decaps_key key;
static mlk_sme2_decaps_workspace ws;
static int16_t output[16][4][256];
static uint8_t ss[256][MLKEM_SSBYTES];
static uint8_t ct[256][MLKEM_INDCCA_CIPHERTEXTBYTES];
static uint8_t expected[256][MLKEM_SSBYTES];

#define CHECK(c) do { if (!(c)) { \
  fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
  return 1; } } while (0)

static int bytes_equal(const void *p, size_t n, unsigned char value)
{
  const unsigned char *b = p;
  size_t i;
  for (i = 0; i < n; i++)
    if (b[i] != value)
      return 0;
  return 1;
}

static int check_rejection(const void *inaccessible)
{
  const mlk_sme2_public_ctx *bad_ctx = inaccessible;
  const int16_t (*bad_poly)[3][256] = inaccessible;
  const uint8_t (*bad_ct)[MLKEM_INDCCA_CIPHERTEXTBYTES] = inaccessible;

  memset(&public_ctx, 0xa5, sizeof(public_ctx));
  memset(&full_ws, 0xa5, sizeof(full_ws));
  memset(&key, 0xa5, sizeof(key));
  memset(&ws, 0xa5, sizeof(ws));
  memset(output, 0xa5, sizeof(output));
  memset(ss, 0xa5, sizeof(ss));

  CHECK(mlk_sme2_check_svl() == MLK_SME2_ERR_SVL);
  CHECK(mlk_sme2_prepare_public(&public_ctx, bad_poly) == MLK_SME2_ERR_SVL);
  CHECK(mlk_sme2_arith_batch_rolling_preloaded(bad_ctx, &ws.arithmetic, 16)
        == MLK_SME2_ERR_SVL);
  CHECK(mlk_sme2_arith_batch_workspace_preloaded(bad_ctx, &full_ws, 16)
        == MLK_SME2_ERR_SVL);
  CHECK(mlk_sme2_arith_batch_workspace(bad_ctx, &full_ws, bad_poly, 16)
        == MLK_SME2_ERR_SVL);
  CHECK(mlk_sme2_arith_batch(bad_ctx, &full_ws, bad_poly, output, 16)
        == MLK_SME2_ERR_SVL);
  CHECK(mlk_sme2_cross_matrix_d3(inaccessible, inaccessible,
                               (int32_t *)output, 1) == MLK_SME2_ERR_SVL);
  CHECK(mlk_sme2_decaps_prepare(&key, inaccessible, NULL)
        == MLK_SME2_DECAPS_ERR_SVL);
  CHECK(bytes_equal(&key, sizeof(key), 0));
  CHECK(mlk_sme2_decaps_batch(inaccessible, &ws, ss, bad_ct, 16, NULL)
        == MLK_SME2_DECAPS_ERR_SVL);

  CHECK(bytes_equal(&public_ctx, sizeof(public_ctx), 0xa5));
  CHECK(bytes_equal(&full_ws, sizeof(full_ws), 0xa5));
  CHECK(bytes_equal(&ws, sizeof(ws), 0xa5));
  CHECK(bytes_equal(output, sizeof(output), 0xa5));
  CHECK(bytes_equal(ss, sizeof(ss), 0xa5));
  return 0;
}

static int check_supported(void)
{
  uint8_t pk[MLKEM_INDCCA_PUBLICKEYBYTES];
  uint8_t sk[MLKEM_INDCCA_SECRETKEYBYTES];
  uint8_t key_coins[2 * MLKEM_SYMBYTES];
  uint8_t coins[MLKEM_SYMBYTES];
  unsigned n, r, i;

  CHECK(mlk_sme2_check_svl() == 0);
  for (i = 0; i < sizeof(key_coins); i++)
    key_coins[i] = (uint8_t)(3u * i + 1u);
  CHECK(mlk_kem_keypair_derand(pk, sk, key_coins, NULL) == 0);
  CHECK(mlk_sme2_decaps_prepare(&key, sk, NULL) == 0);
  for (r = 0; r < 256; r++)
  {
    for (i = 0; i < sizeof(coins); i++)
      coins[i] = (uint8_t)(r + 7u * i);
    CHECK(mlk_kem_enc_derand(ct[r], expected[r], pk, coins, NULL) == 0);
    if ((r & 1u) != 0u)
      ct[r][r % MLKEM_INDCCA_CIPHERTEXTBYTES] ^= 1;
    CHECK(mlk_kem_dec(expected[r], ct[r], sk, NULL) == 0);
  }
  for (n = 16; n <= 256; n += 16)
  {
    memset(&ws, 0, sizeof(ws));
    CHECK(mlk_sme2_decaps_batch(&key, &ws, ss, ct, n, NULL) == 0);
    CHECK(memcmp(ss, expected, (size_t)n * MLKEM_SSBYTES) == 0);
    CHECK(bytes_equal(&ws, sizeof(ws), 0));
  }
  mlk_sme2_decaps_key_clear(&key);
  CHECK(bytes_equal(&key, sizeof(key), 0));
  puts("PASS SVL=512: mixed valid/invalid Decaps, N=16..256; cleanup");
  return 0;
}

int main(void)
{
  const uint64_t actual = svcntsb();
  const long page_size = sysconf(_SC_PAGESIZE);
  void *inaccessible;
  CHECK(page_size > 0);
  inaccessible = mmap(NULL, (size_t)page_size, PROT_NONE,
                      MAP_PRIVATE | MAP_ANON, -1, 0);
  CHECK(inaccessible != MAP_FAILED);
  printf("actual streaming VL: %llu bits\n", (unsigned long long)(8u * actual));
#ifdef TEST_SVL_OVERRIDE
  /* Cover every architectural VL in 128..2048 bits, except supported 512. */
  for (phase1_svl_bytes = 16; phase1_svl_bytes <= 256; phase1_svl_bytes += 16)
  {
    if (phase1_svl_bytes == MLK_SME2_SVL_BYTES)
      continue;
    CHECK(check_rejection(inaccessible) == 0);
    printf("PASS injected SVL=%llu: rejected before operand access; canaries intact\n",
           (unsigned long long)(8u * phase1_svl_bytes));
  }
  phase1_svl_bytes = actual;
#else
  if (actual != MLK_SME2_SVL_BYTES)
    CHECK(check_rejection(inaccessible) == 0);
#endif
  CHECK(munmap(inaccessible, (size_t)page_size) == 0);
  if (actual != MLK_SME2_SVL_BYTES)
  {
    puts("Unsupported host SVL; rejection passed, supported-path tests unavailable");
    return 77;
  }
  return check_supported();
}
