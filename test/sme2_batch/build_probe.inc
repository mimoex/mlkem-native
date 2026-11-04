/* Included once for each configured level, before its monolithic source. */
#include <stdio.h>
#include <string.h>
#include "mlkem/src/common.h"
#include "mlkem/src/kem.h"
#include "mlkem/src/verify.h"
#include "mlkem/src/native/aarch64/sme2/mlkem_sme2_decaps.h"

int MLK_NAMESPACE_K(phase3_build_smoke)(void)
{
  uint8_t pk[MLKEM_INDCCA_PUBLICKEYBYTES], sk[MLKEM_INDCCA_SECRETKEYBYTES];
  uint8_t ct[MLKEM_INDCCA_CIPHERTEXTBYTES], expected[MLKEM_SSBYTES];
  uint8_t coins[2 * MLKEM_SYMBYTES];
  unsigned i;
  for (i = 0; i < sizeof(coins); ++i)
    coins[i] = (uint8_t)(i + MLKEM_K);
  if (mlk_kem_keypair_derand(pk, sk, coins, NULL) != 0 ||
      mlk_kem_enc_derand(ct, expected, pk, coins, NULL) != 0)
    return 1;
#if !defined(MLK_CONFIG_NO_DECAPS_API)
  {
    uint8_t actual[MLKEM_SSBYTES];
    if (mlk_kem_dec(actual, ct, sk, NULL) != 0 ||
        memcmp(actual, expected, sizeof(actual)) != 0)
      return 2;
  }
#if defined(MLK_SYS_AARCH64_SME2) && \
    defined(MLK_CONFIG_USE_NATIVE_BACKEND_ARITH) && MLKEM_K == 3
  {
    static mlk_sme2_decaps_key key;
    static mlk_sme2_decaps_workspace ws;
    uint8_t batch[16][MLKEM_INDCCA_CIPHERTEXTBYTES];
    uint8_t shared[16][MLKEM_SSBYTES], oracle[16][MLKEM_SSBYTES];
    if (mlk_sme2_decaps_prepare(&key, sk, NULL) != 0)
      return 3;
    for (i = 0; i < 16; ++i)
    {
      memcpy(batch[i], ct, sizeof(ct));
      if ((i & 1u) != 0u)
        batch[i][i] ^= 0x80;
      if (mlk_kem_dec(oracle[i], batch[i], sk, NULL) != 0)
        return 4;
    }
    if (mlk_sme2_decaps_batch(&key, &ws, shared, batch, 16, NULL) != 0 ||
        memcmp(shared, oracle, sizeof(shared)) != 0)
      return 5;
    mlk_sme2_decaps_key_clear(&key);
    mlk_sme2_decaps_workspace_clear(&ws);
    for (i = 0; i < sizeof(key); ++i)
      if (((const uint8_t *)&key)[i] != 0)
        return 6;
    for (i = 0; i < sizeof(ws); ++i)
      if (((const uint8_t *)&ws)[i] != 0)
        return 7;
  }
#endif
#endif
  printf("PASS build level=%d normal-KEM", MLK_CONFIG_PARAMETER_SET);
#if defined(MLK_SYS_AARCH64_SME2) && \
    defined(MLK_CONFIG_USE_NATIVE_BACKEND_ARITH) && MLKEM_K == 3 && \
    !defined(MLK_CONFIG_NO_DECAPS_API)
  printf(" SME2-mixed-batch=16");
#endif
  puts("");
  mlk_zeroize(sk, sizeof(sk));
  mlk_zeroize(coins, sizeof(coins));
  return 0;
}
