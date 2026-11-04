#ifndef MLK_NATIVE_AARCH64_SME2_MLKEM_SME2_DECAPS_H
#define MLK_NATIVE_AARCH64_SME2_MLKEM_SME2_DECAPS_H

#include "../../../common.h"
#include "../../../params.h"
#include "../../../kem.h"

#if defined(MLK_SYS_AARCH64_SME2) && \
    defined(MLK_CONFIG_USE_NATIVE_BACKEND_ARITH) && \
    !defined(MLK_CONFIG_NO_DECAPS_API) && \
    MLKEM_K == 3

#include "mlkem_sme2_backend.h"

typedef struct MLK_SME2_ALIGN64 {
  mlk_sme2_public_ctx public_ctx;
  uint8_t dkpke[MLKEM_INDCPA_SECRETKEYBYTES];
  uint8_t hpk[MLKEM_SYMBYTES];
  uint8_t z[MLKEM_SYMBYTES];
} mlk_sme2_decaps_key;

typedef struct MLK_SME2_ALIGN64 {
  mlk_sme2_rolling_workspace arithmetic;
  uint8_t message[MLK_SME2_MAX_BATCH][MLKEM_SYMBYTES];
  uint8_t kr[MLK_SME2_MAX_BATCH][2 * MLKEM_SYMBYTES];
} mlk_sme2_decaps_workspace;

#if defined(MLK_CONFIG_CONTEXT_PARAMETER)
#define MLK_SME2_CONTEXT_PARAMETERS_5(arg0, arg1, arg2, arg3, arg4, context) \
  (arg0, arg1, arg2, arg3, arg4, context)
#else
#define MLK_SME2_CONTEXT_PARAMETERS_5(arg0, arg1, arg2, arg3, arg4, context) \
  (arg0, arg1, arg2, arg3, arg4)
#endif

#define mlk_sme2_decaps_prepare \
  MLK_NAMESPACE_K(sme2_decaps_prepare) MLK_CONTEXT_PARAMETERS_2
#define mlk_sme2_decaps_batch \
  MLK_NAMESPACE_K(sme2_decaps_batch) MLK_SME2_CONTEXT_PARAMETERS_5
#define mlk_sme2_decaps_key_clear MLK_NAMESPACE_K(sme2_decaps_key_clear)
#define mlk_sme2_decaps_workspace_clear \
  MLK_NAMESPACE_K(sme2_decaps_workspace_clear)

#define MLK_SME2_DECAPS_ERR_ARGUMENT (-1000)
#define MLK_SME2_DECAPS_ERR_BATCH    (-1001)
#define MLK_SME2_DECAPS_ERR_BACKEND  (-1002)
#define MLK_SME2_DECAPS_ERR_SVL      (-1003)


MLK_INTERNAL_API
MLK_MUST_CHECK_RETURN_VALUE
int mlk_sme2_decaps_prepare(
    mlk_sme2_decaps_key *key,
    const uint8_t sk[MLKEM_INDCCA_SECRETKEYBYTES],
    MLK_CONFIG_CONTEXT_PARAMETER_TYPE context);

MLK_INTERNAL_API
MLK_MUST_CHECK_RETURN_VALUE
int mlk_sme2_decaps_batch(
    const mlk_sme2_decaps_key *key,
    mlk_sme2_decaps_workspace *ws,
    uint8_t ss[][MLKEM_SSBYTES],
    const uint8_t ct[][MLKEM_INDCCA_CIPHERTEXTBYTES],
    unsigned n,
    MLK_CONFIG_CONTEXT_PARAMETER_TYPE context);

MLK_INTERNAL_API
void mlk_sme2_decaps_key_clear(mlk_sme2_decaps_key *key);

MLK_INTERNAL_API
void mlk_sme2_decaps_workspace_clear(mlk_sme2_decaps_workspace *ws);

#endif

#endif
