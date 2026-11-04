/*
 * Arm SME2 residual-domain arithmetic backend for ML-KEM-768.
 *
 * The public operand is prepared once into a reusable context. Per-batch
 * secret operands are transformed through a depth-3 partial NTT, packed for
 * the SME2 matrix kernel, and reconstructed into coefficient-domain outputs.
 *
 * Mutable state is caller-owned through mlk_sme2_public_ctx and
 * mlk_sme2_workspace.
 */

#include "../../../common.h"

#if defined(MLK_ARITH_BACKEND_AARCH64) && \
    defined(MLK_SYS_AARCH64_SME2) && MLKEM_K == 3

#include "mlkem_sme2_internal.h"

#include <arm_sme.h>

typedef char mlk_sme2_static_assert_caller_owned_public_context_layout[(sizeof(mlk_sme2_public_ctx) == 196672u) ? 1 : -1];
typedef char mlk_sme2_static_assert_caller_owned_batch_workspace_layout[(sizeof(mlk_sme2_workspace) == 1572864u) ? 1 : -1];


// CPUがSME2をサポートしているかどうかを確認する。サポートされていない場合はMLK_SME2_ERR_SVL(-3)を返す。
int mlk_sme2_check_svl(void)
{
  /* RDSVL queries streaming VL even when called in non-streaming mode. */
  return svcntsb() == MLK_SME2_SVL_BYTES ? 0 : MLK_SME2_ERR_SVL;
}

// アセンブリで実装されたSME2行列積和の本体
__arm_new("za")
__arm_locally_streaming
int mlk_sme2_cross_matrix_d3(
    const int16_t *public_gemm,
    const int16_t *secret_gemm,
    int32_t *raw_out,
    unsigned groups)
{
  if (svcntsb() != MLK_SME2_SVL_BYTES)
    return MLK_SME2_ERR_SVL;
  MLK_NAMESPACE(sme2_cross_matrix_d3_body)(
      public_gemm, secret_gemm, raw_out, groups);
  return 0;
}


#else

MLK_EMPTY_CU(aarch64_sme2_backend)

#endif /* MLK_ARITH_BACKEND_AARCH64 && MLK_SYS_AARCH64_SME2 && MLKEM_K == 3 */
