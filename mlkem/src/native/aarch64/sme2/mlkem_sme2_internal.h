/* Internal interfaces for the ML-KEM-768 SME2 translation units. */
#ifndef MLK_NATIVE_AARCH64_SME2_INTERNAL_H
#define MLK_NATIVE_AARCH64_SME2_INTERNAL_H

#include "mlkem_sme2_backend.h"
#include "mlkem_sme2.h"

#if defined(__GNUC__) || defined(__clang__)

#define MLK_SME2_NOINLINE __attribute__((noinline))
#else

#define MLK_SME2_NOINLINE
#endif

#define SME2_MAX_BATCH MLK_SME2_MAX_BATCH

/* Exact arithmetic constants/macros from sme2-exp. */
#define SME2_Q 3329
#define SME2_GROUP 16u

#define SME2_MAX_GROUPS (SME2_MAX_BATCH / SME2_GROUP)

/* These symbols are parameter-set specific, including in multilevel builds. */
#define sme2_residual_mod_q MLK_NAMESPACE_K(sme2_residual_mod_q)
#define sme2_partial_ntt_scalar MLK_NAMESPACE_K(sme2_partial_ntt_scalar)
#define sme2_inverse_reconstruct MLK_NAMESPACE_K(sme2_inverse_reconstruct)
#define sme2_partial_ntt3_neon MLK_NAMESPACE_K(sme2_partial_ntt3_neon)
#define sme2_inv2_roots_centered MLK_NAMESPACE_K(sme2_inv2_roots_centered)

MLK_INTERNAL_DATA_DECLARATION const int16_t sme2_inv2_roots_centered[128];

MLK_INTERNAL_API int sme2_residual_mod_q(int64_t x);
MLK_INTERNAL_API void sme2_partial_ntt_scalar(int16_t a[256],unsigned depth);
MLK_INTERNAL_API MLK_SME2_NOINLINE void sme2_inverse_reconstruct(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    unsigned n
);

/* 3段NTT */
MLK_INTERNAL_API void sme2_partial_ntt3_neon(int16_t a[256]);

static inline mlk_poly *sme2_output_poly(
    mlk_sme2_workspace *ws,
    unsigned r,
    unsigned d)
{
  if (d < MLKEM_K)
    return &ws->arena.out[r].b.vec[d];
  return &ws->arena.out[r].v;
}

extern void MLK_NAMESPACE(sme2_cross_matrix_d3_body)(
    const int16_t *public_gemm,
    const int16_t *secret_gemm,
    int32_t *raw_out,
    unsigned groups)
    __arm_streaming __arm_inout("za");

/* SVL=512: inverse_h[b][d][t][r] -> output_ptrs[d*16+r][32*b+t].
 * ZAを用いた転置（出力時） */
extern void MLK_NAMESPACE(sme2_transpose_output_d3_body)(
    const int16_t *inverse_h,
    int16_t *const output_ptrs[MLK_SME2_DOTS * MLK_SME2_GROUP])
    __arm_streaming __arm_inout("za");

#endif /* MLK_NATIVE_AARCH64_SME2_INTERNAL_H */
