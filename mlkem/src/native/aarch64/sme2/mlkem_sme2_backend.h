#ifndef MLK_NATIVE_AARCH64_SME2_BACKEND_H
#define MLK_NATIVE_AARCH64_SME2_BACKEND_H

#include "../../../common.h"
#include <stdint.h>
#include "../../../poly_k.h"

#define MLK_SME2_N 256u
#define MLK_SME2_K 3u
#define MLK_SME2_DOTS 4u
#define MLK_SME2_MAX_BATCH 256u
#define MLK_SME2_GROUP 16u
#define MLK_SME2_SVL_BYTES 64u
#define MLK_SME2_ERR_SVL (-3)
#define MLK_SME2_MAX_GROUPS \
    (MLK_SME2_MAX_BATCH / MLK_SME2_GROUP)

#if defined(__GNUC__) || defined(__clang__)
#define MLK_SME2_ALIGN64 __attribute__((aligned(64)))
#else
#define MLK_SME2_ALIGN64
#endif

typedef struct {
  int16_t coeffs[MLK_SME2_N];
} mlk_sme2_poly;

typedef struct {
  mlk_sme2_poly vec[MLK_SME2_K];
} mlk_sme2_polyvec;

typedef struct MLK_SME2_ALIGN64 {
  int16_t public_gemm[2][8][MLK_SME2_K][16][4][32];
  int16_t inverse_c[8];
  int16_t inverse_tw[8];
} mlk_sme2_public_ctx;

typedef struct {
  mlk_polyvec b;
  mlk_poly v;
} mlk_sme2_output;

typedef union {
  mlk_polyvec sec_partial[MLK_SME2_MAX_BATCH];
  int32_t raw[MLK_SME2_MAX_GROUPS][8][4][2][256];
  mlk_sme2_output out[MLK_SME2_MAX_BATCH];
} mlk_sme2_workspace_arena;

typedef union {
  int16_t secret_gemm[MLK_SME2_MAX_GROUPS]
                     [8][MLK_SME2_K][16][32];
  int16_t inverse_h[MLK_SME2_MAX_GROUPS][8][4][32][16];
} mlk_sme2_workspace_secondary_arena;


/*
 * Rolling Decaps arithmetic workspace.
 *
 * Each 16-request group gets its own typed lifetime-overlay slot:
 *   sec[16] (24 KiB) -> out[16] (32 KiB).
 * The union is per group, so different groups never rely on simultaneous
 * access to different members of one monolithic union object.  This keeps
 * the overlay type-safe at group granularity while fixing the primary arena
 * at 16 * 32 KiB = 512 KiB for the external 256-request maximum.
 *
 * raw and secondary are one-group scratch reused inside one SME streaming
 * region across the entire batch.
 */
typedef union {
  mlk_polyvec sec[MLK_SME2_GROUP];
  mlk_sme2_output out[MLK_SME2_GROUP];
} mlk_sme2_rolling_group_slot;

typedef struct {
  mlk_sme2_rolling_group_slot group[MLK_SME2_MAX_GROUPS];
} mlk_sme2_rolling_arena;

typedef union {
  int16_t secret_gemm[1][8][MLK_SME2_K][16][32];
  int16_t inverse_h[1][8][4][32][16];
} mlk_sme2_rolling_secondary_arena;

typedef struct MLK_SME2_ALIGN64 {
  mlk_sme2_rolling_arena arena;
  int32_t raw[1][8][4][2][256];
  mlk_sme2_rolling_secondary_arena secondary;
} mlk_sme2_rolling_workspace;

typedef struct MLK_SME2_ALIGN64 {
  /*
   * These three representations have disjoint lifetimes:
   * sec_partial -> raw -> out.
   *
   * Use exact typed union members rather than a byte arena so that callers
   * and stock polynomial helpers retain their native alignment/type contract.
   */
  mlk_sme2_workspace_arena arena;

  /*
   * secret_gemm is dead after residual_gemm returns. inverse_h is first
   * produced by inverse reconstruction, so these two representations are
   * also lifetime-disjoint.
   */
  mlk_sme2_workspace_secondary_arena secondary;
} mlk_sme2_workspace;

#if MLKEM_K != 3
#error "Arm SME2 residual backend is specialized for ML-KEM-768"
#endif

#define mlk_sme2_prepare_public MLK_NAMESPACE_K(sme2_prepare_public)
#define mlk_sme2_check_svl MLK_NAMESPACE_K(sme2_check_svl)
#define mlk_sme2_arith_batch MLK_NAMESPACE_K(sme2_arith_batch)
#define mlk_sme2_arith_batch_workspace \
  MLK_NAMESPACE_K(sme2_arith_batch_workspace)
#define mlk_sme2_arith_batch_workspace_preloaded \
  MLK_NAMESPACE_K(sme2_arith_batch_workspace_preloaded)
#define mlk_sme2_arith_batch_rolling_preloaded \
  MLK_NAMESPACE_K(sme2_arith_batch_rolling_preloaded)

/* The fixed layouts require a 512-bit streaming vector length. All arithmetic
 * entries reject other lengths with MLK_SME2_ERR_SVL before touching buffers.
 * Check the current thread on every call; do not cache a process-wide result.
 */
int mlk_sme2_check_svl(void);

int mlk_sme2_prepare_public(
    mlk_sme2_public_ctx *ctx,
    const int16_t public_coeff[MLK_SME2_DOTS]
                              [MLK_SME2_K]
                              [MLK_SME2_N]);

int mlk_sme2_arith_batch(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    const int16_t secret_coeff[][MLK_SME2_K][MLK_SME2_N],
    int16_t out[][MLK_SME2_DOTS][MLK_SME2_N],
    unsigned n);

/*
 * Internal no-copy entry for protocol layers that consume ws->arena.out directly.
 * The caller must not modify/reuse ws until all requested ws->arena.out slots have
 * been consumed.
 */
int mlk_sme2_arith_batch_workspace(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    const int16_t secret_coeff[][MLK_SME2_K][MLK_SME2_N],
    unsigned n);

/*
 * Internal entry for protocol layers that have already filled
 * ws->arena.sec_partial[0..n) with coefficient-domain secret polynomials.
 * The function transforms sec_partial in place and produces ws->arena.out.
 */
int mlk_sme2_arith_batch_workspace_preloaded(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    unsigned n);

/* Decaps-only rolling arithmetic entry. */
int mlk_sme2_arith_batch_rolling_preloaded(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_rolling_workspace *ws,
    unsigned n);

#endif
