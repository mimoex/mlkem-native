/* Ordinary batch preparation, SME2 dispatch, and copying APIs. */

#include "../../../common.h"

#if defined(MLK_ARITH_BACKEND_AARCH64) && \
    defined(MLK_SYS_AARCH64_SME2) && MLKEM_K == 3

#include "mlkem_sme2_internal.h"

#include <string.h>

static MLK_SME2_NOINLINE void sme2_pack_secret(
    mlk_sme2_workspace *ws,
    unsigned n
);
static MLK_SME2_NOINLINE int sme2_residual_gemm(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    unsigned n
);
static MLK_SME2_NOINLINE void sme2_prepare_secret_batch(
    mlk_sme2_workspace *ws,
    unsigned n,
    const int16_t secret_coeff[][MLKEM_K][MLK_SME2_N]
);
static MLK_SME2_NOINLINE int sme2_arith_batch_internal(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    unsigned n,
    const int16_t secret_coeff[][MLKEM_K][MLK_SME2_N]
);

/* ---- sme2_pack_secret ---- */

/*
 * sec_partial and secret_gemm are disjoint live regions at this point.
 * Expose them as independent restrict-qualified flat pointers so the
 * compiler need not preserve hypothetical source/destination aliasing.
 */
static MLK_SME2_NOINLINE void sme2_pack_secret_restrict(
    const int16_t * restrict src,
    int16_t * restrict dst,
    unsigned n
)
{
  unsigned g,b,k,p,c;

  for (g=0u;g<n/SME2_GROUP;g++)
    for (b=0u;b<8u;b++)
      for (k=0u;k<MLKEM_K;k++)
        for (p=0u;p<16u;p++)
          for (c=0u;c<16u;c++) {
            const unsigned r=16u*g+c;
            const unsigned base=32u*b+2u*p;
            const size_t sbase=
                ((size_t)r*MLKEM_K+k)*MLK_SME2_N+base;
            const size_t dbase=
                ((((size_t)g*8u+b)*MLKEM_K+k)*16u+p)*32u+2u*c;

            dst[dbase+0u]=src[sbase+0u];
            dst[dbase+1u]=src[sbase+1u];
          }
}

static MLK_SME2_NOINLINE void sme2_pack_secret(
    mlk_sme2_workspace *ws,
    unsigned n
)
{
  const int16_t *src=
      &ws->arena.sec_partial[0].vec[0].coeffs[0];
  int16_t *dst=
      &ws->secondary.secret_gemm[0][0][0][0][0];

  sme2_pack_secret_restrict(src,dst,n);
}

/* ---- sme2_residual_gemm ---- */
static MLK_SME2_NOINLINE int sme2_residual_gemm(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    unsigned n
)
{
  return mlk_sme2_cross_matrix_d3(
      &ctx->public_gemm[0][0][0][0][0][0],
      &ws->secondary.secret_gemm[0][0][0][0][0],
      &ws->arena.raw[0][0][0][0][0],
      n/SME2_GROUP);
}

/* ---- sme2_prepare_secret_batch ---- */
static MLK_SME2_NOINLINE void sme2_prepare_secret_batch_inplace(
    mlk_sme2_workspace *ws,
    unsigned n)
{
  unsigned r,k;

  for (r=0u;r<n;r++) {
    for (k=0u;k<MLKEM_K;k++)
      sme2_partial_ntt3_neon(
          ws->arena.sec_partial[r].vec[k].coeffs);
  }

  sme2_pack_secret(ws,n);
}

static MLK_SME2_NOINLINE void sme2_prepare_secret_batch(
    mlk_sme2_workspace *ws,
    unsigned n,
    const int16_t secret_coeff[][MLKEM_K][MLK_SME2_N]
)
{
  unsigned r,k;

  for (r=0u;r<n;r++) {
    for (k=0u;k<MLKEM_K;k++)
      memcpy(
          ws->arena.sec_partial[r].vec[k].coeffs,
          secret_coeff[r][k],
          sizeof(ws->arena.sec_partial[r].vec[k].coeffs));
  }

  sme2_prepare_secret_batch_inplace(ws,n);
}

/* ---- sme2_arith_batch_internal ---- */
static MLK_SME2_NOINLINE int sme2_arith_batch_internal(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    unsigned n,
    const int16_t secret_coeff[][MLKEM_K][MLK_SME2_N]
)
{
  int ret;
  sme2_prepare_secret_batch(ws,n,secret_coeff);
  ret = sme2_residual_gemm(ctx,ws,n);
  if (ret != 0)
    return ret;
  sme2_inverse_reconstruct(ctx,ws,n);
  return 0;
}

int mlk_sme2_arith_batch_workspace_preloaded(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    unsigned n)
{
  int ret;
  if (ctx == NULL || ws == NULL)
    return -1;
  if (n < MLK_SME2_GROUP ||
      n > MLK_SME2_MAX_BATCH ||
      (n % MLK_SME2_GROUP) != 0u)
    return -2;

  if (mlk_sme2_check_svl() != 0)
    return MLK_SME2_ERR_SVL;

  sme2_prepare_secret_batch_inplace(ws,n);
  ret = sme2_residual_gemm(ctx,ws,n);
  if (ret != 0)
    return ret;
  sme2_inverse_reconstruct(ctx,ws,n);
  return 0;
}

int mlk_sme2_arith_batch_workspace(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    const int16_t secret_coeff[][MLK_SME2_K][MLK_SME2_N],
    unsigned n)
{
  if (ctx == NULL || ws == NULL || secret_coeff == NULL)
    return -1;
  if (n < MLK_SME2_GROUP ||
      n > MLK_SME2_MAX_BATCH ||
      (n % MLK_SME2_GROUP) != 0u)
    return -2;

  if (mlk_sme2_check_svl() != 0)
    return MLK_SME2_ERR_SVL;

  return sme2_arith_batch_internal(ctx,ws,n,secret_coeff);
}

int mlk_sme2_arith_batch(
    const mlk_sme2_public_ctx *ctx,
    mlk_sme2_workspace *ws,
    const int16_t secret_coeff[][MLK_SME2_K][MLK_SME2_N],
    int16_t out[][MLK_SME2_DOTS][MLK_SME2_N],
    unsigned n)
{
  unsigned off = 0u;

  if (ctx == NULL || ws == NULL || secret_coeff == NULL || out == NULL)
    return -1;
  if (n < MLK_SME2_GROUP ||
      n > MLK_SME2_MAX_BATCH ||
      (n % MLK_SME2_GROUP) != 0u)
    return -2;

  while (off < n)
  {
    unsigned m;
    unsigned r, d;
    int ret;

    /* The full workspace covers the requested batch in one tile. */
    m = n - off;

    ret = mlk_sme2_arith_batch_workspace(
        ctx, ws, secret_coeff + off, m);
    if (ret != 0)
      return ret;

    for (r = 0u; r < m; r++)
      for (d = 0u; d < MLK_SME2_DOTS; d++)
        mlk_memcpy(
            out[off + r][d],
            sme2_output_poly(ws, r, d)->coeffs,
            sizeof(sme2_output_poly(ws, r, d)->coeffs));

    off += m;
  }

  return 0;
}

#else

MLK_EMPTY_CU(aarch64_sme2_batch)

#endif /* MLK_ARITH_BACKEND_AARCH64 && MLK_SYS_AARCH64_SME2 && MLKEM_K == 3 */
