#include "../../../common.h"

#if defined(MLK_ARITH_BACKEND_AARCH64) && \
    defined(MLK_SYS_AARCH64_SME2) && MLKEM_K == 3

#include "mlkem_sme2_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int16_t coeffs[256];
} mlk_sme2_internal_poly;

typedef struct {
  mlk_sme2_internal_poly vec[MLKEM_K];
} mlk_sme2_internal_polyvec;

typedef struct {
  mlk_sme2_internal_polyvec vec[MLKEM_K];
} mlk_sme2_internal_polymat;

/* C99-compatible compile-time layout assertions.  The modular
 * project build uses -std=c99 -pedantic -Werror. */
typedef char mlk_sme2_static_assert_mlk_sme2_internal_poly_layout[(sizeof(mlk_sme2_internal_poly) == 512u) ? 1 : -1];
typedef char mlk_sme2_static_assert_mlk_sme2_internal_polyvec_layout[(sizeof(mlk_sme2_internal_polyvec) == 1536u) ? 1 : -1];
typedef char mlk_sme2_static_assert_mlk_sme2_internal_polymat_layout[(sizeof(mlk_sme2_internal_polymat) == 4608u) ? 1 : -1];

typedef struct {
  mlk_sme2_internal_polymat at_coeff;
  mlk_sme2_internal_polyvec t_coeff;
  int16_t gamma[16];
  mlk_sme2_internal_poly pub_partial[4][MLKEM_K];
} sme2_prepare_scratch;

static int sme2_residual_discover_gamma(
    sme2_prepare_scratch *prep,
    unsigned depth
);
static int sme2_prepare_public_partial_ntt(
    sme2_prepare_scratch *prep
);
static int sme2_toeplitz_mod_q(int64_t x);
static int16_t sme2_toeplitz_center_q(int64_t x);
static int16_t sme2_toeplitz_entry(
    const int16_t a[32],
    int gamma,
    unsigned row,
    unsigned col);
static int sme2_prepare_public_toeplitz(
    mlk_sme2_public_ctx *ctx,
    sme2_prepare_scratch *prep
);
static int sme2_inverse_center_q(int x);
static int sme2_twisted15(int c);
static void sme2_init_inverse_constants(
    mlk_sme2_public_ctx *ctx
);
static int sme2_prepare_public_internal(
    mlk_sme2_public_ctx *ctx,
    sme2_prepare_scratch *prep
);

/* ---- sme2_residual_discover_gamma ---- */
static int sme2_residual_discover_gamma(
    sme2_prepare_scratch *prep,
    unsigned depth
)
{
  MLK_SME2_ALIGN64 int16_t basis[256];
  const unsigned blocks=1u<<depth;
  const unsigned m=256u>>depth;
  unsigned b,i;

  memset(basis,0,sizeof(basis));
  basis[m]=1;

  sme2_partial_ntt_scalar(basis,depth);

  for (b=0u;b<blocks;b++) {
    const unsigned base=b*m;
    const int gamma=sme2_residual_mod_q(basis[base]);

    for (i=1u;i<m;i++)
      if (sme2_residual_mod_q(basis[base+i])!=0) {
        fprintf(stderr,
            "FAIL gamma structure depth=%u block=%u lane=%u value=%d\n",
            depth,b,i,(int)basis[base+i]);
        return 1;
      }

    if (gamma==0) {
      fprintf(stderr,"FAIL gamma zero depth=%u block=%u\n",depth,b);
      return 1;
    }

    prep->gamma[b]=(int16_t)gamma;
  }

  for (b=0u;b<blocks;b++) {
    int64_t x=1;
    unsigned p;

    for (p=0u;p<blocks;p++)
      x=(x*prep->gamma[b])%SME2_Q;

    if (sme2_residual_mod_q(x)!=SME2_Q-1) {
      fprintf(stderr,
          "FAIL gamma power depth=%u block=%u gamma=%d power=%d\n",
          depth,b,(int)prep->gamma[b],sme2_residual_mod_q(x));
      return 1;
    }
  }

  return 0;
}

/* ---- sme2_prepare_public_partial_ntt ---- */
// 3段NTTを行い、8個の32係数に分解
// 係数を-1664..1664の範囲に中心化する
static int sme2_prepare_public_partial_ntt(
    sme2_prepare_scratch *prep
)
{
  unsigned d,k;
  if (sme2_residual_discover_gamma(prep,3u))
    return 1;

  for (d=0u;d<4u;d++)
    for (k=0u;k<MLKEM_K;k++) {
      if (d<3u)
        prep->pub_partial[d][k]=prep->at_coeff.vec[d].vec[k];
      else
        prep->pub_partial[d][k]=prep->t_coeff.vec[k];

      sme2_partial_ntt3_neon(
          prep->pub_partial[d][k].coeffs);

      /*
      * Center the partial-NTT coefficients before Toeplitz packing.
      * This guarantees |public_gemm| <= 1664 and prevents
      * signed 32-bit ZA accumulator overflow in SMOPA.
      */
      for (unsigned i = 0u; i < 256u; i++) {
        prep->pub_partial[d][k].coeffs[i] =
            sme2_toeplitz_center_q(
                prep->pub_partial[d][k].coeffs[i]);
      }
    }
return 0;
}

/* ---- sme2_toeplitz_mod_q ---- */
static int sme2_toeplitz_mod_q(int64_t x)
{
  int64_t r=x%3329;
  if (r<0) r+=3329;
  return (int)r;
}

/* ---- sme2_toeplitz_center_q ---- */
static int16_t sme2_toeplitz_center_q(int64_t x)
{
  int r=sme2_toeplitz_mod_q(x);
  if (r>1664) r-=3329;
  return (int16_t)r;
}

/* ---- sme2_toeplitz_entry ---- */
// 32係数からテプリッツ行列の要素を計算
static int16_t sme2_toeplitz_entry(
    const int16_t a[32],
    int gamma,
    unsigned row,
    unsigned col)
{
  // row >= col の場合は、a[row-col] を返す
  if (row>=col)
    return a[row-col];

  // row < col の場合は、a[row+32-col] * gamma(mod 3329) を返す
  return sme2_toeplitz_center_q(
      (int64_t)gamma*(int64_t)a[row+32u-col]);
}

/* ---- sme2_prepare_public_toeplitz ---- */
// 2列16行をZレジスタ用にパックする
// ctx->public_gemmに格納する
static int sme2_prepare_public_toeplitz(
    mlk_sme2_public_ctx *ctx,
    sme2_prepare_scratch *prep
)
{
  unsigned pair,b,k,p,rr;
  if (sme2_prepare_public_partial_ntt(prep))
    return 1;

  for (pair=0u;pair<2u;pair++) {
    const unsigned da=2u*pair;
    const unsigned db=da+1u;

    for (b=0u;b<8u;b++) {
      const int gamma=prep->gamma[b];

      for (k=0u;k<MLKEM_K;k++) {
        const int16_t *a=
            &prep->pub_partial[da][k].coeffs[32u*b];
        const int16_t *bb=
            &prep->pub_partial[db][k].coeffs[32u*b];

        for (p=0u;p<16u;p++) {
          const unsigned c0=2u*p;
          const unsigned c1=c0+1u;

          for (rr=0u;rr<16u;rr++) {
            /* dotA, output rows 0..15 */

            // pair 4出力を2個ずつ処理, b 32係数を8個に分割, 
            // k 3行, p 16列, dot 4出力, row 32行, col 2列
            ctx->public_gemm[pair][b][k][p][0][2u*rr+0u]=
                sme2_toeplitz_entry(a,gamma,rr,c0);
            ctx->public_gemm[pair][b][k][p][0][2u*rr+1u]=
                sme2_toeplitz_entry(a,gamma,rr,c1);

            /* dotA, output rows 16..31 */
            ctx->public_gemm[pair][b][k][p][1][2u*rr+0u]=
                sme2_toeplitz_entry(a,gamma,16u+rr,c0);
            ctx->public_gemm[pair][b][k][p][1][2u*rr+1u]=
                sme2_toeplitz_entry(a,gamma,16u+rr,c1);

            /* dotB, output rows 0..15 */
            ctx->public_gemm[pair][b][k][p][2][2u*rr+0u]=
                sme2_toeplitz_entry(bb,gamma,rr,c0);
            ctx->public_gemm[pair][b][k][p][2][2u*rr+1u]=
                sme2_toeplitz_entry(bb,gamma,rr,c1);

            /* dotB, output rows 16..31 */
            ctx->public_gemm[pair][b][k][p][3][2u*rr+0u]=
                sme2_toeplitz_entry(bb,gamma,16u+rr,c0);
            ctx->public_gemm[pair][b][k][p][3][2u*rr+1u]=
                sme2_toeplitz_entry(bb,gamma,16u+rr,c1);
          }
        }
      }
    }
  }
return 0;
}

/* ---- sme2_inverse_center_q ---- */
// 係数を-1664..1664の範囲に中心化する
static int sme2_inverse_center_q(int x)
{
  int r=x%3329;
  if (r<0)
    r+=3329;
  if (r>1664)
    r-=3329;
  return r;
}

/* ---- sme2_twisted15 ---- */
static int sme2_twisted15(int c)
{
  const int64_t n=(int64_t)c*32768;

  if (n>=0)
    return (int)((n+1664)/3329);

  return -(int)(((-n)+1664)/3329);
}

/* ---- sme2_init_inverse_constants ---- */
static void sme2_init_inverse_constants(
    mlk_sme2_public_ctx *ctx
)
{
  unsigned k;
  for (k=0u;k<8u;k++) {
    const int c=sme2_inverse_center_q((int)sme2_inv2_roots_centered[k]);
    const int tw=sme2_twisted15(c);

    if (c<-32768 || c>32767 || tw<-32768 || tw>32767) {
      fprintf(stderr,
          "SME2 inverse constant range failure k=%u c=%d tw=%d\n",
          k,c,tw);
      abort();
    }

    ctx->inverse_c[k]=(int16_t)c;
    ctx->inverse_tw[k]=(int16_t)tw;
  }
}

/* ---- sme2_prepare_public_internal ---- */
static int sme2_prepare_public_internal(
    mlk_sme2_public_ctx *ctx,
    sme2_prepare_scratch *prep
)
{
  /*
   * Build the Toeplitz public representation through the validated depth-3
   * partial-NTT path. Residual gamma values must be derived before Toeplitz
   * packing because each degree-32 block satisfies x^32 = gamma_b.
   */
  return sme2_prepare_public_toeplitz(ctx,prep);
}

int mlk_sme2_prepare_public(
    mlk_sme2_public_ctx *ctx,
    const int16_t public_coeff[MLK_SME2_DOTS]
                              [MLK_SME2_K]
                              [MLK_SME2_N]
)
{
  MLK_SME2_ALIGN64 sme2_prepare_scratch prep;
  unsigned d, k;

  if (ctx == NULL || public_coeff == NULL)
    return -1;

  if (mlk_sme2_check_svl() != 0)
    return MLK_SME2_ERR_SVL;

  for (d = 0; d < MLKEM_K; d++)
    for (k = 0; k < MLKEM_K; k++)
      memcpy(prep.at_coeff.vec[d].vec[k].coeffs,
             public_coeff[d][k],
             sizeof(prep.at_coeff.vec[d].vec[k].coeffs));

  for (k = 0; k < MLKEM_K; k++)
    memcpy(prep.t_coeff.vec[k].coeffs,
           public_coeff[3][k],
           sizeof(prep.t_coeff.vec[k].coeffs));
  sme2_init_inverse_constants(ctx);
  return sme2_prepare_public_internal(ctx,&prep);
}


#else

MLK_EMPTY_CU(aarch64_sme2_public)

#endif /* MLK_ARITH_BACKEND_AARCH64 && MLK_SYS_AARCH64_SME2 && MLKEM_K == 3 */
