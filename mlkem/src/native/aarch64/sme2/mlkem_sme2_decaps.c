/*
* 本番用SME2残余ドメイン再暗号化バックエンドを使用した、
* 同一受信者向けのML-KEM-768バッチ処理復号。
 */

#include "../../../common.h"
#include "../../../params.h"

#if defined(MLK_SYS_AARCH64_SME2) && \
    defined(MLK_CONFIG_USE_NATIVE_BACKEND_ARITH) && \
    !defined(MLK_CONFIG_NO_DECAPS_API) && \
    MLKEM_K == 3

#include "mlkem_sme2_decaps.h"

#include "../../../indcpa.h"
#include "../../../kem.h"
#include "../../../poly.h"
#include "../../../poly_k.h"
#include "../../../sampling.h"
#include "../../../symmetric.h"
#include "../../../verify.h"

static void mlk_sme2_decaps_ntt_to_coeff(mlk_poly *p)
{
  unsigned i;

  mlk_poly_invntt_tomont(p);
  for (i = 0; i < MLKEM_N; i++)
    p->coeffs[i] = mlk_montgomery_reduce((int32_t)p->coeffs[i]);
  mlk_poly_reduce(p);
}

static int mlk_sme2_decaps_prepare_public(
    mlk_sme2_public_ctx *public_ctx,
    const uint8_t pk[MLKEM_INDCCA_PUBLICKEYBYTES])
{
  mlk_polymat at_ntt;
  mlk_polymat at_coeff;
  mlk_polyvec t_ntt;
  mlk_polyvec t_coeff;
  uint8_t seed[MLKEM_SYMBYTES];
  int16_t public_coeff[MLK_SME2_DOTS][MLK_SME2_K][MLK_SME2_N];
  unsigned d, k;
  int ret;

  mlk_polyvec_frombytes(&t_ntt, pk);
  mlk_memcpy(seed, pk + MLKEM_POLYVECBYTES, MLKEM_SYMBYTES);
  mlk_gen_matrix(&at_ntt, seed, 1);

  at_coeff = at_ntt;
  t_coeff = t_ntt;

  for (d = 0; d < MLKEM_K; d++)
  {
    mlk_sme2_decaps_ntt_to_coeff(&t_coeff.vec[d]);
    for (k = 0; k < MLKEM_K; k++)
      mlk_sme2_decaps_ntt_to_coeff(&at_coeff.vec[d].vec[k]);
  }

  for (d = 0; d < MLKEM_K; d++)
    for (k = 0; k < MLKEM_K; k++)
      mlk_memcpy(public_coeff[d][k],
                 at_coeff.vec[d].vec[k].coeffs,
                 sizeof(public_coeff[d][k]));

  for (k = 0; k < MLKEM_K; k++)
    mlk_memcpy(public_coeff[3][k],
               t_coeff.vec[k].coeffs,
               sizeof(public_coeff[3][k]));

  ret = mlk_sme2_prepare_public(public_ctx, public_coeff);

  mlk_zeroize(public_coeff, sizeof(public_coeff));
  mlk_zeroize(seed, sizeof(seed));
  mlk_zeroize(&t_coeff, sizeof(t_coeff));
  mlk_zeroize(&t_ntt, sizeof(t_ntt));
  mlk_zeroize(&at_coeff, sizeof(at_coeff));
  mlk_zeroize(&at_ntt, sizeof(at_ntt));

  return ret;
}

MLK_INTERNAL_API
MLK_MUST_CHECK_RETURN_VALUE
int mlk_sme2_decaps_prepare(
    mlk_sme2_decaps_key *key,
    const uint8_t sk[MLKEM_INDCCA_SECRETKEYBYTES],
    MLK_CONFIG_CONTEXT_PARAMETER_TYPE context)
{
  const uint8_t *pk;
  int ret;

  if (key == NULL || sk == NULL)
    return MLK_SME2_DECAPS_ERR_ARGUMENT;

  mlk_zeroize(key, sizeof(*key));

  if (mlk_sme2_check_svl() != 0)
    return MLK_SME2_DECAPS_ERR_SVL;

  ret = mlk_kem_check_sk(sk, context);
  if (ret != 0)
    return ret;

  pk = sk + MLKEM_INDCPA_SECRETKEYBYTES;

  ret = mlk_sme2_decaps_prepare_public(&key->public_ctx, pk);
  if (ret != 0)
  {
    mlk_zeroize(key, sizeof(*key));
    return ret == MLK_SME2_ERR_SVL ? MLK_SME2_DECAPS_ERR_SVL
                                   : MLK_SME2_DECAPS_ERR_BACKEND;
  }

  mlk_memcpy(key->dkpke, sk, MLKEM_INDCPA_SECRETKEYBYTES);
  mlk_memcpy(key->hpk,
             sk + MLKEM_INDCCA_SECRETKEYBYTES - 2 * MLKEM_SYMBYTES,
             MLKEM_SYMBYTES);
  mlk_memcpy(key->z,
             sk + MLKEM_INDCCA_SECRETKEYBYTES - MLKEM_SYMBYTES,
             MLKEM_SYMBYTES);

  return 0;
}

static void mlk_sme2_decaps_sample_sp(
    mlk_polyvec *out,
    const uint8_t coins[MLKEM_SYMBYTES])
{
  // 通常のmlk_poly_getnoise_eta1_4x()を使用して、(y0,y1,y2)の256係数の3行の多項式ベクトルを生成する
  mlk_poly_getnoise_eta1_4x(
      &out->vec[0], &out->vec[1], &out->vec[2], NULL,
      coins, 0, 1, 2, 0xFF);
}

static void mlk_sme2_decaps_finish_one(
    const mlk_sme2_decaps_key *key,
    mlk_sme2_decaps_workspace *ws,
    uint8_t ss[MLKEM_SSBYTES],
    const uint8_t ct[MLKEM_INDCCA_CIPHERTEXTBYTES],
    unsigned r)
{
  mlk_polyvec ep;
  mlk_poly epp;
  mlk_poly kmsg;
  mlk_polyvec *b = &ws->arithmetic.arena.group[r / MLK_SME2_GROUP]
         .out[r % MLK_SME2_GROUP].b;
  mlk_poly *v = &ws->arithmetic.arena.group[r / MLK_SME2_GROUP]
         .out[r % MLK_SME2_GROUP].v;
  uint8_t reenc[MLKEM_INDCCA_CIPHERTEXTBYTES];
  uint8_t reject_input[MLKEM_SYMBYTES + MLKEM_INDCCA_CIPHERTEXTBYTES];
  uint8_t fail;

  // 通常のmlkem-nativeの関数で処理する
  mlk_poly_getnoise_eta2_4x(
      &ep.vec[0], &ep.vec[1], &ep.vec[2], &epp,
      ws->kr[r] + MLKEM_SYMBYTES, 3, 4, 5, 6);

  mlk_poly_frommsg(&kmsg, ws->message[r]);

  
  mlk_polyvec_add(b, &ep);
  mlk_poly_add(v, &epp);
  mlk_poly_add(v, &kmsg);

  mlk_polyvec_reduce(b);
  mlk_poly_reduce(v);

  mlk_polyvec_compress_du(reenc, b);
  mlk_poly_compress_dv(
      reenc + MLKEM_POLYVECCOMPRESSEDBYTES_DU, v);

  fail = mlk_ct_memcmp(ct, reenc, MLKEM_INDCCA_CIPHERTEXTBYTES);

  mlk_memcpy(reject_input, key->z, MLKEM_SYMBYTES);
  mlk_memcpy(reject_input + MLKEM_SYMBYTES,
             ct,
             MLKEM_INDCCA_CIPHERTEXTBYTES);
  mlk_hash_j(ss, reject_input, sizeof(reject_input));

  mlk_ct_cmov_zero(ss, ws->kr[r], MLKEM_SYMBYTES, fail);

  mlk_zeroize(reject_input, sizeof(reject_input));
  mlk_zeroize(reenc, sizeof(reenc));
  mlk_zeroize(&kmsg, sizeof(kmsg));
  mlk_zeroize(&epp, sizeof(epp));
  mlk_zeroize(&ep, sizeof(ep));
}


static void mlk_sme2_decaps_zeroize_used(
    mlk_sme2_decaps_workspace *ws,
    unsigned n)
{
  const size_t requests = (size_t)n;
  const size_t groups = requests / MLK_SME2_GROUP;

  if (n == MLK_SME2_MAX_BATCH)
  {
    mlk_zeroize(ws, sizeof(*ws));
    return;
  }

  /*
   * Group-local rolling arena:
   *   group[g].sec[16] -> group[g].out[16]
   *
   * Every touched group ends in the larger 32-KiB out lifetime, so wiping
   * the touched group-slot prefix covers all primary-arena bytes used by the
   * earlier 24-KiB secret lifetime as well.  raw and secondary are one-group
   * scratch and are fully reused for every group.
   */
  mlk_zeroize(
      ws->arithmetic.arena.group,
      groups * sizeof(ws->arithmetic.arena.group[0]));
  mlk_zeroize(ws->arithmetic.raw, sizeof(ws->arithmetic.raw));
  mlk_zeroize(
      ws->arithmetic.secondary.inverse_h,
      sizeof(ws->arithmetic.secondary.inverse_h));

  mlk_zeroize(
      ws->message,
      requests * sizeof(ws->message[0]));

  mlk_zeroize(
      ws->kr,
      requests * sizeof(ws->kr[0]));
}

// 提案方式　呼び出し元
MLK_INTERNAL_API
MLK_MUST_CHECK_RETURN_VALUE
int mlk_sme2_decaps_batch(
    const mlk_sme2_decaps_key *key,
    mlk_sme2_decaps_workspace *ws,
    uint8_t ss[][MLKEM_SSBYTES],
    const uint8_t ct[][MLKEM_INDCCA_CIPHERTEXTBYTES],
    unsigned n,
    MLK_CONFIG_CONTEXT_PARAMETER_TYPE context)
{
  unsigned off = 0u;
  int ret = 0;

  if (key == NULL || ws == NULL || ss == NULL || ct == NULL)
    return MLK_SME2_DECAPS_ERR_ARGUMENT;

  if (n < MLK_SME2_GROUP ||
      n > MLK_SME2_MAX_BATCH ||
      (n % MLK_SME2_GROUP) != 0u)
    return MLK_SME2_DECAPS_ERR_BATCH;

  if (mlk_sme2_check_svl() != 0)
    return MLK_SME2_DECAPS_ERR_SVL;

  while (off < n)
  {
    unsigned tile_n;
    unsigned r;

    /* The rolling workspace covers the remaining requests in one tile. */
    tile_n = n - off;

    for (r = 0u; r < tile_n; r++)
    {
      uint8_t g_input[2 * MLKEM_SYMBYTES];

      // 通常の復号
      ret = mlk_indcpa_dec(
          ws->message[r], ct[off + r], key->dkpke, context);
      if (ret != 0)
      {
        mlk_zeroize(g_input, sizeof(g_input));
        goto tile_cleanup;
      }

      mlk_memcpy(g_input, ws->message[r], MLKEM_SYMBYTES);
      mlk_memcpy(
          g_input + MLKEM_SYMBYTES,
          key->hpk,
          MLKEM_SYMBYTES);
      // 通常の共有秘密候補・乱数の導出
      mlk_hash_g(ws->kr[r], g_input, sizeof(g_input));

      //秘密多項式の生成
      // 16要求分が.sec[0..15]に格納される
      mlk_sme2_decaps_sample_sp(
          &ws->arithmetic.arena.group[r / MLK_SME2_GROUP]
              .sec[r % MLK_SME2_GROUP],
          ws->kr[r] + MLKEM_SYMBYTES);

      mlk_zeroize(g_input, sizeof(g_input));
    }
    // バッチ処理のために、ここまでは通常の復号と同じ処理を行う。
    // ここからは、提案方式のバッチ処理を行う。

    // 3段NTTとSME2算術処理
    ret = mlk_sme2_arith_batch_rolling_preloaded(
        &key->public_ctx,
        &ws->arithmetic,
        tile_n);
    if (ret != 0)
    {
      ret = ret == MLK_SME2_ERR_SVL ? MLK_SME2_DECAPS_ERR_SVL
                                    : MLK_SME2_DECAPS_ERR_BACKEND;
      goto tile_cleanup;
    }

    for (r = 0u; r < tile_n; r++)
    // 再暗号化・照合・共有秘密導出
      mlk_sme2_decaps_finish_one(
          key,
          ws,
          ss[off + r],
          ct[off + r],
          r);

tile_cleanup:
    mlk_sme2_decaps_zeroize_used(ws, tile_n);

    if (ret != 0)
      goto cleanup;

    off += tile_n;
  }

  return 0;

cleanup:
  mlk_zeroize(ss, (size_t)n * MLKEM_SSBYTES);
  return ret;
}

MLK_INTERNAL_API
void mlk_sme2_decaps_key_clear(mlk_sme2_decaps_key *key)
{
  if (key != NULL)
    mlk_zeroize(key, sizeof(*key));
}

MLK_INTERNAL_API
void mlk_sme2_decaps_workspace_clear(mlk_sme2_decaps_workspace *ws)
{
  if (ws != NULL)
    mlk_zeroize(ws, sizeof(*ws));
}

#else

MLK_EMPTY_CU(sme2_decaps)

#endif
