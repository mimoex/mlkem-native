#include "prepared_ntt.h"
#include "common.h"
#include "indcpa.h"
#include "kem.h"
#include "params.h"
#include "poly.h"
#include "poly_k.h"
#include "sampling.h"
#include "symmetric.h"
#include "verify.h"

// 同一公開鍵の前計算
int prepare_ntt(ntt_key *key,
                       const uint8_t sk0[MLKEM_INDCCA_SECRETKEYBYTES])
{
  const uint8_t *pk0;
  uint8_t seed[MLKEM_SYMBYTES];
  int ret;

  mlk_zeroize(key, sizeof(*key));
  ret = mlk_kem_check_sk(sk0, NULL);
  if (ret != 0)
    return ret;

  pk0 = sk0 + MLKEM_INDCPA_SECRETKEYBYTES;
  mlk_polyvec_frombytes(&key->t_ntt, pk0);
  mlk_memcpy(seed, pk0 + MLKEM_POLYVECBYTES, MLKEM_SYMBYTES);
  mlk_gen_matrix(&key->at_ntt, seed, 1);
  mlk_memcpy(key->dkpke, sk0, sizeof(key->dkpke));
  mlk_memcpy(key->hpk,
             sk0 + MLKEM_INDCCA_SECRETKEYBYTES - 2u * MLKEM_SYMBYTES,
             MLKEM_SYMBYTES);
  mlk_memcpy(key->z,
             sk0 + MLKEM_INDCCA_SECRETKEYBYTES - MLKEM_SYMBYTES,
             MLKEM_SYMBYTES);
  mlk_zeroize(seed, sizeof(seed));
  return 0;
}

//復号と再暗号化の前処理
// mlk_indcpa_dec()で暗号文からm'を復号し、g()でkrを生成する
int ntt_front(const ntt_key *key,
                     ntt_sb_arena *sb,
                     uint8_t message[][MLKEM_SYMBYTES],
                     uint8_t kr[][2 * MLKEM_SYMBYTES],
                     const uint8_t ct0[][MLKEM_INDCCA_CIPHERTEXTBYTES],
                     unsigned n)
{
  unsigned r;
  for (r = 0; r < n; r++)
  {
    uint8_t g_input[2 * MLKEM_SYMBYTES];
    int ret = mlk_indcpa_dec(message[r], ct0[r], key->dkpke, NULL);
    if (ret != 0)
    {
      mlk_zeroize(g_input, sizeof(g_input));
      return ret;
    }

    mlk_memcpy(g_input, message[r], MLKEM_SYMBYTES);
    mlk_memcpy(g_input + MLKEM_SYMBYTES, key->hpk, MLKEM_SYMBYTES);
    mlk_hash_g(kr[r], g_input, sizeof(g_input));

    mlk_poly_getnoise_eta1_4x(
        &sb->sp[r].vec[0],
        &sb->sp[r].vec[1],
        &sb->sp[r].vec[2],
        NULL,
        kr[r] + MLKEM_SYMBYTES,
        0, 1, 2, 0xFF);

    mlk_zeroize(g_input, sizeof(g_input));
  }
  return 0;
}

void ntt_finish_one(const ntt_key *key,
                           ntt_sb_arena *sb,
                           mlk_poly v[],
                           uint8_t message[][MLKEM_SYMBYTES],
                           uint8_t kr[][2 * MLKEM_SYMBYTES],
                           uint8_t ss[MLKEM_SSBYTES],
                           const uint8_t ct0[MLKEM_INDCCA_CIPHERTEXTBYTES],
                           unsigned r)
{
  MLK_ALIGN mlk_polyvec ep;
  MLK_ALIGN mlk_poly epp;
  MLK_ALIGN mlk_poly kmsg;
  uint8_t reenc[MLKEM_INDCCA_CIPHERTEXTBYTES];
  uint8_t reject_input[MLKEM_SYMBYTES + MLKEM_INDCCA_CIPHERTEXTBYTES];
  uint8_t fail;

  mlk_poly_getnoise_eta2_4x(
      &ep.vec[0], &ep.vec[1], &ep.vec[2], &epp,
      kr[r] + MLKEM_SYMBYTES, 3, 4, 5, 6);

  mlk_poly_frommsg(&kmsg, message[r]);
  mlk_polyvec_add(&sb->b[r], &ep);
  mlk_poly_add(&v[r], &epp);
  mlk_poly_add(&v[r], &kmsg);
  mlk_polyvec_reduce(&sb->b[r]);
  mlk_poly_reduce(&v[r]);

  mlk_polyvec_compress_du(reenc, &sb->b[r]);
  mlk_poly_compress_dv(
      reenc + MLKEM_POLYVECCOMPRESSEDBYTES_DU, &v[r]);

  fail = mlk_ct_memcmp(ct0, reenc, MLKEM_INDCCA_CIPHERTEXTBYTES);
  mlk_memcpy(reject_input, key->z, MLKEM_SYMBYTES);
  mlk_memcpy(reject_input + MLKEM_SYMBYTES,
             ct0, MLKEM_INDCCA_CIPHERTEXTBYTES);
  mlk_hash_j(ss, reject_input, sizeof(reject_input));
  mlk_ct_cmov_zero(ss, kr[r], MLKEM_SYMBYTES, fail);

  mlk_zeroize(reject_input, sizeof(reject_input));
  mlk_zeroize(reenc, sizeof(reenc));
  mlk_zeroize(&kmsg, sizeof(kmsg));
  mlk_zeroize(&epp, sizeof(epp));
  mlk_zeroize(&ep, sizeof(ep));
}

void ntt_wipe_common(ntt_sb_arena *sb,
                            mlk_poly v[],
                            uint8_t message[][MLKEM_SYMBYTES],
                            uint8_t kr[][2 * MLKEM_SYMBYTES],
                            unsigned n)
{
  mlk_zeroize(sb->sp, (size_t)n * sizeof(sb->sp[0]));
  mlk_zeroize(v, (size_t)n * sizeof(v[0]));
  mlk_zeroize(message, (size_t)n * sizeof(message[0]));
  mlk_zeroize(kr, (size_t)n * sizeof(kr[0]));
}

// 従来のNTT
int ntt_legacy_batch(const ntt_key *key,
                            ntt_legacy_workspace *ws,
                            uint8_t ss[][MLKEM_SSBYTES],
                            const uint8_t ct0[][MLKEM_INDCCA_CIPHERTEXTBYTES],
                            unsigned n)
{
  unsigned r, row;
  int ret = ntt_front(key, &ws->sb, ws->message, ws->kr, ct0, n);
  if (ret != 0)
    goto cleanup;

  for (r = 0; r < n; r++)
  {
    MLK_ALIGN mlk_polyvec sp_ntt = ws->sb.sp[r];
    MLK_ALIGN mlk_polyvec_mulcache cache;

    mlk_polyvec_ntt(&sp_ntt);
    mlk_polyvec_mulcache_compute(&cache, &sp_ntt);

    for (row = 0; row < MLKEM_K; row++)
      mlk_polyvec_basemul_acc_montgomery_cached(
          &ws->sb.b[r].vec[row], &key->at_ntt.vec[row],
          &sp_ntt, &cache);

    mlk_polyvec_basemul_acc_montgomery_cached(
        &ws->v[r], &key->t_ntt, &sp_ntt, &cache);

    mlk_polyvec_invntt_tomont(&ws->sb.b[r]);
    mlk_poly_invntt_tomont(&ws->v[r]);

    // 要求を1件処理するたびに、2つの作業領域を消去。
    mlk_zeroize(&cache, sizeof(cache));
    mlk_zeroize(&sp_ntt, sizeof(sp_ntt));
  }

  for (r = 0; r < n; r++)
    ntt_finish_one(key, &ws->sb, ws->v, ws->message, ws->kr,
                   ss[r], ct0[r], r);

cleanup:
  if (ret != 0)
    mlk_zeroize(ss, (size_t)n * MLKEM_SSBYTES);
  ntt_wipe_common(&ws->sb, ws->v, ws->message, ws->kr, n);
  return ret;
}

int ntt_fair_batch(const ntt_key *key,
                          ntt_fair_workspace *ws,
                          uint8_t ss[][MLKEM_SSBYTES],
                          const uint8_t ct0[][MLKEM_INDCCA_CIPHERTEXTBYTES],
                          unsigned n)
{
  unsigned r, row;
  int ret = ntt_front(key, &ws->sb, ws->message, ws->kr, ct0, n);
  if (ret != 0)
    goto cleanup;

  for (r = 0; r < n; r++)
  {
    // 1件の要求を処理するたびに、作業領域内の秘密情報を消去。
    ws->sp_ntt = ws->sb.sp[r];
    mlk_polyvec_ntt(&ws->sp_ntt);
    mlk_polyvec_mulcache_compute(&ws->cache, &ws->sp_ntt);

    for (row = 0; row < MLKEM_K; row++)
      mlk_polyvec_basemul_acc_montgomery_cached(
          &ws->sb.b[r].vec[row], &key->at_ntt.vec[row],
          &ws->sp_ntt, &ws->cache);

    mlk_polyvec_basemul_acc_montgomery_cached(
        &ws->v[r], &key->t_ntt, &ws->sp_ntt, &ws->cache);

    mlk_polyvec_invntt_tomont(&ws->sb.b[r]);
    mlk_poly_invntt_tomont(&ws->v[r]);
  }

  for (r = 0; r < n; r++)
    ntt_finish_one(key, &ws->sb, ws->v, ws->message, ws->kr,
                   ss[r], ct0[r], r);

// バッチ処理最後に作業領域内の秘密情報を消去
cleanup:
  if (ret != 0)
    mlk_zeroize(ss, (size_t)n * MLKEM_SSBYTES);
  ntt_wipe_common(&ws->sb, ws->v, ws->message, ws->kr, n);
  mlk_zeroize(&ws->sp_ntt, sizeof(ws->sp_ntt));
  mlk_zeroize(&ws->cache, sizeof(ws->cache));
  return ret;
}

/* Final O3 public-cache baseline, extracted without arithmetic changes. */
/* These benchmark helpers have the same valid-pointer and batch-size
 * preconditions as the original prepared NTT benchmark. */
int prepare_ntt_public(
    ntt_public_key *key,
    const uint8_t sk0[MLKEM_INDCCA_SECRETKEYBYTES])
{
  unsigned row;
  int ret;

  /* Also destroy the previous public caches if preparation fails. */
  mlk_zeroize(key, sizeof(*key));
  ret = prepare_ntt(&key->base, sk0);
  if (ret != 0)
    return ret;

  /* A^T and t are already in the NTT domain after prepare_ntt(). */
  for (row = 0; row < MLKEM_K; row++)
    mlk_polyvec_mulcache_compute(&key->at_cache[row],
                                &key->base.at_ntt.vec[row]);
  mlk_polyvec_mulcache_compute(&key->t_cache, &key->base.t_ntt);
  return 0;
}

/* The separate arithmetic helper lets the correctness oracle observe all
 * four output polynomials before FO additions, compression and wiping. */
void ntt_public_arith(const ntt_public_key *key,
                             ntt_public_workspace *ws,
                             unsigned n)
{
  unsigned r, row;
  for (r = 0; r < n; r++)
  {
    ws->sp_ntt = ws->sb.sp[r];
    mlk_polyvec_ntt(&ws->sp_ntt);

    /* The native cached product requires its FIRST operand in [0,4095].
     * NTT output is lazy; canonical reduction gives [0,q-1] without changing
     * its Montgomery scale. The cache belongs to the SECOND operand. Swapping
     * operands preserves the commutative product and its R^-1 factor, which
     * invntt_tomont cancels exactly as in the original prepared baseline. */
    mlk_polyvec_reduce(&ws->sp_ntt);
    for (row = 0; row < MLKEM_K; row++)
      mlk_polyvec_basemul_acc_montgomery_cached(
          &ws->sb.b[r].vec[row], &ws->sp_ntt,
          &key->base.at_ntt.vec[row], &key->at_cache[row]);
    mlk_polyvec_basemul_acc_montgomery_cached(
        &ws->v[r], &ws->sp_ntt, &key->base.t_ntt, &key->t_cache);

    mlk_polyvec_invntt_tomont(&ws->sb.b[r]);
    mlk_poly_invntt_tomont(&ws->v[r]);
  }
}

int ntt_public_batch(
    const ntt_public_key *key,
    ntt_public_workspace *ws,
    uint8_t ss[][MLKEM_SSBYTES],
    const uint8_t ct0[][MLKEM_INDCCA_CIPHERTEXTBYTES],
    unsigned n)
{
  unsigned r;
  int ret = ntt_front(&key->base, &ws->sb, ws->message, ws->kr, ct0, n);
  if (ret != 0)
    goto cleanup;

  ntt_public_arith(key, ws, n);
  for (r = 0; r < n; r++)
    ntt_finish_one(&key->base, &ws->sb, ws->v, ws->message, ws->kr,
                   ss[r], ct0[r], r);

cleanup:
  if (ret != 0)
    mlk_zeroize(ss, (size_t)n * MLKEM_SSBYTES);
  ntt_wipe_common(&ws->sb, ws->v, ws->message, ws->kr, n);
  mlk_zeroize(&ws->sp_ntt, sizeof(ws->sp_ntt));
  return ret;
}
