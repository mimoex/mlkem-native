/* Research comparison API: the upstream AArch64 NEON arithmetic is unchanged. */
#ifndef PREPARED_NTT_H
#define PREPARED_NTT_H

#include "common.h"
#include "poly_k.h"

#if !defined(MLK_ARITH_BACKEND_AARCH64) || MLKEM_K != 3
#error "Prepared NTT requires the native AArch64 ML-KEM-768 backend"
#endif

#define MAX_N 256u

typedef struct {
  MLK_ALIGN mlk_polymat at_ntt;
  MLK_ALIGN mlk_polyvec t_ntt;
  uint8_t dkpke[MLKEM_INDCPA_SECRETKEYBYTES];
  uint8_t hpk[MLKEM_SYMBYTES];
  uint8_t z[MLKEM_SYMBYTES];
} ntt_key;

typedef union {
  MLK_ALIGN mlk_polyvec sp[MAX_N];
  MLK_ALIGN mlk_polyvec b[MAX_N];
} ntt_sb_arena;

/* Current corrected baseline: caller workspace only. Arithmetic scratch lives
 * on the stack and is wiped for every request. */
typedef struct {
  ntt_sb_arena sb;
  MLK_ALIGN mlk_poly v[MAX_N];
  uint8_t message[MAX_N][MLKEM_SYMBYTES];
  uint8_t kr[MAX_N][2 * MLKEM_SYMBYTES];
} ntt_legacy_workspace;

/* Fair prepared-NTT baseline: expose the NTT-domain secret and mulcache scratch
 * explicitly, reuse one instance across the entire batch, and wipe it once
 * at the API boundary. */
typedef struct {
  ntt_sb_arena sb;
  MLK_ALIGN mlk_poly v[MAX_N];
  uint8_t message[MAX_N][MLKEM_SYMBYTES];
  uint8_t kr[MAX_N][2 * MLKEM_SYMBYTES];
  MLK_ALIGN mlk_polyvec sp_ntt;
  MLK_ALIGN mlk_polyvec_mulcache cache;
} ntt_fair_workspace;

_Static_assert(sizeof(ntt_legacy_workspace) == 548864u,
               "unexpected legacy prepared-NTT workspace");
_Static_assert(sizeof(ntt_fair_workspace) == 551168u,
               "unexpected fair prepared-NTT workspace");

/* Harness-only preconditions, matching the original benchmark:
 * pointers are non-NULL, disjoint, correctly aligned, and sized for n;
 * n is 16,32,...,256; prepare_ntt succeeded; workspace is exclusively owned.
 * Prepared keys may be shared across calls with separate workspaces.
 * Public/key preparation is excluded from timing. The fair batch reuses one
 * sp_ntt/cache and erases both once at its boundary; live arrays are erased too.
 * Invalid ciphertexts use the upstream implicit-rejection result.
 * Call mlk_zeroize on the complete key/workspace when their lifetime ends.
 * front/finish/wipe and legacy are shared helpers for scheduling ablations.
 */
int prepare_ntt(ntt_key *key,
                       const uint8_t sk0[MLKEM_INDCCA_SECRETKEYBYTES]);

int ntt_front(const ntt_key *key,
                     ntt_sb_arena *sb,
                     uint8_t message[][MLKEM_SYMBYTES],
                     uint8_t kr[][2 * MLKEM_SYMBYTES],
                     const uint8_t ct0[][MLKEM_INDCCA_CIPHERTEXTBYTES],
                     unsigned n);

void ntt_finish_one(const ntt_key *key,
                           ntt_sb_arena *sb,
                           mlk_poly v[],
                           uint8_t message[][MLKEM_SYMBYTES],
                           uint8_t kr[][2 * MLKEM_SYMBYTES],
                           uint8_t ss[MLKEM_SSBYTES],
                           const uint8_t ct0[MLKEM_INDCCA_CIPHERTEXTBYTES],
                           unsigned r);

void ntt_wipe_common(ntt_sb_arena *sb,
                            mlk_poly v[],
                            uint8_t message[][MLKEM_SYMBYTES],
                            uint8_t kr[][2 * MLKEM_SYMBYTES],
                            unsigned n);

int ntt_legacy_batch(const ntt_key *key,
                            ntt_legacy_workspace *ws,
                            uint8_t ss[][MLKEM_SSBYTES],
                            const uint8_t ct0[][MLKEM_INDCCA_CIPHERTEXTBYTES],
                            unsigned n);

int ntt_fair_batch(const ntt_key *key,
                          ntt_fair_workspace *ws,
                          uint8_t ss[][MLKEM_SSBYTES],
                          const uint8_t ct0[][MLKEM_INDCCA_CIPHERTEXTBYTES],
                          unsigned n);

/* Final O3 baseline: cache the public operand; same harness preconditions. */
typedef struct {
  ntt_key base;
  MLK_ALIGN mlk_polyvec_mulcache at_cache[MLKEM_K];
  MLK_ALIGN mlk_polyvec_mulcache t_cache;
} ntt_public_key;

typedef struct {
  ntt_sb_arena sb;
  MLK_ALIGN mlk_poly v[MAX_N];
  uint8_t message[MAX_N][MLKEM_SYMBYTES];
  uint8_t kr[MAX_N][2 * MLKEM_SYMBYTES];
  MLK_ALIGN mlk_polyvec sp_ntt;
} ntt_public_workspace;

_Static_assert(sizeof(ntt_key) == 7360u,
               "unexpected original prepared-NTT key");
_Static_assert(sizeof(ntt_public_key) == 10432u,
               "unexpected public-cache prepared-NTT key");
_Static_assert(sizeof(ntt_public_workspace) == 550400u,
               "unexpected public-cache prepared-NTT workspace");

int prepare_ntt_public(
    ntt_public_key *key,
    const uint8_t sk0[MLKEM_INDCCA_SECRETKEYBYTES]);

void ntt_public_arith(const ntt_public_key *key,
                             ntt_public_workspace *ws,
                             unsigned n);

int ntt_public_batch(
    const ntt_public_key *key,
    ntt_public_workspace *ws,
    uint8_t ss[][MLKEM_SSBYTES],
    const uint8_t ct0[][MLKEM_INDCCA_CIPHERTEXTBYTES],
    unsigned n);

#endif
