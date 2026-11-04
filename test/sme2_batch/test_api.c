/* Exercise the real, separately linked API, including rejection boundaries. */
#include <limits.h>
#include <sys/mman.h>
#include <unistd.h>
#include "api_test_support.h"

static int rejected_arguments(const void *inaccessible)
{
  unsigned mask, n, extra;
  const mlk_sme2_decaps_key *bad_key = inaccessible;
  const uint8_t (*bad_ct)[MLKEM_INDCCA_CIPHERTEXTBYTES] = inaccessible;
  reset_guards();
  for (mask = 1; mask < 16; mask++)
    for (extra = 0; extra < 2; extra++)
    {
      CHECK(mlk_sme2_decaps_batch(
          (mask & 1u) ? NULL : bad_key,
          (mask & 2u) ? NULL : &workspace.value,
          (mask & 4u) ? NULL : output.value,
          (mask & 8u) ? NULL : bad_ct,
          extra ? 0u : 16u, NULL) == MLK_SME2_DECAPS_ERR_ARGUMENT);
      CHECK(all_bytes(&workspace, sizeof(workspace), 0xa5));
      CHECK(all_bytes(&output, sizeof(output), 0xa5));
    }
  for (n = 0; n <= 272; n++)
  {
    if (n >= 16 && n <= 256 && n % 16 == 0)
      continue;
    CHECK(mlk_sme2_decaps_batch(bad_key, &workspace.value, output.value,
          bad_ct, n, NULL) == MLK_SME2_DECAPS_ERR_BATCH);
    CHECK(all_bytes(&workspace, sizeof(workspace), 0xa5));
    CHECK(all_bytes(&output, sizeof(output), 0xa5));
  }
  CHECK(mlk_sme2_decaps_batch(bad_key, &workspace.value, output.value,
        bad_ct, UINT_MAX, NULL) == MLK_SME2_DECAPS_ERR_BATCH);
  CHECK(mlk_sme2_decaps_batch(bad_key, &workspace.value, output.value,
        bad_ct, UINT_MAX - 15u, NULL) == MLK_SME2_DECAPS_ERR_BATCH);
  CHECK(all_bytes(&workspace, sizeof(workspace), 0xa5));
  CHECK(all_bytes(&output, sizeof(output), 0xa5));

  CHECK(mlk_sme2_decaps_prepare(NULL, inaccessible, NULL)
        == MLK_SME2_DECAPS_ERR_ARGUMENT);
  CHECK(mlk_sme2_decaps_prepare(NULL, NULL, NULL)
        == MLK_SME2_DECAPS_ERR_ARGUMENT);
  CHECK(mlk_sme2_decaps_prepare(&key.value, NULL, NULL)
        == MLK_SME2_DECAPS_ERR_ARGUMENT);
  CHECK(all_bytes(&key, sizeof(key), 0xa5));
  puts("PASS API rejection: all NULL combinations; 259 invalid sizes incl UINT_MAX; protected input pages untouched");
  return 0;
}

static int rejected_secret_key(void)
{
  uint8_t invalid_sk[MLKEM_INDCCA_SECRETKEYBYTES];
  int expected;
  memcpy(invalid_sk, sk, sizeof(invalid_sk));
  invalid_sk[sizeof(invalid_sk) - 2u * MLKEM_SYMBYTES] ^= 1;
  expected = mlk_kem_check_sk(invalid_sk, NULL);
  CHECK(expected != 0);
  memset(&key.value, 0xa5, sizeof(key.value));
  CHECK(mlk_sme2_decaps_prepare(&key.value, invalid_sk, NULL) == expected);
  CHECK(all_bytes(&key.value, sizeof(key.value), 0));
  CHECK(guards_intact() == 0);
  puts("PASS prepare: malformed sk error propagated and entire key cleared");
  return 0;
}

static int sequential_calls(void)
{
  unsigned seed, scenario, step, n;
  for (seed = 1; seed <= 2; seed++)
  {
    reset_guards();
    CHECK(make_fixture(seed) == 0);
    CHECK(rejected_secret_key() == 0);
    CHECK(mlk_sme2_decaps_prepare(&key.value, sk, NULL) == 0);
    expected_key = key.value;
    /* Deliberately never reset ws between calls. Ascending, alternating
     * small/large, and descending batches cover stale-tail reuse. */
    for (scenario = 0; scenario < 3; scenario++)
    {
      CHECK(select_scenario(scenario) == 0);
      for (step = 0; step < 16; step++)
      {
        n = scenario == 0 ? 16u * (step + 1u) :
            scenario == 1 ? 16u * ((step & 1u) ? 16u - step / 2u :
                                                         step / 2u + 1u) :
                            256u - 16u * step;
        memset(output.value, 0xa5, sizeof(output.value));
        expected_cleanup(n);
        CHECK(mlk_sme2_decaps_batch(&key.value, &workspace.value,
                                  output.value, ct, n, NULL) == 0);
        CHECK(cleanup_matches(n, 1) == 0);
      }
    }
    mlk_sme2_decaps_workspace_clear(&workspace.value);
    CHECK(all_bytes(&workspace.value, sizeof(workspace.value), 0));
    mlk_sme2_decaps_workspace_clear(&workspace.value);
    mlk_sme2_decaps_workspace_clear(NULL);
    mlk_sme2_decaps_key_clear(&key.value);
    CHECK(all_bytes(&key.value, sizeof(key.value), 0));
    mlk_sme2_decaps_key_clear(&key.value);
    mlk_sme2_decaps_key_clear(NULL);
    CHECK(guards_intact() == 0);
  }
  puts("PASS API: 2 keys x 3 scenarios x all 16 sizes = 96 calls / 13056 shared secrets; sequential shrink/grow reuse; active wipe and tail canaries; idempotent NULL-safe clear");
  return 0;
}

int main(void)
{
  const long page_size = sysconf(_SC_PAGESIZE);
  void *inaccessible;
  CHECK(page_size > 0);
  inaccessible = mmap(NULL, (size_t)page_size, PROT_NONE,
                      MAP_PRIVATE | MAP_ANON, -1, 0);
  CHECK(inaccessible != MAP_FAILED);
  CHECK(rejected_arguments(inaccessible) == 0);
  CHECK(munmap(inaccessible, (size_t)page_size) == 0);
  CHECK(mlk_sme2_check_svl() == 0);
  CHECK(sequential_calls() == 0);
  return 0;
}
