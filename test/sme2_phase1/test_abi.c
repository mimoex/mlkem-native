/*
 * Copyright (c) The mlkem-native project authors
 * SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../../mlkem/src/native/aarch64/sme2/mlkem_sme2_backend.h"

extern unsigned MLK_NAMESPACE(sme2_phase1_check_abi)(
    const int16_t *public_gemm, const int16_t *secret_gemm, int32_t *raw_out,
    unsigned groups) __arm_streaming __arm_inout("za");

/* The canaries must live across the streaming call, not across SMSTART. */
__arm_new("za") __arm_locally_streaming
static unsigned check_abi(const int16_t *public_gemm,
                          const int16_t *secret_gemm, int32_t *raw_out,
                          unsigned groups)
{
  return MLK_NAMESPACE(sme2_phase1_check_abi)(public_gemm, secret_gemm,
                                            raw_out, groups);
}

static MLK_SME2_ALIGN64 int16_t public_gemm[2][8][3][16][4][32];
static MLK_SME2_ALIGN64 int16_t secret_gemm[2][8][3][16][32];
static MLK_SME2_ALIGN64 int32_t raw[2][8][4][2][256];

int main(void)
{
  uint64_t svl_bytes;
  unsigned groups;
  unsigned failures = 0;
  size_t i;

  __asm__ volatile("rdsvl %0, #1" : "=r"(svl_bytes));
  if (svl_bytes != 64)
  {
    fprintf(stderr, "ABI test requires SVL=512 bits, got %llu bits\n",
            (unsigned long long)(8 * svl_bytes));
    return 1;
  }

  for (i = 0; i < sizeof(public_gemm) / sizeof(int16_t); i++)
  {
    ((int16_t *)public_gemm)[i] = 1;
  }
  for (i = 0; i < sizeof(secret_gemm) / sizeof(int16_t); i++)
  {
    ((int16_t *)secret_gemm)[i] = 2;
  }

  for (groups = 0; groups <= 2; groups++)
  {
    unsigned mask;
    unsigned mismatches = 0;
    size_t used = groups * sizeof(raw[0]) / sizeof(int32_t);
    memset(raw, 0, sizeof(raw));
    mask = check_abi(&public_gemm[0][0][0][0][0][0],
                     &secret_gemm[0][0][0][0][0], &raw[0][0][0][0][0],
                     groups);
    for (i = 0; i < sizeof(raw) / sizeof(int32_t); i++)
    {
      /* Every element accumulates 96 products of 1 * 2. */
      if (((int32_t *)raw)[i] != (i < used ? 192 : 0))
      {
        mismatches++;
      }
    }
    printf("groups=%u d8-d15_clobber_mask=0x%02x raw_mismatches=%u\n",
           groups, mask, mismatches);
    failures += mask != 0 || mismatches != 0;
  }
  printf("SMOPA ABI and raw arithmetic: %s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
