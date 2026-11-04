#ifndef MLK_NATIVE_AARCH64_SME2_MLKEM_SME2_H
#define MLK_NATIVE_AARCH64_SME2_MLKEM_SME2_H

#include "../../../common.h"
#include <stdint.h>

#define mlk_sme2_cross_matrix_d3 MLK_NAMESPACE(sme2_cross_matrix_d3)

int mlk_sme2_cross_matrix_d3(const int16_t *public_gemm,
                              const int16_t *secret_gemm,
                              int32_t *raw_out,
                              unsigned groups);

#endif
