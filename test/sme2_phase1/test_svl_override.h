/* Test translation unit only: replace the VL query, never the guard itself. */
#ifndef SME2_PHASE1_TEST_SVL_OVERRIDE_H
#define SME2_PHASE1_TEST_SVL_OVERRIDE_H
#include <arm_sme.h>
extern uint64_t phase1_svl_bytes;
#define svcntsb() phase1_svl_bytes
#endif
