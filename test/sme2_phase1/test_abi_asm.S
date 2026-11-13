/*
 * Copyright (c) The mlkem-native project authors
 * SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT
 */

#include "../../mlkem/src/common.h"

/* Streaming, shared-ZA caller. Return bit i if d(8+i) was clobbered. */
.text
.p2align 2
.globl MLK_ASM_NAMESPACE(sme2_phase1_check_abi)
MLK_ASM_NAMESPACE(sme2_phase1_check_abi):
    stp x29, x30, [sp, #-80]!
    mov x29, sp
    stp d8, d9, [sp, #16]
    stp d10, d11, [sp, #32]
    stp d12, d13, [sp, #48]
    stp d14, d15, [sp, #64]

    /* Distinct full-width values catch partial and whole-register writes. */
.macro SET_CANARY REG, VALUE
    mov x9, #\VALUE
    movk x9, #0x1234, lsl #16
    movk x9, #0xabcd, lsl #32
    movk x9, #0x5678, lsl #48
    fmov \REG, x9
.endm
    SET_CANARY d8, 0x7008
    SET_CANARY d9, 0x7009
    SET_CANARY d10, 0x700a
    SET_CANARY d11, 0x700b
    SET_CANARY d12, 0x700c
    SET_CANARY d13, 0x700d
    SET_CANARY d14, 0x700e
    SET_CANARY d15, 0x700f

    bl MLK_ASM_NAMESPACE(sme2_cross_matrix_d3_body)

    mov w0, #0
.macro CHECK_CANARY REG, VALUE, BIT
    mov x9, #\VALUE
    movk x9, #0x1234, lsl #16
    movk x9, #0xabcd, lsl #32
    movk x9, #0x5678, lsl #48
    fmov x10, \REG
    cmp x10, x9
    cset w11, ne
    orr w0, w0, w11, lsl #\BIT
.endm
    CHECK_CANARY d8, 0x7008, 0
    CHECK_CANARY d9, 0x7009, 1
    CHECK_CANARY d10, 0x700a, 2
    CHECK_CANARY d11, 0x700b, 3
    CHECK_CANARY d12, 0x700c, 4
    CHECK_CANARY d13, 0x700d, 5
    CHECK_CANARY d14, 0x700e, 6
    CHECK_CANARY d15, 0x700f, 7

    ldp d8, d9, [sp, #16]
    ldp d10, d11, [sp, #32]
    ldp d12, d13, [sp, #48]
    ldp d14, d15, [sp, #64]
    ldp x29, x30, [sp], #80
    ret

#if defined(__ELF__)
.section .note.GNU-stack,"",@progbits
#endif
