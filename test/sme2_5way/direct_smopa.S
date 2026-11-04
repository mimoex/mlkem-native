/* Direct 256x256 negacyclic convolution for four ML-KEM-768 matrix-vector
 * outputs and 16 independent requests.  Apple M4 / SVL=512 only.
 * Inputs: x0 = prepared Toeplitz int16 [2][8][3][128][4][32]
 *         x1 = packed input int16 [3][128][32]
 *         x2 = output int32 [8][4][2][256]
 * The function is called by a locally-streaming __arm_inout("za") bridge.
 * Copyright: original experimental benchmark source, not mlkem-native. */
#if defined(__APPLE__) && defined(__aarch64__)
.text
.p2align 2
.macro BW_DIRECT_PAIR
    zero {za0.s, za1.s}
    zero {za2.s, za3.s}
    mov x27, #384                 /* 3 polynomial terms * 128 coefficient pairs */
.Linner_\@:
    ld1h {z9.h}, p0/z, [x4]
    ld1h {z1.h}, p0/z, [x8]
    ld1h {z2.h}, p0/z, [x8, #1, mul vl]
    ld1h {z3.h}, p0/z, [x8, #2, mul vl]
    ld1h {z4.h}, p0/z, [x8, #3, mul vl]
    smopa za0.s, p0/m, p0/m, z1.h, z9.h
    smopa za1.s, p0/m, p0/m, z2.h, z9.h
    smopa za2.s, p0/m, p0/m, z3.h, z9.h
    smopa za3.s, p0/m, p0/m, z4.h, z9.h
    add x4, x4, #64
    add x8, x8, #256
    subs x27, x27, #1
    b.ne .Linner_\@
.endm
.macro BW_DIRECT_STORE TILE
    mov w12, #0
.Lstore_\@:
    mova {z28.s-z31.s}, \TILE\().s[w12, 0:3]
    st1w {z28.s}, p1, [x2]
    st1w {z29.s}, p1, [x2, #1, mul vl]
    st1w {z30.s}, p1, [x2, #2, mul vl]
    st1w {z31.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    add w12, w12, #4
    cmp w12, #16
    b.lt .Lstore_\@
.endm
.globl _bench5_direct_smopa
_bench5_direct_smopa:
    sub sp, sp, #0x90
    stp x19, x20, [sp, #0]
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    ptrue p0.h
    ptrue p1.s
    mov x19, x0                  /* public (pair0, rowblock0) */
    mov x20, x1                  /* packed 16-request vector */
    mov x21, x2                  /* raw rowblock0 */
    mov x22, #8                  /* 8 blocks x 32 output coefficients */
.Louter:
    mov x4, x20
    mov x8, x19
    BW_DIRECT_PAIR
    mov x2, x21
    BW_DIRECT_STORE za0h
    BW_DIRECT_STORE za1h
    BW_DIRECT_STORE za2h
    BW_DIRECT_STORE za3h

    mov x4, x20
    mov x11, #192
    add x8, x19, x11, lsl #12   /* pair1: 786432-byte stride */
    BW_DIRECT_PAIR
    add x2, x21, #1, lsl #12   /* second pair: raw block +4096 bytes */
    BW_DIRECT_STORE za0h
    BW_DIRECT_STORE za1h
    BW_DIRECT_STORE za2h
    BW_DIRECT_STORE za3h

    mov x11, #24
    add x19, x19, x11, lsl #12  /* next 32-row public block: +98304 */
    add x21, x21, #2, lsl #12  /* next raw block: +8192 */
    subs x22, x22, #1
    b.ne .Louter
    ldp x19, x20, [sp, #0]
    ldp x21, x22, [sp, #16]
    ldp x23, x24, [sp, #32]
    ldp x25, x26, [sp, #48]
    ldp x27, x28, [sp, #64]
    add sp, sp, #0x90
    ret
#endif
