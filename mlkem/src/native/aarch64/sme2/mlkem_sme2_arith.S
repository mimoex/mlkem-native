#include "../../../common.h"

#if defined(MLK_ARITH_BACKEND_AARCH64) && \
    defined(MLK_SYS_AARCH64_SME2) && \
    !defined(MLK_CONFIG_MULTILEVEL_NO_SHARED) && \
    (defined(MLK_CONFIG_MULTILEVEL_WITH_SHARED) || MLKEM_K == 3)

/*
 * ML-KEM-768向けSME2行列積
 *
 * 3段打切りNTT後に残る32係数多項式乗算を行列積として表現し、
 * SME2のSMOPA命令を用いて16要求をまとめて処理する。
 *
 * 1 group = 16要求
 * 1 groupにつき8個のdegree-32残存多項式を処理する。
 *
 * 入力:
 *   x0 : 事前計算済み公開値（public_gemm）の先頭
 *   x1 : 16要求単位に配置した秘密値（secret_gemm）の先頭
 *   x2 : 32-bit SMOPA累積結果（raw_out）の先頭
 *   x3 : 処理するgroup数
 *
 * SMOPAでは16-bit × 16-bitの積を32-bit ZAタイルへ累積する。
 */

.text
.p2align 2


/*
 * SME2カーネルで使用する述語を初期化する。
 *
 * p0.h:
 *   制御レジスタ。
 *   secretのLD1HおよびSMOPAで全halfword laneを有効にする。
 *
 * pn8.h:
 *   SME2 multi-vector LD1H用の制御・カウントレジスタ。
 *
 * pn9.s:
 *   SME2 multi-vector ST1W用の制御・カウントレジスタ。
 */
.macro S5234_ENTER
    ptrue p0.h
    ptrue pn8.h
    ptrue pn9.s
.endm


/*
 * 1組の行列積を計算する。
 *
 * x4:
 *   secret側の現在位置。
 *   1反復で64 B（512 bit = z4.h 1本）読み込む。
 *
 * x8:
 *   public側の現在位置。
 *   1反復で256 B（512 bit × 4本）読み込む。
 *
 * za0.s～za3.s:
 *   4個の32-bit ZAタイル。
 *   4系列の積和結果を並列に保持する。
 *
 * 48回ループでSMOPAを累積する。
 */
.macro S5234_PAIR_GEMM
    /* 4個のZAタイルをゼロ埋め */
    zero {za0.s, za1.s}
    zero {za2.s, za3.s}

    mov x27, #48    // 32係数を２係数ごとに処理し、k=3のため、48回のループ

.Ls5234_kp_\@:
    /*
     * secret側から16-bit係数を1ベクトル読み込む。
     * M4のSVL=512 bitでは32個の16bit(.h(ハーフワード))を保持する。
     */
    /* z0: 出力Aの係数0~15
     * z1: 出力Aの係数16~31
     * z2: 出力Bの係数0~15
     * z3: 出力Bの係数16~31
     * z4: 16要求それぞれの2係数
     * z28-z31: 一時データ保存用;
     */
    ld1h {z4.h}, p0/z, [x4] // 16要求 × 2係数

    /*
     * public側から連続する4本のZレジスタを一括ロードする。
     *
     * 4 × 512 bit = 256 B
     */
    ld1h {z0.h-z3.h}, pn8/z, [x8]   // 公開値4ベクトル

    /*
     * public × secretの外積を4個のZAタイルへ並列に累積する。
     *
     * 各SMOPAは16-bit × 16-bitを2-wayで積和し、
     * 32-bitのZAタイルへ加算する。
     */
    smopa za0.s, p0/m, p0/m, z0.h, z4.h
    smopa za1.s, p0/m, p0/m, z1.h, z4.h
    smopa za2.s, p0/m, p0/m, z2.h, z4.h
    smopa za3.s, p0/m, p0/m, z3.h, z4.h

    /* 次のsecret/publicデータへ進む。 */
    add x4, x4, #64
    add x8, x8, #256

    subs x27, x27, #1
    b.ne .Ls5234_kp_\@
.endm


/*
 * 1個のZAタイルをraw出力へ書き出す。
 *
 * ZA:
 *   書き出すZAタイル。
 *
 * PTR:
 *   32-bit raw出力先。
 *
 * ZAタイルから4本のZレジスタをまとめて取り出し、
 * SME2 multi-vector ST1Wで256 Bずつ出力する。
 *
 * 512-bit SVLでは32-bit要素が16個なので、
 * 16×16のZA.sタイル全体は
 *
 *     16 × 16 × 4 B = 1024 B
 *
 * となる。
 */
.macro S5234_STORE_TILE ZA, PTR
    mov w12, #0     // w12はZAから読み出す行の指定

.Ls5234_store_\@:
    /*
     * ZAから連続する4本分の32-bitデータを
     * z28～z31へまとめて取り出す。
     */
    mova {z28.s-z31.s}, \ZA\().s[w12, 0:3]

    /*
     * 4本のZレジスタをmulti-vector storeで一括出力する。
     * 1回あたり4 × 64 B = 256 B。
     */
    st1w {z28.s-z31.s}, pn9, [\PTR]

    add \PTR, \PTR, #256    // 出力先ポインタを256バイト進める
    add w12, w12, #4        // ZAの読み出し位置を4行進める

    cmp w12, #16        //ZAの読み出し位置を16と比較
    b.lt .Ls5234_store_\@   // 16未満なら繰り返す(4ループ)
.endm


/*
 * SME2 residual-domain kernel本体。
 *
 * NAME:
 *   生成する関数名。
 *
 * STORE:
 *   非0の場合、ZAの計算結果をraw_outへ書き出す。
 *
 * 外側ループ:
 *   16要求 = 1 group
 *
 * 内側ループ:
 *   3段打切りNTTで得られる8個のdegree-32残存多項式
 */
.macro S5234_FUNC NAME, STORE

.p2align 2
.globl \NAME
\NAME:
    /* group数が0なら何もせず終了する。 */
    cbz x3, .Ls5234_done_\@

    /*
     * 汎用レジスタをスタックに退避する。
     */
    sub sp, sp, #0x90
    stp x19, x20, [sp, #0]
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]

    S5234_ENTER

    /*
     * 関数引数をgroupループ用レジスタへ保持する。
     */
    mov x19, x0              /* 事前計算済み公開値の先頭 */
    mov x20, x1              /* secret groupの先頭 */
    mov x21, x2              /* raw出力groupの先頭 */
    mov x22, x3              /* 残りgroup数 */


/*
 * 16要求を1 groupとして処理する。
 */
.Ls5234_group_\@:
    mov x23, x19             /* 現在のpublic block先頭 */
    mov x24, x20             /* 現在のsecret block先頭 */
    mov x25, x21             /* 現在のraw出力block先頭 */

    /*
     * 3段打切りNTTによって得られる8個の
     * degree-32残存多項式を順に処理する。
     */
    mov x26, #8


.Ls5234_block_\@:
    /*
     * pair0: dot0, dot1
     *
     * 同じsecret blockに対する前半の公開値を用いて
     * 4個のZAタイルへ積和する。
     */
    mov x4, x24
    mov x8, x23
    S5234_PAIR_GEMM

    .if \STORE
        /*
         * 4個のZAタイルを書き出す。
         *
         * 1 ZA.s tile = 1024 B
         * 4 tiles      = 4096 B
         */
        mov x2, x25
        S5234_STORE_TILE za0h, x2
        S5234_STORE_TILE za1h, x2
        S5234_STORE_TILE za2h, x2
        S5234_STORE_TILE za3h, x2
    .endif


    /*
     * pair1: dot2, dot3
     *
     * secret側はpair0と同じblockを再利用する。
     * public側はpair0から98304 B先の領域を使用する。
     *
     * 24 << 12 = 98304 B
     * 後半2出力の計算
     */
    mov x4, x24
    mov x11, #24
    add x8, x23, x11, lsl #12
    S5234_PAIR_GEMM

    .if \STORE
        /*
         * pair1のraw出力はpair0の4096 B後に配置する。
         */
        add x2, x25, #1, lsl #12
        S5234_STORE_TILE za0h, x2
        S5234_STORE_TILE za1h, x2
        S5234_STORE_TILE za2h, x2
        S5234_STORE_TILE za3h, x2
    .endif


    /*
     * 次の32係数多項式のブロックへ進む。
     *
     * public : +12288 B
     * secret : +3072 B
     * raw    : +8192 B
     */
    add x23, x23, #3, lsl #12
    add x24, x24, #3072
    add x25, x25, #2, lsl #12

    subs x26, x26, #1
    b.ne .Ls5234_block_\@


    /*
     * 8 blockの処理が終了したので、次の16要求groupへ進む。
     *
     * secret:
     *   6 << 12 = 24576 B / group
     *
     * raw:
     *   16 << 12 = 65536 B / group
     */
    mov x11, #6
    add x20, x20, x11, lsl #12

    mov x11, #16
    add x21, x21, x11, lsl #12

    subs x22, x22, #1
    b.ne .Ls5234_group_\@


    /* callee-savedレジスタを復元する。 */
    ldp x19, x20, [sp, #0]
    ldp x21, x22, [sp, #16]
    ldp x23, x24, [sp, #32]
    ldp x25, x26, [sp, #48]
    ldp x27, x28, [sp, #64]
    add sp, sp, #0x90

.Ls5234_done_\@:
    ret

.endm


/*
 * 3段打切りNTT後の32x32多項式行列積を実行し、
 * ZA累積結果をraw_outへ保存する実体関数を生成する。
 */
S5234_FUNC MLK_ASM_NAMESPACE(sme2_cross_matrix_d3_body), 1


/* BEGIN EXPERIMENT_ZA_PACK_D3 */
/*
 * SMOPAに入力する要求側データを、要求ごとの配置から係数ペアごとの配置へ変更する
 *
 * 入力　要求ごとに並んでいる係数
 * 要求0: y0,0 y0,1 y0,2 y0,3 ... y0,31
 * 要求1: y1,0 y1,1 y1,2 y1,3 ... y1,31
 * ...
 * 
 * 出力　各要求の2係数ずつのペアを作る
 * pair0: y0,0 y0,1 y1,0 y1,1 ... y15,0 y15,1
 * pair1: y0,2 y0,3 y1,2 y1,3 ... y15,2 y15,3
 * ...
 * 
 */
.p2align 2
.globl MLK_ASM_NAMESPACE(sme2_pack_secret_d3_body)
MLK_ASM_NAMESPACE(sme2_pack_secret_d3_body):
    /* 16個の32-bit要素すべてを有効化 */
    ptrue p0.s

    mov x4, x0                 // 入力側のブロック先頭 
    mov x5, x1                 // 出力側のブロック先頭
    mov x6, #8                 // 32x32係数 8ブロックのカウンタ
.Lsme2_pack_block:      
    mov x7, x4                 // 入力側
    mov x8, x5                 // 出力側
    mov x9, #3                 // ML-KEM-768のk=3用カウンタ
.Lsme2_pack_poly:       // 32x32多項式の並び替え
    /* ZAの水平方向へ16要求をロード */
    mov x10, x7
    mov w12, #0
.Lsme2_pack_row:
    ld1w {za0h.s[w12, 0]}, p0/z, [x10]  // メモリからZAに行でロード(za0hは行、.sで32bit単位で扱うため16bit変数2個)
    add x10, x10, #1536       // 3多項式 * 256係数 * 2バイト/req
    add w12, w12, #1
    cmp w12, #16
    b.lo .Lsme2_pack_row

    /* 垂直方向から読み出して保存する */
    mov x10, x8
    mov w12, #0
.Lsme2_pack_col:
    st1w {za0v.s[w12, 0]}, p0, [x10]    // ZAからメモリに列でストア(ZA0Vは垂直、.s32bit単位のため、y0,0 y0,1 y1,0 y1,1のようにできる)
    add x10, x10, #64
    add w12, w12, #1
    cmp w12, #16
    b.lo .Lsme2_pack_col

    add x7, x7, #512      // 次の256係数にポインタを移動
    add x8, x8, #1024     // 次の出力に移動
    subs x9, x9, #1         // 次の多項式の処理を行う k--
    b.ne .Lsme2_pack_poly

    add x4, x4, #64           // 次の32z32多項式に移動
    add x5, x5, #3072       
    subs x6, x6, #1
    b.ne .Lsme2_pack_block
    ret
/* END EXPERIMENT_ZA_PACK_D3 */


.p2align 2
.globl MLK_ASM_NAMESPACE(sme2_transpose_output_d3_body)
MLK_ASM_NAMESPACE(sme2_transpose_output_d3_body):
    ptrue p0.h, vl16        /* 入力: 16要求（16 halfword）だけロード */
    ptrue p1.h              /* 出力: 32係数（32 halfword）すべて保存 */

    mov w4, #0              /* d = 0..3（3個のb + 1個のv） */
.Lsme2_out_d:
    add x6, x0, x4, lsl #10 /* inverse_h[0][d][0][0]: d*1024 B */
    mov w5, #0              /* b = 0..7 */
    mov x8, #0              /* 出力係数のbyte offset = 64*b */
.Lsme2_out_b:
    mov x7, x6              /* 当該[b][d]先頭 */
    mov w12, #0             /* t = 0..31 */
.Lsme2_out_load:
    /* 16要求の係数tをZAの水平スライスtへ読み込む。                */
    /* 残り16 laneは /z によって0となる。                           */
    ld1h {za0h.h[w12, 0]}, p0/z, [x7]
    add x7, x7, #32        /* 16要求 * 2 B = 32 B/係数行 */
    add w12, w12, #1
    cmp w12, #32
    b.lo .Lsme2_out_load

    add x9, x1, x4, lsl #7 /* &out[d][0]: d*16ポインタ*8 B */
    mov w12, #0            /* r = 0..15 */
.Lsme2_out_store:
    ldr x10, [x9], #8      /* out[d][r] */
    add x10, x10, x8       /* &out[d][r][32*b] */
    /* ZAの垂直スライスrを読み出し、32係数を一括保存。              */
    st1h {za0v.h[w12, 0]}, p1, [x10]
    add w12, w12, #1
    cmp w12, #16
    b.lo .Lsme2_out_store

    add x6, x6, #4096      /* 次のbへ: 4出力 * 1024 B */
    add x8, x8, #64        /* 次のbへ: 32係数 * 2 B */
    add w5, w5, #1
    cmp w5, #8
    b.lo .Lsme2_out_b

    add w4, w4, #1
    cmp w4, #4
    b.lo .Lsme2_out_d
    ret


#endif
