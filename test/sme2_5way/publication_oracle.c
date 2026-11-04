/*
 * Publication-grade full correctness gate for the SME2 5-way benchmark.
 *
 * Reuses the exact benchmark implementations from bench5.c, but replaces its
 * main() and checks every request / output / coefficient against the independent
 * 64-bit schoolbook negacyclic oracle in math_core.h.
 *
 * This file is test-only. It does not modify production mlkem-native sources.
 */
#define main bench5_hidden_main
#include "bench5.c"
#undef main

static int pub5_build_oracle(b5_state *s, unsigned n)
{
    unsigned r;
    for (r = 0; r < n; r++)
        bw_oracle_one((const int16_t (*)[3][256])s->pub,
                      s->sec[r], s->reference[r]);
    return 0;
}

static int pub5_check_mode_against_oracle(
    b5_state *s, int mode, unsigned n, unsigned trial)
{
    unsigned r, d, i;
    if (b5_eval(s, mode, n) != 0) {
        fprintf(stderr, "FAIL publication oracle: eval mode=%s trial=%u N=%u\n",
                b5_names[mode], trial, n);
        return -1;
    }

    for (r = 0; r < n; r++)
        for (d = 0; d < 4; d++)
            for (i = 0; i < 256; i++) {
                const int got = bw_mod(s->out[r][d][i]);
                const int want = bw_mod(s->reference[r][d][i]);
                if (got != want) {
                    fprintf(stderr,
                            "FAIL publication oracle mode=%s trial=%u N=%u "
                            "req=%u d=%u coeff=%u got=%d want=%d\n",
                            b5_names[mode], trial, n, r, d, i, got, want);
                    return -1;
                }
            }
    return 0;
}

int main(int argc, char **argv)
{
    static const unsigned ns[] = {16u, 80u, 256u};
    static const uint64_t seeds[] = {
        UINT64_C(0x5AE21234),
        UINT64_C(0x9E3779B97F4A7C15),
        UINT64_C(0xD1B54A32D192ED03)
    };
    b5_state s;
    unsigned trial, ni;
    int mode;

    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--preflight-only"))) {
        fprintf(stderr, "Usage: %s [--preflight-only]\n", argv[0]);
        return 2;
    }
    if (b5_preflight() != 0)
        return 1;
    if (argc == 2)
        return 0;

    b5_state_init(&s);

    for (trial = 0; trial < sizeof(seeds) / sizeof(seeds[0]); trial++) {
        bw_fill_input(s.pub, s.sec, BW_MAX, seeds[trial]);
        if (b5_prepare(&s) != 0) {
            fprintf(stderr, "FAIL publication oracle: public prepare trial=%u\n", trial);
            b5_state_free(&s);
            return 1;
        }

        for (ni = 0; ni < sizeof(ns) / sizeof(ns[0]); ni++) {
            const unsigned n = ns[ni];
            pub5_build_oracle(&s, n);
            for (mode = 0; mode < B5_MODES; mode++) {
                if (pub5_check_mode_against_oracle(&s, mode, n, trial) != 0) {
                    b5_state_free(&s);
                    return 1;
                }
            }
            printf("PASS publication oracle trial=%u N=%u: "
                   "A/B/C/D/E == independent schoolbook for all "
                   "%u requests x 4 outputs x 256 coefficients\n",
                   trial, n, n);
            fflush(stdout);
        }
    }

    b5_state_free(&s);
    puts("PASS publication oracle: 3 input seeds; N=16,80,256; all five methods");
    return 0;
}
