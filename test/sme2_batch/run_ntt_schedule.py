#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from datetime import datetime
from pathlib import Path

from benchlib import (
    compile_benchmark,
    paired_abba,
    parse_run,
    repo_root,
    write_metadata,
)


def main():
    ap = argparse.ArgumentParser(
        description="Cross-request AArch64 NTT scheduling ablation"
    )
    ap.add_argument("--root")
    ap.add_argument("--mode", choices=["quick", "final"], default="quick")
    ap.add_argument("--output", type=Path, help="new result directory")
    ap.add_argument("--batches", help="comma-separated multiples of 16, maximum 256")
    args = ap.parse_args()

    root = repo_root(__file__, args.root)
    source = Path(__file__).with_name("bench_ntt_schedule.c")

    if args.mode == "quick":
        batches = [80, 128, 160, 192, 224, 256]
        samples, inner, warmup, boots = 31, 8, 2, 5000
    else:
        batches = [80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]
        samples, inner, warmup, boots = 301, 8, 2, 20000

    if args.batches is not None:
        try:
            batches = [int(n) for n in args.batches.split(",")]
        except ValueError:
            ap.error("--batches must contain integers")
        if (not batches or len(set(batches)) != len(batches) or
                any(n < 16 or n > 256 or n % 16 for n in batches)):
            ap.error("--batches must be distinct multiples of 16 from 16 to 256")

    result = args.output.resolve() if args.output else (
        root
        / "test/build/sme2_batch"
        / ("ntt-schedule-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    result.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix="mlkem_ntt_schedule_") as td:
        binary = Path(td) / "bench"
        command = compile_benchmark(root, source, binary)

        _, meta = parse_run(binary, root, 6, 128, 1, 0)
        write_metadata(root, binary, result, command, {
            "mode": args.mode, "batches": batches, "samples": samples,
            "inner": inner, "warmup": warmup, "bootstrap_reps": boots,
            "workspace": meta,
            "comparison_seeds": [0x52530041, 0x52530081, 0x52530161, 0x52530414, 0x52530815, 0x52531616],
        })

        print("=== schedule-ablation locks ===")
        print("same executable / same source revision / same front+finish")
        print(
            "all NTT variants use the same existing mlkem-native "
            "NTT/basemul/INTT primitives"
        )
        print("only cross-request arithmetic scheduling changes")
        print(
            f"workspace bytes: SME2={meta['sme2_ws']} "
            f"fair={meta['ntt_fair_ws']} "
            f"phase4={meta['ntt_phase4_ws']} "
            f"phase8={meta['ntt_phase8_ws']} "
            f"phase16={meta['ntt_phase16_ws']}"
        )

        comparisons = [
            (4, 3, "NTT phase4 A vs fair request-major NTT B", 0x52530041),
            (5, 3, "NTT phase8 A vs fair request-major NTT B", 0x52530081),
            (6, 3, "NTT phase16 A vs fair request-major NTT B", 0x52530161),
            (1, 4, "SME2 A vs NTT phase4 B", 0x52530414),
            (1, 5, "SME2 A vs NTT phase8 B", 0x52530815),
            (1, 6, "SME2 A vs NTT phase16 B", 0x52531616),
        ]

        for va, vb, label, seed in comparisons:
            paired_abba(
                binary=binary,
                root=root,
                batches=batches,
                samples=samples,
                inner=inner,
                warmup=warmup,
                variant_a=va,
                variant_b=vb,
                label=label,
                seed=seed,
                result_dir=result,
                bootstrap_reps=boots,
            )

    print(f"\nresults: {result}")


if __name__ == "__main__":
    main()
