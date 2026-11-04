#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter

from benchlib import (
    compile_benchmark,
    paired_abba,
    parse_run,
    repo_root,
    write_metadata,
)


def run_with_progress(
    *,
    label: str,
    progress_interval: float,
    **paired_kwargs,
) -> None:
    """Run one ABBA comparison; benchlib reports progress at block boundaries."""
    started = perf_counter()
    print(f"\n[start] {label}", flush=True)
    status = "failed"
    try:
        paired_abba(
            label=label,
            progress_interval=progress_interval,
            **paired_kwargs,
        )
        status = "completed"
    finally:
        elapsed = perf_counter() - started
        print(f"[{status}] {label}: {elapsed:.1f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SME2 vs fair prepared-NTT batch benchmark"
    )
    ap.add_argument("--root")
    ap.add_argument("--mode", choices=["quick", "final"], default="quick")
    ap.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Report completed ABBA blocks at this interval (default: 10; 0 disables)",
    )
    ap.add_argument("--output", type=Path, help="new result directory")
    ap.add_argument("--batches", help="comma-separated multiples of 16, maximum 256")
    args = ap.parse_args()
    if not math.isfinite(args.progress_interval) or args.progress_interval < 0:
        ap.error("--progress-interval must be a finite number >= 0")

    root = repo_root(__file__, args.root)
    source = Path(__file__).with_name("bench_ntt_fairness.c")

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
        / ("fairness-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    result.mkdir(parents=True, exist_ok=False)

    total_started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="mlkem_sme2_fairness_") as td:
        binary = Path(td) / "bench"

        print("[start] compile benchmark", flush=True)
        t0 = perf_counter()
        command = compile_benchmark(root, source, binary)
        print(f"[completed] compile benchmark: {perf_counter() - t0:.1f} s", flush=True)

        print("[start] metadata probe", flush=True)
        t0 = perf_counter()
        _, meta = parse_run(binary, root, 3, 128, 1, 0)
        print(f"[completed] metadata probe: {perf_counter() - t0:.1f} s", flush=True)

        write_metadata(root, binary, result, command, {
            "mode": args.mode, "batches": batches, "samples": samples,
            "inner": inner, "warmup": warmup, "bootstrap_reps": boots,
            "workspace": meta,
            "comparison_seeds": [0x52520034],
        })

        print("=== fairness locks ===", flush=True)
        print("same executable / same source revision / same front+finish", flush=True)
        print("public/key preparation is outside the timed batch paths", flush=True)
        print(
            "fair NTT reuses one sp_ntt + mulcache scratch across the batch "
            "and wipes it once at the API boundary",
            flush=True,
        )
        print(f"key bytes: SME2={meta['sme2_key']} NTT={meta['ntt_key']}", flush=True)
        print(
            f"workspace bytes: SME2={meta['sme2_ws']} "
            f"NTT-legacy={meta['ntt_legacy_ws']} "
            f"NTT-fair={meta['ntt_fair_ws']}",
            flush=True,
        )

        common = dict(
            binary=binary,
            root=root,
            batches=batches,
            samples=samples,
            inner=inner,
            warmup=warmup,
            result_dir=result,
            bootstrap_reps=boots,
        )
        print(
            f"\nconfiguration: mode={args.mode}, batches={len(batches)}, "
            f"samples={samples}, inner={inner}, warmup={warmup}, bootstrap={boots}",
            flush=True,
        )

        run_with_progress(
            label="SME2 A vs fair prepared NTT B",
            progress_interval=args.progress_interval,
            variant_a=1,
            variant_b=3,
            seed=0x52520034,
            **common,
        )

    print(f"\nresults: {result}", flush=True)
    print(f"total elapsed: {perf_counter() - total_started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
