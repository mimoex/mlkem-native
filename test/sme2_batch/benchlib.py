#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter


def repo_root(script_file: str, override: str | None) -> Path:
    if override:
        root = Path(override).expanduser().resolve()
    else:
        root = Path(script_file).resolve().parents[2]
    if not (root / "mlkem/mlkem_native.c").is_file():
        raise RuntimeError(f"not an mlkem-native repository root: {root}")
    return root


def run(cmd, *, cwd: Path, quiet: bool = False):
    p = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=180,
    )
    if not quiet:
        print("+", " ".join(str(x) for x in cmd))
        if p.stdout:
            print(p.stdout, end="")
        if p.stderr:
            print(p.stderr, end="", file=sys.stderr)
    if p.returncode != 0:
        if quiet:
            if p.stdout:
                print(p.stdout, end="")
            if p.stderr:
                print(p.stderr, end="", file=sys.stderr)
        raise RuntimeError("command failed: " + " ".join(str(x) for x in cmd))
    return p


def compile_benchmark(root: Path, source: Path, binary: Path) -> list[str]:
    command = [
        "clang",
        "-O3",
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-mcpu=apple-m4",
        "-I",
        root / "mlkem",
        "-I",
        root / "mlkem/src",
        "-DMLK_CONFIG_PARAMETER_SET=768",
        "-DMLK_CONFIG_USE_NATIVE_BACKEND_ARITH",
        '-DMLK_CONFIG_ARITH_BACKEND_FILE="native/aarch64/meta.h"',
        "-DMLK_CONFIG_NO_RANDOMIZED_API",
        source,
        root / "test/sme2_batch/prepared_ntt.c",
        root / "mlkem/mlkem_native.c",
        root / "mlkem/mlkem_native_asm.S",
        "-o",
        binary,
    ]
    run(command, cwd=root)
    return [str(x) for x in command]


def write_metadata(root: Path, binary: Path, result: Path,
                   command: list[str], conditions: dict) -> None:
    sources = sorted(p for folder in (root / "mlkem", root / "test/sme2_batch")
                     for p in folder.rglob("*")
                     if p.suffix in (".c", ".h", ".S", ".py", ".inc"))
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": run(["git", "rev-parse", "HEAD"], cwd=root, quiet=True).stdout.strip(),
        "source_status": run(["git", "status", "--porcelain=v1"], cwd=root, quiet=True).stdout,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "compiler": run([command[0], "--version"], cwd=root, quiet=True).stdout,
        "compile_command": command,
        "platform": platform.platform(),
        "cpu": run(["sysctl", "-n", "machdep.cpu.brand_string"], cwd=root, quiet=True).stdout.strip(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "svl_bits": 512,
        "conditions": conditions,
        "input": {"key_seed": "0x243f6a88", "enc_seed": "0x85a308d3 ^ (0x9e3779b9 * (i + 1))"},
        "timing": "mach_continuous_time; ns/request; preparation excluded; same executable and inputs; ABBA/BAAB",
    }
    (result / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_run(
    binary: Path,
    root: Path,
    variant: int,
    n: int,
    inner: int,
    warmup: int,
):
    p = run(
        [binary, variant, n, inner, warmup],
        cwd=root,
        quiet=True,
    )
    line = p.stdout.strip().splitlines()[-1]
    vals = {}
    for item in line.split():
        if "=" in item:
            k, v = item.split("=", 1)
            vals[k] = v
    return float(vals["ns_per_req"]), vals


def percentile(xs, q):
    ys = sorted(xs)
    pos = q * (len(ys) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ys[lo]
    f = pos - lo
    return ys[lo] * (1.0 - f) + ys[hi] * f


def bootstrap_median_ci(xs, reps: int, seed: int):
    rng = random.Random(seed)
    n = len(xs)
    boots = []
    for _ in range(reps):
        boots.append(
            statistics.median([xs[rng.randrange(n)] for _ in range(n)])
        )
    return percentile(boots, 0.025), percentile(boots, 0.975)


def paired_abba(
    *,
    binary: Path,
    root: Path,
    batches,
    samples: int,
    inner: int,
    warmup: int,
    variant_a: int,
    variant_b: int,
    label: str,
    seed: int,
    result_dir: Path,
    bootstrap_reps: int,
    progress_interval: float = 10.0,
):
    """ABBA測定を実行し、ブロック境界で進捗を表示する。

    progress_interval=0で進捗表示を無効化できる。
    測定順序・統計処理・CSV形式には変更を加えない。
    """
    if not math.isfinite(progress_interval) or progress_interval < 0:
        raise ValueError("progress_interval must be a finite number >= 0")

    rng = random.Random(seed)
    stats = {
        n: {"a": [], "b": [], "d": [], "ratio": [], "wins": 0}
        for n in batches
    }
    rows = []

    total_runs = samples * len(batches) * 4
    print(f"\n=== paired: {label} ===", flush=True)
    print("paired_delta = A - B; negative means A wins", flush=True)
    print(
        "publication-stable = bootstrap 95% CI excludes zero "
        "AND all 3 chronological epoch medians have the same sign",
        flush=True,
    )
    print(
        f"[measure] {samples} blocks, {len(batches)} batch sizes, "
        f"{total_runs:,} subprocess runs",
        flush=True,
    )

    started = perf_counter()
    next_progress = started + progress_interval
    for block in range(samples):
        order_n = list(batches)
        rng.shuffle(order_n)

        for visit, n in enumerate(order_n):
            order = (
                (variant_b, variant_a, variant_a, variant_b)
                if ((block + n // 16) & 1)
                else (variant_a, variant_b, variant_b, variant_a)
            )
            av, bv = [], []

            for slot, variant in enumerate(order):
                value, _ = parse_run(
                    binary, root, variant, n, inner, warmup
                )
                tag = "A" if variant == variant_a else "B"
                (av if tag == "A" else bv).append(value)
                rows.append(
                    {
                        "comparison": label,
                        "batch": n,
                        "block": block,
                        "visit": visit,
                        "slot": slot,
                        "variant": tag,
                        "ns_per_req": value,
                    }
                )

            a = statistics.fmean(av)
            b = statistics.fmean(bv)
            d = a - b
            st = stats[n]
            st["a"].append(a)
            st["b"].append(b)
            st["d"].append(d)
            st["ratio"].append(b / a)
            if d < 0:
                st["wins"] += 1

        # 1ブロック内の全バッチ計測が終わった位置でのみ進捗を表示する。
        # Nの順序はブロックごとにシャッフルされるため、N単位の完了表示は不適切。
        if progress_interval > 0:
            now = perf_counter()
            if now >= next_progress or block + 1 == samples:
                done_blocks = block + 1
                done_runs = done_blocks * len(batches) * 4
                elapsed = now - started
                print(
                    f"[progress] {label}: "
                    f"block {done_blocks}/{samples} "
                    f"({100.0 * done_blocks / samples:.1f}%), "
                    f"runs {done_runs:,}/{total_runs:,}, "
                    f"elapsed {elapsed:.1f} s",
                    flush=True,
                )
                next_progress = now + progress_interval

    elapsed_measure = perf_counter() - started
    print(
        f"[measure completed] {label}: {elapsed_measure:.1f} s; "
        "computing bootstrap CIs and writing CSV...",
        flush=True,
    )

    summary = []
    for n in batches:
        st = stats[n]
        med_a = statistics.median(st["a"])
        med_b = statistics.median(st["b"])
        med_d = statistics.median(st["d"])
        med_ratio = statistics.median(st["ratio"])
        lo, hi = bootstrap_median_ci(
            st["d"], bootstrap_reps, seed ^ n ^ 0x6A09E667
        )

        cut1 = samples // 3
        cut2 = 2 * samples // 3
        epochs = [
            statistics.median(st["d"][:cut1]),
            statistics.median(st["d"][cut1:cut2]),
            statistics.median(st["d"][cut2:]),
        ]

        if hi < 0:
            direction = "A win"
            stable = all(x < 0 for x in epochs)
        elif lo > 0:
            direction = "B win"
            stable = all(x > 0 for x in epochs)
        else:
            direction = "inconclusive"
            stable = False

        decision = (
            direction
            if direction == "inconclusive"
            else direction
            + (", epoch-stable" if stable else ", EPOCH-UNSTABLE")
        )

        epoch_text = ",".join(f"{x:+.3f}" for x in epochs)
        print(
            f"N={n:3d} A={med_a:10.3f} B={med_b:10.3f} "
            f"delta={med_d:+9.3f} 95%CI=[{lo:+.3f},{hi:+.3f}] "
            f"paired_speedup={med_ratio:.5f}x "
            f"wins={st['wins']}/{samples} "
            f"epochs=[{epoch_text}] {decision}",
            flush=True,
        )

        summary.append(
            {
                "comparison": label,
                "batch": n,
                "a_median_ns_req": med_a,
                "b_median_ns_req": med_b,
                "paired_delta_median_ns_req": med_d,
                "ci_low": lo,
                "ci_high": hi,
                "paired_speedup_median": med_ratio,
                "wins": st["wins"],
                "samples": samples,
                "epoch1": epochs[0],
                "epoch2": epochs[1],
                "epoch3": epochs[2],
                "decision": decision,
            }
        )

    safe = "".join(
        c if c.isalnum() else "_" for c in label.lower()
    ).strip("_")
    raw_path = result_dir / f"{safe}_raw.csv"
    summary_path = result_dir / f"{safe}_summary.csv"

    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"raw CSV: {raw_path}", flush=True)
    print(f"summary CSV: {summary_path}", flush=True)
    return summary
