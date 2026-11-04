#!/usr/bin/env python3
"""Publication-grade wrapper for test/sme2_5way/run.py.

What this adds over the ordinary five-way runner:
  * clean-tree / commit provenance gate (override only with --allow-dirty)
  * SHA-256 manifest for benchmark + SME2 production sources
  * three independent mathematical/layout unit tests
  * full five-way independent-schoolbook oracle over every coefficient
  * repeated 301-block final sessions
  * higher-resolution C-vs-D paired bootstrap reanalysis
  * explicit cross-session replication gates for Table 4 (N=256)

Install this file and publication_oracle.c in test/sme2_5way/ and run from the
repository root, or invoke it by path. Production files are never modified.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import statistics as st
import subprocess
import sys
import time

PRIMARY = "C_ntt3_sme2"
BASELINE = "D_native_full_ntt"
ALLOWED_RESULT_PREFIX = "test/build/sme2_5way/"

CRITICAL_REL = [
    "test/sme2_5way/bench5.c",
    "test/sme2_5way/direct_core.h",
    "test/sme2_5way/direct_smopa.S",
    "test/sme2_5way/math_core.h",
    "test/sme2_5way/native_ntt.c",
    "test/sme2_5way/ntt_bridge.h",
    "test/sme2_5way/ntt_impl.h",
    "test/sme2_5way/partial_scalar.h",
    "test/sme2_5way/portable_bridge.c",
    "test/sme2_5way/run.py",
    "test/sme2_5way/sanity_scalar.c",
    "test/sme2_5way/test_direct_layout.c",
    "test/sme2_5way/test_partial.c",
    "test/sme2_5way/publication_oracle.c",
    "test/sme2_5way/run_publication.py",
    "mlkem/mlkem_native.c",
    "mlkem/mlkem_native_asm.S",
    "mlkem/src/native/aarch64/sme2/mlkem_sme2_backend.c",
    "mlkem/src/native/aarch64/sme2/mlkem_sme2_batch.c",
    "mlkem/src/native/aarch64/sme2/mlkem_sme2_inverse.c",
    "mlkem/src/native/aarch64/sme2/mlkem_sme2_ntt.c",
    "mlkem/src/native/aarch64/sme2/mlkem_sme2_public.c",
    "mlkem/src/native/aarch64/sme2/mlkem_sme2_rolling.c",
    "mlkem/src/native/aarch64/sme2/mlkem_sme2_internal.h",
    "mlkem/src/native/aarch64/sme2/mlkem_sme2_backend.h",
    "mlkem/src/native/aarch64/sme2/mlkem_sme2_arith.S",
]


def run(cmd: list[str], cwd: Path, *, log: Path | None = None,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log is not None:
        with log.open("a", encoding="utf-8") as f:
            f.write("+ " + " ".join(cmd) + "\n")
            f.write(p.stdout)
            if p.stdout and not p.stdout.endswith("\n"):
                f.write("\n")
    if p.stdout:
        print(p.stdout, end="" if p.stdout.endswith("\n") else "\n")
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p


def output(cmd: list[str], cwd: Path) -> str:
    return subprocess.run(cmd, cwd=cwd, text=True, check=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT).stdout.strip()


def root_of(script: Path) -> Path:
    # test/sme2_5way/run_publication.py => checkout root
    root = script.resolve().parents[2]
    if not (root / "mlkem/mlkem_native.c").is_file():
        raise RuntimeError("run_publication.py must live under test/sme2_5way/")
    return root


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def source_manifest(root: Path) -> dict[str, str]:
    missing = [rel for rel in CRITICAL_REL if not (root / rel).is_file()]
    if missing:
        raise RuntimeError("missing critical files:\n  " + "\n  ".join(missing))
    return {rel: sha256(root / rel) for rel in CRITICAL_REL}


def git_status_lines(root: Path) -> list[str]:
    txt = output(["git", "status", "--porcelain=v1", "--untracked-files=all"], root)
    return [line for line in txt.splitlines() if line.strip()]


def status_path(line: str) -> str:
    # porcelain v1: XY<space>path, or rename "old -> new".
    p = line[3:] if len(line) >= 4 else line
    if " -> " in p:
        p = p.split(" -> ", 1)[1]
    return p.strip().strip('"')


def unexpected_dirty(lines: list[str]) -> list[str]:
    bad = []
    for line in lines:
        p = status_path(line)
        if p.startswith(ALLOWED_RESULT_PREFIX):
            continue
        bad.append(line)
    return bad


def compile_small_test(root: Path, cc: str, src: str, out: Path,
                       log: Path) -> None:
    flags = [
        "-O3", "-std=c99", "-Wall", "-Wextra", "-Werror",
        "-mcpu=apple-m4", "-I", "test/sme2_5way",
    ]
    run([cc] + flags + [src, "-o", str(out)], root, log=log)
    run([str(out)], root, log=log)


def build_full_oracle(root: Path, cc: str, work: Path, log: Path) -> Path:
    script_dir = root / "test/sme2_5way"
    incs = ["-I", ".", "-I", "mlkem", "-I", "mlkem/src", "-I", str(script_dir)]
    common = [
        "-O3", "-std=c99", "-Wall", "-Wextra", "-Werror", "-mcpu=apple-m4",
        "-D_POSIX_C_SOURCE=200809L", "-DMLK_CONFIG_PARAMETER_SET=768",
        "-DMLK_CONFIG_NO_RANDOMIZED_API",
    ] + incs
    native = common + [
        "-DMLK_CONFIG_USE_NATIVE_BACKEND_ARITH",
        '-DMLK_CONFIG_ARITH_BACKEND_FILE="native/aarch64/meta.h"',
    ]
    portable = common + ["-DMLK_CONFIG_NAMESPACE_PREFIX=BENCH5_PORTABLE"]
    jobs = [
        ("publication_oracle", "test/sme2_5way/publication_oracle.c", native),
        ("native_ntt", "test/sme2_5way/native_ntt.c", native),
        ("native_library", "mlkem/mlkem_native.c", native),
        ("native_asm", "mlkem/mlkem_native_asm.S", native),
        ("direct_smopa", "test/sme2_5way/direct_smopa.S", native),
        ("portable_bridge", "test/sme2_5way/portable_bridge.c", portable),
        ("portable_library", "mlkem/mlkem_native.c", portable),
    ]
    objs = []
    for name, src, flags in jobs:
        obj = work / f"{name}.o"
        run([cc] + flags + ["-c", src, "-o", str(obj)], root, log=log)
        objs.append(str(obj))
    exe = work / "publication_oracle"
    run([cc, "-mcpu=apple-m4", "-o", str(exe)] + objs, root, log=log)
    return exe


def collapse_pair_rows(raw: Path, method: str, baseline: str):
    with raw.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    per: dict[tuple[int, int, str], list[float]] = {}
    for row in rows:
        m = row["method"]
        if m not in (method, baseline):
            continue
        key = (int(row["n"]), int(row["block"]), m)
        per.setdefault(key, []).append(float(row["ns_per_req"]))
    med = {k: st.median(v) for k, v in per.items() if len(v) == 2}
    ns = sorted({k[0] for k in med})
    out = {}
    for n in ns:
        blocks = sorted({k[1] for k in med if k[0] == n})
        common = [b for b in blocks
                  if (n, b, method) in med and (n, b, baseline) in med]
        a = [med[(n, b, method)] for b in common]
        d = [med[(n, b, baseline)] for b in common]
        diffs = [x - y for x, y in zip(a, d)]
        out[n] = (a, d, diffs)
    return out


def percentile_sorted(xs: list[float], q: float) -> float:
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def bootstrap_median_ci(xs: list[float], reps: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(xs)
    boots = []
    for _ in range(reps):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        boots.append(st.median(sample))
    boots.sort()
    return percentile_sorted(boots, 0.025), percentile_sorted(boots, 0.975)


def analyze_primary(raw: Path, dest: Path, boots: int, session: int):
    pairs = collapse_pair_rows(raw, PRIMARY, BASELINE)
    fieldnames = [
        "session", "n", "blocks", "C_median_ns_per_req", "D_median_ns_per_req",
        "speedup_D_over_C", "paired_delta_median", "ci95_low", "ci95_high",
        "wins_C", "epoch1", "epoch2", "epoch3", "ci_epoch_stable_C_win",
    ]
    rows = []
    for n, (c, d, diffs) in sorted(pairs.items()):
        ci = bootstrap_median_ci(diffs, boots,
                                 0xC0FFEE + 1009 * session + 17 * n)
        chunks = [diffs[i * len(diffs) // 3:(i + 1) * len(diffs) // 3]
                  for i in range(3)]
        epochs = [st.median(ch) for ch in chunks]
        stable = ci[1] < 0 and all(x < 0 for x in epochs)
        row = {
            "session": session,
            "n": n,
            "blocks": len(diffs),
            "C_median_ns_per_req": st.median(c),
            "D_median_ns_per_req": st.median(d),
            "speedup_D_over_C": st.median(d) / st.median(c),
            "paired_delta_median": st.median(diffs),
            "ci95_low": ci[0],
            "ci95_high": ci[1],
            "wins_C": sum(x < 0 for x in diffs),
            "epoch1": epochs[0],
            "epoch2": epochs[1],
            "epoch3": epochs[2],
            "ci_epoch_stable_C_win": stable,
        }
        rows.append(row)
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return rows


def read_summary_n256(summary: Path, session: int):
    with summary.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if int(r["n"]) != 256:
            continue
        out.append({
            "session": session,
            "method": r["method"],
            "median_ns_per_req": float(r["median_ns_per_req"]),
            "speedup_vs_native": float(r["speedup_vs_native"]),
            "paired_delta_ns_per_req": float(r["median_paired_delta_ns_per_req"]),
            "ci95_low": float(r["ci95_low"]),
            "ci95_high": float(r["ci95_high"]),
            "wins": int(r["wins"]),
            "epoch_and_ci_stable": r["epoch_and_ci_stable"].lower() == "true",
        })
    return out


def parse_results_path(text: str, root: Path) -> Path:
    m = re.findall(r"^RESULTS:\s*(.+)$", text, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("underlying run.py did not print RESULTS path")
    p = Path(m[-1].strip())
    if not p.is_absolute():
        p = root / p
    if not p.is_dir():
        raise RuntimeError(f"reported results directory missing: {p}")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", type=int, default=3,
                    help="independent final sessions (default: 3)")
    ap.add_argument("--samples", type=int, default=301)
    ap.add_argument("--target-us", type=int, default=2500)
    ap.add_argument("--batches", default="16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256")
    ap.add_argument("--bootstrap", type=int, default=10000,
                    help="bootstrap replications for C-vs-D reanalysis")
    ap.add_argument("--cooldown", type=int, default=60,
                    help="seconds between sessions")
    ap.add_argument("--seed", default="0x5AE21234")
    ap.add_argument("--cc", default=os.environ.get("CC", "clang"))
    ap.add_argument("--expect-head", default=None,
                    help="optional exact/abbreviated git commit required")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="development only: permit non-result working-tree changes")
    ap.add_argument("--skip-oracle", action="store_true",
                    help="skip full schoolbook oracle; mandatory ZA regression still runs")
    ap.add_argument("--skip-unit-tests", action="store_true")
    ap.add_argument("--output", type=Path, help="new publication result directory")
    args = ap.parse_args()

    if args.sessions < 1 or args.samples < 1 or args.bootstrap < 1000:
        raise SystemExit("invalid sessions/samples/bootstrap")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("publication run requires Apple Silicon macOS")

    root = root_of(Path(__file__))
    base_seed = int(args.seed, 0)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = args.output.resolve() if args.output else root / "test/build/sme2_5way" / f"publication-{stamp}"
    outdir.mkdir(parents=True, exist_ok=False)
    log = outdir / "publication.log"

    head = output(["git", "rev-parse", "HEAD"], root)
    branch = output(["git", "branch", "--show-current"], root)
    if args.expect_head:
        want = output(["git", "rev-parse", args.expect_head], root)
        if head != want:
            raise RuntimeError(f"HEAD mismatch: have {head}, expected {want}")

    status0 = git_status_lines(root)
    dirty0 = unexpected_dirty(status0)
    if dirty0 and not args.allow_dirty:
        raise RuntimeError(
            "publication gate requires a clean source tree. Unexpected status:\n  " +
            "\n  ".join(dirty0) +
            "\nCommit/stash the harness and source changes, or use --allow-dirty only for development."
        )

    manifest0 = source_manifest(root)
    provenance = {
        "timestamp": stamp,
        "head": head,
        "branch": branch,
        "status_at_start": status0,
        "allow_dirty": args.allow_dirty,
        "compiler": output([args.cc, "--version"], root),
        "uname": output(["uname", "-a"], root),
        "platform": platform.platform(),
        "sessions": args.sessions,
        "samples": args.samples,
        "target_us": args.target_us,
        "batches": args.batches,
        "bootstrap": args.bootstrap,
        "cooldown": args.cooldown,
        "base_seed": base_seed,
        "critical_sha256": manifest0,
    }
    (outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    print(f"=== publication provenance ===\nhead={head}\nbranch={branch}\nresults={outdir}")
    if dirty0:
        print("WARNING: --allow-dirty active; this is not publication-clean")

    work = outdir / "preflight-build"
    work.mkdir()

    if not args.skip_unit_tests:
        print("\n=== independent unit tests ===")
        compile_small_test(root, args.cc, "test/sme2_5way/sanity_scalar.c",
                           work / "sanity_scalar", log)
        compile_small_test(root, args.cc, "test/sme2_5way/test_direct_layout.c",
                           work / "test_direct_layout", log)
        compile_small_test(root, args.cc, "test/sme2_5way/test_partial.c",
                           work / "test_partial", log)

    print("\n=== mandatory ZA regression and publication oracle ===")
    oracle = build_full_oracle(root, args.cc, work, log)
    oracle_args = ["--preflight-only"] if args.skip_oracle else []
    run([str(oracle)] + oracle_args, root, log=log)

    all_primary = []
    all_table4 = []
    session_dirs = []

    for sess in range(1, args.sessions + 1):
        if sess > 1 and args.cooldown:
            print(f"\n=== cooldown {args.cooldown}s before session {sess} ===", flush=True)
            time.sleep(args.cooldown)

        sess_seed = (base_seed + (sess - 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        print(f"\n=== final session {sess}/{args.sessions} seed=0x{sess_seed:016x} ===")
        cmd = [
            sys.executable, "test/sme2_5way/run.py",
            "--mode", "final",
            "--samples", str(args.samples),
            "--target-us", str(args.target_us),
            "--batches", args.batches,
            "--seed", hex(sess_seed),
            "--cc", args.cc,
            "--output", str(outdir / f"measurement-{sess:02d}"),
        ]
        p = run(cmd, root, log=log)
        src_result = parse_results_path(p.stdout, root)
        dst = outdir / f"session-{sess:02d}"
        shutil.copytree(src_result, dst)
        session_dirs.append(dst)

        primary_rows = analyze_primary(dst / "raw.csv",
                                       dst / "primary_C_vs_D_10k.csv",
                                       args.bootstrap, sess)
        all_primary.extend(primary_rows)
        all_table4.extend(read_summary_n256(dst / "summary.csv", sess))

        # Re-check source immutability after each session.
        if source_manifest(root) != manifest0:
            raise RuntimeError(f"critical source hash changed during session {sess}")
        if output(["git", "rev-parse", "HEAD"], root) != head:
            raise RuntimeError(f"git HEAD changed during session {sess}")

    # Combined primary replication table.
    primary_fields = list(all_primary[0].keys())
    with (outdir / "primary_C_vs_D_all_sessions.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=primary_fields)
        w.writeheader(); w.writerows(all_primary)

    table4_fields = list(all_table4[0].keys())
    with (outdir / "table4_N256_all_sessions.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=table4_fields)
        w.writeheader(); w.writerows(all_table4)

    # Cross-session descriptive replication gate for C vs D.
    ns = sorted({int(r["n"]) for r in all_primary})
    replication = []
    for n in ns:
        rows = [r for r in all_primary if int(r["n"]) == n]
        speeds = [float(r["speedup_D_over_C"]) for r in rows]
        deltas = [float(r["paired_delta_median"]) for r in rows]
        stable = [bool(r["ci_epoch_stable_C_win"]) for r in rows]
        replication.append({
            "n": n,
            "sessions": len(rows),
            "all_sessions_stable_C_win": all(stable),
            "all_session_deltas_negative": all(x < 0 for x in deltas),
            "all_session_speedups_gt_1": all(x > 1 for x in speeds),
            "median_session_speedup": st.median(speeds),
            "min_session_speedup": min(speeds),
            "max_session_speedup": max(speeds),
            "median_session_paired_delta": st.median(deltas),
            "min_session_paired_delta": min(deltas),
            "max_session_paired_delta": max(deltas),
        })

    rep_fields = list(replication[0].keys())
    with (outdir / "replication_gate.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rep_fields)
        w.writeheader(); w.writerows(replication)

    n256 = next((r for r in replication if int(r["n"]) == 256), None)
    table4_pass = bool(n256 and n256["all_sessions_stable_C_win"] and
                       n256["all_session_deltas_negative"] and
                       n256["all_session_speedups_gt_1"])
    curve_stable = [int(r["n"]) for r in replication
                    if r["all_sessions_stable_C_win"]]

    gate = {
        "table4_N256_replication_pass": table4_pass,
        "stable_C_over_D_all_sessions_at_N": curve_stable,
        "criterion": (
            "For N=256, every independent session must have a negative paired "
            "median delta C-D, speedup D/C > 1, bootstrap 95% CI below zero, "
            "and all three chronological epoch medians below zero."
        ),
        "note": (
            "Cross-session min/median/max are descriptive. Sessions are not "
            "pooled into a single pseudo-independent bootstrap sample."
        ),
    }
    (outdir / "publication_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest1 = source_manifest(root)
    status1 = git_status_lines(root)
    if manifest1 != manifest0:
        raise RuntimeError("critical source hashes changed during publication run")
    if output(["git", "rev-parse", "HEAD"], root) != head:
        raise RuntimeError("git HEAD changed during publication run")
    dirty1 = unexpected_dirty(status1)
    if not args.allow_dirty and dirty1:
        raise RuntimeError("unexpected source-tree changes appeared during run:\n  " +
                           "\n  ".join(dirty1))

    print("\n=== publication replication summary: C vs D ===")
    print(f"{'N':>5} {'sessions':>8} {'all-stable':>11} {'median x':>10} "
          f"{'min x':>10} {'max x':>10} {'median delta':>14}")
    for r in replication:
        print(f"{r['n']:5d} {r['sessions']:8d} "
              f"{str(r['all_sessions_stable_C_win']):>11} "
              f"{r['median_session_speedup']:10.5f} "
              f"{r['min_session_speedup']:10.5f} "
              f"{r['max_session_speedup']:10.5f} "
              f"{r['median_session_paired_delta']:14.3f}")

    print("\n=== Table 4 replication gate (N=256) ===")
    print("PASS" if table4_pass else "FAIL / NOT REPLICATED")
    print("Stable C>D in every session at N:", curve_stable)
    print("RESULTS:", outdir)
    return 0 if table4_pass else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
