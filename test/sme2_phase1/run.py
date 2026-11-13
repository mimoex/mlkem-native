#!/usr/bin/env python3
"""Build and run SME2 Phase 1 correctness gates, without measuring performance."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True


def source_hashes(root: Path) -> dict[str, str]:
    paths = list((root / "mlkem").rglob("*"))
    for directory in ("test/sme2_phase1", "test/sme2_5way"):
        paths.extend((root / directory).glob("*"))
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
        if p.is_file() and p.suffix in {".c", ".h", ".S", ".py"}
    }


def load_runner(root: Path, name: str):
    path = root / "test/sme2_5way" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"phase1_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default=os.environ.get("CC", "clang"))
    parser.add_argument("--output", type=Path, help="new directory (must not exist)")
    parser.add_argument("--suite", choices=("all", "svl", "abi", "inverse", "preflight"),
                        default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d-%H%M%S-%f")
    out = (args.output or root / "test/build/sme2_phase1" / f"check-{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = source_hashes(root)
    (out / "sources-before.json").write_text(json.dumps(before, indent=2) + "\n")

    def run(command: list[str], name: str) -> None:
        print("+", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=root, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=180)
        with (out / "commands.log").open("a") as log:
            log.write("+ " + " ".join(command) + "\n" + completed.stdout)
            log.write(f"exit={completed.returncode}\n")
        (out / f"{name}.log").write_text(completed.stdout)
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        completed.check_returncode()

    common = [args.cc, "-O3", "-std=c99", "-Wall", "-Wextra", "-Werror",
              "-mcpu=apple-m4", "-I", ".", "-I", "mlkem", "-I", "mlkem/src",
              "-DMLK_CONFIG_PARAMETER_SET=768", "-DMLK_CONFIG_NO_RANDOMIZED_API"]
    native = common + ["-DMLK_CONFIG_USE_NATIVE_BACKEND_ARITH",
                       '-DMLK_CONFIG_ARITH_BACKEND_FILE="native/aarch64/meta.h"']
    asm = out / "native_asm.o"
    run([args.cc, "--version"], "compiler")
    run(native + ["-c", "mlkem/mlkem_native_asm.S", "-o", str(asm)], "build-asm")

    if args.suite in ("all", "svl"):
        for label, injected in (("svl", False), ("svl-injected", True)):
            library = out / f"{label}-library.o"
            test = out / f"{label}-test.o"
            flags = (["-include", "test/sme2_phase1/test_svl_override.h"]
                     if injected else [])
            run(native + flags + ["-c", "mlkem/mlkem_native.c", "-o", str(library)],
                f"build-{label}-library")
            flags = ["-DTEST_SVL_OVERRIDE"] if injected else []
            run(native + flags + ["-c", "test/sme2_phase1/test_svl.c", "-o", str(test)],
                f"build-{label}-test")
            exe = out / label
            run([args.cc, "-mcpu=apple-m4", str(test), str(library), str(asm),
                 "-o", str(exe)], f"link-{label}")
            run([str(exe)], label)

    if args.suite in ("all", "abi"):
        exe = out / "abi"
        run(native + ["test/sme2_phase1/test_abi.c", "test/sme2_phase1/test_abi_asm.S",
                      str(asm), "-o", str(exe)], "build-abi")
        run([str(exe)], "abi")

    if args.suite in ("all", "inverse"):
        exe = out / "inverse16"
        run(native + ["test/sme2_phase1/test_inverse16.c", str(asm), "-o", str(exe)],
            "build-inverse16")
        run([str(exe)], "inverse16")

    if args.suite in ("all", "preflight"):
        for test in ("sanity_scalar", "test_partial", "test_direct_layout"):
            exe = out / test
            run(common + [f"test/sme2_5way/{test}.c", "-o", str(exe)], f"build-{test}")
            run([str(exe)], test)
        bench_dir = out / "fiveway"
        bench_dir.mkdir()
        bench = load_runner(root, "run").build(root, args.cc, bench_dir,
                                               out / "build-fiveway.log")
        run([str(bench), "--mode", "check"], "fiveway-check")
        oracle_dir = out / "oracle"
        oracle_dir.mkdir()
        oracle = load_runner(root, "run_publication").build_full_oracle(
            root, args.cc, oracle_dir, out / "build-oracle.log")
        run([str(oracle), "--preflight-only"], "oracle-preflight")
        run([str(oracle)], "oracle-full")

    after = source_hashes(root)
    (out / "sources-after.json").write_text(json.dumps(after, indent=2) + "\n")
    if before != after:
        raise RuntimeError("source changed during verification; rerun on a stable tree")
    if list(out.rglob("*.csv")):
        raise RuntimeError("correctness-only runner unexpectedly produced a CSV")
    (out / "PASS.txt").write_text(f"PASS suite={args.suite}; no performance measurements\n")
    print(f"PASS: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
