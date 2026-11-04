#!/usr/bin/env python3
"""Verify the final public-cache baseline, then run paired ABBA blocks."""
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import uuid

sys.dont_write_bytecode = True
from benchlib import paired_abba


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(root: Path) -> dict[str, str]:
    paths = list((root / "mlkem").rglob("*"))
    for directory in ("sme2_batch", "sme2_phase1", "sme2_5way"):
        paths.extend((root / "test" / directory).glob("*"))
    return {
        str(p.relative_to(root)): digest(p)
        for p in sorted(paths)
        if p.is_file() and p.suffix in {".c", ".h", ".S", ".py", ".md"}
    }


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory; must not exist")
    parser.add_argument("--cc", default="clang")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--protected-manifest", type=Path,
                        help="JSON containing existing_results path/SHA-256 map")
    parser.add_argument("--mode", choices=("quick", "final"), default="quick")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out = (args.output or root / "test/build/sme2_batch" /
           f"public-cache-{stamp}-{uuid.uuid4().hex[:8]}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = sources(root)
    protected = {}
    if args.protected_manifest:
        protected = json.loads(args.protected_manifest.read_text())["existing_results"]
    else:
        for directory in ("sme2_batch", "sme2_5way", "sme2_pke"):
            for path in (root / "test" / directory / "results").rglob("*"):
                if path.is_file() and out not in path.parents:
                    protected[str(path.relative_to(root))] = digest(path)

    def assert_unchanged() -> None:
        if sources(root) != before:
            raise RuntimeError("source changed during run; measurements are not accepted")
        for name, expected_hash in protected.items():
            if digest(root / name) != expected_hash:
                raise RuntimeError(f"pre-existing result changed: {name}")

    assert_unchanged()
    config = dict(batches=list(range(16, 257, 16)), blocks=31 if args.mode == "quick" else 301, inner=8, warmup=2,
                  bootstrap_reps=5000 if args.mode == "quick" else 20000, seed=0x52520034,
                  variant_a=1, variant_b=5,
                  label="SME2 A vs public-cache prepared NTT B",
                  preparation_samples=31, preparation_inner=16,
                  preparation_warmup=2, verify_only=args.verify_only,
                  timing="same fairness batch API boundary; key/public preparation excluded",
                  sanitizers="address,undefined; native assembly not instrumented")
    manifest = dict(started=datetime.now().astimezone().isoformat(),
                    configuration=config, sources_before=before,
                    protected_results=protected, commands=[], dependencies={},
                    platform=platform.platform(), python=sys.version,
                    invocation=sys.argv, status="in progress")

    def save_manifest() -> None:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(command, name, *, env=None, timeout=300, expected_returncode=0) -> str:
        command = [str(arg) for arg in command]
        print("+", shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout)
        (out / f"{name}.log").write_text(result.stdout)
        manifest["commands"].append(dict(argv=command, log=f"{name}.log",
                                          exit_code=result.returncode))
        save_manifest()
        if expected_returncode is None and result.returncode:
            manifest.setdefault("unavailable_metadata", []).append(name)
            print(f"Metadata unavailable ({name}); see saved log.", flush=True)
        elif expected_returncode is not None and result.returncode != expected_returncode:
            print(result.stdout, end="", flush=True)
            raise RuntimeError(f"{name}: exit {result.returncode}, expected {expected_returncode}")
        return result.stdout

    native = [args.cc, "-O3", "-std=c99", "-Wall", "-Wextra", "-Werror",
              "-mcpu=apple-m4", "-I", str(root / "mlkem"),
              "-I", str(root / "mlkem/src"),
              "-DMLK_CONFIG_PARAMETER_SET=768",
              "-DMLK_CONFIG_USE_NATIVE_BACKEND_ARITH",
              '-DMLK_CONFIG_ARITH_BACKEND_FILE="native/aarch64/meta.h"',
              "-DMLK_CONFIG_NO_RANDOMIZED_API"]
    driver = root / "test/sme2_batch/bench_ntt_public_cache.c"
    test = root / "test/sme2_batch/test_ntt_public_cache.c"
    prepared = root / "test/sme2_batch/prepared_ntt.c"
    library = root / "mlkem/mlkem_native.c"
    asm = root / "mlkem/mlkem_native_asm.S"
    manifest["compiler_flags"] = native
    try:
        manifest["compiler_version"] = run([args.cc, "--version"], "compiler")
        compiler_path = Path(shutil.which(args.cc) or args.cc).resolve()
        manifest["compiler_path"] = str(compiler_path)
        manifest["compiler_sha256"] = digest(compiler_path)
        run(["uname", "-a"], "uname")
        run(["sw_vers"], "os")
        run(["sysctl", "hw.model", "hw.machine", "hw.memsize", "hw.ncpu",
             "machdep.cpu.brand_string"], "hardware", expected_returncode=None)
        run(["pmset", "-g", "batt"], "power-before", expected_returncode=None)
        run(["git", "rev-parse", "HEAD"], "git-head")
        run(["git", "status", "--short"], "git-status-before")
        run(["git", "diff", "--binary"], "git-diff-before")
        run(["git", "diff", "--cached", "--binary"], "git-index-before")
        for name in before:
            dest = out / "sources" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, dest)

        # Record actual compiler-discovered dependencies, including SDK headers.
        for source in (driver, test, prepared, library, asm):
            dep_text = run(native + ["-M", "-MT", "phase2", str(source)],
                           "deps-" + source.stem)
            deps = shlex.split(dep_text.replace("\\\n", " "))[1:]
            for name in deps:
                path = Path(name)
                if not path.is_absolute():
                    path = root / path
                key = str(path)
                if key in manifest["dependencies"]:
                    continue
                sha = digest(path)
                dest = out / "dependencies" / sha / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
                manifest["dependencies"][key] = dict(sha256=sha,
                    saved_as=str(dest.relative_to(out)))

        binary = out / "bench"
        check = out / "correctness"
        run(native + [str(test), str(prepared), str(library), str(asm), "-o", str(check)],
            "build-correctness")
        print(run([check], "correctness"), end="", flush=True)
        negative = run([check, "--corrupt-cache"], "negative-cache",
                       expected_returncode=1)
        if "equal_mod_q(out->coeffs[i], swapped.coeffs[i])" not in negative:
            raise RuntimeError("negative cache control did not fail at the product oracle")
        sanitized = out / "correctness-sanitized"
        sanitize = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                    "-fno-sanitize-recover=all"]
        run(native + sanitize + [str(test), str(prepared), str(library), str(asm),
                                 "-o", str(sanitized)], "build-sanitized")
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0:halt_on_error=1",
                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        manifest["sanitizer_environment"] = {k: env[k] for k in
                                             ("ASAN_OPTIONS", "UBSAN_OPTIONS")}
        print(run([sanitized], "sanitized", env=env), end="", flush=True)
        print(run([sys.executable, "-B", root / "test/sme2_phase1/run.py",
                   "--cc", args.cc, "--output", out / "phase1-regression"],
                  "phase1-regression", timeout=600), end="", flush=True)
        run(native + [str(driver), str(prepared), str(library), str(asm), "-o", str(binary)],
            "build-benchmark")
        for source, name in ((driver, "driver"), (library, "library"),
                             (driver.with_name("bench_ntt_fairness.c"), "old-driver")):
            run(native + ["-S", str(source), "-o", out / f"{name}.s"],
                "assembly-" + name)
        run(native + ["-E", str(asm), "-o", out / "native-asm.s"],
            "assembly-native")
        run(["otool", "-tvV", binary], "disassembly")
        print(run([binary, "--metadata"], "metadata"), end="", flush=True)
        assert_unchanged()
        (out / "CORRECTNESS_PASS.txt").write_text(
            "Phase 2 native + ASan/UBSan and unchanged Phase 1 regression PASS\n")
        print("Correctness gates PASS; performance is now permitted.", flush=True)
        if not args.verify_only:
            measurement = out / "measurement"
            measurement.mkdir(exist_ok=False)
            with (out / "abba.log").open("x") as log:
                with contextlib.redirect_stdout(Tee(sys.stdout, log)):
                    paired_abba(binary=binary, root=root,
                                batches=config["batches"], samples=config["blocks"],
                                inner=8, warmup=2, variant_a=1, variant_b=5,
                                label=config["label"], seed=config["seed"],
                                result_dir=measurement, bootstrap_reps=config["bootstrap_reps"],
                                progress_interval=10)
            rows = []
            for sample in range(config["preparation_samples"]):
                output = run([binary, "--prepare", config["preparation_inner"],
                              config["preparation_warmup"]], f"prepare-{sample:02d}")
                values = dict(item.split("=", 1) for item in
                              output.strip().splitlines()[-1].split() if "=" in item)
                rows.append(dict(sample=sample, **values))
            with (out / "preparation.csv").open("x", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            manifest["preparation_summary"] = {
                key: statistics.median(float(row[key]) for row in rows)
                for key in rows[0] if key.endswith("ns_per_key")}
        run(["pmset", "-g", "batt"], "power-after", expected_returncode=None)
        assert_unchanged()
        manifest["sources_after"] = sources(root)
        manifest["binaries"] = {p.name: digest(p) for p in (binary, check, sanitized)}
        manifest["protected_results_verified_unchanged"] = len(protected)
        manifest["status"] = "PASS"
        (out / "PASS.txt").write_text(f"PASS; {config['blocks']} ABBA blocks\n" if not args.verify_only
                                      else "PASS; verification only\n")
    except BaseException as error:
        manifest["status"] = "FAILED"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["finished"] = datetime.now().astimezone().isoformat()
        save_manifest()
    print(f"Results: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
