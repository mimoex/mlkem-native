#!/usr/bin/env python3
"""Compile and execute the Phase 3 configuration matrix, without timing it."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory; never overwritten")
    parser.add_argument("--cc", default=os.environ.get("CC", "clang"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    flags = ["-O3", "-std=c99", "-Wall", "-Wextra", "-Werror", "-I.",
             "-Imlkem", "-Imlkem/src", "-DMLK_CONFIG_NO_RANDOMIZED_API",
             "-DMLK_CONFIG_NAMESPACE_PREFIX=phase3"]
    native = ["-DMLK_CONFIG_USE_NATIVE_BACKEND_ARITH",
              '-DMLK_CONFIG_ARITH_BACKEND_FILE="native/aarch64/meta.h"']
    results = []

    def run(cmd: list[str], name: str) -> str:
        p = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=180)
        with (out / "commands.log").open("a") as f:
            f.write(shlex.join(cmd) + "\n" + p.stdout +
                    f"\nexit={p.returncode}\n")
        (out / f"{name}.log").write_text(p.stdout)
        if p.returncode:
            raise RuntimeError(f"{name} failed: {p.stdout}")
        return p.stdout

    def driver(path: Path, levels: tuple[int, ...], multi: bool) -> None:
        names = [f"phase3{level if multi else ''}_phase3_build_smoke"
                 for level in levels]
        path.write_text("\n".join(f"int {name}(void);" for name in names) +
                        "\nint main(void) { return " +
                        " | ".join(f"{name}()" for name in names) + "; }\n")

    def check(binary: Path, case: str, expect_sme: bool, multi: bool) -> None:
        text = run(["nm", "-g", str(binary)], case + "-symbols")
        suffix = "phase3768_sme2_decaps_batch" if multi else "phase3_sme2_decaps_batch"
        if (suffix in text) != expect_sme:
            raise AssertionError(f"{case}: unexpected SME2 API symbol visibility")
        if "phase3512_sme2_decaps_" in text or "phase31024_sme2_decaps_" in text:
            raise AssertionError(f"{case}: unsupported level exposed SME2 Decaps")
        run([str(binary)], case + "-runtime")
        results.append({"case": case, "sme2_decaps": expect_sme, "result": "PASS"})
        print(f"PASS {case}", flush=True)

    run([args.cc, "--version"], "compiler")
    source = out / "probe.c"
    source.write_text('#include "test/sme2_batch/build_probe.inc"\n')
    for arch, archflags in (("sme2", ["-mcpu=apple-m4", *native]),
                            ("neon", ["-march=armv8-a", *native]),
                            ("portable", ["-march=armv8-a"])):
        for level in (512, 768, 1024):
            case = f"single-{arch}-{level}"
            entry = out / (case + "-main.c")
            driver(entry, (level,), False)
            binary = out / case
            cmd = [args.cc, *flags, *archflags,
                   f"-DMLK_CONFIG_PARAMETER_SET={level}", str(source), str(entry),
                   "mlkem/mlkem_native.c", "mlkem/mlkem_native_asm.S", "-o", str(binary)]
            run(cmd, case + "-build")
            check(binary, case, arch == "sme2" and level == 768, False)

    # Shared assembly contains every level's stock kernels plus one SME2 kernel.
    # Verify each possible owner; C's 768 extension is independent of that owner.
    for owner in (512, 768, 1024):
        case = f"multitu-shared-{owner}"
        base = [args.cc, *flags, "-mcpu=apple-m4", *native,
                "-DMLK_CONFIG_MULTILEVEL_BUILD"]
        objects = []
        for level in (512, 768, 1024):
            mode = "WITH_SHARED" if level == owner else "NO_SHARED"
            config = [f"-DMLK_CONFIG_PARAMETER_SET={level}",
                      f"-DMLK_CONFIG_MULTILEVEL_{mode}"]
            for tag, src in (("library", "mlkem/mlkem_native.c"), ("probe", str(source))):
                obj = out / f"{case}-{level}-{tag}.o"
                run([*base, *config, "-c", src, "-o", str(obj)], obj.stem)
                objects.append(str(obj))
        asm = out / (case + "-asm.o")
        run([*base, f"-DMLK_CONFIG_PARAMETER_SET={owner}",
             "-DMLK_CONFIG_MULTILEVEL_WITH_SHARED", "-c", "mlkem/mlkem_native_asm.S",
             "-o", str(asm)], asm.stem)
        entry = out / (case + "-main.c")
        driver(entry, (512, 768, 1024), True)
        binary = out / case
        run([args.cc, *objects, str(asm), str(entry), "-o", str(binary)], case + "-link")
        check(binary, case, True, True)

    for order in itertools.permutations((512, 768, 1024)):
        case = "scu-" + "-".join(map(str, order))
        text = []
        for i, level in enumerate(order):
            mode = "WITH_SHARED" if i == 0 else "NO_SHARED"
            if i == 0:
                text.append("#define MLK_CONFIG_MONOBUILD_KEEP_SHARED_HEADERS")
            if i == 2:
                text.append("#undef MLK_CONFIG_MONOBUILD_KEEP_SHARED_HEADERS")
            text.extend([f"#define MLK_CONFIG_PARAMETER_SET {level}",
                         f"#define MLK_CONFIG_MULTILEVEL_{mode}",
                         '#include "test/sme2_batch/build_probe.inc"',
                         '#include "mlkem/mlkem_native.c"',
                         f"#undef MLK_CONFIG_MULTILEVEL_{mode}",
                         "#undef MLK_CONFIG_PARAMETER_SET"])
        scu = out / (case + ".c")
        scu.write_text("\n".join(text) + "\n")
        entry = out / (case + "-main.c")
        driver(entry, order, True)
        asm = out / f"multitu-shared-{order[0]}-asm.o"
        binary = out / case
        run([args.cc, *flags, "-mcpu=apple-m4", *native,
             "-DMLK_CONFIG_MULTILEVEL_BUILD", str(scu), str(entry), str(asm),
             "-o", str(binary)], case + "-build")
        check(binary, case, True, True)

    for option, extra in (("context", ["-DMLK_CONFIG_CONTEXT_PARAMETER",
                                       "-DMLK_CONFIG_CONTEXT_PARAMETER_TYPE=void*"]),
                          ("no-decaps", ["-DMLK_CONFIG_NO_DECAPS_API"])):
        case = "single-sme2-768-" + option
        entry = out / (case + "-main.c")
        driver(entry, (768,), False)
        binary = out / case
        run([args.cc, *flags, "-mcpu=apple-m4", *native,
             "-DMLK_CONFIG_PARAMETER_SET=768", *extra,
             str(source), str(entry), "mlkem/mlkem_native.c", "mlkem/mlkem_native_asm.S",
             "-o", str(binary)], case + "-build")
        check(binary, case, option != "no-decaps", False)
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "artifacts-sha256.json").write_text(json.dumps({
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir()) if p.is_file()}, indent=2) + "\n")
    print(f"PASS build matrix: {len(results)} configurations", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
