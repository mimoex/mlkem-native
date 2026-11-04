#!/usr/bin/env python3
"""Build and run SME2 correctness tests, with SCU and separate objects."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="clang")
    parser.add_argument("--build-dir", type=Path, default=Path("test/build/sme2"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = (root / args.build_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "-mcpu=apple-m4", "-std=c99", "-Wall", "-Wextra", "-Werror",
             "-I.", "-Imlkem", "-Imlkem/src", "-DMLK_CONFIG_PARAMETER_SET=768",
             "-DMLK_CONFIG_USE_NATIVE_BACKEND_ARITH",
             '-DMLK_CONFIG_ARITH_BACKEND_FILE="native/aarch64/meta.h"',
             "-DMLK_CONFIG_NO_RANDOMIZED_API"]

    def run(cmd: list[str]) -> None:
        print("+", shlex.join(cmd), flush=True)
        subprocess.run(cmd, cwd=root, check=True, timeout=180)

    run([args.cc, "--version"])
    tests = ["test_api", "test_prepared_ntt", "test_ntt_public_cache",
             "test_output_transpose"]
    for layout in ("scu", "objects"):
        if layout == "scu":
            sources = [root / "mlkem/mlkem_native.c", root / "mlkem/mlkem_native_asm.S"]
        else:
            sources = []
            for directory in ("mlkem/src", "mlkem/src/fips202",
                              "mlkem/src/native/aarch64/src",
                              "mlkem/src/native/aarch64/sme2"):
                sources.extend(sorted((root / directory).glob("*.c")))
                sources.extend(sorted((root / directory).glob("*.S")))
        sources.append(root / "test/sme2_batch/prepared_ntt.c")
        objects = []
        for source in sources:
            obj = out / (layout + "-" + source.stem + ".o")
            run([args.cc, *flags, "-c", str(source), "-o", str(obj)])
            objects.append(str(obj))
        for test in tests:
            binary = out / (test + "-" + layout)
            run([args.cc, *flags, "test/sme2_batch/" + test + ".c",
                 *objects, "-o", str(binary)])
            run([str(binary)])
    print("PASS all SME2 correctness tests (SCU and separate objects)", flush=True)


if __name__ == "__main__":
    main()
