#!/usr/bin/env python3
"""Build and benchmark five ML-KEM-768 polynomial arithmetic strategies.

The direct SME2 and the installed SME2 backend require Apple M4/SVL=512.
Run from the SME2 research checkout with `python3 test/sme2_5way/run.py --mode ...`.
No changes to production files are made.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics as st
import subprocess
import sys

VARIANTS = [
    "A_direct_sme2", "B_ntt3_scalar", "C_ntt3_sme2",
    "D_native_full_ntt", "E_portable_c_full_ntt",
]


def call(args: list[str], cwd: Path, log: Path | None = None):
    print("+", " ".join(args), flush=True)
    p = subprocess.run(args, cwd=cwd, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log is not None:
        with log.open("a", encoding="utf-8") as f:
            f.write("+ " + " ".join(args) + "\n")
            f.write(p.stdout + "\n")
    if p.returncode != 0:
        print(p.stdout[-9000:], file=sys.stderr)
        raise RuntimeError(f"command exited {p.returncode}: {args[0]}")
    return p.stdout


def root_of(path: Path) -> Path:
    # test/sme2_5way/run.py => checkout root
    root = path.resolve().parents[2]
    must = [
        "mlkem/mlkem_native.c", "mlkem/mlkem_native_asm.S",
        "mlkem/src/native/aarch64/sme2/mlkem_sme2_backend.h",
        "mlkem/src/native/aarch64/sme2/mlkem_sme2_backend.c",
    ]
    missing = [a for a in must if not (root / a).is_file()]
    if missing:
        raise RuntimeError("Not installed in mlkem-native SME2 research checkout: " + ", ".join(missing))
    return root


def build(root: Path, cc: str, work: Path, logfile: Path) -> Path:
    script_dir = root / "test/sme2_5way"
    incs = ["-I", ".", "-I", "mlkem", "-I", "mlkem/src", "-I", str(script_dir)]
    common = ["-O3", "-std=c99", "-Wall", "-Wextra", "-Werror", "-mcpu=apple-m4",
              "-D_POSIX_C_SOURCE=200809L", "-DMLK_CONFIG_PARAMETER_SET=768",
              "-DMLK_CONFIG_NO_RANDOMIZED_API"] + incs
    native = common + ["-DMLK_CONFIG_USE_NATIVE_BACKEND_ARITH",
                       '-DMLK_CONFIG_ARITH_BACKEND_FILE="native/aarch64/meta.h"']
    portable = common + ["-DMLK_CONFIG_NAMESPACE_PREFIX=BENCH5_PORTABLE"]
    jobs = [
        ("bench5", "test/sme2_5way/bench5.c", native),
        ("native_ntt", "test/sme2_5way/native_ntt.c", native),
        ("native_library", "mlkem/mlkem_native.c", native),
        ("native_asm", "mlkem/mlkem_native_asm.S", native),
        ("direct_smopa", "test/sme2_5way/direct_smopa.S", native),
        ("portable_bridge", "test/sme2_5way/portable_bridge.c", portable),
        ("portable_library", "mlkem/mlkem_native.c", portable),
    ]
    objects=[]
    for name, source, flags in jobs:
        obj = work / (name + ".o")
        call([cc] + flags + ["-c", source, "-o", str(obj)], root, logfile)
        objects.append(str(obj))
    exe = work / "bench5"
    call([cc, "-mcpu=apple-m4", "-o", str(exe)] + objects, root, logfile)
    return exe


def median_2_by_block(rows):
    vals={}
    for row in rows:
        key=(int(row["n"]),int(row["block"]),row["method"])
        vals.setdefault(key,[]).append(float(row["ns_per_req"]))
    return {k:st.median(v) for k,v in vals.items() if len(v)==2}


def ci_median(values, seed=0xB5, nboot=2000):
    rng=random.Random(seed)
    n=len(values)
    sample=[]
    for _ in range(nboot):
        v=[values[rng.randrange(n)] for _ in range(n)]
        sample.append(st.median(v))
    sample.sort()
    return sample[int(0.025*nboot)],sample[int(0.975*nboot)]


def summarize(raw: Path, dest: Path):
    with raw.open(newline="", encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    av=median_2_by_block(rows)
    bsz=sorted({n for n,_,_ in av})
    bs=sorted({block for _,block,_ in av})
    all_results=[]
    for n in bsz:
        for m in VARIANTS:
            common=[b for b in bs if (n,b,m) in av and (n,b,"D_native_full_ntt") in av]
            if not common:raise RuntimeError(f"missing baseline: N={n} method={m}")
            vs=[av[(n,b,m)] for b in common]
            native=[av[(n,b,"D_native_full_ntt")] for b in common]
            diffs=[a-d for a,d in zip(vs,native)]
            ci=ci_median(diffs,seed=0xB5 + n*17 + VARIANTS.index(m))
            chunks=[diffs[i*len(diffs)//3:(i+1)*len(diffs)//3] for i in range(3)]
            epoch=[st.median(ch) for ch in chunks]
            stable=(ci[1]<0 and all(v<0 for v in epoch)) or (ci[0]>0 and all(v>0 for v in epoch))
            res=dict(n=n,method=m,samples=len(common),median_ns_per_req=st.median(vs),
                     native_median_ns_per_req=st.median(native),median_paired_delta_ns_per_req=st.median(diffs),
                     ci95_low=ci[0],ci95_high=ci[1],speedup_vs_native=st.median(native)/st.median(vs),
                     wins=sum(v<0 for v in diffs),
                     epoch1=epoch[0],epoch2=epoch[1],epoch3=epoch[2],
                     epoch_and_ci_stable=stable)
            all_results.append(res)
    if not all_results:raise RuntimeError("no output rows")
    with dest.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(all_results[0]))
        w.writeheader();w.writerows(all_results)
    print("\nSpeedup vs native AArch64 full NTT (greater than 1 = faster):")
    print(f"{'batch':>5s} " + " ".join(f"{v.split('_')[0]:>12}" for v in VARIANTS))
    for n in bsz:
        rr={r["method"]:r for r in all_results if r["n"]==n}
        print(f"{n:5d} " + " ".join(f"{rr[v]['speedup_vs_native']:12.5f}" for v in VARIANTS))
    print("\nFull summary:",dest)
    print("Note: all measurements are arithmetic-only (4 output polynomials, 12 products/request).")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",choices=["check","quick","final"],default="check")
    ap.add_argument("--cc",default=os.environ.get("CC","clang"))
    ap.add_argument("--batches",default=None,help="16,32,48,...; 16 multiples, maximum 256")
    ap.add_argument("--samples",type=int,default=None)
    ap.add_argument("--target-us",type=int,default=None,help="timing target for each method in one pass")
    ap.add_argument("--seed",default="0x5AE21234")
    ap.add_argument("--output", type=Path, help="new result directory")
    args=ap.parse_args()
    if platform.system()!="Darwin" or platform.machine()!="arm64":
        raise SystemExit("Full benchmark needs Apple Silicon macOS; on other hosts use the independent C mathematical tests.")
    root=root_of(Path(__file__))
    seed=int(args.seed,0)
    timestamp=dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir=args.output.resolve() if args.output else root/"test/build/sme2_5way"/f"{args.mode}-{timestamp}"
    result_dir.mkdir(parents=True,exist_ok=False)
    logfile=result_dir/"build.log"
    defaults={
        "check":"16,80,256", "quick":"16,32,48,64,80,96,112,128,144,160,192,224,256",
        "final":"16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256"
    }
    batches=args.batches or defaults[args.mode]
    samples=args.samples or (31 if args.mode=="quick" else 301)
    target=args.target_us or (1500 if args.mode=="quick" else 2500)
    meta={"date":timestamp,"platform":platform.platform(),"compiler":args.cc,"mode":args.mode,
          "batches":batches,"samples":samples,"target_us":target,"seed":seed,
          "note":"SIMD CPU path portable-C uses mlkem_native.c compiled separately with BENCH5_PORTABLE namespace and native arithmetic disabled"}
    for field,cmd in [("head",["git","rev-parse","HEAD"]),
                      ("branch",["git","branch","--show-current"]),
                      ("status",["git","status","--short"]),
                      ("clang_version",[args.cc,"--version"]),
                      ("os",["uname","-a"])]:
        try:meta[field]=subprocess.run(cmd,cwd=root,encoding="utf-8",stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.strip()
        except OSError as ex:meta[field]=repr(ex)
    (result_dir/"metadata.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    exe=build(root,args.cc,result_dir,logfile)
    cmd=[str(exe),"--mode","check" if args.mode=="check" else "bench",
         "--samples",str(samples),"--batches",batches,"--target-us",str(target),
         "--seed",str(seed),"--csv",str(result_dir/"raw.csv")]
    text=call(cmd,root,result_dir/"execution.log")
    print(text)
    if args.mode!="check":summarize(result_dir/"raw.csv",result_dir/"summary.csv")
    print("RESULTS:",result_dir)

if __name__=="__main__":main()
