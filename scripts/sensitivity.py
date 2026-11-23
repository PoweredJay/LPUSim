#!/usr/bin/env python3
"""Latency sweep: halve and double every assumed [A] value alone and record ResNet-50 cycles.

    python3 scripts/sensitivity.py --bin build/resnet     (or: make sensitivity)

Writes results/latency_sensitivity.csv, which scripts/plots.py --only sensitivity draws. The
values and tags come from the baseline run's latency_used.csv; the SXM width (a floorplan
distance in hops, also [A]) is read from src/lpusim.h. Latencies are whole cycles, so a value of
1 has no half. Exits non-zero if a run fails its violation, IQ or logits check.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# latency_used.csv row -> (resnet --lat name, label on the figure)
OPS = {
    "MEM Read": ("mem_read", "MEM read"),
    "MEM Write": ("mem_write", "MEM write"),
    "MEM Gather": ("mem_gather", "MEM gather"),
    "VXM add": ("vxm_add", "VXM add"),
    "VXM mul": ("vxm_mul", "VXM mul"),
    "VXM convert int32->int8": ("vxm_convert", "VXM convert"),
    "VXM relu": ("vxm_relu", "VXM relu"),
    "VXM max": ("vxm_max", "VXM max"),
    "MXM LW": ("mxm_lw", "MXM load weights (LW)"),
    "MXM IW (buffer)": ("mxm_iw", "MXM install weights (IW)"),
    "MXM ABC": ("mxm_abc", "MXM matmul (ABC)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "resnet"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--latency", default=os.path.join(ROOT, "planning", "instruction-timing-table.csv"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        def run(tag, *flags):
            out = os.path.join(tmp, tag)
            os.mkdir(out)
            p = subprocess.run([a.bin, "--depth", "50", "--latency", a.latency, "--out", out, *flags],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if p.returncode != 0:
                sys.exit(f"error: run {tag} ({' '.join(flags)}) failed:\n" + "\n".join(p.stdout.splitlines()[-6:]))
            with open(os.path.join(out, "resnet50_summary.json")) as f:
                return json.load(f)["cycles"]

        base = run("baseline")
        with open(os.path.join(tmp, "baseline", "latency_used.csv"), newline="") as f:
            used = list(csv.DictReader(f))
        with open(os.path.join(ROOT, "src", "lpusim.h"), encoding="utf-8") as f:
            hops = int(re.search(r"int sxm_hops = (\d+)", f.read()).group(1))

        params = [("sxm_hops", "SXM crossing", hops, "hops", "A")]
        for r in used:
            if "A" in r["tag"].split("/"):      # [A], or part [A] such as A/P
                if r["op"] not in OPS:
                    sys.exit(f"error: no --lat name known for {r['op']}; add it to OPS in scripts/sensitivity.py")
                params.append((*OPS[r["op"]], int(r["d_func"]), "cycles", r["tag"]))

        def flags(key, v):
            return ["--sxm-hops", str(v)] if key == "sxm_hops" else ["--lat", f"{key}={v}"]

        jobs = {}
        with ThreadPoolExecutor(a.jobs) as ex:
            for key, _, v, _, _ in params:
                if v == 0:
                    continue            # 0 halved or doubled is still 0
                if v % 2 == 0:
                    jobs[key, "half"] = ex.submit(run, key + "_half", *flags(key, v // 2))
                jobs[key, "double"] = ex.submit(run, key + "_double", *flags(key, v * 2))
        cyc = {k: j.result() for k, j in jobs.items()}

    rows = []
    for key, label, v, unit, tag in params:
        half = base if v == 0 else cyc.get((key, "half"))
        dbl = base if v == 0 else cyc[key, "double"]
        ends = [base, dbl] + ([half] if half is not None else [])
        rows.append([key, label, tag, unit, v, base, "" if half is None else half, dbl, max(ends) - min(ends)])
    rows.sort(key=lambda r: -r[-1])
    os.makedirs(a.results, exist_ok=True)
    out = os.path.join(a.results, "latency_sensitivity.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow("key,instruction,tag,unit,assumed,baseline_cycles,half_cycles,double_cycles,swing_cycles".split(","))
        w.writerows(rows)
    print(f"wrote {os.path.relpath(out, ROOT)} (baseline {base:,} cycles, {len(cyc) + 1} runs, all bit-exact)")


if __name__ == "__main__":
    main()
