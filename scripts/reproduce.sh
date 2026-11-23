#!/usr/bin/env bash
# Rebuild everything and regenerate every number in README.md / results/REPORT.md.
#
#   make                      (from the repo root; same as scripts/reproduce.sh)
#   B=/tmp/lpu-build make     build somewhere else (e.g. if the checkout is on a noexec mount)
#   PYTHON=/path/to/python3 make   a Python 3 that has matplotlib (pip install -r requirements.txt)
#
# Steps: latency CSV from the xlsx -> build -> unit + layer tests -> ResNet-50/101/152 ->
# sensitivity runs -> roofline sweeps -> figures -> results/REPORT.md. Exits non-zero if any
# test fails, any run reports a violation / underflow / mismatch, or any step errors.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
B="${B:-$ROOT/build}"
PYTHON="${PYTHON:-python3}"
PYTHON="${PYTHON/#\~/$HOME}"   # zsh leaves a literal ~ in `make PYTHON=~/...`
B="${B/#\~/$HOME}"
JOBS="${JOBS:-$( (sysctl -n hw.ncpu || nproc) 2>/dev/null || echo 4)}"
RES="$ROOT/results"
LAT="$ROOT/planning/instruction-timing-table.csv"
# Roofline loads: powers of two, plus M = 100 / 320 / 400 so the paper's 200 / 639 / 799 ops/B have
# simulated partners. M = 4096 (paper 8,179) does not fit one MEM bank.
MS="1,2,4,8,16,32,64,100,128,256,320,400,512,1024,2048"

if ! "$PYTHON" -c "import matplotlib" 2>/dev/null; then
    echo "error: '$PYTHON' has no matplotlib. Run: $PYTHON -m pip install -r requirements.txt" >&2
    echo "       or point PYTHON at an interpreter that has it: PYTHON=/path/to/python3 make" >&2
    exit 1
fi

step() { printf '\n==> %s\n' "$*"; }

step "latency table: planning/instruction-timing-table.xlsx -> .csv"
"$PYTHON" -I "$ROOT/planning/xlsx_to_csv.py"

step "build ($B)"
make -C "$ROOT/src" -j"$JOBS" B="$B" all

# results/ is generated: clear what this script writes so nothing stale survives
rm -rf "$RES"/*.csv "$RES"/*.json "$RES"/tests.log "$RES"/REPORT.md "$RES"/sensitivity "$RES"/figures "$RES"/logs
mkdir -p "$RES/sensitivity" "$RES/logs"
cd "$ROOT/src"   # the binaries' default paths are relative to src/

step "tests"
{ "$B/tests" && "$B/tests_layers"; } 2>&1 | tee "$RES/tests.log"

step "ResNet-50/101/152, sensitivity runs and roofline sweeps (in parallel; logs in results/logs/)"
pids=()
run() { local log="$RES/logs/$1.log"; shift; "$@" >"$log" 2>&1 & pids+=($!); }
run resnet50          "$B/resnet" --depth 50  --latency "$LAT" --out "$RES"
run resnet101         "$B/resnet" --depth 101 --latency "$LAT" --out "$RES" --alias-weights
run resnet152         "$B/resnet" --depth 152 --latency "$LAT" --out "$RES" --alias-weights
run resnet50_mem1     "$B/resnet" --depth 50  --latency "$LAT" --out "$RES/sensitivity" --tag _mem1 --mem-single-issue
run resnet50_sxm14    "$B/resnet" --depth 50  --latency "$LAT" --out "$RES/sensitivity" --tag _sxm14 --sxm-hops 14
run resnet50_nooverlap "$B/resnet" --depth 50 --latency "$LAT" --out "$RES/sensitivity" --tag _nooverlap --no-layer-overlap
run roofline          "$B/roofline" --latency "$LAT" --out "$RES" --m "$MS"
run roofline_wide     "$B/roofline" --latency "$LAT" --out "$RES" --m "$MS" --csv roofline_wide_weights.csv \
                      --place "weights over 32 slices (MEM12-43):12:43:0:11"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for f in "$RES"/logs/*.log; do printf '\n--- %s\n' "$(basename "$f")"; grep -E 'simulated|BIT-EXACT|MISMATCH|PROJECTION|violation' "$f" | head -6 || true; done
if [ "$fail" -ne 0 ]; then echo "error: a run failed (see results/logs/)" >&2; exit 1; fi

step "figures and report"
"$PYTHON" -I "$ROOT/scripts/plots.py"
"$PYTHON" -I "$ROOT/scripts/report.py"
printf '\nDone. Numbers: results/REPORT.md   Figures: results/figures/\n'
