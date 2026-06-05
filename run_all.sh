#!/usr/bin/env bash
# ----------------------------------------------------------------------------------
# Generate the data and figures for the four 4.8.8 variants
# ----------------------------------------------------------------------------------
#
# Prerequisite: pip install -e 
# Then:         ./run_all.sh
#
# Chains (scan -> plot), in order:
#   1. Threshold (MWPM + BP+matching) threshold_scan.py   -> threshold_plot.py
#   2. Sub-threshold scaling        subthreshold_scan.py  -> subthreshold_plot.py
#   3. Distance + timelike          spatial+timelike data -> distance_plot.py
#
# Toggles:
#   SKIP_SCANS=1         skip the two long Monte-Carlo scans; just (re)plot.
#   WORKERS=N            parallel workers for the threshold scan.
#   THRESH_SHOTS=N       shots per threshold cell (default 1,000,000).
#   THRESH_LS="4 6 8 12" threshold lattice sizes (default "4 6 8 12").
#   CROSSING_PAIR="A B"  the two L curves used for the threshold crossing.
#   ST_LS / ST_PLIST / ST_MWPM_SHOTS / ST_BP_SHOTS  sub-threshold.
#
# Fear not, all scans are resumable (re-running skips finished cells). 
# ----------------------------------------------------------------------------------

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export DYN488_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if ! python3 -c "import builders" 2>/dev/null; then
    echo "error: the 4.8.8 modules in src/ are not importable." >&2
    echo "       check that $ROOT/src exists and contains builders.py." >&2
    exit 1
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PLOTS="$ROOT/scans_and_plots"
LOG_DIR="$ROOT/results/logs"
mkdir -p "$LOG_DIR" "$ROOT/results/threshold" "$ROOT/results/timelike" \
         "$ROOT/results/subthreshold"

SKIP_SCANS="${SKIP_SCANS:-0}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 4)}}"
THRESH_SHOTS="${THRESH_SHOTS:-1000000}"
CROSSING_PAIR="${CROSSING_PAIR:-6 8}"
ST_LS="${ST_LS:-6 8 10 12}"
ST_PLIST="${ST_PLIST:-0.0007 0.001 0.0015 0.002 0.0025}"
ST_MWPM_SHOTS="${ST_MWPM_SHOTS:-10000000}"
ST_BP_SHOTS="${ST_BP_SHOTS:-100000}"

step () {
    local label="$1"; shift
    local logfile="$LOG_DIR/$(echo "$label" | tr ' /' '__').log"
    printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$label"
    if "$@" >"$logfile" 2>&1; then
        printf '  ok  -> %s\n' "$logfile"
    else
        printf '  FAILED -> see %s\n' "$logfile"
        tail -25 "$logfile" >&2
        exit 1
    fi
}

printf '\n=== 4.8.8: generating data + result figures ===\n'
printf 'root:        %s\n' "$ROOT"
printf 'workers:     %s   shots/cell (threshold): %s\n' "$WORKERS" "$THRESH_SHOTS"
printf 'skip scans:  %s\n' "$SKIP_SCANS"

# 1. Threshold (MWPM + BP+matching): scan -> plot
if [[ "$SKIP_SCANS" != "1" ]]; then
    step "threshold: BP+matching scan" \
        python3 "$PLOTS/threshold_scan.py" --workers "$WORKERS" --shots "$THRESH_SHOTS"
fi
step "threshold: per-round crossing plot" \
    python3 "$PLOTS/threshold_plot.py" --crossing-pair $CROSSING_PAIR

# 2. Sub-threshold scaling: scan -> plot
if [[ "$SKIP_SCANS" != "1" ]]; then
    step "subthreshold: 4-variant scan" \
            python3 "$PLOTS/subthreshold_scan.py" \
                --variants all --L $ST_LS --p-list $ST_PLIST \
                --mwpm-shots "$ST_MWPM_SHOTS" --bp-shots "$ST_BP_SHOTS" \
                --workers "$WORKERS" --resume
                
fi
step "subthreshold: teraquop extrapolation" \
    python3 "$PLOTS/subthreshold_plot_teraquop.py"

# 3. Distance + timelike: generate data (spatial + timelike) -> plot
if [[ "$SKIP_SCANS" != "1" ]]; then
    step "distance: spatial d = L (all variants)" \
        python3 "$PLOTS/spatial_distance.py"
    step "distance: timelike bounds (L=4 for the figure)" \
        env TIMELIKE_LS=4 python3 "$PLOTS/timelike_distance.py"
fi
step "figure: two-panel distance + timelike" \
    python3 "$PLOTS/distance_plot.py"

printf '\n=== Done! ===\n'