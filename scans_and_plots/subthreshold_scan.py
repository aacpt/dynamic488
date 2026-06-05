"""
Sub-threshold scaling scans for all four circuit variants.

Goal: measure LER(L, p) for L in at p well below threshold so we
can extrapolate the dynamic-circuit's power-law-decay-with-d.
"""

from __future__ import annotations

import argparse, pickle, os, sys, time
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import results_dir

import numpy as np
import stim
import pymatching


# Variant builders - return (circuit, lattice) ready to sample
def build_with_reset(L, n_qec, p):
    from reset import make_circuit_minweight
    from observable_helper import attach_observable
    c, lat = make_circuit_minweight(L, L, n_qec_rounds=n_qec, p=p,
                                    n_warmup_periods=2, n_tail_periods=2,
                                    drop_topological=True)
    n = lat['n_qubits']
    c.append('MX', list(range(n)))
    qubits = lat['qubits']
    C = [qubits[(i, 0, 0)] for i in range(L)] + [qubits[(i, 0, 2)] for i in range(L)]
    L_p = stim.PauliString(n)
    for q in C:
        L_p[q] = 'X'
    c, _ = attach_observable(c, L_p, 0)
    return c


def build_no_reset(L, n_qec, p):
    from noreset import make_circuit
    from observable_helper import attach_observable
    # Noreset's make_circuit doesn't accept (or need) drop_topological 
    c, lat = make_circuit(L, L, n_qec_rounds=n_qec, p=p,
                          n_warmup_periods=2, n_tail_periods=2)
    n = lat['n_qubits']
    c.append('MX', list(range(n)))
    qubits = lat['qubits']
    C = [qubits[(i, 0, 0)] for i in range(L)] + [qubits[(i, 0, 2)] for i in range(L)]
    L_p = stim.PauliString(n)
    for q in C:
        L_p[q] = 'X'
    c, _ = attach_observable(c, L_p, 0)
    return c


def build_ancilla(L, n_qec, p):
    from ancilla import make_ancilla_circuit_with_detectors
    from observable_helper import attach_observable
    c, lat, _ = make_ancilla_circuit_with_detectors(L, L, n_qec=n_qec, p=p,
                                                       n_warmup_periods=2, n_tail_periods=2,
                                                       drop_topological=True)
    n_data = lat['n_data']
    n_total = lat['n_total']
    c.append('MX', list(range(n_data)))
    qubits = lat['qubits']
    C = [qubits[(i, 0, 0)] for i in range(L)] + [qubits[(i, 0, 2)] for i in range(L)]
    L_p = stim.PauliString(n_total)
    for q in C:
        L_p[q] = 'X'
    c, _ = attach_observable(c, L_p, 0)
    return c


def build_pipelined(L, n_qec, p):
    from pipelined import make_pipelined_ancilla_circuit_with_detectors
    from observable_helper import attach_observable
    c, lat, _ = make_pipelined_ancilla_circuit_with_detectors(L, L, n_qec=n_qec, p=p,
                                                                  n_warmup_periods=2, n_tail_periods=2,
                                                                  drop_topological=True)
    n_data = lat['n_data']
    n_total = lat['n_total']
    c.append('MX', list(range(n_data)))
    qubits = lat['qubits']
    C = [qubits[(i, 0, 0)] for i in range(L)] + [qubits[(i, 0, 2)] for i in range(L)]
    L_p = stim.PauliString(n_total)
    for q in C:
        L_p[q] = 'X'
    c, _ = attach_observable(c, L_p, 0)
    return c


VARIANTS = {
    'with_reset': build_with_reset,
    'no_reset':   build_no_reset,
    'ancilla':    build_ancilla,
    'pipelined':  build_pipelined,
}


# Constants / helpers
BP_ITERS = 20  # BP iterations, matching the threshold scan


def _ler_se(errs, n):
    n = max(n, 1)
    ler = errs / n
    se = float(np.sqrt(ler * (1 - ler) / n)) if 0 < ler < 1 else 1.0 / n
    return ler, se


# Fixed-shot sampler - decode a fixed number of shots with each decoder
def fixed_sample(circuit, mwpm_shots, bp_shots, batch_size=100_000):
    """Decode a FIXED number of shots with each decoder, no adaptive stopping:
    `mwpm_shots` with MWPM (cheap) and the first `bp_shots` of the same sample
    with BP+matching (slow, so given its own smaller fixed count). The two
    decoders thus see the same shots up to bp_shots. Set bp_shots=0 to skip BP.

    Returns (mwpm_shots, errs_mwpm, errs_bp, bp_done, elapsed_seconds).
    """
    from beliefmatching import BeliefMatching
    # No ignore_decomposition_failures here 
    dem = circuit.detector_error_model(
        decompose_errors=True, allow_gauge_detectors=False)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    bm = (BeliefMatching.from_detector_error_model(dem, max_bp_iters=BP_ITERS)
          if bp_shots > 0 else None)
    sampler = circuit.compile_detector_sampler()

    done = 0
    bp_done = 0
    errs_mwpm = 0
    errs_bp = 0
    t0 = time.time()
    while done < mwpm_shots:
        n = min(batch_size, mwpm_shots - done)
        det, obs = sampler.sample(shots=n, separate_observables=True)
        pred_m = matcher.decode_batch(det)
        errs_mwpm += int(np.sum(np.any(pred_m != obs, axis=1)))
        # BP decodes the same shots, but only up to its own fixed count.
        if bm is not None and bp_done < bp_shots:
            m = min(n, bp_shots - bp_done)
            pred_b = bm.decode_batch(det[:m])
            errs_bp += int(np.sum(np.any(pred_b != obs[:m], axis=1)))
            bp_done += m
        done += n

    return mwpm_shots, errs_mwpm, errs_bp, bp_done, time.time() - t0


# Builds the circuit, runs the fixed both-decoder sample, returns a
#   self-contained result dict (or a {'failed': ...} dict on error)
def run_cell(job):
    variant, L, p, mwpm_shots, bp_shots, batch_size = job
    n_qec = max(L, 3)
    t0 = time.time()
    try:
        circuit = VARIANTS[variant](L, n_qec, p)
        build_t = time.time() - t0
        shots, errs_mwpm, errs_bp, bp_done, sample_t = fixed_sample(
            circuit, mwpm_shots, bp_shots, batch_size)
    except Exception as e:
        return {'variant': variant, 'L': L, 'p': p, 'n_qec': n_qec,
                'failed': str(e)[:300]}
    ler_m, se_m = _ler_se(errs_mwpm, shots)
    ler_b, se_b = _ler_se(errs_bp, bp_done)   # BP uses its own fixed shot count
    return {
        'variant': variant, 'L': L, 'p': p, 'n_qec': n_qec, 'shots': shots,
        # legacy keys (== MWPM) kept so the existing plotter keeps working
        'errors': errs_mwpm, 'ler': ler_m, 'se': se_m,
        # explicit per-decoder keys (BP has its own shot count, bp_shots)
        'errors_mwpm': errs_mwpm, 'ler_mwpm': ler_m, 'se_mwpm': se_m,
        'bp_shots': bp_done,
        'errors_bp': errs_bp, 'ler_bp': ler_b, 'se_bp': se_b,
        'build_seconds': build_t, 'sample_seconds': sample_t,
    }


# Resumable result store
def load_results(path):
    if not os.path.exists(path):
        return []
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_results(path, results):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(results, f)
    os.replace(tmp, path)


def already_done(results, variant, L, p):
    for r in results:
        if r['variant'] == variant and r['L'] == L and abs(r['p'] - p) < 1e-9:
            return True
    return False


# Main scan loop
def run_scan(args):
    out_path = str(results_dir('subthreshold') / args.output)
    results = load_results(out_path)
    if results:
        print(f"Loaded {len(results)} existing cells from {out_path}")

    variants = args.variants
    if 'all' in variants:
        variants = ['with_reset', 'no_reset', 'ancilla', 'pipelined']

    cells = [(v, L, p) for v in variants for L in args.L for p in args.p_list]

    # Build the job list, skipping cells already present in the checkpoint
    jobs = []
    skipped = 0
    for variant, L, p in cells:
        if already_done(results, variant, L, p):
            skipped += 1
            continue
        jobs.append((variant, L, p, args.mwpm_shots, args.bp_shots,
                     args.batch_size))

    print(f"Plan: {len(cells)} cells "
          f"({len(variants)} variants × {len(args.L)} L × {len(args.p_list)} p); "
          f"{skipped} already done, {len(jobs)} to run on {args.workers} worker(s)")
    print(f"Fixed shots per cell: MWPM {args.mwpm_shots:,}, "
          f"BP {args.bp_shots:,}")
    print(flush=True)
    if not jobs:
        print("Nothing to do; all cells already in results.")
        return

    t0_global = time.time()
    done = 0

    def handle(r):
        nonlocal done
        done += 1
        results.append(r)
        save_results(out_path, results)          # checkpoint after every cell
        tag = f"[{done}/{len(jobs)}] {r['variant']:<10} L={r['L']} p={r['p']:.4f}"
        if 'failed' in r:
            print(f"  {tag}  FAILED: {r['failed']}", flush=True)
            return
        elapsed = (time.time() - t0_global) / 60
        print(f"  {tag}  MWPM {r['errors_mwpm']:>6} err/{r['shots']:,} "
              f"LER={r['ler_mwpm']:.2e}±{r['se_mwpm']:.1e}  "
              f"BP {r['errors_bp']:>5} err/{r['bp_shots']:,} "
              f"LER={r['ler_bp']:.2e}±{r['se_bp']:.1e}  "
              f"({r['sample_seconds']:.0f}s)  [elapsed {elapsed:.1f} min]",
              flush=True)

    if args.workers == 1:
        for j in jobs:
            handle(run_cell(j))
    else:
        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(run_cell, jobs):
                handle(r)

    print()
    print(f"Total: {(time.time() - t0_global) / 60:.1f} min")
    print(f"Wrote {out_path}")


# CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+',
                        choices=['with_reset', 'no_reset', 'ancilla', 'pipelined', 'all'],
                        default=['all'],
                        help='Which variants to scan (default: all four)')
    parser.add_argument('--L', nargs='+', type=int, default=[6, 8, 10, 12],
                        help='Lattice sizes (default: 6 8 10 12)')
    parser.add_argument('--p-list', nargs='+', type=float,
                        default=[0.001],
                        help='Physical error rates (default: 1e-3 1.5e-3 2e-3)')
    parser.add_argument('--mwpm-shots', type=int, default=10_000_000,
                        help='Fixed shots decoded with MWPM per cell (default: 1e7). '
                             'Cheap, so high — gives good points down to LER ~1e-6.')
    parser.add_argument('--bp-shots', type=int, default=100_000,
                        help='Fixed shots decoded with BP+matching per cell '
                             '(default: 1e5). BP is ~10-100x slower than MWPM, so '
                             'this sets the wall-clock; set 0 to skip BP entirely.')
    parser.add_argument('--workers', type=int,
                        default=max(1, (os.cpu_count() or 2) - 1),
                        help='Parallel worker processes, one cell each '
                             '(default: cpu_count-1). Use 1 for serial/debugging.')
    parser.add_argument('--batch-size', type=int, default=100_000,
                        help='Shots per Stim sampler call (default: 100k). Smaller '
                             'keeps per-worker memory down when many workers run.')
    parser.add_argument('--output', default='subthreshold_results_all_variants.pkl',
                        help='Output pickle filename (written to results/subthreshold/)')
    parser.add_argument('--resume', action='store_true',
                        help='If the output pickle exists, skip cells already done')

    args = parser.parse_args()
    run_scan(args)


if __name__ == '__main__':
    main()