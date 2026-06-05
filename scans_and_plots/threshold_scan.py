"""
Threshold scan, all four variants, two decoders.

Computes per-shot LER vs (p, L) for with-reset, no-reset, ancilla un-pipelined,
and ancilla pipelined. Each cell is sampled once and decoded with both plain
MWPM (pymatching) and BP+matching (beliefmatching) on the decomposed DEM, so
both decoders see the same shots. Runs cells in parallel across processes and
writes results incrementally so the scan is resumable (re-running skips
finished cells).
"""

import os, sys, time, pickle, argparse
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)

import stim, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from reset import make_circuit_minweight
from ancilla import make_ancilla_circuit_with_detectors
from pipelined import make_pipelined_ancilla_circuit_with_detectors
from observable_helper import attach_observable
from builders import build_memory_circuit
from paths import results_dir


# Circuit builders (one per variant). Each returns a noisy memory circuit with
#   the horizontal X-cycle observable attached
def _h_observable(c, lat, L, n_total):
    qubits = lat['qubits']
    C = [qubits[(i, 0, 0)] for i in range(L)] + [qubits[(i, 0, 2)] for i in range(L)]
    Lp = stim.PauliString(n_total)
    for q in C:
        Lp[q] = 'X'
    return attach_observable(c, Lp, observable_index=0)


def build_with_reset(L, n_qec, p):
    c, lat = make_circuit_minweight(L, L, n_qec, p=p,
                                    n_warmup_periods=2, n_tail_periods=2,
                                    drop_topological=True)
    n = lat['n_qubits']
    c.append('MX', list(range(n)))
    c, _ = _h_observable(c, lat, L, n)
    return c


def build_no_reset(L, n_qec, p):
    c, _, _ = build_memory_circuit(L, n_qec, p=p,
                                   n_warmup_periods=2, n_tail_periods=2)
    return c


def build_ancilla(L, n_qec, p):
    c, lat, _ = make_ancilla_circuit_with_detectors(
        L, L, n_qec=n_qec, p=p,
        n_warmup_periods=2, n_tail_periods=2, drop_topological=True)
    n_data = lat['n_data']
    n_total = lat['n_total']
    c.append('MX', list(range(n_data)))
    c, _ = _h_observable(c, lat, L, n_total)
    return c


def build_pipelined(L, n_qec, p):
    c, lat, _ = make_pipelined_ancilla_circuit_with_detectors(
        L, L, n_qec=n_qec, p=p,
        n_warmup_periods=2, n_tail_periods=2, drop_topological=True)
    n_data = lat['n_data']
    n_total = lat['n_total']
    c.append('MX', list(range(n_data)))
    c, _ = _h_observable(c, lat, L, n_total)
    return c


BUILDERS = {
    'with_reset': build_with_reset,
    'no_reset':   build_no_reset,
    'ancilla':    build_ancilla,
    'pipelined':  build_pipelined,
}


# Scan grid
def _env_floats(name, default):
    v = os.environ.get(name)
    return [float(x) for x in v.split()] if v else default

def _env_ints(name, default):
    v = os.environ.get(name)
    return [int(x) for x in v.split()] if v else default

P_VALUES = _env_floats('THRESH_PLIST',
                       [1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3, 5e-3, 7e-3, 8e-3])
L_VALUES = _env_ints('THRESH_LS', [4, 6, 8, 12])
N_SHOTS = 10_000
BATCH = 500  # shots per sampler.sample() call (memory cap)
BP_ITERS = 20

OUT_PATH = str(results_dir('threshold') / 'threshold_results.pkl')


def _ler_se(errs, n):
    ler = errs / n
    se = float(np.sqrt(ler * (1 - ler) / n)) if 0 < ler < 1 else 1.0 / n
    return ler, se


def estimate_ler_both(variant, L, p, n_shots, batch, base_seed):
    """
    Sample once and decode with both Pymatching and BP+matching.
    Returns (n_shots, errs_mwpm, errs_bp).
    """
    import pymatching
    from beliefmatching import BeliefMatching
    circuit = BUILDERS[variant](L, L, p)
    dem = circuit.detector_error_model(decompose_errors=True,
                                       allow_gauge_detectors=False)
    mwpm = pymatching.Matching.from_detector_error_model(dem)
    bm = BeliefMatching.from_detector_error_model(dem, max_bp_iters=BP_ITERS)

    shots_done = 0
    errs_mwpm = 0
    errs_bp = 0
    seed = base_seed
    while shots_done < n_shots:
        n = min(batch, n_shots - shots_done)
        sampler = circuit.compile_detector_sampler(seed=seed)
        seed += 1
        dets, obs = sampler.sample(shots=n, separate_observables=True)
        pred_mwpm = mwpm.decode_batch(dets)
        pred_bp = bm.decode_batch(dets)
        errs_mwpm += int((pred_mwpm[:, 0] != obs[:, 0]).sum())
        errs_bp += int((pred_bp[:, 0] != obs[:, 0]).sum())
        shots_done += n
    return n_shots, errs_mwpm, errs_bp


def run_cell(args):
    variant, L, p, n_shots, batch, base_seed = args
    t0 = time.time()
    try:
        ns, errs_mwpm, errs_bp = estimate_ler_both(variant, L, p, n_shots, batch, base_seed)
    except Exception as e:
        return {'variant': variant, 'L': L, 'p': p,
                'error': str(e)[:200], 'dt': time.time() - t0}
    ler_m, se_m = _ler_se(errs_mwpm, ns)
    ler_b, se_b = _ler_se(errs_bp, ns)
    return {'variant': variant, 'L': L, 'p': p, 'shots': ns,
            'errors_mwpm': errs_mwpm, 'ler_mwpm': ler_m, 'se_mwpm': se_m,
            'errors_bp': errs_bp, 'ler_bp': ler_b, 'se_bp': se_b,
            'dt': time.time() - t0}


def load_existing(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {}


def already_done(results, variant, L, p):
    if variant not in results:
        return False
    if str(L) not in results[variant]:
        return False
    return any(r['p'] == p for r in results[variant][str(L)])


def write_results(path, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(results, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help='parallel worker processes')
    ap.add_argument('--shots', type=int, default=N_SHOTS,
                    help='shots per cell (default 1,000,000)')
    ap.add_argument('--batch', type=int, default=BATCH,
                    help='shots per sampler call (memory cap)')
    args = ap.parse_args()

    results = load_existing(OUT_PATH)

    jobs = []
    seed = 7000
    for variant in BUILDERS:
        for L in L_VALUES:
            for p in P_VALUES:
                if already_done(results, variant, L, p):
                    continue
                jobs.append((variant, L, p, args.shots, args.batch, seed))
                seed += 100

    print(f'Threshold scan (MWPM + BP+matching): {len(jobs)} cells to run on '
          f'{args.workers} worker(s), {args.shots:,} shots/cell',
          flush=True)
    if not jobs:
        print('Nothing to do; all cells already done.')
        return

    t0_global = time.time()
    done = 0

    def handle(r):
        nonlocal done
        done += 1
        if 'error' in r:
            print(f'  [{done}/{len(jobs)}] FAIL  {r["variant"]:<10} '
                  f'L={r["L"]} p={r["p"]:.4f}  {r["error"]}', flush=True)
            return
        v, L = r['variant'], r['L']
        results.setdefault(v, {}).setdefault(str(L), []).append({
            'p': r['p'], 'shots': r['shots'],
            'ler_mwpm': r['ler_mwpm'], 'se_mwpm': r['se_mwpm'], 'errors_mwpm': r['errors_mwpm'],
            'ler_bp': r['ler_bp'], 'se_bp': r['se_bp'], 'errors_bp': r['errors_bp'],
        })
        results[v][str(L)].sort(key=lambda x: x['p'])
        write_results(OUT_PATH, results)
        elapsed = (time.time() - t0_global) / 60
        print(f'  [{done}/{len(jobs)}] {r["variant"]:<10} '
              f'L={r["L"]} p={r["p"]:.4f}:  '
              f'MWPM {r["ler_mwpm"]:.5f}±{r["se_mwpm"]:.5f}  '
              f'BP {r["ler_bp"]:.5f}±{r["se_bp"]:.5f}  '
              f'({r["dt"]/60:.1f} min)  [elapsed {elapsed:.1f} min]',
              flush=True)

    if args.workers == 1:
        for j in jobs:
            handle(run_cell(j))
    else:
        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(run_cell, jobs):
                handle(r)

    print(f'\nTotal: {(time.time() - t0_global) / 60:.1f} min')
    print(f'Saved to {OUT_PATH}')


if __name__ == '__main__':
    main()