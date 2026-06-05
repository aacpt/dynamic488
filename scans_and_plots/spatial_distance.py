"""
Measures the spatial code distance d = L for all four variants using two
independent methods, and writes the result to a pickle consumed by
distance_plot.py (left panel of Fig. 4).

  (i)  shortest_graphlike_error          - minimum-weight graphlike logical
                                            fault (ignores hyperedges).
  (ii) search_for_undetectable_logical_errors  - full circuit-fault search,
                                            no graphlike approximation
                                            (explores hyperedges too).

Assumptions:
  - n_warmup_periods = n_tail_periods = 2. At 1/1 the graphlike decomposition
    produces a boundary artifact (apparent d=1) that the full search correctly
    resolves as d = L.
  - n_qec = L noisy periods.
  - X-basis memory experiment; logical X-bar on the horizontal non-contractible
    cycle (support 2L).
"""

import os
import sys
import time
import pickle

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

import stim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from reset import make_circuit_minweight                       # reset
from noreset import make_circuit as make_noreset_circuit           # no-reset
from ancilla import make_ancilla_circuit_with_detectors           # ancilla
from pipelined import (                                   # pipelined
    make_pipelined_ancilla_circuit_with_detectors,
)
from observable_helper import attach_observable
from paths import results_dir


# Circuit builders. Each returns a noisy memory circuit with the horizontal
#   X-bar observable attached and final destructive MX readout. All use
#   n_warmup = n_tail = 2 and the X-cycle on row j = 0 (support 2L)
N_WARMUP = 2
N_TAIL = 2
P = 1e-3


def _attach_horizontal_X(circuit, lat, L, n_total):
    """
    Append destructive MX on all data qubits and attach X-bar on the
    horizontal non-contractible cycle (N and S qubits of row j = 0).
    """
    qubits = lat['qubits']
    cycle = [qubits[(i, 0, 0)] for i in range(L)] + \
            [qubits[(i, 0, 2)] for i in range(L)]
    pauli = stim.PauliString(n_total)
    for q in cycle:
        pauli[q] = 'X'
    circuit, _ = attach_observable(circuit, pauli, observable_index=0)
    return circuit


def build_reset(L, n_qec):
    c, lat = make_circuit_minweight(
        L, L, n_qec, p=P,
        n_warmup_periods=N_WARMUP, n_tail_periods=N_TAIL,
        init_basis='X', drop_topological=True)
    n = lat['n_qubits']
    c.append('MX', list(range(n)))
    return _attach_horizontal_X(c, lat, L, n)


def build_no_reset(L, n_qec):
    c, lat = make_noreset_circuit(
        L, L, n_qec, p=P,
        n_warmup_periods=N_WARMUP, n_tail_periods=N_TAIL,
        init_basis='X')
    n = lat['n_qubits']
    c.append('MX', list(range(n)))
    return _attach_horizontal_X(c, lat, L, n)


def build_ancilla(L, n_qec):
    c, lat, _ = make_ancilla_circuit_with_detectors(
        L, L, n_qec=n_qec, p=P,
        n_warmup_periods=N_WARMUP, n_tail_periods=N_TAIL,
        drop_topological=True, cutoff=24)
    c.append('MX', list(range(lat['n_data'])))
    return _attach_horizontal_X(c, lat, L, lat['n_total'])


def build_pipelined(L, n_qec):
    c, lat, _ = make_pipelined_ancilla_circuit_with_detectors(
        L, L, n_qec=n_qec, p=P,
        n_warmup_periods=N_WARMUP, n_tail_periods=N_TAIL,
        drop_topological=True, cutoff=24)
    c.append('MX', list(range(lat['n_data'])))
    return _attach_horizontal_X(c, lat, L, lat['n_total'])


# (canonical_key, display_label, builder) - canonical keys match the variant
#   keys used by timelike_distance.py and distance_plot.py
VARIANTS = [
    ('with_reset',          'reset',             build_reset),
    ('no_reset',            'no-reset',          build_no_reset),
    ('ancilla_unpipelined', 'ancilla',           build_ancilla),
    ('ancilla_pipelined',   'ancilla pipelined', build_pipelined),
]


# Distance methods
def d_graphlike(circuit):
    """Method (i): minimum-weight graphlike logical fault."""
    return len(circuit.shortest_graphlike_error())


def d_full(circuit):
    """Method (ii): full undetectable-error search, no graphlike approximation."""
    err = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=8,
        dont_explore_edges_with_degree_above=20,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True)
    return len(err)


GRAPHLIKE_LS = [int(x) for x in
                os.environ.get('GRAPHLIKE_LS', '2 4 6 8 10 12').split()]
FULL_LS = {
    'with_reset':          [2, 4] + ([6] if os.environ.get('RESET_FULL_L6') else []),
    'no_reset':            [2, 4],
    'ancilla_unpipelined': [2, 4],
    'ancilla_pipelined':   [2, 4],
}


def main():
    print("=" * 74)
    print("Spatial distance d = L : both methods, all four variants")
    print(f"(n_warmup = n_tail = {N_WARMUP}, n_qec = L, p = {P})")
    print("=" * 74)

    all_ok = True
    # graphlike[key][L] = d ; full[key][L] = d
    graphlike = {key: {} for key, _, _ in VARIANTS}
    full = {key: {} for key, _, _ in VARIANTS}

    # Method (i): shortest graphlike error 
    print("\n[i] shortest_graphlike_error  (expect d = L)")
    print(f"  {'variant':<18} " + "  ".join(f"L={L}" for L in GRAPHLIKE_LS))
    for key, label, build in VARIANTS:
        row = []
        for L in GRAPHLIKE_LS:
            c = build(L, L)
            d = d_graphlike(c)
            graphlike[key][L] = d
            ok = (d == L)
            all_ok &= ok
            row.append(f"{d}{'✓' if ok else '✗'}")
        print(f"  {label:<18} " + "   ".join(f"{r:<4}" for r in row))

    # Method (ii): full undetectable-error search 
    print("\n[ii] search_for_undetectable_logical_errors  (no graphlike approx.)")
    for key, label, build in VARIANTS:
        cells = []
        for L in FULL_LS[key]:
            t0 = time.time()
            c = build(L, L)
            d = d_full(c)
            full[key][L] = d
            ok = (d == L)
            all_ok &= ok
            cells.append(f"L={L}: d={d}{'✓' if ok else '✗'} ({time.time()-t0:.1f}s)")
        print(f"  {label:<18} " + "   ".join(cells))


    out_pkl = str(results_dir('spatial') / 'spatial_distance.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'p': P, 'n_warmup': N_WARMUP, 'n_tail': N_TAIL,
            'graphlike_ls': GRAPHLIKE_LS,
            'full_ls': {k: list(v) for k, v in FULL_LS.items()},
            'labels': {key: label for key, label, _ in VARIANTS},
            'graphlike': graphlike,
            'full': full,
        }, f)
    print(f"\nWrote {out_pkl}")

    print("\n" + "=" * 74)
    print(f"RESULT: {'ALL PASS — d = L confirmed by both methods' if all_ok else 'FAIL'}")
    print("CSS X<->Z symmetry implies the same distance for the Z-type logicals.")
    print("=" * 74)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())