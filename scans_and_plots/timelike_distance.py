"""
Timelike-distance bounds for all four variants.

Defines the measure_timelike algorithm (DEM graph analysis,
Claes 2507.08069 Appendix B) and the driver that sweeps the four variants
over L with n_qec = L, computing d_graph (upper bound) and d_hyper (lower
bound) on the timelike distance d_t.
"""

import os
import sys
import pickle
import heapq
from collections import defaultdict

import stim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from paths import results_dir
from reset import make_circuit_minweight
from noreset import make_circuit as make_noreset_circuit
from ancilla import make_ancilla_circuit_with_detectors
from pipelined import make_pipelined_ancilla_circuit_with_detectors


# Timelike-distance algorithm (Claes 2507.08069 Appendix B)

def parse_dem_to_two_graphs(dem):
    """
    Convert DEM into two graphs.

    g_graph: hyperedges (errors flipping >= 3 detectors) are ignored entirely.
             Shortest paths in this graph give an upper bound on d_t (because
             we have fewer edges available to take shortcuts through).
    g_hyper: hyperedges are expanded into a complete graph on their |h|
             vertices. Shortest paths give a lower bound on d_t (because
             clique expansion adds shortcut edges).
    """
    g_graph = defaultdict(list)
    g_hyper = defaultdict(list)
    detector_coords = {}
    n_dets = dem.num_detectors

    eid = 0
    hyperedge_count = 0
    coord_offset = [0.0] * 16

    def walk(d, scope_offset):
        nonlocal eid, hyperedge_count
        offset = list(scope_offset)
        for inst in d:
            if isinstance(inst, stim.DemRepeatBlock):
                body = inst.body_copy()
                for _ in range(inst.repeat_count):
                    walk(body, offset)
            elif isinstance(inst, stim.DemInstruction):
                if inst.type == 'shift_detectors':
                    args = inst.args_copy()
                    for k, v in enumerate(args):
                        if k < len(offset):
                            offset[k] += v
                elif inst.type == 'detector':
                    args = inst.args_copy()
                    if inst.targets_copy():
                        det_idx = inst.targets_copy()[0].val
                        coords = tuple(args[k] + (offset[k] if k < len(offset) else 0)
                                        for k in range(len(args)))
                        detector_coords[det_idx] = coords
                elif inst.type == 'error':
                    targets = inst.targets_copy()
                    dets = [t.val for t in targets if t.is_relative_detector_id()]
                    if len(dets) == 0:
                        pass
                    elif len(dets) == 1:
                        # Boundary edge: in both graphs
                        g_graph[dets[0]].append((None, eid))
                        g_hyper[dets[0]].append((None, eid))
                    elif len(dets) == 2:
                        # Pairwise edge: in both graphs
                        g_graph[dets[0]].append((dets[1], eid))
                        g_graph[dets[1]].append((dets[0], eid))
                        g_hyper[dets[0]].append((dets[1], eid))
                        g_hyper[dets[1]].append((dets[0], eid))
                    else:
                        # Hyperedge:
                        #   Ignored in g_graph (no edges added)
                        #   Expanded to clique in g_hyper
                        hyperedge_count += 1
                        for i in range(len(dets)):
                            for j in range(i + 1, len(dets)):
                                g_hyper[dets[i]].append((dets[j], eid))
                                g_hyper[dets[j]].append((dets[i], eid))
                    eid += 1
    walk(dem, coord_offset)
    return dict(g_graph), dict(g_hyper), n_dets, detector_coords, hyperedge_count


def parse_dem_to_graph(dem):
    """Wrapper: returns only the clique-expanded graph (i.e., the d_hyper graph)."""
    g_graph, g_hyper, n_dets, coords, n_hyper = parse_dem_to_two_graphs(dem)
    return g_hyper, n_dets, coords, n_hyper


def shortest_path_between_sets(graph, sources, targets, n_dets):
    """
    Shortest path from any source detector to any target.
    Returns the minimum number of faults (= edge count, all edges weight 1).
    """
    # Multi-source BFS
    BOUNDARY = -1
    dist = {}
    pq = []
    for s in sources:
        dist[s] = 0
        heapq.heappush(pq, (0, s))
    target_set = set(targets)

    best = float('inf')
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        if u in target_set:
            best = min(best, d)
            continue
        if d >= best:
            continue
        if u == BOUNDARY:
            continue  # boundary has no outgoing edges
        for v, _eid in graph.get(u, []):
            if v is None:
                v = BOUNDARY
            nd = d + 1
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return best


def default_time_from_coords(c, sub_rounds_per_period=6):
    """
    Default time extraction (deliberately conservative).

    Returns period * sub_rounds_per_period + sub_round under the assumption
    that the coords are [period_idx, sub_round, ...] - this is the
    convention used by both the ancilla-free reset circuit and the
    Davydova-style ancilla-based circuit.

    The detector time is read directly from coords[0] (period_idx). There is
    no autodetection of coordinate conventions.

    For other coord conventions (e.g. noreset's [x, y, time]) the caller
    has to pass an explicit time_from_coords callable to measure_timelike.
    """
    if len(c) >= 2:
        return int(round(c[0])) * sub_rounds_per_period + int(round(c[1]))
    return None


def measure_timelike(circuit, n_warmup_periods, n_main_periods, n_tail_periods,
                       sub_rounds_per_period=6, delta=1,
                       time_from_coords=None):
    """
    Measure both timelike-distance bounds.
    The relationship is d_graph >= d_t >= d_hyper.
    """
    dem = circuit.detector_error_model(decompose_errors=False)
    g_graph, g_hyper, n_dets, coords, hyperedge_count = parse_dem_to_two_graphs(dem)

    if not coords:
        return {'d_graph': None, 'd_hyper': None, 'hyperedge_count': hyperedge_count}

    # Time = period_idx * sub_rounds_per_period + sub_round
    # (default; can be overridden via time_from_coords parameter)
    if time_from_coords is None:
        time_from_coords = lambda c: default_time_from_coords(c, sub_rounds_per_period)
    times = {}
    for d, c in coords.items():
        if len(c) >= 2:
            t = time_from_coords(c)
            if t is not None:
                times[d] = t
    if not times:
        return {'d_graph': None, 'd_hyper': None, 'hyperedge_count': hyperedge_count}

    main_start = n_warmup_periods * sub_rounds_per_period
    main_end = (n_warmup_periods + n_main_periods) * sub_rounds_per_period

    early = [d for d, t in times.items()
             if main_start <= t < main_start + delta * sub_rounds_per_period]
    late = [d for d, t in times.items()
            if main_end - delta * sub_rounds_per_period <= t < main_end]

    if not early or not late:
        early = [d for d, t in times.items() if main_start <= t < main_start + 6]
        late = [d for d, t in times.items() if main_end - 6 <= t < main_end]
    if not early or not late:
        return {'d_graph': None, 'd_hyper': None, 'hyperedge_count': hyperedge_count}

    d_graph = shortest_path_between_sets(g_graph, early, late, n_dets)
    d_hyper = shortest_path_between_sets(g_hyper, early, late, n_dets)
    return {
        'd_graph': d_graph,
        'd_hyper': d_hyper,
        'hyperedge_count': hyperedge_count,
    }


# Driver: sweep the four variants over L with n_qec = L, write the pkl

# Canonical settings
# Spatial sizes to sweep. n_qec is tied to L (n_qec = L), the standard
#   operating point. The timelike bounds are L-independent (set by n_qec/time),
#   so the slope of d_t vs n_qec (= L) is the per-Floquet-period fault rate
L_VALUES = [int(x) for x in os.environ.get('TIMELIKE_LS', '4 6 8 10 12').split()]
P = 0.001
N_WARMUP = 2
N_TAIL = 2


def time_extractor_reset(coords):
    """
    With-reset emits coords [period_idx, sub_round, last_meas].
    Avoid auto-detection picking up last_meas.
    """
    return int(round(coords[0])) * 6 + int(round(coords[1]))


def time_extractor_xyt(coords):
    """No-reset emits [x, y, absolute_subround]; time is the last coord."""
    return int(round(coords[-1]))


def build_with_reset(L, n_qec, p):
    c, _ = make_circuit_minweight(L, L, n_qec_rounds=n_qec, p=p,
                                    n_warmup_periods=N_WARMUP,
                                    n_tail_periods=N_TAIL,
                                    drop_topological=True)
    return c, time_extractor_reset


def build_no_reset(L, n_qec, p):
    c, _ = make_noreset_circuit(L, L, n_qec_rounds=n_qec, p=p,
                                  n_warmup_periods=N_WARMUP,
                                  n_tail_periods=N_TAIL,
                                  init_basis='X')
    return c, time_extractor_xyt


def build_ancilla_unpipelined(L, n_qec, p):
    c, _, _ = make_ancilla_circuit_with_detectors(
        L, L, n_qec=n_qec, p=p,
        n_warmup_periods=N_WARMUP, n_tail_periods=N_TAIL,
        drop_topological=True)
    return c, time_extractor_reset


def build_ancilla_pipelined(L, n_qec, p):
    c, _, _ = make_pipelined_ancilla_circuit_with_detectors(
        L, L, n_qec=n_qec, p=p,
        n_warmup_periods=N_WARMUP, n_tail_periods=N_TAIL,
        drop_topological=True)
    return c, time_extractor_reset


VARIANTS = {
    # name -> (builder, ticks_per_period)
    # ticks_per_period is what to pass to measure_timelike's
    #   sub_rounds_per_period argument ("time-coord units per Floquet period")
    # All four variants index time as period*6 + sub_round: the reset and
    #   ancilla circuits emit detector coords [period, sub_round, meas_idx]
    #   (use coords[0]*6 + coords[1]); the no-reset circuit emits
    #   [x, y, absolute_subround] (use coords[-1]). Either way, 6 sub-rounds
    #   per period
    'with_reset':           (build_with_reset,           6),
    'no_reset':             (build_no_reset,             6),
    'ancilla_unpipelined':  (build_ancilla_unpipelined,  6),
    'ancilla_pipelined':    (build_ancilla_pipelined,    6),
}


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

    print(f"Measuring timelike bounds at p={P}, n_warmup={N_WARMUP}, "
          f"n_tail={N_TAIL}, n_qec = L, L in {L_VALUES}\n")

    # results[variant_name] = list of per-L rows (one row per L, with n_qec = L)
    results = {name: [] for name in VARIANTS}

    print(f"  {'variant':<22s} {'L':>3s} {'n_qec':>5s} {'dets':>5s} "
          f"{'d_graph':>7s} {'d_hyper':>7s} {'hyperE':>7s}")
    print('-' * 70)
    for L in L_VALUES:
        n_qec = L
        for variant_name, (builder, ticks_per_period) in VARIANTS.items():
            try:
                c, time_extr = builder(L, n_qec, P)
                bounds = measure_timelike(c, N_WARMUP, n_qec, N_TAIL,
                                            sub_rounds_per_period=ticks_per_period,
                                            time_from_coords=time_extr)
                d_g = bounds['d_graph']
                d_h = bounds['d_hyper']
                he = bounds['hyperedge_count']
                print(f"  {variant_name:<22s} {L:>3d} {n_qec:>5d} "
                      f"{c.num_detectors:>5d} {str(d_g):>7s} {str(d_h):>7s} {he:>7d}")
                results[variant_name].append({
                    'L': L,
                    'n_qec': n_qec,
                    'n_detectors': c.num_detectors,
                    'd_graph': d_g,
                    'd_hyper': d_h,
                    'hyperedge_count': he,
                })
            except Exception as e:
                print(f"  {variant_name:<22s} {L:>3d} {n_qec:>5d}: FAIL {str(e)[:80]}")
        print()

    # Slope of d_t vs n_qec (= L) across the sweep: faults per Floquet period
    # The bounds are L-independent (d_t is set by n_qec), so with n_qec = L the
    #   fit of d_t against n_qec gives the per-period fault rate directly
    print("Linear fit slopes (faults per Floquet period, n_qec = L):")
    print(f"  {'variant':<22s} {'slope_graph':>11s} {'slope_hyper':>11s} {'tight?':>7s}")
    print('-' * 56)
    slopes = {}
    for variant_name, rows in results.items():
        xs = np.array([r['n_qec'] for r in rows], dtype=float)
        g = np.array([r['d_graph'] if r['d_graph'] is not None else np.nan
                       for r in rows], dtype=float)
        h = np.array([r['d_hyper'] if r['d_hyper'] is not None else np.nan
                       for r in rows], dtype=float)
        g_valid = ~np.isnan(g)
        h_valid = ~np.isnan(h)
        slope_g = (float(np.polyfit(xs[g_valid], g[g_valid], 1)[0])
                   if g_valid.sum() >= 2 else float('nan'))
        slope_h = (float(np.polyfit(xs[h_valid], h[h_valid], 1)[0])
                   if h_valid.sum() >= 2 else float('nan'))
        tight = '✓' if (g_valid.sum() == h_valid.sum() and
                         np.allclose(g[g_valid], h[h_valid])) else ''
        slopes[variant_name] = {'slope_graph': slope_g, 'slope_hyper': slope_h}
        print(f"  {variant_name:<22s} {slope_g:>11.2f} {slope_h:>11.2f} {tight:>7s}")
    print()

    out_pkl = str(results_dir('timelike') / 'timelike_bounds.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'L_values': L_VALUES, 'p': P,
            'n_warmup': N_WARMUP, 'n_tail': N_TAIL,
            'n_qec_equals_L': True,
            'data': results,
            'slopes': slopes,
        }, f)
    print(f"Wrote {out_pkl}")