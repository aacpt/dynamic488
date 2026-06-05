"""
Reset ancilla-free 4.8.8 dynamic Floquet code.

Combines the reset circuit builder (CX-M-R-CX gadget) with the
min-weight detector basis (greedy XOR reduction of the flow_generators
output). The shared schedule/colouring live in common.py.
"""

import stim
from collections import defaultdict
from lattice import build_lattice
from common import SCHEDULE, qubit_color


def _build_noiseless_skeleton(lat, n_periods, init_basis):
    """
    Build the noiseless circuit body (no detectors, no observables).
    Used both for flow extraction and as a template for the noisy version.
    """
    n = lat['n_qubits']
    coord_of_q = {v: k for k, v in lat['qubits'].items()}

    def color(q):
        i, j, d = coord_of_q[q]
        return qubit_color(i, j, d)

    circuit = stim.Circuit()
    for q in range(n):
        x, y = lat['coords'][q]
        circuit.append('QUBIT_COORDS', [q], [x, y])
    circuit.append('R', list(range(n)))
    if init_basis == 'X':
        circuit.append('H', list(range(n)))
    elif init_basis != 'Z':
        raise ValueError(f"init_basis must be 'X' or 'Z', got {init_basis!r}")
    circuit.append('TICK')

    meas_log = []  # one entry per measurement: (sub_round, c, flavour, edge_key, meas_qubit)
    for r in range(6 * n_periods):
        c, flavour = SCHEDULE[r % 6]
        is_A_round = r % 6 in (0, 2, 4)
        ctrls, tgts, meas_q = [], [], []
        edges_used = []
        for q1, q2 in lat['edges_by_color'][c]:
            meas = q1 if (color(q1) == 0) == is_A_round else q2
            non_meas = q2 if meas == q1 else q1
            if flavour == 'Z':
                ctrls.append(non_meas); tgts.append(meas)
            else:  # X-flavour: measured qubit is the control
                ctrls.append(meas); tgts.append(non_meas)
            meas_q.append(meas)
            edges_used.append(tuple(sorted([q1, q2])))
        cx_pairs = [v for pair in zip(ctrls, tgts) for v in pair]

        # Open CX
        circuit.append('CX', cx_pairs); circuit.append('TICK')
        # Measure (basis depends on flavour)
        circuit.append('M' if flavour == 'Z' else 'MX', meas_q); circuit.append('TICK')
        # Reset (basis matches measurement)
        circuit.append('R' if flavour == 'Z' else 'RX', meas_q); circuit.append('TICK')
        # Close CX (same as open)
        circuit.append('CX', cx_pairs); circuit.append('TICK')

        for ek, mq in zip(edges_used, meas_q):
            meas_log.append((r, c, flavour, ek, mq))
    return circuit, meas_log


def _is_identity(pauli_string):
    s = str(pauli_string).replace('+', '').replace('-', '').replace('_', '')
    return s == ''


def _extract_detector_flows(plain_circuit):
    """
    Use stim.Circuit.flow_generators() to find every linear combination of
    measurements that is deterministically zero. Returns sorted lists of
    measurement indices (each list is one detector).
    """
    flows = plain_circuit.flow_generators()
    out = []
    for f in flows:
        if _is_identity(f.input_copy()) and _is_identity(f.output_copy()):
            recs = list(f.measurements_copy())
            if recs:
                out.append(sorted(recs))
    return out


def make_circuit(Lx, Ly, n_qec_rounds, p=0.0,
                 n_warmup_periods=2, n_tail_periods=2, init_basis='X'):
    """
    Build the reset ancilla-free 4.8.8 dynamic Floquet circuit.

    Layout:
      - 1 init layer (R^n; H^n if init_basis='X')
      - n_warmup_periods noiseless periods (no detectors fire from warmup-only
        events because those measurements are still part of the global
        flow_generators set, but the NOISE injection only happens in main).
      - n_qec_rounds noisy periods.
      - n_tail_periods noiseless periods (so that final-period detectors
        compare against deterministic boundary measurements).

    The circuit has DETECTOR statements but no OBSERVABLE_INCLUDE -
    observable_helper.attach_observable() adds a logical operator.
    """
    lat = build_lattice(Lx, Ly)
    n = lat['n_qubits']
    coord_of_q = {v: k for k, v in lat['qubits'].items()}

    def color(q):
        i, j, d = coord_of_q[q]
        return qubit_color(i, j, d)

    n_periods = n_warmup_periods + n_qec_rounds + n_tail_periods

    # Step 1: build the noiseless skeleton, run flow analysis, collect detector flows
    plain, meas_log = _build_noiseless_skeleton(lat, n_periods, init_basis)
    detector_flows = _extract_detector_flows(plain)

    # Index detectors by their LAST measurement so we know where to emit them
    det_by_last = defaultdict(list)
    for ds in detector_flows:
        det_by_last[max(ds)].append(ds)

    # Step 2: rebuild the circuit, this time with noise + DETECTOR statements
    circuit = stim.Circuit()
    for q in range(n):
        x, y = lat['coords'][q]
        circuit.append('QUBIT_COORDS', [q], [x, y])
    circuit.append('R', list(range(n)))
    if init_basis == 'X':
        circuit.append('H', list(range(n)))
    circuit.append('TICK')

    measurement_count = 0
    for r in range(6 * n_periods):
        c, flavour = SCHEDULE[r % 6]
        is_A_round = r % 6 in (0, 2, 4)
        period_idx = r // 6
        is_main = n_warmup_periods <= period_idx < n_warmup_periods + n_qec_rounds
        noisy = is_main and p > 0

        ctrls, tgts, meas_q, non_meas_q = [], [], [], []
        for q1, q2 in lat['edges_by_color'][c]:
            meas = q1 if (color(q1) == 0) == is_A_round else q2
            non_meas = q2 if meas == q1 else q1
            if flavour == 'Z':
                ctrls.append(non_meas); tgts.append(meas)
            else:
                ctrls.append(meas); tgts.append(non_meas)
            meas_q.append(meas)
            non_meas_q.append(non_meas)
        cx_pairs = [v for pair in zip(ctrls, tgts) for v in pair]

        # Noise model (matches make_circuit_minweight below, the path used for results):
        #   - DEPOLARIZE2(p) after each CX
        #   - DEPOLARIZE1(p) on idle (non-measured) qubits during the M/R ticks
        #   - M(p)/MX(p) classical readout error (Stim noisy measurement): the
        #     recorded bit is flipped with probability p
        #   - X_ERROR (or Z_ERROR) after R (or RX): state-preparation error
        # In this ancilla-free schedule every data qubit is on every colour's
        #   check edges, so each CX tick has no idle qubits, and the non_meas_q
        #   during the M/R ticks are the idle qubits

        circuit.append('CX', cx_pairs)
        if noisy: circuit.append('DEPOLARIZE2', cx_pairs, p)
        circuit.append('TICK')

        if noisy:
            circuit.append('DEPOLARIZE1', non_meas_q, p)
        if flavour == 'Z':
            circuit.append('M', meas_q, p if noisy else 0)
        else:
            circuit.append('MX', meas_q, p if noisy else 0)
        circuit.append('TICK')

        circuit.append('R' if flavour == 'Z' else 'RX', meas_q)
        if noisy:
            circuit.append('X_ERROR' if flavour == 'Z' else 'Z_ERROR', meas_q, p)
            circuit.append('DEPOLARIZE1', non_meas_q, p)
        circuit.append('TICK')

        circuit.append('CX', cx_pairs)
        if noisy: circuit.append('DEPOLARIZE2', cx_pairs, p)
        circuit.append('TICK')

        first_idx = measurement_count
        measurement_count += len(meas_q)
        for last_meas in range(first_idx, measurement_count):
            for ds in det_by_last.get(last_meas, []):
                rels = [r2 - measurement_count for r2 in ds]
                tgts_rec = [stim.target_rec(rr) for rr in rels]
                circuit.append('DETECTOR', tgts_rec, [period_idx, r % 6, last_meas])

    return circuit, lat


def reduce_basis(detectors, max_iters=50):
    """
    Greedy pairwise XOR reduction. Returns a basis spanning the same GF(2)
    subspace with smaller total support.
    """
    basis = [set(d) for d in detectors]
    n = len(basis)
    for _ in range(max_iters):
        changed = 0
        # Inverted index: measurement -> set of basis indices that include it
        inv = defaultdict(set)
        for i, b in enumerate(basis):
            for m in b:
                inv[m].add(i)
        for i in range(n):
            bi = basis[i]
            if not bi:
                continue
            best_j = None
            best_size = len(bi)
            cand = set()
            for m in bi:
                cand |= inv[m]
            cand.discard(i)
            for j in cand:
                bj = basis[j]
                if not bj:
                    continue
                xor = bi ^ bj
                if 0 < len(xor) < best_size:
                    best_size = len(xor)
                    best_j = j
            if best_j is not None:
                basis[i] = bi ^ basis[best_j]
                changed += 1
        if changed == 0:
            break
    return [tuple(sorted(b)) for b in basis if b]


def _local_cutoff(reduced):
    """Cutoff for removing topological detectors."""
    LOCAL_STEP = 4
    uniq = sorted({len(d) for d in reduced})
    cut = uniq[0] if uniq else 0
    for a, b in zip(uniq, uniq[1:]):
        if b - a > LOCAL_STEP:
            break
        cut = b
    return cut


def make_circuit_minweight(Lx, Ly, n_qec_rounds, p=0.0,
                            n_warmup_periods=2, n_tail_periods=2,
                            init_basis='X', drop_topological=True,
                            topological_size_cutoff=None):
    """
    Build the reset circuit with min-weight detector basis.

    Parameters:
      drop_topological: if True, drop detectors of size > the cutoff.
                         These are Wilson-loop detectors on the torus that
                         can't be reduced by local XOR. Dropping them makes
                         the DEM graphlike at the cost of a strictly local
                         (rather than complete) detector basis.
      topological_size_cutoff: detectors above this size are considered
                                topological. If None (default), the cutoff is
                                auto-detected per circuit as the top of the
                                local-size cluster (the first gap above the
                                small local detectors), which stays correct as
                                L grows. Pass an int to override.
    """
    lat = build_lattice(Lx, Ly)
    n = lat['n_qubits']
    coord_of_q = {v: k for k, v in lat['qubits'].items()}

    def color(q):
        i, j, d = coord_of_q[q]
        return qubit_color(i, j, d)

    n_periods = n_warmup_periods + n_qec_rounds + n_tail_periods
    plain, _ = _build_noiseless_skeleton(lat, n_periods, init_basis)

    raw_dets = _extract_detector_flows(plain)
    reduced = reduce_basis(raw_dets)
    if drop_topological:
        if topological_size_cutoff is None:
            cut = _local_cutoff(reduced)
        else:
            cut = topological_size_cutoff
        kept = [d for d in reduced if len(d) <= cut]
        dropped = len(reduced) - len(kept)
        # Topological detectors are rare, should be a small minority
        if reduced and dropped > 0.25 * len(reduced):
            raise RuntimeError(
                "make_circuit_minweight: size-based topological cutoff would "
                f"drop {dropped}/{len(reduced)} detectors at Lx={Lx}, Ly={Ly} "
                f"(cutoff={cut}). That's too many, something's funky.")
        reduced = kept

    det_by_last = defaultdict(list)
    for ds in reduced:
        det_by_last[max(ds)].append(ds)

    circuit = stim.Circuit()
    for q in range(n):
        x, y = lat['coords'][q]
        circuit.append('QUBIT_COORDS', [q], [x, y])
    circuit.append('R', list(range(n)))
    if init_basis == 'X':
        circuit.append('H', list(range(n)))
    circuit.append('TICK')

    measurement_count = 0
    for r in range(6 * n_periods):
        c, flavor = SCHEDULE[r % 6]
        is_A_round = r % 6 in (0, 2, 4)
        period_idx = r // 6
        is_main = n_warmup_periods <= period_idx < n_warmup_periods + n_qec_rounds
        noisy = is_main and p > 0

        ctrls, tgts, meas_q, non_meas_q = [], [], [], []
        for q1, q2 in lat['edges_by_color'][c]:
            meas = q1 if (color(q1) == 0) == is_A_round else q2
            non_meas = q2 if meas == q1 else q1
            if flavor == 'Z':
                ctrls.append(non_meas); tgts.append(meas)
            else:
                ctrls.append(meas); tgts.append(non_meas)
            meas_q.append(meas); non_meas_q.append(non_meas)
        cx_pairs = [v for pair in zip(ctrls, tgts) for v in pair]

        circuit.append('CX', cx_pairs)
        if noisy: circuit.append('DEPOLARIZE2', cx_pairs, p)
        circuit.append('TICK')

        # Classical readout error (Stim noisy measurement): the underlying
        #   projection is correct and the recorded bit is flipped with
        #   probability p
        if noisy:
            circuit.append('DEPOLARIZE1', non_meas_q, p)
        if flavor == 'Z':
            circuit.append('M', meas_q, p if noisy else 0)
        else:
            circuit.append('MX', meas_q, p if noisy else 0)
        circuit.append('TICK')

        circuit.append('R' if flavor == 'Z' else 'RX', meas_q)
        if noisy:
            # Post-reset Pauli error 
            circuit.append('X_ERROR' if flavor == 'Z' else 'Z_ERROR', meas_q, p)
            circuit.append('DEPOLARIZE1', non_meas_q, p)
        circuit.append('TICK')

        circuit.append('CX', cx_pairs)
        if noisy: circuit.append('DEPOLARIZE2', cx_pairs, p)
        circuit.append('TICK')

        first_idx = measurement_count
        measurement_count += len(meas_q)
        for last_meas in range(first_idx, measurement_count):
            for ds in det_by_last.get(last_meas, []):
                rels = [r2 - measurement_count for r2 in ds]
                circuit.append('DETECTOR',
                               [stim.target_rec(rr) for rr in rels],
                               [period_idx, r % 6, last_meas])

    return circuit, lat


if __name__ == '__main__':
    # Quick verification
    print("Quick determinism check (Reset 4.8.8, L=Lx=Ly=4, n_qec=3, p=0)")
    circuit, lat = make_circuit(4, 4, 3, p=0.0)
    print(f"  measurements={circuit.num_measurements}, "
          f"detectors={circuit.num_detectors}")
    dem = circuit.detector_error_model(allow_gauge_detectors=False)
    print(f"  noiseless DEM: {dem.num_detectors} detectors, "
          f"{dem.num_errors} errors - determinism {'OK' if dem.num_errors == 0 else 'FAIL'}")


if __name__ == '__main__':
    print("Min-weight Reset 4.8.8: quick sanity check")
    for L in [4, 6]:
        for drop in [False, True]:
            c, lat = make_circuit_minweight(L, L, 3, p=0.0, drop_topological=drop)
            dem0 = c.detector_error_model(allow_gauge_detectors=False)
            print(f"  L={L} drop_topological={drop}: {c.num_detectors} dets, "
                  f"p=0 errs={dem0.num_errors}")