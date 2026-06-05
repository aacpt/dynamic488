"""
Ancilla-based 4.8.8 Floquet code.

Combines the ancilla lattice/circuit builder, the flow-based detector
extraction + min-weight reduction, and the detectors-with-determinism-check
builder. The shared schedule/colouring live in common.py.
"""

import stim
from collections import defaultdict, Counter
from lattice import build_lattice
from common import SCHEDULE, qubit_color


def build_ancilla_lattice(Lx, Ly):
    """
    Extend the data-qubit lattice with one ancilla per edge.

    Returns dict with:
      n_data: number of data qubits
      n_ancilla: number of ancilla qubits
      n_total: n_data + n_ancilla
      qubits, coords, edges, edges_by_color, plaquettes: as in
        lattice.build_lattice (data qubits indexed 0..n_data-1)
      ancilla_of_edge: dict mapping (q1, q2) edge tuple → ancilla qubit index
      ancilla_coords: dict mapping ancilla index → (x, y) display coordinates
    """
    lat = build_lattice(Lx, Ly)
    n_data = lat['n_qubits']

    # Allocate one ancilla per edge
    # For deterministic ordering (helpful for debugging), we iterate edges
    #   in color order r -> g -> b, and within each color in the same order
    #   build_lattice produced
    ancilla_of_edge = {}
    ancilla_coords = {}
    next_anc = n_data
    for color in 'rgb':
        for q1, q2 in lat['edges_by_color'][color]:
            edge_key = (q1, q2)
            ancilla_of_edge[edge_key] = next_anc
            # Display ancilla at the midpoint of its edge in space
            x1, y1 = lat['coords'][q1]
            x2, y2 = lat['coords'][q2]
            ancilla_coords[next_anc] = ((x1 + x2) / 2, (y1 + y2) / 2)
            next_anc += 1

    n_ancilla = next_anc - n_data
    return {
        **lat,
        'n_data': n_data,
        'n_ancilla': n_ancilla,
        'n_total': next_anc,
        'ancilla_of_edge': ancilla_of_edge,
        'ancilla_coords': ancilla_coords,
    }


def build_ancilla_circuit(Lx, Ly, n_qec, p=0.0,
                            n_warmup_periods=2, n_tail_periods=2,
                            basis='X'):
    """
    Build the ancilla-based 4.8.8 dynamic circuit, no detectors yet.

    Args:
      Lx, Ly: lattice dimensions
      n_qec: number of NOISY QEC periods
      p: depolarizing noise rate (0 = noiseless)
      n_warmup_periods, n_tail_periods: noiseless periods at start/end
      basis: 'X' or 'Z' for initial state preparation
    """
    lat = build_ancilla_lattice(Lx, Ly)
    n_data = lat['n_data']
    n_total = lat['n_total']
    coord_of_q = {v: k for k, v in lat['qubits'].items()}

    def color(q):
        i, j, d = coord_of_q[q]
        return qubit_color(i, j, d)  # 0 = A, 1 = B (the ancilla-free
                                     #   2-coloring isn't needed for the
                                     #   ancilla version, but kept for
                                     #   consistency / lols

    n_periods = n_warmup_periods + n_qec + n_tail_periods
    circuit = stim.Circuit()

    # Place qubit coordinates (data first, then ancillas)
    for q in range(n_data):
        x, y = lat['coords'][q]
        circuit.append('QUBIT_COORDS', [q], [x, y])
    for q in range(n_data, n_total):
        x, y = lat['ancilla_coords'][q]
        circuit.append('QUBIT_COORDS', [q], [x, y])

    # Initialize all DATA qubits in the chosen basis
    # (Ancillas will be initialized fresh each sub-round, so don't bother now)
    circuit.append('R', list(range(n_total)))
    if basis == 'X':
        circuit.append('H', list(range(n_data)))
    circuit.append('TICK')

    # Main loop
    for r in range(6 * n_periods):
        c, flavor = SCHEDULE[r % 6]
        period_idx = r // 6
        is_main = n_warmup_periods <= period_idx < n_warmup_periods + n_qec
        noisy = is_main and p > 0

        # For each edge of the active color, do the syndrome extraction
        # Collect all the operations into parallel layers so they happen
        #   simultaneously across all edges of this color
        edges = lat['edges_by_color'][c]
        ancillas = [lat['ancilla_of_edge'][(q1, q2)] for (q1, q2) in edges]

        # Layer 1: reset all ancillas (basis-matched)
        if flavor == 'X':
            circuit.append('RX', ancillas)
        else:
            circuit.append('R', ancillas)
        # Reset noise: a flip in the orthogonal basis after preparation
        if noisy:
            err = 'Z_ERROR' if flavor == 'X' else 'X_ERROR'
            circuit.append(err, ancillas, p)
            # All non-ancilla qubits are idle during reset; also all other
            #   ancillas (those not associated with the current color) are idle
            # For simplicity, idle = every qubit not in ancillas
            idle = [q for q in range(n_total) if q not in set(ancillas)]
            circuit.append('DEPOLARIZE1', idle, p)
        circuit.append('TICK')

        # Layer 2: first CX (data → ancilla for ZZ; ancilla -> data for XX)
        cx_pairs_1 = []
        for (q1, q2), anc in zip(edges, ancillas):
            if flavor == 'Z':
                cx_pairs_1.extend([q1, anc])  # CX(q1, anc)
            else:  # X-flavor
                cx_pairs_1.extend([anc, q1])  # CX(anc, q1)
        circuit.append('CX', cx_pairs_1)
        if noisy:
            circuit.append('DEPOLARIZE2', cx_pairs_1, p)
            # Idle qubits during this CX tick get single-qubit depolarising noise
            idle = [q for q in range(n_total) if q not in set(cx_pairs_1)]
            if idle:
                circuit.append('DEPOLARIZE1', idle, p)
        circuit.append('TICK')

        # Layer 3: second CX (data -> ancilla for ZZ; ancilla -> data for XX)
        cx_pairs_2 = []
        for (q1, q2), anc in zip(edges, ancillas):
            if flavor == 'Z':
                cx_pairs_2.extend([q2, anc])  # CX(q2, anc)
            else:
                cx_pairs_2.extend([anc, q2])
        circuit.append('CX', cx_pairs_2)
        if noisy:
            circuit.append('DEPOLARIZE2', cx_pairs_2, p)
            # Idle qubits during this CX tick get single-qubit depolarising noise
            idle = [q for q in range(n_total) if q not in set(cx_pairs_2)]
            if idle:
                circuit.append('DEPOLARIZE1', idle, p)
        circuit.append('TICK')

        # Layer 4: measure ancillas in the matching basis with M(p)/MX(p)
        #   classical readout error
        if noisy:
            idle = [q for q in range(n_total) if q not in set(ancillas)]
            circuit.append('DEPOLARIZE1', idle, p)
        if flavor == 'X':
            circuit.append('MX', ancillas, p if noisy else 0)
        else:
            circuit.append('M', ancillas, p if noisy else 0)
        circuit.append('TICK')

    return circuit, lat


def _is_identity(pauli_string):
    """Check if a stim.PauliString is the identity (all 0 or all I)."""
    return all(pauli_string[q] == 0 for q in range(len(pauli_string)))


def extract_detector_flows(circuit):
    """Extract I → I flows with non-empty measurement support."""
    flows = circuit.flow_generators()
    detectors = []
    for f in flows:
        ip, op = f.input_copy(), f.output_copy()
        if _is_identity(ip) and _is_identity(op):
            mids = list(f.measurements_copy())
            if mids:
                detectors.append(sorted(mids))
    return detectors


def reduce_basis(detectors, max_iters=50):
    """
    Greedy XOR reduction to min-weight basis.

    Repeatedly: for each pair (d_i, d_j) with i < j, if (d_i XOR d_j) has
    smaller support than max(|d_i|, |d_j|), replace the larger with the XOR.
    Continue until no more reductions are found.
    """
    dets = [set(d) for d in detectors]
    for _ in range(max_iters):
        changed = False
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                xor_set = dets[i] ^ dets[j]
                bigger = i if len(dets[i]) > len(dets[j]) else j
                if len(xor_set) < len(dets[bigger]) and len(xor_set) > 0:
                    dets[bigger] = xor_set
                    changed = True
        if not changed:
            break
    return [sorted(d) for d in dets if d]


def make_ancilla_circuit_with_detectors(Lx, Ly, n_qec, p=0.0,
                                          n_warmup_periods=2,
                                          n_tail_periods=2, basis='X',
                                          drop_topological=False, cutoff=12):
    """
    Build the noisy circuit with detectors emitted from the min-weight basis.

    Builds a noiseless skeleton, extracts flow detectors and reduce to min-weight 
    basis, optionally drops topological detectors (size > cutoff) for a graphlike 
    DEM, and builds the noisy circuit, emitting DETECTOR statements from the basis.
    """
    # Build noiseless skeleton
    plain, lat = build_ancilla_circuit(Lx, Ly, n_qec, p=0.0,
                                         n_warmup_periods=n_warmup_periods,
                                         n_tail_periods=n_tail_periods,
                                         basis=basis)

    # Extract + reduce
    raw = extract_detector_flows(plain)
    reduced = reduce_basis(raw)

    # Filter
    if drop_topological:
        det_basis = [d for d in reduced if len(d) <= cutoff]
    else:
        det_basis = reduced

    # Build noisy circuit, emitting detectors at the right times
    n_data = lat['n_data']
    n_total = lat['n_total']
    coord_of_q = {v: k for k, v in lat['qubits'].items()}

    def color(q):
        if q < n_data:
            i, j, d = coord_of_q[q]
            return qubit_color(i, j, d)
        return None

    n_periods = n_warmup_periods + n_qec + n_tail_periods

    # Group detectors by their LAST measurement index (i.e., the moment they
    #   become deterministic and should be emitted)
    det_by_last = defaultdict(list)
    for ds in det_basis:
        det_by_last[max(ds)].append(ds)

    circuit = stim.Circuit()
    for q in range(n_data):
        x, y = lat['coords'][q]
        circuit.append('QUBIT_COORDS', [q], [x, y])
    for q in range(n_data, n_total):
        x, y = lat['ancilla_coords'][q]
        circuit.append('QUBIT_COORDS', [q], [x, y])
    circuit.append('R', list(range(n_total)))
    if basis == 'X':
        circuit.append('H', list(range(n_data)))
    circuit.append('TICK')

    measurement_count = 0
    for r in range(6 * n_periods):
        c, flavor = SCHEDULE[r % 6]
        period_idx = r // 6
        is_main = n_warmup_periods <= period_idx < n_warmup_periods + n_qec
        noisy = is_main and p > 0
        edges = lat['edges_by_color'][c]
        ancillas = [lat['ancilla_of_edge'][(q1, q2)] for (q1, q2) in edges]
        idle_set = set(range(n_total)) - set(ancillas)
        idle_list = sorted(idle_set)

        # Reset ancillas
        if flavor == 'X':
            circuit.append('RX', ancillas)
        else:
            circuit.append('R', ancillas)
        if noisy:
            err = 'Z_ERROR' if flavor == 'X' else 'X_ERROR'
            circuit.append(err, ancillas, p)
            circuit.append('DEPOLARIZE1', idle_list, p)
        circuit.append('TICK')

        # First CX
        cx1 = []
        for (q1, q2), anc in zip(edges, ancillas):
            if flavor == 'Z':
                cx1.extend([q1, anc])
            else:
                cx1.extend([anc, q1])
        circuit.append('CX', cx1)
        if noisy:
            circuit.append('DEPOLARIZE2', cx1, p)
            # Idle qubits during this CX tick get DEPOLARIZE1(p) 
            # During CX1 only one endpoint per edge plus its ancilla are active,
            #   the other endpoint and all other-colour ancillas are idle
            idle_cx1 = [q for q in range(n_total) if q not in set(cx1)]
            if idle_cx1:
                circuit.append('DEPOLARIZE1', idle_cx1, p)
        circuit.append('TICK')

        # Second CX
        cx2 = []
        for (q1, q2), anc in zip(edges, ancillas):
            if flavor == 'Z':
                cx2.extend([q2, anc])
            else:
                cx2.extend([anc, q2])
        circuit.append('CX', cx2)
        if noisy:
            circuit.append('DEPOLARIZE2', cx2, p)
            idle_cx2 = [q for q in range(n_total) if q not in set(cx2)]
            if idle_cx2:
                circuit.append('DEPOLARIZE1', idle_cx2, p)
        circuit.append('TICK')

        # Measure ancillas (M(p) / MX(p) classical readout-error model)
        if noisy:
            circuit.append('DEPOLARIZE1', idle_list, p)
        if flavor == 'X':
            circuit.append('MX', ancillas, p if noisy else 0)
        else:
            circuit.append('M', ancillas, p if noisy else 0)

        # Track measurements and emit detectors that close at this sub-round
        first_idx = measurement_count
        measurement_count += len(ancillas)
        for last_meas in range(first_idx, measurement_count):
            for ds in det_by_last.get(last_meas, []):
                rels = [r2 - measurement_count for r2 in ds]
                circuit.append('DETECTOR',
                                [stim.target_rec(rr) for rr in rels],
                                [period_idx, r % 6, last_meas])
        circuit.append('TICK')

    return circuit, lat, det_basis


if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    print("Building ancilla-based 4.8.8 dynamic circuit, L=2, n_qec=1, p=0.0")
    circuit, lat = build_ancilla_circuit(2, 2, n_qec=1, p=0.0)
    print(f"  n_data: {lat['n_data']}")
    print(f"  n_ancilla: {lat['n_ancilla']}")
    print(f"  n_total: {lat['n_total']}")
    print(f"  edges by color: r={len(lat['edges_by_color']['r'])}, "
          f"g={len(lat['edges_by_color']['g'])}, "
          f"b={len(lat['edges_by_color']['b'])}")
    print(f"  circuit ticks: {circuit.num_ticks}")
    print(f"  circuit measurements: {circuit.num_measurements}")
    print()
    print("Per-period sanity: 6 sub-rounds × 8 edges/color = 48 measurements/period")
    print(f"  Total periods: 1 warmup + 1 main + 1 tail = 3")
    print(f"  Expected total measurements: 3 × 48 = 144")
    print(f"  Actual: {circuit.num_measurements}")
    print()
    print("=== First 30 lines of circuit ===")
    text = str(circuit)
    for line in text.split('\n')[:30]:
        print(f"  {line}")


if __name__ == '__main__':
    # Step 1: build noiseless circuit at L=2
    print("=== Step 1: extract detectors from noiseless L=2 ancilla circuit ===")
    circuit, lat = build_ancilla_circuit(2, 2, n_qec=3, p=0.0,
                                           n_warmup_periods=1, n_tail_periods=1)
    print(f"L=2 ancilla, n_qec=3 (5 periods): {circuit.num_measurements} measurements")

    raw_detectors = extract_detector_flows(circuit)
    raw_sizes = Counter(len(d) for d in raw_detectors)
    print(f"raw detectors from flow_generators: {len(raw_detectors)}")
    print(f"  size histogram: {dict(sorted(raw_sizes.items()))}")
    print(f"  max size: {max(len(d) for d in raw_detectors)}")
    print(f"  total support: {sum(len(d) for d in raw_detectors)}")

    print()
    print("=== Step 2: reduce to min-weight basis ===")
    reduced = reduce_basis(raw_detectors)
    red_sizes = Counter(len(d) for d in reduced)
    print(f"reduced detectors: {len(reduced)}")
    print(f"  size histogram: {dict(sorted(red_sizes.items()))}")
    print(f"  max size: {max(len(d) for d in reduced)}")
    print(f"  total support: {sum(len(d) for d in reduced)}")

    print()
    print("=== Step 3: compare to auxiliary-free version ===")
    # Build the ancilla-free version at the same parameters
    from reset import make_circuit_minweight
    af_circuit, af_lat = make_circuit_minweight(2, 2, 3, p=0.0,
                                                  n_warmup_periods=1,
                                                  n_tail_periods=1,
                                                  drop_topological=False)
    print(f"  auxiliary-free at same params: {af_circuit.num_detectors} detectors")
    # That comparison gives us the # of detectors, sizes should match
    af_sizes = Counter()
    for inst in af_circuit:
        if inst.name == 'DETECTOR':
            af_sizes[len(inst.targets_copy())] += 1
    print(f"  auxiliary-free size histogram: {dict(sorted(af_sizes.items()))}")
    print()
    print(f"  Ancilla version basis: {dict(sorted(red_sizes.items()))}")
    print(f"  Aux-free version dets: {dict(sorted(af_sizes.items()))}")
    print()
    if red_sizes == af_sizes:
        print("  ✓ Detector size distributions match!")
    else:
        print("  ! Distributions differ - call the cops!")


if __name__ == '__main__':
    print("=== Determinism check at L=2, n_qec=3, p=0 ===")
    c, lat, basis = make_ancilla_circuit_with_detectors(2, 2, 3, p=0.0,
                                                          drop_topological=False)
    print(f"Circuit: {c.num_measurements} meas, {c.num_detectors} detectors, "
          f"{len(basis)} basis elements")
    dem = c.detector_error_model(allow_gauge_detectors=False)
    print(f"DEM at p=0: {dem.num_errors} errors")
    if dem.num_errors == 0:
        print("✓ Noiseless circuit is deterministic - every detector fires predictably")
    else:
        print("✗ NON-DETERMINISTIC - there is probably a bug in the circuit construction")
        print("  First few error instructions:")
        for inst in dem:
            if inst.type == 'error':
                print(f"    {inst}")
                break

    # Also check at L=4 and L=6
    for L in (4, 6):
        print(f"\n=== Determinism check at L={L} ===")
        c, lat, basis = make_ancilla_circuit_with_detectors(L, L, 3, p=0.0)
        dem = c.detector_error_model(allow_gauge_detectors=False)
        print(f"L={L}: {c.num_detectors} detectors, DEM errors at p=0: {dem.num_errors}")
        if dem.num_errors == 0:
            print(f"  ✓ Deterministic")
        else:
            print(f"  ✗ FAILED")