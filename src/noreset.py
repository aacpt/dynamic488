"""
No-reset ancilla-free 4.8.8 Floquet code.

SImilar to arXiv:2512.17999v2, without the H-wrapping:
  - For a ZZ check on edge (a, b):  CX a->b -> M  b -> CX a->b
  - For an XX check on edge (a, b): CX b->a -> MX b -> CX b->a (direct MX)

Heisenberg picture for the XX gadget:
  X_a X_b  --CX b->a-->  X_a · (X_b X_a) = X_b
  -> MX b reads X_a X_b on the pre-CX state.
"""

import stim
from collections import defaultdict
from lattice import build_lattice


# Schedule
SCHEDULE = [('r', 'X'), ('g', 'Z'), ('b', 'X'), ('r', 'Z'), ('g', 'X'), ('b', 'Z')]


def find_plaquette_inferences(lat, sub_color):
    """
    Return a dict mapping (plaq_color, plaq_idx) -> list of edges of sub_color
    around that plaquette. These edges' check outcomes multiply to give the plaquette
    stabilizer in the current Pauli basis.
    """
    edge_set = set(tuple(sorted([q1, q2])) for q1, q2 in lat['edges_by_color'][sub_color])
    inferences = {}
    for plaq_color, plist in lat['plaquettes'].items():
        if plaq_color == sub_color:
            # Around a same-coloured plaquette there are no edges of that colour
            continue
        for plaq_idx, verts in enumerate(plist):
            edges_on = []
            n = len(verts)
            for k in range(n):
                e = tuple(sorted([verts[k], verts[(k+1) % n]]))
                if e in edge_set:
                    edges_on.append(e)
            if edges_on:
                inferences[(plaq_color, plaq_idx)] = edges_on
    return inferences


def make_circuit(Lx, Ly, n_qec_rounds, p=0.0, n_warmup_periods=2, n_tail_periods=2,
                 init_basis='X'):
    """
    Build the no-reset ancilla-free 4.8.8 circuit.

    Structure:
      - Init: R all qubits in Z, then H all to make |+⟩^n  (init_basis='X')
              or just R all qubits to make |0⟩^n           (init_basis='Z').
      - Warmup: n_warmup_periods × 6 noiseless sub-rounds (no detectors).
      - Main: n_qec_rounds × 6 noisy sub-rounds with detectors.
      - Tail: n_tail_periods × 6 noiseless sub-rounds with detectors.

    init_basis='X' is for when the logical is X̄ over a horizontal cycle
    (X-basis memory test). init_basis='Z' is for Z̄ memory tests.
    """
    lat = build_lattice(Lx, Ly)
    n = lat['n_qubits']
    
    circuit = stim.Circuit()
    
    # Qubit coords
    for q in range(n):
        x, y = lat['coords'][q]
        circuit.append('QUBIT_COORDS', [q], [x, y])
    
    # Initial state |+⟩^n or |0⟩^n
    circuit.append('R', list(range(n)))
    if init_basis == 'X':
        circuit.append('H', list(range(n)))
    elif init_basis != 'Z':
        raise ValueError(f"init_basis must be 'X' or 'Z', got {init_basis!r}")
    circuit.append('TICK')
    
    # Tracking
    measurement_count = [0]
    # plaq_history[(plaq_color, plaq_idx, pauli)] = list of (sub_round_idx, [global_rec_indices])
    plaq_history = defaultdict(list)
    
    # Pre-compute, for each sub-round template, the plaquette inferences and edge orderings
    # For consistency, fix an ordering of edges per sub-round
    sub_inferences = {}   # color -> {(plaq_color, plaq_idx): [edge_keys]}
    edge_pos = {}         # color -> {edge_key: position_index_in_sub_round}
    for color in 'rgb':
        edges = lat['edges_by_color'][color]
        edge_pos[color] = {tuple(sorted([q1, q2])): k for k, (q1, q2) in enumerate(edges)}
        sub_inferences[color] = find_plaquette_inferences(lat, color)
    
    def emit_subround(sub_round_idx, noisy):
        c, pauli = SCHEDULE[sub_round_idx % 6]
        edges = lat['edges_by_color'][c]
        # control = q1, target = q2 (q2 is the qubit being measured)
        cx_pairs = []
        controls = []
        targets = []
        for q1, q2 in edges:
            cx_pairs.extend([q1, q2])
            controls.append(q1)
            targets.append(q2)
        both = controls + targets
        
        if pauli == 'X':
            # Direct MX measurement: CX q2->q1 -> MX q2 -> CX q2->q1
            # The CX direction is flipped relative to the ZZ branch so the
            #   measured qubit (q2) is the control and propagates X to q1;
            #   X_q1·X_q2 -> X_q2 under conjugation, so MX q2 reads X_q1·X_q2
            #   on the pre-CX state
            cx_pairs_x = []  # (control=q2 measured, target=q1 non-measured)
            for q1, q2 in edges:
                cx_pairs_x.extend([q2, q1])

            circuit.append('CX', cx_pairs_x)
            if noisy:
                circuit.append('DEPOLARIZE2', cx_pairs_x, p)
            circuit.append('TICK')

            if noisy:
                # M(p) classical readout error: the recorded bit is
                #   flipped with probability p, but the post-measurement state
                #   is the true projection. Depolarize the non-measured
                #   qubit during measurement (same as the ZZ branch).
                circuit.append('DEPOLARIZE1', controls, p)
            circuit.append('MX', targets, p if noisy else 0)
            mc_start = measurement_count[0]
            measurement_count[0] += len(targets)
            circuit.append('TICK')

            circuit.append('CX', cx_pairs_x)
            if noisy:
                circuit.append('DEPOLARIZE2', cx_pairs_x, p)
            circuit.append('TICK')
        else:
            # CX -> M target -> CX
            circuit.append('CX', cx_pairs)
            if noisy:
                circuit.append('DEPOLARIZE2', cx_pairs, p)
            circuit.append('TICK')
            
            if noisy:
                # M(p) classical readout error 
                circuit.append('DEPOLARIZE1', controls, p)
            circuit.append('M', targets, p if noisy else 0)
            mc_start = measurement_count[0]
            measurement_count[0] += len(targets)
            circuit.append('TICK')
            
            circuit.append('CX', cx_pairs)
            if noisy:
                circuit.append('DEPOLARIZE2', cx_pairs, p)
            circuit.append('TICK')
        
        # Build edge -> global_rec_index map for this sub-round
        edge_recs = {}
        for k, (q1, q2) in enumerate(edges):
            ek = tuple(sorted([q1, q2]))
            edge_recs[ek] = mc_start + k
        return c, pauli, edge_recs
    
    def emit_detectors_after(sub_round_idx, edge_recs, c, pauli):
        """
        For each plaquette × Pauli inferable at this sub-round, build a detector
        comparing to the previous inference of the same plaquette × pauli.
        """
        for (plaq_color, plaq_idx), edges_on in sub_inferences[c].items():
            key = (plaq_color, plaq_idx, pauli)
            cur_recs = [edge_recs[ek] for ek in edges_on]
            history = plaq_history[key]
            if history:
                prev_round, prev_recs = history[-1]
                gap = sub_round_idx - prev_round
                # Only emit detector if gap is 4 (since gap-2 detectors are randomized
                #   by the sub-round between). Actually we determine validity dynamically:
                #   we'd like to check if Pb(X) etc commutes with all intervening sub-rounds,
                #   but here we use the theoretically derived schedule of valid pairs
                # In the period-6 CSS schedule, valid pairs are gap-4 only — see header
                if gap == 4:
                    all_recs = cur_recs + prev_recs
                    rels = [r - measurement_count[0] for r in all_recs]
                    targets_rec = [stim.target_rec(rr) for rr in rels]
                    # Add coords for visualization: use plaquette centroid + sub-round index
                    coord = _plaq_centroid(lat, plaq_color, plaq_idx)
                    circuit.append('DETECTOR', targets_rec, [coord[0], coord[1], sub_round_idx])
            history.append((sub_round_idx, cur_recs))
    
    # WARMUP: noiseless, no detectors
    n_warmup_sr = 6 * n_warmup_periods
    for r in range(n_warmup_sr):
        c, pauli, edge_recs = emit_subround(r, noisy=False)
        for (plaq_color, plaq_idx), edges_on in sub_inferences[c].items():
            key = (plaq_color, plaq_idx, pauli)
            cur_recs = [edge_recs[ek] for ek in edges_on]
            plaq_history[key].append((r, cur_recs))

    # MAIN: noisy sub-rounds with detectors
    for r in range(n_warmup_sr, n_warmup_sr + 6 * n_qec_rounds):
        c, pauli, edge_recs = emit_subround(r, noisy=True)
        emit_detectors_after(r, edge_recs, c, pauli)

    # TAIL: noiseless, with detectors (so any noise from the last noisy round
    #   is detected when its plaquette is re-measured 4 sub-rounds later)
    tail_start = n_warmup_sr + 6 * n_qec_rounds
    for r in range(tail_start, tail_start + 6 * n_tail_periods):
        c, pauli, edge_recs = emit_subround(r, noisy=False)
        emit_detectors_after(r, edge_recs, c, pauli)

    return circuit, lat


def _plaq_centroid(lat, plaq_color, plaq_idx):
    verts = lat['plaquettes'][plaq_color][plaq_idx]
    xs = [lat['coords'][v][0] for v in verts]
    ys = [lat['coords'][v][1] for v in verts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


if __name__ == '__main__':
    Lx, Ly = 4, 4
    n_qec = 12
    p = 0.001
    
    circuit, lat = make_circuit(Lx, Ly, n_qec, p)
    print(f"Lattice: {Lx}x{Ly} stations → {lat['n_qubits']} qubits")
    print(f"  edges per colour: r={len(lat['edges_by_color']['r'])}, "
          f"g={len(lat['edges_by_color']['g'])}, "
          f"b={len(lat['edges_by_color']['b'])}")
    print(f"  plaquettes: r={len(lat['plaquettes']['r'])} "
          f"g={len(lat['plaquettes']['g'])} "
          f"b={len(lat['plaquettes']['b'])}")
    print(f"Circuit: {circuit.num_measurements} measurements, "
          f"{circuit.num_detectors} detectors, "
          f"{circuit.num_observables} observables")
    
    # Test that the DEM compiles (i.e. all detectors are deterministic in noiseless circuit)
    print("\nChecking detector validity (compiling DEM at p=0)...")
    circuit_noiseless, _ = make_circuit(Lx, Ly, n_qec, p=0.0)
    try:
        # Use approximate_disjoint_errors=True, won't matter at p=0
        dem = circuit_noiseless.detector_error_model(allow_gauge_detectors=False)
        print(f"DEM compiles at p=0. Detectors: {dem.num_detectors}, errors: {dem.num_errors}")
    except Exception as e:
        print(f"DEM error at p=0: {e}")