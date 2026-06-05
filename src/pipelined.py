"""
Pipelined ancilla-based 4.8.8 dynamic Floquet code.

Combines the 8-TICK pipelined circuit builder with its detectors-with-
determinism-check builder. Builds on the ancilla lattice/detector helpers
(ancilla.py) and the shared schedule (common.py).
"""

import stim
from collections import defaultdict, deque
from lattice import build_lattice
from common import SCHEDULE
from ancilla import build_ancilla_lattice, extract_detector_flows, reduce_basis


def compute_data_qubit_parity(lat):
    """
    BFS 2-coloring of the data-qubit graph. Returns dict q->{0,1}.
    Raises ValueError if the graph isn't bipartite.
    """
    n_data = lat['n_qubits']
    adj = {i: set() for i in range(n_data)}
    for c in 'rgb':
        for q1, q2 in lat['edges_by_color'][c]:
            adj[q1].add(q2)
            adj[q2].add(q1)
    parity = {}
    for start in range(n_data):
        if start in parity:
            continue
        parity[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in parity:
                    parity[v] = 1 - parity[u]
                    q.append(v)
                elif parity[v] == parity[u]:
                    raise ValueError(
                        "4.8.8 data-qubit graph is not bipartite.")
    return parity


# Color -> TICK offset within the 8-TICK cycle
# r at offset 0, g at offset 1, b at offset 2

COLOR_OFFSET = {'r': 0, 'g': 1, 'b': 2}


def tick_for(sr_in_period, op_phase):
    """
    For a sub-round position within a period (0..5) and an op_phase in
    {R, CX_a, CX_b, M}, return the TICK offset within the 8-TICK cycle.
    """
    color, _ = SCHEDULE[sr_in_period]
    α = COLOR_OFFSET[color]
    # k=0 for first appearance of this color (sr 0,1,2), k=1 for second (sr 3,4,5)
    k = 0 if sr_in_period < 3 else 1
    base = (α + 4 * k) % 8
    op_offsets = {'R': 0, 'CX_a': 1, 'CX_b': 2, 'M': 3}
    return (base + op_offsets[op_phase]) % 8


def absolute_tick(sr_global, op_phase):
    """Compute absolute TICK index (0-based) for (sr_global, op_phase)."""
    sr_in_period = sr_global % 6
    cycle_idx = sr_global // 6
    R_tick = tick_for(sr_in_period, 'R')
    op_tick = tick_for(sr_in_period, op_phase)
    if op_tick < R_tick:
        op_tick += 8  # operation is in next 8-TICK cycle
    return cycle_idx * 8 + op_tick


def build_pipelined_ancilla_circuit(Lx, Ly, n_qec, p=0.0,
                                       n_warmup_periods=1,
                                       n_tail_periods=1,
                                       basis='X'):
    """
    Build the pipelined ancilla-based 4.8.8 dynamic memory circuit.

    Returns (circuit, lattice_dict). Circuit has no detectors yet.
    Detectors are added in pipelined_ancilla_detectors.py.

    Standard circuit-level depolarising noise, injected only in
    the main region.
    """
    lat = build_ancilla_lattice(Lx, Ly)
    n_data = lat['n_data']
    n_total = lat['n_total']
    parity = compute_data_qubit_parity(lat)

    # Pre-compute canonical edges for each sub-round position
    canonical = []
    for sr_in_period in range(6):
        c, flavor = SCHEDULE[sr_in_period]
        edges = lat['edges_by_color'][c]
        cl = []
        for (q1, q2) in edges:
            qa, qb = (q1, q2) if parity[q1] == 0 else (q2, q1)
            anc = lat['ancilla_of_edge'][(q1, q2)]
            cl.append((qa, qb, anc, flavor))
        canonical.append(cl)

    n_periods = n_warmup_periods + n_qec + n_tail_periods
    total_subrounds = n_periods * 6

    last_sr = total_subrounds - 1
    n_total_ticks = absolute_tick(last_sr, 'M') + 1

    # Build per-TICK buckets: ticks[t] = list of (op_phase, sr_global, is_noisy)
    ticks = [[] for _ in range(n_total_ticks)]
    for sr_global in range(total_subrounds):
        period_idx = sr_global // 6
        is_main = n_warmup_periods <= period_idx < n_warmup_periods + n_qec
        noisy = is_main and p > 0
        for op_phase in ('R', 'CX_a', 'CX_b', 'M'):
            t = absolute_tick(sr_global, op_phase)
            ticks[t].append((op_phase, sr_global, noisy))

    circuit = stim.Circuit()

    # Place qubit coordinates
    for q in range(n_data):
        x, y = lat['coords'][q]
        circuit.append('QUBIT_COORDS', [q], [x, y])
    for q in range(n_data, n_total):
        x, y = lat['ancilla_coords'][q]
        circuit.append('QUBIT_COORDS', [q], [x, y])

    # Initialize all qubits in basis
    circuit.append('R', list(range(n_total)))
    if basis == 'X':
        circuit.append('H', list(range(n_data)))
    circuit.append('TICK')

    # Emit each TICK
    for t in range(n_total_ticks):
        ops = ticks[t]
        cx_a_pairs = []  # CX targets for CX_a (parity-0 ↔ ancilla)
        cx_b_pairs = []
        cx_a_noise = []
        cx_b_noise = []
        r_targets = []
        rx_targets = []
        m_targets = []
        mx_targets = []
        x_err_after_r = []
        z_err_after_r = []
        x_err_before_m = []
        z_err_before_m = []
        # Track whether this tick has an R or M (to decide if idle noise applies)
        #   and whether ANY of the operations in this tick are noisy
        tick_has_r_or_m = False
        tick_is_noisy = False

        for (op_phase, sr_global, noisy) in ops:
            sr_in_period = sr_global % 6
            cl = canonical[sr_in_period]
            flavor = cl[0][3]
            ancillas = [a for (qa, qb, a, fl) in cl]

            if op_phase in ('R', 'M'):
                tick_has_r_or_m = True
            if noisy:
                tick_is_noisy = True

            if op_phase == 'R':
                if flavor == 'X':
                    rx_targets.extend(ancillas)
                    if noisy:
                        z_err_after_r.extend(ancillas)
                else:
                    r_targets.extend(ancillas)
                    if noisy:
                        x_err_after_r.extend(ancillas)
            elif op_phase == 'CX_a':
                cx_pairs = []
                for (qa, qb, anc, fl) in cl:
                    if fl == 'Z':
                        cx_pairs.extend([qa, anc])
                    else:
                        cx_pairs.extend([anc, qa])
                cx_a_pairs.extend(cx_pairs)
                if noisy:
                    cx_a_noise.extend(cx_pairs)
            elif op_phase == 'CX_b':
                cx_pairs = []
                for (qa, qb, anc, fl) in cl:
                    if fl == 'Z':
                        cx_pairs.extend([qb, anc])
                    else:
                        cx_pairs.extend([anc, qb])
                cx_b_pairs.extend(cx_pairs)
                if noisy:
                    cx_b_noise.extend(cx_pairs)
            elif op_phase == 'M':
                if flavor == 'X':
                    if noisy:
                        z_err_before_m.extend(ancillas)
                    mx_targets.extend(ancillas)
                else:
                    if noisy:
                        x_err_before_m.extend(ancillas)
                    m_targets.extend(ancillas)

        # Idle noise: DEPOLARIZE1(p) on every qubit not acted on by an
        #   R / M / RX / MX / CX operation during a noisy tick.
        idle_targets = []
        if tick_is_noisy:
            active = set()
            active.update(r_targets)
            active.update(rx_targets)
            active.update(m_targets)
            active.update(mx_targets)
            active.update(cx_a_pairs)
            active.update(cx_b_pairs)
            idle_targets = [q for q in range(n_total) if q not in active]

        # Emit
        if r_targets:
            circuit.append('R', r_targets)
        if rx_targets:
            circuit.append('RX', rx_targets)
        if x_err_after_r:
            circuit.append('X_ERROR', x_err_after_r, p)
        if z_err_after_r:
            circuit.append('Z_ERROR', z_err_after_r, p)
        if cx_a_pairs:
            circuit.append('CX', cx_a_pairs)
        if cx_a_noise:
            circuit.append('DEPOLARIZE2', cx_a_noise, p)
        if cx_b_pairs:
            circuit.append('CX', cx_b_pairs)
        if cx_b_noise:
            circuit.append('DEPOLARIZE2', cx_b_noise, p)
        # M(p) / MX(p) for classical readout error. Emit in
        #   consecutive-status chunks so the record-index ordering matches the
        #   original M(m_targets); MX(mx_targets) exactly
        def _emit_chunked(targets, gate, noisy_set):
            if not targets:
                return
            buf = [targets[0]]
            cur_noisy = (targets[0] in noisy_set)
            for q in targets[1:]:
                is_n = (q in noisy_set)
                if is_n == cur_noisy:
                    buf.append(q)
                else:
                    circuit.append(gate, buf, p if cur_noisy else 0)
                    buf = [q]
                    cur_noisy = is_n
            circuit.append(gate, buf, p if cur_noisy else 0)
        _emit_chunked(m_targets,  'M',  set(x_err_before_m))
        _emit_chunked(mx_targets, 'MX', set(z_err_before_m))
        if idle_targets:
            circuit.append('DEPOLARIZE1', idle_targets, p)
        circuit.append('TICK')

    return circuit, lat


def make_pipelined_ancilla_circuit_with_detectors(
        Lx, Ly, n_qec, p=0.0,
        n_warmup_periods=1, n_tail_periods=1,
        basis='X', drop_topological=False, cutoff=24):
    """
    Build the noisy pipelined circuit with detectors emitted from
    the min-weight basis.
    """

    # Build noiseless pipelined skeleton
    plain, lat = build_pipelined_ancilla_circuit(
        Lx, Ly, n_qec, p=0.0,
        n_warmup_periods=n_warmup_periods,
        n_tail_periods=n_tail_periods,
        basis=basis)

    # Extract + reduce detector basis
    raw = extract_detector_flows(plain)
    reduced = reduce_basis(raw)

    # Filter
    if drop_topological:
        det_basis = [d for d in reduced if len(d) <= cutoff]
    else:
        det_basis = reduced

    # Re-emit pipelined circuit with detectors at the right TICKs
    # We need to know, for each (sr_global, measurement_index_within_sr),
    #   the global measurement index in the pipelined emission order
    n_data = lat['n_data']
    n_total = lat['n_total']
    parity = compute_data_qubit_parity(lat)

    canonical = []
    for sr_in_period in range(6):
        c, flavor = SCHEDULE[sr_in_period]
        edges = lat['edges_by_color'][c]
        cl = []
        for (q1, q2) in edges:
            qa, qb = (q1, q2) if parity[q1] == 0 else (q2, q1)
            anc = lat['ancilla_of_edge'][(q1, q2)]
            cl.append((qa, qb, anc, flavor))
        canonical.append(cl)

    n_periods = n_warmup_periods + n_qec + n_tail_periods
    total_subrounds = n_periods * 6

    # Order of measurements:
    # In the un-pipelined version, measurements occur in sub-round order
    #   (sr=0's M's happen first, then sr=1's, ...). The detector flows
    #   are derived from THAT measurement ordering.
    # In the pipelined version, M's happen in TICK order, which differs
    # So when emitting detectors, we need to translate measurement indices
    #   from un-pipelined order to pipelined order

    # Build a translation: unpiped_meas_idx -> pipelined_meas_idx
    # In un-pipelined: measurements happen in order
    #   sr=0: meas indices [0, 1, ..., n_e-1]
    #   sr=1: [n_e, n_e+1, ..., 2n_e-1]
    #   ...
    # In pipelined: measurements happen in TICK order. At each TICK with
    #   one or more M ops, we emit them in the order their sub-rounds appear

    n_e = len(canonical[0])  # edges per color (assumes uniform)

    # First, figure out which TICK each sub-round's M happens at, and the
    #   order of M's within a TICK
    M_tick_per_sr = {}
    for sr_global in range(total_subrounds):
        M_tick_per_sr[sr_global] = absolute_tick(sr_global, 'M')

    # Group sub-rounds by their M tick, sorted by sub-round index for
    #   deterministic ordering within a TICK
    by_tick = defaultdict(list)
    for sr_global, t in M_tick_per_sr.items():
        by_tick[t].append(sr_global)
    for t in by_tick:
        by_tick[t].sort()

    # Now compute the pipelined measurement order

    # Trace the emission to get the exact measurement order
    pipelined_order = []  # list of (sr_global, edge_idx_within_sr) in emission order
    for t in range(max(by_tick) + 1):
        srs_at_t = by_tick.get(t, [])
        # Z-flavor first, then X-flavor (matching build_pipelined emit)
        for flavor in ('Z', 'X'):
            for sr in srs_at_t:
                sr_in_period = sr % 6
                cl = canonical[sr_in_period]
                fl = cl[0][3]
                if fl != flavor:
                    continue
                for edge_idx in range(len(cl)):
                    pipelined_order.append((sr, edge_idx))

    # Translation: unpiped_meas_idx -> pipelined_meas_idx
    # Un-pipelined: sub-round sr_global, edge_idx -> idx = sr_global * n_e + edge_idx
    unpiped_to_piped = {}
    for piped_idx, (sr_g, edge_idx) in enumerate(pipelined_order):
        unpiped_idx = sr_g * n_e + edge_idx
        unpiped_to_piped[unpiped_idx] = piped_idx

    # Sanity check
    if len(unpiped_to_piped) != total_subrounds * n_e:
        raise ValueError(
            f"Translation table size mismatch: "
            f"{len(unpiped_to_piped)} vs expected {total_subrounds * n_e}")

    # Translate detector basis from un-pipelined indices to pipelined ones
    det_basis_piped = []
    for ds in det_basis:
        translated = sorted(unpiped_to_piped[m] for m in ds)
        det_basis_piped.append(translated)

    # Group by last (in pipelined order) for emission timing
    det_by_last_piped = defaultdict(list)
    for ds in det_basis_piped:
        det_by_last_piped[max(ds)].append(ds)

    # Re-emit the pipelined circuit but with detectors and noise
    circuit = stim.Circuit()

    # Place qubit coordinates
    for q in range(n_data):
        x, y = lat['coords'][q]
        circuit.append('QUBIT_COORDS', [q], [x, y])
    for q in range(n_data, n_total):
        x, y = lat['ancilla_coords'][q]
        circuit.append('QUBIT_COORDS', [q], [x, y])

    # Initialize all qubits in basis
    circuit.append('R', list(range(n_total)))
    if basis == 'X':
        circuit.append('H', list(range(n_data)))
    circuit.append('TICK')

    # Build per-TICK ops list (same as in build_pipelined_ancilla_circuit)
    last_sr = total_subrounds - 1
    n_total_ticks = absolute_tick(last_sr, 'M') + 1

    ticks_ops = [[] for _ in range(n_total_ticks)]
    for sr_global in range(total_subrounds):
        period_idx = sr_global // 6
        is_main = n_warmup_periods <= period_idx < n_warmup_periods + n_qec
        noisy = is_main and p > 0
        for op_phase in ('R', 'CX_a', 'CX_b', 'M'):
            t = absolute_tick(sr_global, op_phase)
            ticks_ops[t].append((op_phase, sr_global, noisy))

    measurement_count = 0
    for t in range(n_total_ticks):
        ops = ticks_ops[t]

        # Sort ops by phase order (R, CX_a, CX_b, M) and within phase by sr
        #   for deterministic emission
        cx_a_pairs = []
        cx_b_pairs = []
        cx_a_noise = []
        cx_b_noise = []
        r_z_targets = []  # R targets (Z-basis)
        r_x_targets = []  # RX targets
        m_z_targets = []  # M targets
        m_x_targets = []  # MX targets
        x_err_after_r = []
        z_err_after_r = []
        x_err_before_m = []
        z_err_before_m = []

        # Track the order of measurements within this TICK to emit detectors
        #   in correct order
        m_emission_order = []  # list of (flavor, sr_global, ancilla_list)

        # Track which qubits are touched by ANY op in this TICK (active set);
        #   all others are "idle" and accumulate DEPOLARIZE1(p)
        # Whether to add idle noise: only if the TICK contains a NOISY R or M
        #   (matching the un-pipelined model which adds DEPOLARIZE1 only on
        #   R-TICKs and M-TICKs, not on CX-TICKs)
        active_qubits = set()
        tick_has_noisy_r_or_m = False

        for (op_phase, sr_global, noisy) in ops:
            sr_in_period = sr_global % 6
            cl = canonical[sr_in_period]
            flavor = cl[0][3]
            ancillas = [a for (qa, qb, a, fl) in cl]

            if op_phase == 'R':
                if flavor == 'X':
                    r_x_targets.extend(ancillas)
                    if noisy:
                        z_err_after_r.extend(ancillas)
                else:
                    r_z_targets.extend(ancillas)
                    if noisy:
                        x_err_after_r.extend(ancillas)
                active_qubits.update(ancillas)
                if noisy:
                    tick_has_noisy_r_or_m = True
            elif op_phase == 'CX_a':
                cx_pairs = []
                for (qa, qb, anc, fl) in cl:
                    if fl == 'Z':
                        cx_pairs.extend([qa, anc])
                    else:
                        cx_pairs.extend([anc, qa])
                cx_a_pairs.extend(cx_pairs)
                if noisy:
                    cx_a_noise.extend(cx_pairs)
                active_qubits.update(cx_pairs)
            elif op_phase == 'CX_b':
                cx_pairs = []
                for (qa, qb, anc, fl) in cl:
                    if fl == 'Z':
                        cx_pairs.extend([qb, anc])
                    else:
                        cx_pairs.extend([anc, qb])
                cx_b_pairs.extend(cx_pairs)
                if noisy:
                    cx_b_noise.extend(cx_pairs)
                active_qubits.update(cx_pairs)
            elif op_phase == 'M':
                if flavor == 'X':
                    if noisy:
                        z_err_before_m.extend(ancillas)
                    m_x_targets.extend(ancillas)
                    m_emission_order.append(('X', sr_global, ancillas))
                else:
                    if noisy:
                        x_err_before_m.extend(ancillas)
                    m_z_targets.extend(ancillas)
                    m_emission_order.append(('Z', sr_global, ancillas))
                active_qubits.update(ancillas)
                if noisy:
                    tick_has_noisy_r_or_m = True

        # Compute idle qubits for this TICK
        idle_list = sorted(set(range(n_total)) - active_qubits)

        # Emit
        if r_z_targets:
            circuit.append('R', r_z_targets)
        if r_x_targets:
            circuit.append('RX', r_x_targets)
        if x_err_after_r:
            circuit.append('X_ERROR', x_err_after_r, p)
        if z_err_after_r:
            circuit.append('Z_ERROR', z_err_after_r, p)
        # Add idle noise once per TICK (matching un-pipelined which adds it
        #   alongside R-flips and M-flips). Only if there's a noisy R or M
        if tick_has_noisy_r_or_m and idle_list:
            circuit.append('DEPOLARIZE1', idle_list, p)
        if cx_a_pairs:
            circuit.append('CX', cx_a_pairs)
        if cx_a_noise:
            circuit.append('DEPOLARIZE2', cx_a_noise, p)
        if cx_b_pairs:
            circuit.append('CX', cx_b_pairs)
        if cx_b_noise:
            circuit.append('DEPOLARIZE2', cx_b_noise, p)
        # M(p)/MX(p) for classical readout error. Emit in
        #   consecutive-status chunks so the record-index ordering matches the
        #   original M(m_z_targets); MX(m_x_targets) order exactly (because
        #   detectors below depend on m_emission_sorted matching this order)
        def _emit_chunked(targets, gate, noisy_set):
            if not targets:
                return
            buf = [targets[0]]
            cur_noisy = (targets[0] in noisy_set)
            for q in targets[1:]:
                is_n = (q in noisy_set)
                if is_n == cur_noisy:
                    buf.append(q)
                else:
                    circuit.append(gate, buf, p if cur_noisy else 0)
                    buf = [q]
                    cur_noisy = is_n
            circuit.append(gate, buf, p if cur_noisy else 0)
        _emit_chunked(m_z_targets, 'M',  set(x_err_before_m))
        _emit_chunked(m_x_targets, 'MX', set(z_err_before_m))

        # Now emit detectors that close at this TICK
        # The pipelined emission order (Z-flavor first, then X-flavor; within
        #   each flavor sorted by sub-round index) was used to build the
        #   unpiped->piped translation table above. We do the same here to assign
        #   each measurement its global index
        m_emission_sorted = sorted(m_emission_order,
                                    key=lambda x: (0 if x[0] == 'Z' else 1, x[1]))
        # First, count how many measurements happen in this TICK
        meas_indices_in_tick = []  # list of (sr_global, edge_idx, global_idx)
        n_meas_this_tick = sum(len(ancillas) for (_, _, ancillas) in m_emission_sorted)
        # measurement_count BEFORE this TICK's M's = pre_count
        pre_count = measurement_count
        # post_count = pre_count + n_meas_this_tick
        post_count = pre_count + n_meas_this_tick
        # Now iterate through measurements in pipelined order, assign indices
        idx_offset = pre_count
        for (flavor, sr_global, ancillas) in m_emission_sorted:
            sr_in_period = sr_global % 6
            period_idx = sr_global // 6
            for edge_idx in range(len(ancillas)):
                global_meas_idx = idx_offset
                # Check if any detectors close at this measurement index
                for ds in det_by_last_piped.get(global_meas_idx, []):
                    # rec offsets are relative to the END of the current
                    #   circuit (i.e. after all measurements in this TICK have
                    #   been recorded). At that point the most recent meas is
                    #   at global index post_count - 1
                    rels = [m - post_count for m in ds]
                    circuit.append(
                        'DETECTOR',
                        [stim.target_rec(rr) for rr in rels],
                        [period_idx, sr_in_period, global_meas_idx])
                idx_offset += 1
        measurement_count = post_count
        circuit.append('TICK')

    return circuit, lat, det_basis_piped


if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    # Sanity test
    print("=== Pipelined 4.8.8 ancilla, L=4, n_qec=4, p=0.0 ===")
    c, lat = build_pipelined_ancilla_circuit(4, 4, n_qec=4, p=0.0)
    n_ticks = sum(1 for inst in c.flattened() if inst.name == 'TICK')
    print(f"  qubits: {c.num_qubits}")
    print(f"  TICKs:  {n_ticks}")
    print(f"  measurements: {c.num_measurements}")
    print()

    from ancilla import build_ancilla_circuit
    c_unp, _ = build_ancilla_circuit(4, 4, n_qec=4, p=0.0)
    n_unp = sum(1 for inst in c_unp.flattened() if inst.name == 'TICK')
    n_periods = 6  # 1 warmup + 4 qec + 1 tail
    print(f"  Un-pipelined TICKs (same params): {n_unp}")
    print(f"  Pipelining speedup: {n_unp / n_ticks:.2f}×")
    print(f"  Per period: {n_ticks/n_periods:.1f} pipelined vs "
          f"{n_unp/n_periods:.1f} un-pipelined")
    print()
    print("Asymptotic per-sub-round (varying n_qec):")
    print(f"  {'n_qec':>5} {'pipelined':>10} {'unpiped':>10} {'speedup':>8}")
    for n_qec in (1, 2, 4, 8, 16):
        c1, _ = build_pipelined_ancilla_circuit(4, 4, n_qec=n_qec)
        c2, _ = build_ancilla_circuit(4, 4, n_qec=n_qec)
        n1 = sum(1 for inst in c1.flattened() if inst.name == 'TICK')
        n2 = sum(1 for inst in c2.flattened() if inst.name == 'TICK')
        print(f"  {n_qec:>5} {n1:>10} {n2:>10} {n2/n1:>8.2f}×")


if __name__ == '__main__':
    print("=== Pipelined ancilla 4.8.8: determinism check ===")
    print()
    for L in (2, 4, 6):
        print(f"L={L}:")
        c, lat, basis = make_pipelined_ancilla_circuit_with_detectors(
            L, L, 3, p=0.0, drop_topological=False)
        n_ticks = sum(1 for inst in c.flattened() if inst.name == 'TICK')
        print(f"  qubits: {c.num_qubits}, "
              f"measurements: {c.num_measurements}, "
              f"detectors: {c.num_detectors}, "
              f"TICKs: {n_ticks}")

        try:
            dem = c.detector_error_model(allow_gauge_detectors=False)
            print(f"  DEM at p=0: {dem.num_errors} errors")
            if dem.num_errors == 0:
                print(f"  ✓ DETERMINISTIC")
            else:
                print(f"  ✗ NON-DETERMINISTIC — bug in pipelined construction")
                # Show first error
                for inst in dem:
                    if inst.type == 'error':
                        print(f"    First error: {inst}")
                        break
        except Exception as e:
            print(f"  ✗ DEM failed: {type(e).__name__}: {e}")
        print()