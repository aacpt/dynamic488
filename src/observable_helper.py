"""
Helper for setting up OBSERVABLE_INCLUDE in a circuit.

Given a stim Circuit and a logical Pauli L_init that should be a stabiliser of
the post-init state, find the set of measurement records whose XOR equals
L_init's eigenvalue (deterministically, in the noiseless case).

Strip the init prefix, use stim's flow_generators() on the body, then solve
a small GF(2) linear system to find which generators combine to the desired
flow X^C -> 1.
"""

import stim
import numpy as np


def _pauli_to_bits(p, n_q):
    """
    Encode a PauliString with up to n_q qubits as a 2*n_q-dim binary vector
    (concat of X-part and Z-part). Y is encoded as X+Z.
    """
    bits = np.zeros(2 * n_q, dtype=np.int8)
    for i in range(min(len(p), n_q)):
        v = p[i]
        if v == 1:
            bits[i] = 1            # X
        elif v == 2:
            bits[i] = 1
            bits[n_q + i] = 1      # Y = XZ
        elif v == 3:
            bits[n_q + i] = 1      # Z
    return bits


def _gf2_solve(A, b):
    """Solve Ax = b over GF(2). Returns x or None if infeasible."""
    A = A.copy() % 2
    b = b.copy() % 2
    m, ncols = A.shape
    aug = np.concatenate([A, b.reshape(-1, 1)], axis=1)
    rank = 0
    pivot_cols = []
    for col in range(ncols):
        pr = None
        for row in range(rank, m):
            if aug[row, col] == 1:
                pr = row
                break
        if pr is None:
            continue
        aug[[rank, pr]] = aug[[pr, rank]]
        for row in range(m):
            if row != rank and aug[row, col] == 1:
                aug[row] = (aug[row] + aug[rank]) % 2
        pivot_cols.append(col)
        rank += 1
    for row in range(rank, m):
        if aug[row, -1] != 0:
            return None
    x = np.zeros(ncols, dtype=np.int8)
    for i, c in enumerate(pivot_cols):
        x[c] = aug[i, -1]
    return x


def strip_init_prefix(circuit, init_gates=('R', 'H', 'RX', 'RY', 'RZ', 'X', 'Z', 'Y',
                                            'S', 'S_DAG', 'SQRT_X', 'SQRT_Y',
                                            'SQRT_Z', 'C_XYZ', 'C_ZYX')):
    """
    Strip leading QUBIT_COORDS + init Clifford gates up to the first TICK.

    This is needed because stim's flow_generators only finds non-trivial-input
    flows when the circuit body doesn't contain a state-resetting prefix.
    """
    body = stim.Circuit()
    init_done = False
    for inst in list(circuit):
        if inst.name == 'QUBIT_COORDS':
            continue
        if not init_done:
            if inst.name in init_gates:
                continue
            if inst.name == 'TICK':
                init_done = True
                continue
        body.append(inst)
    return body


def find_observable_records(circuit, logical_pauli, forbidden_records=None):
    """
    Find which measurement records' XOR gives logical_pauli's eigenvalue.

    Args:
        circuit: stim.Circuit including init prefix and (optionally) destructive
                 readout at the end.
        logical_pauli: stim.PauliString - the logical operator at the start of
                 the circuit body (post-init).
        forbidden_records: optional iterable of record indices that MUST NOT
                 appear in the result. These are typically warmup records or
                 other "noiseless" records that, if flipped by an error, would
                 not be detectable.

    Returns:
        sorted list of measurement record indices (0-indexed across whole circuit)
        whose XOR equals the logical's eigenvalue. None if no such combination exists.
    """
    body = strip_init_prefix(circuit)
    flows = body.flow_generators()
    non_trivial = [f for f in flows if f.input_copy().weight > 0]
    detector_flows = [f for f in flows if f.input_copy().weight == 0
                      and f.output_copy().weight == 0]

    # Combine non-trivial-input flows + detector flows. Detector flows let us
    #   "swap" records around without changing the input/output structure
    all_flows = non_trivial + detector_flows
    n_logical = len(non_trivial)  # first n_logical flows are the "real" ones

    n_q = len(logical_pauli)
    target_bits = _pauli_to_bits(logical_pauli, n_q)
    in_mat = np.array([_pauli_to_bits(f.input_copy(), n_q) for f in all_flows],
                      dtype=np.int8).T
    out_mat = np.array([_pauli_to_bits(f.output_copy(), n_q) for f in all_flows],
                       dtype=np.int8).T

    A_blocks = [in_mat, out_mat]
    b_blocks = [target_bits, np.zeros(2 * n_q, dtype=np.int8)]

    # Add forbidden-record constraints
    if forbidden_records:
        forbidden_set = set(forbidden_records)
        for r in forbidden_set:
            row = np.array(
                [1 if r in f.measurements_copy() else 0 for f in all_flows],
                dtype=np.int8,
            )
            A_blocks.append(row.reshape(1, -1))
            b_blocks.append(np.zeros(1, dtype=np.int8))

    A = np.concatenate(A_blocks, axis=0)
    b = np.concatenate(b_blocks)
    x = _gf2_solve(A, b)
    if x is None:
        return None

    record_set = set()
    for i, val in enumerate(x):
        if val == 1:
            for r in all_flows[i].measurements_copy():
                if r in record_set:
                    record_set.remove(r)
                else:
                    record_set.add(r)
    return sorted(record_set)


def attach_observable(circuit, logical_pauli, observable_index=0,
                       forbidden_records=None):
    """
    Find the correct records and append OBSERVABLE_INCLUDE to the circuit.

    Returns the (mutated) circuit and the record indices used.
    """
    record_indices = find_observable_records(
        circuit, logical_pauli, forbidden_records=forbidden_records,
    )
    if record_indices is None:
        raise ValueError("No flow combination produces the requested logical.")
    total_meas = circuit.num_measurements
    rels = [stim.target_rec(r - total_meas) for r in record_indices]
    circuit.append('OBSERVABLE_INCLUDE', rels, [observable_index])
    return circuit, record_indices


if __name__ == '__main__':
    from noreset import make_circuit
    L_, n_qec = 2, 1
    circuit, lat = make_circuit(L_, L_, n_qec, p=0.0)
    n = lat['n_qubits']
    qubits = lat['qubits']
    C = [qubits[(i, 0, 0)] for i in range(L_)] + [qubits[(i, 0, 2)] for i in range(L_)]

    circuit.append('MX', list(range(n)))
    L_init = stim.PauliString(n)
    for q in C:
        L_init[q] = 'X'

    circuit, recs = attach_observable(circuit, L_init, observable_index=0)
    print(f"Found {len(recs)} records for OBSERVABLE_INCLUDE")
    print(f"Circuit has {circuit.num_observables} observables")

    # Verify
    try:
        dem = circuit.detector_error_model(allow_gauge_detectors=False)
        print(f"DEM compiles: {dem.num_detectors} detectors, {dem.num_errors} errors, "
              f"{dem.num_observables} observables")
    except Exception as e:
        print(f"DEM FAIL: {e}")