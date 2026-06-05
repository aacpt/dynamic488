"""
Unified circuit builders for the four Floquet variants.

Each build_* returns a noisy memory circuit on an L x L torus with the
horizontal non-contractible X-cycle attached as logical observable 0, ready to
sample/decode. BUILDERS maps the canonical variant names used throughout
the paper to these functions, and build_circuit(variant, ...) is a single
entry point.

    from builders import build_circuit, BUILDERS
    circuit = build_circuit('with_reset', L=4, n_qec=4, p=1e-3)

Variant names: 'with_reset' (reset dynamic), 'no_reset' (no-reset dynamic),
'ancilla' (ancilla, un-pipelined), 'pipelined' (ancilla, 8-TICK pipelined).
"""
import stim

from reset import make_circuit_minweight
from noreset import make_circuit as _noreset_make_circuit
from ancilla import make_ancilla_circuit_with_detectors
from pipelined import (
    make_pipelined_ancilla_circuit_with_detectors,
)
from observable_helper import attach_observable


def _h_observable(circuit, lat, L, n_total):
    """
    Attach the horizontal X-cycle (X on the N and S qubits of row j=0) as
    logical observable 0. Returns (circuit, observable_records).
    """
    qubits = lat['qubits']
    C = [qubits[(i, 0, 0)] for i in range(L)] + [qubits[(i, 0, 2)] for i in range(L)]
    pauli = stim.PauliString(n_total)
    for q in C:
        pauli[q] = 'X'
    return attach_observable(circuit, pauli, observable_index=0)


def build_memory_circuit(L, n_qec, p=0.001, n_warmup_periods=1, n_tail_periods=1):
    """
    No-reset memory circuit on an L x L torus with the horizontal cycle as l
    ogical X. 
    """
    circuit, lat = _noreset_make_circuit(
        L, L, n_qec, p=p,
        n_warmup_periods=n_warmup_periods, n_tail_periods=n_tail_periods,
    )
    n = lat['n_qubits']
    circuit.append('MX', list(range(n)))
    circuit, recs = _h_observable(circuit, lat, L, n)
    return circuit, lat, recs


def build_with_reset(L, n_qec, p, n_warmup_periods=2, n_tail_periods=2):
    """Reset dynamic circuit (min-weight detector basis)."""
    c, lat = make_circuit_minweight(
        L, L, n_qec, p=p,
        n_warmup_periods=n_warmup_periods, n_tail_periods=n_tail_periods,
        drop_topological=True)
    n = lat['n_qubits']
    c.append('MX', list(range(n)))
    c, _ = _h_observable(c, lat, L, n)
    return c


def build_no_reset(L, n_qec, p, n_warmup_periods=2, n_tail_periods=2):
    """No-reset dynamic circuit."""
    c, _, _ = build_memory_circuit(
        L, n_qec, p=p,
        n_warmup_periods=n_warmup_periods, n_tail_periods=n_tail_periods)
    return c


def build_ancilla(L, n_qec, p, n_warmup_periods=2, n_tail_periods=2):
    """Ancilla-based, un-pipelined circuit."""
    c, lat, _ = make_ancilla_circuit_with_detectors(
        L, L, n_qec=n_qec, p=p,
        n_warmup_periods=n_warmup_periods, n_tail_periods=n_tail_periods,
        drop_topological=True, cutoff=24)
    c.append('MX', list(range(lat['n_data'])))
    c, _ = _h_observable(c, lat, L, lat['n_total'])
    return c


def build_pipelined(L, n_qec, p, n_warmup_periods=2, n_tail_periods=2):
    """Ancilla-based, 8-TICK pipelined circuit."""
    c, lat, _ = make_pipelined_ancilla_circuit_with_detectors(
        L, L, n_qec=n_qec, p=p,
        n_warmup_periods=n_warmup_periods, n_tail_periods=n_tail_periods,
        drop_topological=True, cutoff=24)
    c.append('MX', list(range(lat['n_data'])))
    c, _ = _h_observable(c, lat, L, lat['n_total'])
    return c


BUILDERS = {
    'with_reset': build_with_reset,
    'no_reset':   build_no_reset,
    'ancilla':    build_ancilla,
    'pipelined':  build_pipelined,
}

PRETTY_NAME = {
    'with_reset': 'reset dynamic',
    'no_reset':   'no-reset dynamic',
    'ancilla':    'ancilla un-pipelined',
    'pipelined':  'ancilla pipelined',
}


def build_circuit(variant, L, n_qec, p, **kw):
    """Build any variant's memory circuit by name (see BUILDERS)."""
    if variant not in BUILDERS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(BUILDERS)}")
    return BUILDERS[variant](L, n_qec, p, **kw)