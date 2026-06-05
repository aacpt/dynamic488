"""
Shared definitions for the 4.8.8 dynamic Floquet code.

The period-6 CSS sub-round schedule and the 2-colouring of the qubit set
define the code itself, so they are shared by all four circuit variants
(reset, noreset, ancilla, pipelined).
"""


SCHEDULE = [('r', 'X'), ('g', 'Z'), ('b', 'X'),
            ('r', 'Z'), ('g', 'X'), ('b', 'Z')]


def qubit_color(i, j, d):
    """
    2-colouring of the qubit set: 0 = A, 1 = B. Verified bipartite for the
    4.8.8 edge graph.
    """
    return (i + j + (1 if d in (1, 3) else 0)) % 2