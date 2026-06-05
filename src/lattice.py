"""
Square-octagon 4.8.8 lattice geometry.

Convention (truncated square tiling):
    - Stations sit on a square lattice (Lx, Ly), with one small green square per station.
    - Each station's small square has 4 qubits: vertices at directions N (0), E (1), S (2), W (3).
    - Octagons live on the FACES of the underlying square lattice. Octagon (a, b) is bordered
      by the four stations (a, b), (a+1, b), (a+1, b+1), (a, b+1).
    - Octagons are 2-coloured by face parity: red if a+b even, blue if a+b odd.
    - Edges:
        * 4 small-square edges per station (NE, SE, SW, NW sides of the green square).
          Each goes between the green square and one of the 4 surrounding octagons.
          Edge colour = the colour NOT shared, so:
            green↔red octagon  -> blue edge
            green↔blue octagon -> red edge
        * 1 east octagon-octagon edge per station + 1 north octagon-octagon edge per station.
          These all separate a red octagon from a blue octagon, so colour = green.
    - Vertex degree 3: each vertex is on 1 green square + 2 octagons (one red, one blue),
      with 3 edges of 3 different colours.
    - Plaquette weights: green=4, red=8, blue=8.

Lx and Ly should both be even to keep the 2-coloring of octagons periodic.
"""

from collections import defaultdict


def build_lattice(Lx: int, Ly: int):
    assert Lx % 2 == 0 and Ly % 2 == 0, "Lx and Ly must be even for the 2-coloring to wrap nicely."
    
    # Qubit indexing: (i, j, d) -> int, d in {0:N, 1:E, 2:S, 3:W}
    qubits = {}
    for i in range(Lx):
        for j in range(Ly):
            for d in range(4):
                qubits[(i, j, d)] = len(qubits)
    
    n_qubits = len(qubits)
    
    # Coordinates (for visualization). Place small squares at "stations" with a small offset
    coords = {}
    for (i, j, d), q in qubits.items():
        # Station center at (4*i, 4*j). Small square has half-width 1
        cx, cy = 4 * i, 4 * j
        if d == 0:    # N
            x, y = cx, cy + 1
        elif d == 1:  # E
            x, y = cx + 1, cy
        elif d == 2:  # S
            x, y = cx, cy - 1
        else:         # W
            x, y = cx - 1, cy
        coords[q] = (x, y)
    
    # Edges: each is (q1, q2, colour). Colour is one of 'r', 'g', 'b'
    # The edge colour matches the check colour (rXX checks live on red edges, etc)
    
    edges = []
    # Track edges-by-colour for the schedule
    edges_by_color = {'r': [], 'g': [], 'b': []}
    
    def oct_color_at_face(a, b):
        return 'r' if (a + b) % 2 == 0 else 'b'
    
    def green_oct_edge_color(oct_c):
        # Edge between green square and octagon of color oct_c
        return 'b' if oct_c == 'r' else 'r'
    
    for i in range(Lx):
        for j in range(Ly):
            # The 4 sides of the green square at station (i, j)
            # Side NE (between N and E): borders octagon at face (i, j) [NE diagonal of station]
            # Side SE (between E and S): borders octagon at face (i, j-1) [SE diagonal]
            # Side SW (between S and W): borders octagon at face (i-1, j-1) [SW diagonal]
            # Side NW (between W and N): borders octagon at face (i-1, j) [NW diagonal]
            
            sides = [
                ((0, 1), (i,           j          )),  # NE side -> NE octagon
                ((1, 2), (i,           (j-1)%Ly  )),   # SE side -> SE octagon
                ((2, 3), ((i-1)%Lx,    (j-1)%Ly  )),   # SW side -> SW octagon
                ((3, 0), ((i-1)%Lx,    j          )),  # NW side -> NW octagon
            ]
            for (d1, d2), face in sides:
                oc = oct_color_at_face(*face)
                ec = green_oct_edge_color(oc)
                q1 = qubits[(i, j, d1)]
                q2 = qubits[(i, j, d2)]
                edges.append((q1, q2, ec))
                edges_by_color[ec].append((q1, q2))
            
            # East octagon-octagon edge: (i, j, E) — (i+1, j, W). Colour = green
            q1 = qubits[(i, j, 1)]
            q2 = qubits[((i+1)%Lx, j, 3)]
            edges.append((q1, q2, 'g'))
            edges_by_color['g'].append((q1, q2))
            
            # North octagon-octagon edge: (i, j, N) — (i, j+1, S). Colour = green
            q1 = qubits[(i, j, 0)]
            q2 = qubits[(i, (j+1)%Ly, 2)]
            edges.append((q1, q2, 'g'))
            edges_by_color['g'].append((q1, q2))
    
    # Plaquettes
    plaquettes = {'r': [], 'g': [], 'b': []}
    
    # Green squares: 1 per station, 4 vertices each
    for i in range(Lx):
        for j in range(Ly):
            verts = [qubits[(i, j, d)] for d in range(4)]  # N, E, S, W
            plaquettes['g'].append(verts)
    
    # Octagons: 1 per face, 8 vertices each
    # Octagon at face (a, b) has corners going around as:
    #   (a, b, N), (a, b+1, S), (a, b+1, E), (a+1, b+1, W),
    #   (a+1, b+1, S), (a+1, b, N), (a+1, b, W), (a, b, E)
    for a in range(Lx):
        for b in range(Ly):
            ap1 = (a + 1) % Lx
            bp1 = (b + 1) % Ly
            verts = [
                qubits[(a, b, 0)],     # N of (a, b)
                qubits[(a, bp1, 2)],   # S of (a, b+1)
                qubits[(a, bp1, 1)],   # E of (a, b+1)
                qubits[(ap1, bp1, 3)], # W of (a+1, b+1)
                qubits[(ap1, bp1, 2)], # S of (a+1, b+1)
                qubits[(ap1, b, 0)],   # N of (a+1, b)
                qubits[(ap1, b, 3)],   # W of (a+1, b)
                qubits[(a, b, 1)],     # E of (a, b)
            ]
            color = oct_color_at_face(a, b)
            plaquettes[color].append(verts)
    
    return {
        'Lx': Lx, 'Ly': Ly,
        'n_qubits': n_qubits,
        'qubits': qubits,
        'coords': coords,
        'edges': edges,
        'edges_by_color': edges_by_color,
        'plaquettes': plaquettes,
    }


def sanity_check(lat):
    """Verify the lattice has the right combinatorial structure."""
    n = lat['n_qubits']
    Lx, Ly = lat['Lx'], lat['Ly']
    expected_qubits = 4 * Lx * Ly
    assert n == expected_qubits, f"qubits: {n} != {expected_qubits}"
    
    # Each qubit should have degree 3
    deg = [0] * n
    for q1, q2, _ in lat['edges']:
        deg[q1] += 1
        deg[q2] += 1
    bad = [(i, d) for i, d in enumerate(deg) if d != 3]
    assert not bad, f"vertices with wrong degree: {bad[:10]}"
    
    # Each edge should be on exactly 2 plaquettes
    edge_pl_count = defaultdict(int)
    for color, plist in lat['plaquettes'].items():
        for verts in plist:
            n_v = len(verts)
            for k in range(n_v):
                a, b = verts[k], verts[(k+1) % n_v]
                key = tuple(sorted([a, b]))
                edge_pl_count[key] += 1
    bad_edges = {k: v for k, v in edge_pl_count.items() if v != 2}
    assert not bad_edges, f"edges on wrong number of plaquettes: {list(bad_edges.items())[:5]}"
    
    # Each edge in edges should appear in exactly 1 plaquette pair (covered)
    edge_set = set(tuple(sorted([q1, q2])) for q1, q2, _ in lat['edges'])
    pl_edge_set = set(edge_pl_count.keys())
    assert edge_set == pl_edge_set, "edge sets don't match between edges and plaquettes"
    
    # Edge colour count: at each vertex, exactly one edge of each colour
    incident_colors = defaultdict(set)
    for q1, q2, c in lat['edges']:
        incident_colors[q1].add(c)
        incident_colors[q2].add(c)
    bad = [(q, cs) for q, cs in incident_colors.items() if cs != {'r', 'g', 'b'}]
    assert not bad, f"vertices with wrong colour set: {bad[:5]}"
    
    # Plaquette weights
    for v in lat['plaquettes']['g']:
        assert len(v) == 4
    for v in lat['plaquettes']['r']:
        assert len(v) == 8
    for v in lat['plaquettes']['b']:
        assert len(v) == 8
    
    print(f"Lattice 4.8.8 with Lx={Lx}, Ly={Ly}: {n} qubits, "
          f"{len(lat['edges'])} edges, "
          f"{sum(len(p) for p in lat['plaquettes'].values())} plaquettes "
          f"({len(lat['plaquettes']['r'])}r + {len(lat['plaquettes']['g'])}g "
          f"+ {len(lat['plaquettes']['b'])}b). Sanity OK.")


if __name__ == '__main__':
    for L in [(2, 2), (2, 4), (4, 4)]:
        lat = build_lattice(*L)
        sanity_check(lat)
        for c in 'rgb':
            ne = len(lat['edges_by_color'][c])
            print(f"  {c}-edges: {ne}")