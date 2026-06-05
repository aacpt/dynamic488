"""
Two-panel spatial and timelike distance figure for the paper (Figure 4).

Writes the two-panel figure to figures/two_panel_row.png and .pdf.
"""

from __future__ import annotations

import pickle

import os, sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import results_dir, figures_dir

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})


# Okabe-Ito palette
DISPLAY = {
    'with_reset': dict(label='reset', color='#0072B2', marker='o', x_offset=-0.06),
    'no_reset': dict(label='no-reset', color='#009E73', marker='s', x_offset=-0.02),
    'ancilla_unpipelined': dict(label='ancilla-based un-pipelined', color='#D55E00',
                                marker='^', x_offset=+0.02),
    'ancilla_pipelined': dict(label='ancilla-based pipelined', color='#E69F00',
                              marker='D', x_offset=+0.06),
}
VARIANT_ORDER = ['with_reset', 'no_reset', 'ancilla_unpipelined', 'ancilla_pipelined']

C_RESET = '#0072B2'
C_OTHER = '#D55E00'

# n_qec window for the right panel (closed forms exact here)
NQEC_MIN = 4


# Data loading
def load_spatial():
    """Read spatial_distance.pkl -> {variant: {L: d}} (graphlike sweep)."""
    with open(results_dir('spatial') / 'spatial_distance.pkl', 'rb') as f:
        raw = pickle.load(f)
    return raw['graphlike']


def load_timelike_rows():
    """
    Read timelike_bounds.pkl -> (block, L_used) where block maps
    variant -> list of row dicts {n_qec, d_graph, d_hyper}.
    """
    with open(results_dir('timelike') / 'timelike_bounds.pkl', 'rb') as f:
        raw = pickle.load(f)
    if 'data' in raw:
        # current schema (timelike_distance.py): data[variant] = list of rows
        #   {L, n_qec, d_graph, d_hyper, ...} swept with n_qec = L, which is
        #   already the {variant -> rows} block plot_timelike expects
        return raw['data'], raw.get('L_values')
    if 'data_by_L' in raw:
        by_L = raw['data_by_L']
        L_used = 4 if 4 in by_L else sorted(by_L)[0]
        return by_L[L_used], L_used
    # legacy synthetic pkl
    blk = raw['L4_complete']
    return {v: blk[v]['rows'] for v in blk}, 4


# Left panel: spatial distance vs L
def plot_spatial(ax, graphlike):
    """
    All four 4.8.8 variants give d_spatial = L, computed with Stim's
    shortest_graphlike_error on the horizontal non-contractible X-cycle and
    cross-checked at small L with the full undetectable-error search
    (see spatial_distance.py).
    """
    all_Ls = sorted({L for variant in VARIANT_ORDER
                     for L in graphlike.get(variant, {})})
    L_max = max(all_Ls) if all_Ls else 12

    for variant in VARIANT_ORDER:
        info = DISPLAY[variant]
        dmap = graphlike.get(variant, {})
        if not dmap:
            continue
        Ls = np.array(sorted(dmap), dtype=float)
        ds = np.array([dmap[int(L)] for L in Ls], dtype=float)
        ax.plot(Ls + info['x_offset'], ds,
                marker=info['marker'], markersize=8,
                color=info['color'], linewidth=2.0,
                markerfacecolor=info['color'],
                markeredgecolor='white', markeredgewidth=0.8,
                label=info['label'])

    L_ref = np.linspace(0.0, L_max + 1.5, 100)
    ax.plot(L_ref, L_ref, color='black', linestyle=':',
            linewidth=1.2, alpha=0.55, zorder=0, label=r'$d = L$')
    ax.plot(L_ref, L_ref / 2, color='#CC2222', linestyle='--',
            linewidth=1.3, alpha=0.7, zorder=0,
            label=r'$d = L/2$ (dynamic honeycomb)')

    ax.set_xlabel(r'Code distance $L$', fontsize=12)
    ax.set_ylabel(r'Spatial distance $d_{\mathrm{spatial}}$', fontsize=12)
    ax.set_title(r'(a) Spatial distance preservation', fontsize=16,
                 loc='center', fontfamily='serif', pad=12)
    ax.set_xlim(0.0, L_max + 1.5)
    ax.set_ylim(-0.2, L_max + 1.0)
    ax.set_xticks([0] + all_Ls)
    ax.set_yticks(list(range(0, L_max + 1, 2)))
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.93)
    ax.set_aspect('equal', adjustable='box')


# Right panel: timelike bounds vs n_qec
def plot_timelike(ax, block):
    """
    Reset: separated bracket (d_graph slope 3, d_hyper slope 2).
    No-reset == ancilla (un-pipelined and pipelined): single tight curve,
    slope 3/2. Only n_qec >= NQEC_MIN is shown (closed forms exact).
    """

    def rows(variant):
        rs = [r for r in block[variant] if r['n_qec'] >= NQEC_MIN]
        n = np.array([r['n_qec'] for r in rs], dtype=float)
        g = np.array([r['d_graph'] for r in rs], dtype=float)
        h = np.array([r['d_hyper'] for r in rs], dtype=float)
        return n, g, h

    # Reset
    n, g, h = rows('with_reset')
    ax.plot(n, g, 'o-', color=C_RESET, ms=7, lw=2.0,
            markerfacecolor=C_RESET, markeredgecolor='white', markeredgewidth=0.8,
            label=r'reset $d_{\mathrm{graph}}$')
    ax.plot(n, h, 'o--', color=C_RESET, ms=7, lw=1.8,
            markerfacecolor='white', markeredgecolor=C_RESET, markeredgewidth=1.6,
            label=r'reset $d_{\mathrm{hyper}}$')
    ax.fill_between(n, h, g, color=C_RESET, alpha=0.12, zorder=0)

    # No-reset / both ancilla-based
    n2, g2, _ = rows('no_reset')
    ax.plot(n2, g2, 's-', color=C_OTHER, ms=6.5, lw=2.0,
            markerfacecolor=C_OTHER, markeredgecolor='white', markeredgewidth=0.8,
            label=r'no-reset $=$ ancilla-based ($d_{\mathrm{graph}}{=}d_{\mathrm{hyper}}$)')

    # Exact closed-form guide lines 
    nn = np.linspace(NQEC_MIN, 12, 200)
    ax.plot(nn, 3 * nn - 5, color=C_RESET, lw=0.9, ls=':', zorder=1)
    ax.plot(nn, 2 * nn - 3, color=C_RESET, lw=0.9, ls=':', zorder=1)
    ax.plot(nn, 1.5 * nn - 2, color=C_OTHER, lw=0.9, ls=':', zorder=1)

    ax.set_xlabel(r'Noisy Floquet periods $n_{\mathrm{qec}}$', fontsize=12)
    ax.set_ylabel(r'Shortest fault chain: early $\to$ late detectors', fontsize=12)
    ax.set_title(r'(b) Timelike bounds', fontsize=16,
                 loc='center', fontfamily='serif', pad=12)
    ax.set_xlim(3.8, 12.6)
    ax.set_ylim(2, 36)
    ax.set_xticks([4, 6, 8, 10, 12])
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9.3, framealpha=0.94)


def _line_angle_deg(ax, slope):
    p0 = ax.transData.transform_point((0.0, 0.0))
    p1 = ax.transData.transform_point((1.0, slope))
    return np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))


def add_slope_labels(ax):
    placements = [
        # slope, intercept, x_mid, y_offset_above_line, label, colour
        (3.0,  -5.0,  7.5, 1.4,
         r'$3\,n_{\mathrm{qec}} - 5$  (slope $=3$)',     C_RESET),
        (2.0,  -3.0,  8.5, 1.1,
         r'$2\,n_{\mathrm{qec}} - 3$  (slope $=2$)',     C_RESET),
        (1.5,  -2.0,  9.5, 0.9,
         r'$\frac{3}{2}\,n_{\mathrm{qec}} - 2$  (slope $=3/2$)', C_OTHER),
    ]
    for m, c, x_mid, dy, label, colour in placements:
        y_line = m * x_mid + c
        ax.text(x_mid, y_line + dy, label,
                color=colour, fontsize=10.5,
                rotation=_line_angle_deg(ax, m),
                rotation_mode='anchor',
                ha='center', va='bottom',
                transform=ax.transData)


def main():
    graphlike = load_spatial()
    block, L_used = load_timelike_rows()
    print(f"Spatial: variants {list(graphlike)}; "
          f"timelike: using L={L_used} block")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6.3))
    plot_spatial(ax_l, graphlike)
    plot_timelike(ax_r, block)

    fig.tight_layout()
    add_slope_labels(ax_r)  # after layout: rotation uses finalized transform
    out_dir = figures_dir()
    for ext in ('png', 'pdf'):
        out = out_dir / f'two_panel_row.{ext}'
        fig.savefig(out, dpi=160, bbox_inches='tight')
        print(f'Wrote {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()