"""
Spacetime volume for all four variants.

Operation durations follow the superconducting model in 
nature.com/articles/s41534-025-00998-y: H=20ns, CZ=40ns, M=600ns, 
and unconditional reset is either R=500ns for slow resets or 
R=100ns for fast resets.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


T_H  =  20  # H executed as one sqrt(X) pulse + virtual-Z framings
T_CZ =  40
T_M  = 600
T_R_slow = 500
T_R_fast = 100


def period_time_ns(variant, T_R):
    if variant in ('reset_fixed', 'reset_matched', 'ancilla_unpiped'):
        t_XX = T_H + T_CZ + T_M + T_R + T_CZ + T_H
        t_ZZ = T_H + T_CZ + T_H + T_M + T_R + T_H + T_CZ + T_H
        return 3 * t_XX + 3 * t_ZZ
    elif variant == 'no_reset':
        t_sub = T_H + T_CZ + T_H + T_M + T_H + T_CZ + T_H
        return 6 * t_sub
    elif variant == 'ancilla_piped':
        T_CX = 2 * T_H + T_CZ
        T_MX = 2 * T_H + T_M
        T_RX = T_R + T_H
        layers = [
            max(T_CX, T_R), max(T_CX, T_R),
            max(T_CX, T_MX), max(T_CX, T_MX),
            T_RX, max(T_CX, T_RX),
            max(T_CX, T_M, T_R), max(T_CX, T_MX, T_RX),
            max(T_CX, T_M), T_M,
        ]
        return sum(layers)
    raise ValueError(variant)


def n_periods(variant, L):
    """
    Number of Floquet periods. For the asymptotic time-matched reset,
    we use the continuous bound n*_qec = 3L/4 (no integer ceiling).
    """
    if variant == 'reset_matched':
        return 0.75 * L
    return L


def qubit_count(variant, L):
    if variant in ('reset_fixed', 'reset_matched', 'no_reset'):
        return 4 * L * L
    return 10 * L * L


def V(variant, L, T_R):
    return qubit_count(variant, L) * n_periods(variant, L) * period_time_ns(variant, T_R)


# Styling 
COL_RESET    = '#0072B2'
COL_NORESET  = '#009E73'
COL_ANC_UP   = '#D55E00'
COL_ANC_PIP  = '#E69F00'

# (legend label, colour, marker, linestyle)
STYLE = {
    'reset_fixed':      ('reset',
                          COL_RESET,   'o', '-'),
    'reset_matched':    ('reset (asymptotically timelike distance-matched)',
                          COL_RESET,   'o', '--'),
    'no_reset':         ('no-reset',
                          COL_NORESET, 's', '-'),
    'ancilla_unpiped':  ('ancilla-based un-pipelined',
                          COL_ANC_UP,  '^', '-'),
    'ancilla_piped':    ('ancilla-based pipelined',
                          COL_ANC_PIP, 'D', '-'),
}
ORDER = ['reset_fixed', 'reset_matched', 'no_reset',
         'ancilla_unpiped', 'ancilla_piped']

ALPHA_SLOW = 1.0
ALPHA_FAST = 0.35


def make_overlay_plot(outpath):
    L_values = [2, 4, 6, 8, 12]
    L_ticks = [2, 4, 6, 8, 12]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8))

    # Panel (a): V(L) vs L 
    for key in ORDER:
        label, col, mk, ls = STYLE[key]
        Vs_slow = [V(key, L, T_R_slow) for L in L_values]
        ax1.plot(L_values, Vs_slow, marker=mk, linestyle=ls,
                  color=col, markersize=8, linewidth=2,
                  markerfacecolor='white', markeredgewidth=1.5,
                  markeredgecolor=col, alpha=ALPHA_SLOW,
                  label=label)
        if key == 'no_reset':
            continue  # No-reset is reset-time-independent
        Vs_fast = [V(key, L, T_R_fast) for L in L_values]
        ax1.plot(L_values, Vs_fast, marker=mk, linestyle=ls,
                  color=col, markersize=8, linewidth=2,
                  markerfacecolor='white', markeredgewidth=1.5,
                  markeredgecolor=col, alpha=ALPHA_FAST)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1.7, 14)
    ax1.set_xlabel(r'Code distance $L$', family='serif')
    ax1.set_ylabel(r'Spacetime volume $V(L) = Q \cdot n_{\mathrm{qec}}'
                    r' \cdot t_{\mathrm{period}}$ [qubit$\cdot$ns]',
                    family='serif')
    ax1.set_title('(a) Spacetime volume', family='serif', loc='center', pad=10)
    ax1.grid(True, which='major', linestyle=':', alpha=0.4)
    ax1.legend(loc='upper left', fontsize=9.5, framealpha=0.94,
                prop={'family': 'serif'})
    ax1.xaxis.set_major_locator(FixedLocator(L_ticks))
    ax1.xaxis.set_minor_locator(NullLocator())
    ax1.set_xticklabels([str(L) for L in L_ticks], family='serif')

    # Panel (b): ratios vs ancilla un-pipelined 
    # Reference line — same x-range as the other curves (L=2..12)
    ref_x = L_values
    ref_y = [1.0] * len(L_values)
    ax2.plot(ref_x, ref_y, color=COL_ANC_UP, linewidth=2,
              linestyle='-', alpha=0.85,
              marker=STYLE['ancilla_unpiped'][2],
              markersize=8, markerfacecolor='white',
              markeredgewidth=1.5, markeredgecolor=COL_ANC_UP)

    for key in ORDER:
        label, col, mk, ls = STYLE[key]
        if key == 'ancilla_unpiped':
            ax2.plot([], [], marker=mk, linestyle=ls, color=col,
                      markersize=8, linewidth=2,
                      markerfacecolor='white', markeredgewidth=1.5,
                      markeredgecolor=col, label=label)
            continue
        ratios_slow = [V(key, L, T_R_slow) / V('ancilla_unpiped', L, T_R_slow)
                        for L in L_values]
        ax2.plot(L_values, ratios_slow, marker=mk, linestyle=ls,
                  color=col, markersize=8, linewidth=2,
                  markerfacecolor='white', markeredgewidth=1.5,
                  markeredgecolor=col, alpha=ALPHA_SLOW,
                  label=label)
        ratios_fast = [V(key, L, T_R_fast) / V('ancilla_unpiped', L, T_R_fast)
                        for L in L_values]
        ax2.plot(L_values, ratios_fast, marker=mk, linestyle=ls,
                  color=col, markersize=8, linewidth=2,
                  markerfacecolor='white', markeredgewidth=1.5,
                  markeredgecolor=col, alpha=ALPHA_FAST)

    # Numerical value labels
    anc_slow = V('ancilla_unpiped', 12, T_R_slow)
    anc_fast = V('ancilla_unpiped', 12, T_R_fast)
    rs_val  = V('reset_fixed',   12, T_R_slow) / anc_slow    # 0.400
    rm_val  = V('reset_matched', 12, T_R_slow) / anc_slow    # 0.300
    nr_slow = V('no_reset',      12, T_R_slow) / anc_slow    # 0.271
    nr_fast = V('no_reset',      12, T_R_fast) / anc_fast    # 0.383
    ap_slow = V('ancilla_piped', 12, T_R_slow) / anc_slow    # 0.745
    ap_fast = V('ancilla_piped', 12, T_R_fast) / anc_fast    # 0.799

    X_LABEL = 13.5
    labels = [
        (ap_fast, ap_fast + 0.022, COL_ANC_PIP, ALPHA_FAST),
        (ap_slow, ap_slow + 0.022, COL_ANC_PIP, ALPHA_SLOW),
        (rs_val,  rs_val  + 0.022, COL_RESET,   ALPHA_SLOW),
        (nr_fast, nr_fast + 0.010, COL_NORESET, ALPHA_FAST),
        (rm_val,  rm_val  + 0.022, COL_RESET,   ALPHA_SLOW),
        (nr_slow, nr_slow + 0.018, COL_NORESET, ALPHA_SLOW),
    ]
    for val, y, col, alpha in labels:
        ax2.text(X_LABEL, y, f'{val:.3f}',
                  family='serif', fontsize=9,
                  color=col, alpha=max(alpha, 0.55),
                  ha='left', va='center',
                  bbox=dict(boxstyle='round,pad=0.15',
                            facecolor='white', edgecolor='none',
                            alpha=0.85))

    ax2.set_xscale('log')
    ax2.set_xlim(1.7, 15.8)
    ax2.set_ylim(0.2, 1.18)
    ax2.set_xlabel(r'Code distance $L$', family='serif')
    ax2.set_ylabel(r'$V_{\mathrm{variant}} \,/\, V_{\mathrm{ancilla\text{-}based\ un\text{-}pipelined}}$',
                    family='serif')
    ax2.set_title('(b) Relative ratio', family='serif', loc='center', pad=10)
    ax2.grid(True, which='major', linestyle=':', alpha=0.4)
    ax2.xaxis.set_major_locator(FixedLocator(L_ticks))
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.set_xticklabels([str(L) for L in L_ticks], family='serif')

    ax1.text(0.98, 0.02,
              'dark:  slow reset ($R{=}500$ ns)\n'
              'light: fast reset ($R{=}100$ ns)',
              transform=ax1.transAxes,
              fontsize=8.5, family='serif',
              ha='right', va='bottom',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='lightgrey', alpha=0.9))

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'Saved {outpath}')


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from paths import figures_dir
    OUT = figures_dir()
    make_overlay_plot(OUT / 'spacetime_volume.png')
    make_overlay_plot(OUT / 'spacetime_volume.pdf')

    # Print numerical values 
    print(f'\nAsymptotic ratios (constant in L under continuous n*_qec = 3L/4) '
          f'at T_H={T_H} ns:')
    print(f'  reset                                  : '
          f'{V("reset_fixed",   12, T_R_slow) / V("ancilla_unpiped", 12, T_R_slow):.4f} '
          f'(slow & fast)')
    print(f'  reset (asymp timelike dist-matched)    : '
          f'{V("reset_matched", 12, T_R_slow) / V("ancilla_unpiped", 12, T_R_slow):.4f} '
          f'(slow & fast)')
    print(f'  no-reset slow                          : '
          f'{V("no_reset",      12, T_R_slow) / V("ancilla_unpiped", 12, T_R_slow):.4f}')
    print(f'  no-reset fast                          : '
          f'{V("no_reset",      12, T_R_fast) / V("ancilla_unpiped", 12, T_R_fast):.4f}')
    print(f'  ancilla-based un-pipelined             : 1.0000 (reference)')
    print(f'  ancilla-based pipelined slow           : '
          f'{V("ancilla_piped", 12, T_R_slow) / V("ancilla_unpiped", 12, T_R_slow):.4f}')
    print(f'  ancilla-based pipelined fast           : '
          f'{V("ancilla_piped", 12, T_R_fast) / V("ancilla_unpiped", 12, T_R_fast):.4f}')