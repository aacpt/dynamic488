"""
Sub-threshold scaling extrapolated to the teraquop regime (LER = 1e-12).

Fit: log10(LER) = c + slope * (L/2),  Λ = 10**(-slope).
Teraquop distance: L* = 2 * (log10(1e-12) - c) / slope, rounded up to even.
Footprint: Q = 4 L*^2 (dynamic, data-qubit only) or 10 L*^2 (ancilla-based).
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import results_dir, figures_dir

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

TARGET = 1e-12
Y_FLOOR = 3e-13  # bottom of the y-axis; dashed fits are extended down to here

OKABE_ITO = {'with_reset': '#0072B2', 'no_reset': '#009E73',
             'ancilla': '#D55E00', 'pipelined': '#E69F00'}
PRETTY = {'with_reset': 'reset', 'no_reset': 'no-reset',
          'ancilla': 'ancilla-based un-pipelined', 'pipelined': 'ancilla-based pipelined'}
MARKERS = {'with_reset': 'o', 'no_reset': 's', 'ancilla': '^', 'pipelined': 'D'}
QUBITS_PER_L2 = {'with_reset': 4, 'no_reset': 4, 'ancilla': 10, 'pipelined': 10}
SHORT = {'with_reset': 'reset', 'no_reset': 'no-reset',
         'ancilla': 'ancilla', 'pipelined': 'pipelined'}
ORDER = ['with_reset', 'no_reset', 'ancilla', 'pipelined']


def load():
    with open(results_dir('subthreshold') / 'subthreshold_results_all_variants.pkl', 'rb') as f:
        results = pickle.load(f)
    results = [r for r in results if 'failed' not in r and r.get('errors', 0) > 0]
    by_p_v = {}
    for r in results:
        by_p_v.setdefault((round(r['p'], 6), r['variant']), []).append(
            (r['L'], r['ler'], r['se']))
    return by_p_v


def fit_line(rows):
    """
    rows: list of (L, ler, se). Returns (slope, intercept, Lambda) on the
    log10(LER) vs L/2 line, or None if <2 usable points.
    """
    rows = sorted(rows)
    Ls = np.array([r[0] for r in rows], dtype=float)
    lers = np.array([r[1] for r in rows])
    if len(Ls) < 2:
        return None
    log_lers = np.log10(np.maximum(lers, 1e-300))
    slope, intercept = np.polyfit(Ls / 2.0, log_lers, 1)
    return slope, intercept, float(10 ** (-slope))


def teraquop_L(slope, intercept):
    if slope >= 0:  # not suppressing - no teraquop crossing
        return None
    Lstar = 2.0 * (np.log10(TARGET) - intercept) / slope
    return int(np.ceil(Lstar / 2.0) * 2)  # round up to an even distance


def main():
    by_p_v = load()
    all_ps = sorted({p for (p, _) in by_p_v})
    ps = [p for p in all_ps if abs(p - 1e-3) < 1e-9]  # only the p = 1e-3 panel
    if not ps:
        ps = all_ps[1:-1] or all_ps  # fall back: drop lowest/highest p

    # Vertical stack: one full-width panel per p, single continuous log y-axis
    #   from the data down to the teraquop target
    full_width = 5.4 * 2
    fig, axes = plt.subplots(len(ps), 1, figsize=(full_width, 4.0 * len(ps)),
                             squeeze=False)
    axes = axes[:, 0]

    out = {}
    for ax, p in zip(axes, ps):
        data_max, x_right = 1e-12, 12
        for v in ORDER:
            rows = by_p_v.get((p, v), [])
            if len(rows) < 2:
                continue
            rows.sort()
            Ls = np.array([r[0] for r in rows], dtype=float)
            lers = np.array([r[1] for r in rows])
            ses = np.array([r[2] for r in rows])
            colour, mk = OKABE_ITO[v], MARKERS[v]

            ax.errorbar(Ls, lers, yerr=ses, marker=mk, ls='none', color=colour,
                        markersize=7.5, markerfacecolor='white',
                        markeredgewidth=1.6, capsize=3, zorder=5)
            data_max = max(data_max, lers.max())

            fit = fit_line(list(zip(Ls, lers, ses)))
            if fit is None:
                continue
            slope, intercept, lam = fit
            Lstar = teraquop_L(slope, intercept)
            out.setdefault(f'{p:.4g}', {})[v] = {
                'Lambda': lam, 'Lstar': Lstar,
                'qubits': (QUBITS_PER_L2[v] * Lstar ** 2) if Lstar else None}

            # Solid over the measured range, dashed extrapolation down to L*
            L_solid = np.linspace(Ls.min(), Ls.max(), 30)
            ax.plot(L_solid, 10 ** (intercept + slope * L_solid / 2),
                    color=colour, lw=1.7, zorder=3)
            if Lstar:
                # Extend the dashed extrapolation past the 1e-12 crossing all the
                #   way down to the x-axis (y = Y_FLOOR).
                L_floor = 2.0 * (np.log10(Y_FLOOR) - intercept) / slope
                x_right = max(x_right, L_floor)
                L_dash = np.linspace(Ls.max(), L_floor, 120)
                ax.plot(L_dash, 10 ** (intercept + slope * L_dash / 2),
                        color=colour, lw=1.3, ls='--', alpha=0.85, zorder=3)
                # Solid colour marker (no outline) at the exact 1e-12 crossing, so
                #   it sits on the curve; the reported L* (box) rounds up to even.
                L_cross = 2.0 * (np.log10(TARGET) - intercept) / slope
                ax.plot([L_cross], [TARGET], marker=mk, color=colour, markersize=9,
                        markerfacecolor=colour, markeredgecolor=colour, zorder=6)

        ax.set_yscale('log')
        ax.set_ylim(Y_FLOOR, min(data_max * 3, 1.0))
        ax.set_xlim(5, x_right + 1)
        ax.axhline(TARGET, color='0.4', ls=':', lw=1.2, zorder=1)
        ax.text(0.012, TARGET * 1.6, 'teraquop  $10^{-12}$',
                transform=ax.get_yaxis_transform(), fontsize=8.5,
                color='0.3', va='bottom')
        ax.set_title(fr'$p = {p * 1000:g}\times 10^{{-3}}$', fontsize=12.5)
        ax.grid(True, which='both', alpha=0.25)
        ax.set_xlabel('Code distance $L$', fontsize=11.5)
        ax.set_ylabel('Logical error rate per round', fontsize=11)

        if ax is axes[0]:
            handles = [plt.Line2D([], [], color=OKABE_ITO[v], marker=MARKERS[v],
                                  ls='-', markerfacecolor='white', markeredgewidth=1.5,
                                  label=PRETTY[v]) for v in ORDER if (p, v) in by_p_v]
            ax.legend(handles=handles, fontsize=8.5, loc='lower left', framealpha=0.9)

        tag = f'{p:.4g}'
        if tag in out:
            lines = [r'$L^*$ to $10^{-12}$:']
            for v in ORDER:
                if v in out[tag] and out[tag][v]['Lstar']:
                    q = out[tag][v]['qubits']
                    lines.append(rf'  {SHORT[v]}: $L^*\!=\!{out[tag][v]["Lstar"]}$ '
                                 rf'(${q/1000:.1f}\mathrm{{k}}$ q)')
            ax.text(0.985, 0.96, '\n'.join(lines), transform=ax.transAxes,
                    ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                              edgecolor='gray', alpha=0.9))

    fig.tight_layout()
    fig.savefig(figures_dir() / 'subthreshold_teraquop.png', dpi=140,
                bbox_inches='tight')
    with open(results_dir('subthreshold') / 'subthreshold_teraquop.pkl', 'wb') as f:
        pickle.dump(out, f)
    print('Wrote subthreshold_teraquop.png and subthreshold_teraquop.pkl')
    for tag, d in out.items():
        print(f'\np = {tag}:')
        for v in ORDER:
            if v in d and d[v]['Lstar']:
                print(f'  {v:<12} Lambda={d[v]["Lambda"]:.2f}  '
                      f'L*={d[v]["Lstar"]}  qubits={d[v]["qubits"]:,}')


if __name__ == '__main__':
    main()
