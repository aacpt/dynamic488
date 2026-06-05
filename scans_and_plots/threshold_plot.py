"""
Per-round threshold plots for the four variants.

For each (variant, L, p) raw point with errors and shots, convert the
per-shot logical error rate to a per-QEC-round rate using:

    p_shot  = errors / shots
    p_round = (1 - (1 - 2 * p_shot) ** (1 / rounds)) / 2

where rounds = L (this scan uses n_qec = L). This is the convention
that respects the per-shot saturation at 1/2 — for an n_qec-round memory
experiment with per-round failure rate r, the probability of an odd
number of logical flips is P = (1 - (1-2r)^n) / 2, and inverting that
gives the formula above.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
# The shared modules (paths, ...) live in the sibling src/ folder.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import results_dir, figures_dir

RESULTS_DIR = results_dir('threshold')
PLOTS_DIR = figures_dir()


OI_COLOURS = {
    'with_reset': '#0072B2',  # blue
    'no_reset':   '#009E73',  # bluish green
    'ancilla':    '#D55E00',  # vermillion
    'pipelined':  '#E69F00',  # orange
}
MARKER_FOR_L = {4: 'o', 6: 's', 8: '^', 10: 'D', 12: 'v', 16: 'P', 18: 'X'}
PRETTY_NAME = {
    'with_reset': 'reset',
    'no_reset':   'no-reset',
    'ancilla':    'ancilla un-pipelined',
    'pipelined':  'ancilla pipelined',
}
VARIANT_ORDER = ['with_reset', 'no_reset', 'ancilla', 'pipelined']
ROUNDS_CONVENTION = 'n_qec = L'   # one QEC round = one Floquet period

BP_LIGHTEN = 0.55


def lighten(hex_color: str, amount: float = BP_LIGHTEN) -> str:
    """Blend a hex colour with white. amount=0 -> unchanged, amount=1 -> white."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f'#{r:02X}{g:02X}{b:02X}'


# Per-shot -> per-round conversion 
def per_shot_to_per_round(p_shot: float, rounds: int) -> float:
    """
    P_shot = (1 - (1-2r)^n) / 2  =>  r = (1 - (1-2*P_shot)^(1/n)) / 2.

    Saturates correctly: P_shot -> 1/2 maps to r -> 1/2. For small P_shot,
    r ~ P_shot / n.
    """
    if p_shot >= 0.5:
        return 0.5
    if p_shot <= 0.0:
        return 0.0
    return 0.5 * (1.0 - (1.0 - 2.0 * p_shot) ** (1.0 / rounds))


def per_shot_se_to_per_round_se(p_shot: float, se_shot: float,
                                 rounds: int) -> float:
    """
    Propagate a per-shot binomial SE through the per-round map.

    dr/dP_shot = (1/n) (1 - 2 P_shot)^(1/n - 1).
    """
    if p_shot >= 0.5 - 1e-12 or p_shot <= 1e-12:
        return 0.0
    deriv = (1.0 / rounds) * (1.0 - 2.0 * p_shot) ** (1.0 / rounds - 1.0)
    return deriv * se_shot


# Crossing finder
def find_crossing(p_arr: np.ndarray,
                  r_lo: np.ndarray,
                  r_hi: np.ndarray,
                  log_space: bool = True):
    """
    Return the p at which (r_lo - r_hi) changes sign by linear interp.

    All three arrays are aligned on the same p grid (length >= 2).
    Returns None if the difference does not change sign anywhere in the
    window.
    """
    if log_space:
        # Floor to avoid log(0). Anything well below the smallest
        #   observable rate at the given shot count is treated the same
        floor = 1e-9
        a = np.log(np.maximum(r_lo, floor))
        b = np.log(np.maximum(r_hi, floor))
    else:
        a, b = r_lo, r_hi
    diff = a - b
    for i in range(len(p_arr) - 1):
        d1, d2 = diff[i], diff[i + 1]
        if d1 == 0:
            return float(p_arr[i])
        if d1 * d2 < 0:
            t = d1 / (d1 - d2)
            return float(p_arr[i] + t * (p_arr[i + 1] - p_arr[i]))
    return None


def bin_crossing(rows_lo, rows_hi, L_lo, L_hi,
                       n_bootstrap, rng):
    """
    Resample errors ~ Binomial(shots, p_shot) for both L_lo and L_hi
    curves, recompute per-round LER at every p, and find the crossing.
    """
    p_lo = {r['p']: r for r in rows_lo}
    p_hi = {r['p']: r for r in rows_hi}
    p_common = sorted(set(p_lo) & set(p_hi))
    if len(p_common) < 2:
        return None, None, 0, None
    p_arr = np.array(p_common)
    shots_lo = np.array([p_lo[p]['shots'] for p in p_common])
    shots_hi = np.array([p_hi[p]['shots'] for p in p_common])
    errs_lo  = np.array([p_lo[p]['errors'] for p in p_common])
    errs_hi  = np.array([p_hi[p]['errors'] for p in p_common])

    pshot_lo = errs_lo / shots_lo
    pshot_hi = errs_hi / shots_hi
    rround_lo = np.array([per_shot_to_per_round(x, L_lo) for x in pshot_lo])
    rround_hi = np.array([per_shot_to_per_round(x, L_hi) for x in pshot_hi])
    point = find_crossing(p_arr, rround_lo, rround_hi)

    samples = []
    for _ in range(n_bootstrap):
        e_lo = rng.binomial(shots_lo, pshot_lo)
        e_hi = rng.binomial(shots_hi, pshot_hi)
        ps_lo = e_lo / shots_lo
        ps_hi = e_hi / shots_hi
        rr_lo = np.array([per_shot_to_per_round(x, L_lo) for x in ps_lo])
        rr_hi = np.array([per_shot_to_per_round(x, L_hi) for x in ps_hi])
        c = find_crossing(p_arr, rr_lo, rr_hi)
        if c is not None:
            samples.append(c)
    samples = np.array(samples)
    if len(samples) < 10:
        return point, None, len(samples), point
    return float(samples.mean()), float(samples.std(ddof=1)), len(samples), point


# Load / process
def load_data():
    """
    Return (raw_mwpm, raw_bp): two {variant: {L_str: [rows]}} dicts in the
    single-decoder shape (each row has p/ler/se/errors/shots), one per
    decoder.
    """
    path = RESULTS_DIR / 'threshold_results.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    raw_mwpm, raw_bp = {}, {}
    for variant, by_L in data.items():
        for L_str, rows in by_L.items():
            for r in rows:
                if 'ler_mwpm' in r:
                    raw_mwpm.setdefault(variant, {}).setdefault(L_str, []).append(
                        {'p': r['p'], 'ler': r['ler_mwpm'], 'se': r['se_mwpm'],
                         'errors': r['errors_mwpm'], 'shots': r['shots']})
                elif 'ler' in r:  
                    raw_mwpm.setdefault(variant, {}).setdefault(L_str, []).append(
                        {'p': r['p'], 'ler': r['ler'], 'se': r.get('se', 0.0),
                         'errors': r.get('errors', 0), 'shots': r['shots']})
                if 'ler_bp' in r:
                    raw_bp.setdefault(variant, {}).setdefault(L_str, []).append(
                        {'p': r['p'], 'ler': r['ler_bp'], 'se': r['se_bp'],
                         'errors': r['errors_bp'], 'shots': r['shots']})
    return raw_mwpm, raw_bp


def compute_per_round(raw):
    """Augment each row with 'p_round' and 'se_round' fields."""
    out = {}
    for variant, by_L in raw.items():
        out[variant] = {}
        for L_str, rows in by_L.items():
            L = int(L_str)
            new_rows = []
            for r in rows:
                if 'shots' in r and 'errors' in r and r['shots'] > 0:
                    n = r['shots']; e = r['errors']
                    p_s = e / n
                    se_s = float(np.sqrt(p_s * (1 - p_s) / n))
                else:
                    n = None
                    p_s = r['ler']
                    se_s = r.get('se', 0.0)
                p_r = per_shot_to_per_round(p_s, L)

                if n is not None and n > 0:
                    z = 1.0
                    denom = 1.0 + z*z / n
                    center = (p_s + z*z / (2*n)) / denom
                    margin = (z * np.sqrt(p_s * (1 - p_s) / n
                                          + z*z / (4 * n * n))) / denom
                    p_lo = max(0.0, center - margin)
                    p_hi = min(0.499999, center + margin)
                    pr_lo = per_shot_to_per_round(p_lo, L)
                    pr_hi = per_shot_to_per_round(p_hi, L)
                    yerr_minus = max(0.0, p_r - pr_lo)
                    yerr_plus  = max(0.0, pr_hi - p_r)
                    se_r = 0.5 * (yerr_minus + yerr_plus)
                else:
                    se_r = per_shot_se_to_per_round_se(p_s, se_s, L)
                    yerr_minus = yerr_plus = se_r

                new_rows.append({**r,
                                 'p_shot': p_s, 'se_shot': se_s,
                                 'p_round': p_r, 'se_round': se_r,
                                 'yerr_minus_round': yerr_minus,
                                 'yerr_plus_round': yerr_plus})
            new_rows.sort(key=lambda x: x['p'])
            out[variant][L_str] = new_rows
    return out


def crossing_all_variants(processed, L_lo=None, L_hi=None,
                          n_bootstrap=2000, seed=0):
    """
    Per-variant LER-curve crossing. When L_lo/L_hi are None (the default),
    each variant uses the smallest and largest L curve it actually has, so
    the scan works with whatever distances are present. Passing an explicit
    pair forces those two L values and warns if a variant lacks them.
    """
    rng = np.random.default_rng(seed)
    auto = L_lo is None or L_hi is None
    fits = {}
    for variant in VARIANT_ORDER:
        if variant not in processed:
            continue
        by_L = processed[variant]
        Ls = sorted(int(k) for k in by_L)

        if auto:
            if len(Ls) < 2:
                fits[variant] = {
                    'p_th': None, 'p_th_se': None,
                    'crossing_pair': [Ls[0] if Ls else None, None],
                    'rounds_convention': ROUNDS_CONVENTION,
                    'method': 'per_round_curve_crossing_binomial_bootstrap',
                    'n_bootstrap': n_bootstrap,
                    'n_bootstrap_success': 0,
                    'warning': f'need >=2 distinct L for a crossing; '
                               f'only L={Ls} present',
                }
                continue
            lo, hi = Ls[0], Ls[-1]
        else:
            lo, hi = L_lo, L_hi
            if str(lo) not in by_L or str(hi) not in by_L:
                fits[variant] = {
                    'p_th': None, 'p_th_se': None,
                    'crossing_pair': [lo, hi],
                    'rounds_convention': ROUNDS_CONVENTION,
                    'method': 'per_round_curve_crossing_binomial_bootstrap',
                    'n_bootstrap': n_bootstrap,
                    'n_bootstrap_success': 0,
                    'warning': f'L={lo} or L={hi} data missing '
                               f'(present: L={Ls})',
                }
                continue

        rows_lo = by_L[str(lo)]
        rows_hi = by_L[str(hi)]
        mean, se, n_succ, point = bin_crossing(
            rows_lo, rows_hi, lo, hi, n_bootstrap, rng)
        warnings = []
        if point is None:
            # Decide which direction to extend
            common_p = sorted(set(r['p'] for r in rows_lo) & set(r['p'] for r in rows_hi))
            if not common_p:
                warnings.append('No common p grid; rerun with overlapping p values.')
            else:
                p_max = max(common_p)
                last_lo = next(r['p_round'] for r in rows_lo if r['p'] == p_max)
                last_hi = next(r['p_round'] for r in rows_hi if r['p'] == p_max)
                direction = 'upward' if last_lo > last_hi else 'downward'
                warnings.append(
                    f"No L={lo}/L={hi} crossing bracketed in the scanned p range. "
                    f"Extend scan {direction}.")
        if point is not None and n_succ < 0.5 * n_bootstrap:
            warnings.append(
                f"Only {n_succ}/{n_bootstrap} bootstrap samples bracketed a crossing; "
                f"SE may be unreliable.")
        fits[variant] = {
            'p_th': mean if mean is not None else point,
            'p_th_se': se,
            'p_th_point_estimate': point,
            'crossing_pair': [lo, hi],
            'rounds_convention': ROUNDS_CONVENTION,
            'method': 'per_round_curve_crossing_binomial_bootstrap',
            'n_bootstrap': n_bootstrap,
            'n_bootstrap_success': int(n_succ),
            'warning': '; '.join(warnings) if warnings else None,
        }
    return fits


# Plot line + shaded band
def plot_curve_with_error_band(ax, x, y, yerr_minus, yerr_plus, *,
                               colour, marker, label, markersize=7,
                               markerfacecolor='white', markeredgewidth=1.4,
                               linewidth=1.6, alpha=0.95, band_alpha=0.16,
                               y_floor=0.0):
    """
    Plot a curve with a shaded band.

    The input errors may be asymmetric. The lower edge is clipped to
    `y_floor`, which is nice for log-y axes.
    """
    lower = np.maximum(y - yerr_minus, y_floor)
    upper = np.maximum(y + yerr_plus, y_floor)

    ax.fill_between(
        x, lower, upper,
        color=colour, alpha=band_alpha, linewidth=0, zorder=1,
    )
    ax.plot(
        x, y,
        marker=marker, color=colour, markersize=markersize,
        markerfacecolor=markerfacecolor, markeredgecolor=colour,
        markeredgewidth=markeredgewidth, linestyle='-', linewidth=linewidth,
        alpha=alpha, label=label, zorder=2,
    )

def plot_separated(processed, fits, out_path, p_max_plot=0.008,
                   processed_bp=None, fits_bp=None, xmax_pct=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    PANEL_LABELS = ['(a)', '(b)', '(c)', '(d)']

    def draw_panel_curves(ax, by_L, colour, decoder_tag):
        for L_str in sorted(by_L.keys(), key=int):
            L = int(L_str)
            rows = by_L[L_str]
            ps = np.array([r['p'] for r in rows])
            prs = np.array([r['p_round'] for r in rows])
            ymin = np.array([r.get('yerr_minus_round', r['se_round']) for r in rows])
            ymax = np.array([r.get('yerr_plus_round',  r['se_round']) for r in rows])
            mask = ps <= p_max_plot
            order = np.argsort(ps[mask])
            tag = '' if decoder_tag == 'MWPM' else f', {decoder_tag}'
            plot_curve_with_error_band(
                ax, ps[mask][order] * 100, prs[mask][order],
                ymin[mask][order], ymax[mask][order],
                colour=colour, marker=MARKER_FOR_L.get(L, 'o'),
                markersize=7, markerfacecolor='white', markeredgewidth=1.5,
                linewidth=1.6, band_alpha=0.16, y_floor=1e-12,
                label=f'L = {L}{tag}',
            )

    # Crossing bands
    _ref_se = (fits.get('ancilla') or {}).get('p_th_se')
    bar_hw = (_ref_se * 100) if _ref_se else 0.003   # half-width, x-axis % units

    legend_drawn = False  # one shared key, on the first visible panel only
    for ax, variant, panel_label in zip(axes, VARIANT_ORDER, PANEL_LABELS):
        if variant not in processed:
            ax.set_visible(False)
            continue
        colour = OI_COLOURS[variant]
        by_L = processed[variant]
        fit = fits.get(variant)

        # BP first so it sits behind MWPM visually
        if processed_bp is not None and variant in processed_bp:
            draw_panel_curves(ax, processed_bp[variant], lighten(colour), 'BP')
        draw_panel_curves(ax, by_L, colour, 'MWPM')

        text_lines = []
        if fit and fit['p_th'] is not None:
            p_th = fit['p_th']; p_se = fit.get('p_th_se') or 0.0
            ax.axvspan(p_th * 100 - bar_hw, p_th * 100 + bar_hw,
                       color=colour, alpha=0.40, zorder=0)
            se_str = (rf"\pm {p_se*100:.5f}\%" if p_se else r"\text{ (SE n/a)}")
            text_lines.append(rf"MWPM: $p_{{\rm th}} = {p_th*100:.3f}\% {se_str}$")
        if fits_bp is not None:
            f_bp = fits_bp.get(variant)
            if f_bp and f_bp['p_th'] is not None:
                p_th_bp = f_bp['p_th']; p_se_bp = f_bp.get('p_th_se') or 0.0
                ax.axvspan(p_th_bp * 100 - bar_hw, p_th_bp * 100 + bar_hw,
                           color=lighten(colour), alpha=0.45, zorder=0)
                se_str_bp = (rf"\pm {p_se_bp*100:.5f}\%" if p_se_bp else r"\text{ (SE n/a)}")
                text_lines.append(rf"BP+matching: $p_{{\rm th}} = {p_th_bp*100:.3f}\% {se_str_bp}$")
        if text_lines:
            ax.text(
                0.97, 0.04, '\n'.join(text_lines),
                transform=ax.transAxes,
                ha='right', va='bottom', fontsize=10.5,
                bbox=dict(boxstyle='round,pad=0.4', fc='white',
                          ec=colour, alpha=0.94, lw=1.0),
            )
        elif fit and fit.get('warning'):
            ax.text(0.97, 0.04, fit['warning'],
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=9, wrap=True,
                    bbox=dict(boxstyle='round,pad=0.4', fc='#fff5e6',
                              ec='#cc8800', alpha=0.94, lw=1.0))

        ax.set_xlabel(r'Physical error rate $p$  (%)', fontsize=11)
        ax.set_ylabel(r'Logical error rate per QEC round', fontsize=11)
        ax.set_title(f'{panel_label}  {PRETTY_NAME[variant]}',
                     fontsize=15, color='black',
                     fontfamily='serif', pad=10)
        ax.set_xlim(0.1, xmax_pct if xmax_pct is not None
                    else p_max_plot * 100 + 0.02)
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 0.5)

        if not legend_drawn:
            all_Ls = sorted({int(k) for by_L_v in processed.values()
                             for k in by_L_v})
            shape_handles = [
                Line2D([], [], marker=MARKER_FOR_L.get(L, 'o'),
                       linestyle='none', markerfacecolor='white',
                       markeredgecolor='0.35', markeredgewidth=1.4,
                       markersize=7, label=f'$L = {L}$')
                for L in all_Ls
            ]
            decoder_handles = [
                Line2D([], [], color='0.25', lw=3, label='MWPM')]
            if processed_bp:
                decoder_handles.append(
                    Line2D([], [], color='0.7', lw=3, label='BP+matching'))
            handles = shape_handles + decoder_handles
            ax.legend(handles=handles, loc='upper left', fontsize=9,
                      ncol=2 if len(handles) > 5 else 1, framealpha=0.9,
                      handlelength=1.6, columnspacing=1.0, labelspacing=0.3,
                      borderpad=0.4)
            legend_drawn = True
        ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    fig.savefig(str(out_path).replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


# Main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--crossing-pair', nargs=2, type=int, default=None,
                    metavar=('L_LO', 'L_HI'),
                    help='Which two L curves to use for the crossing. '
                         'Default: auto-select the smallest and largest L '
                         'present for each variant.')
    ap.add_argument('--bootstrap', type=int, default=2000,
                    help='Number of binomial bootstrap samples (default 2000).')
    ap.add_argument('--seed', type=int, default=0, help='RNG seed.')
    args = ap.parse_args()

    if args.crossing_pair is not None:
        L_lo, L_hi = sorted(args.crossing_pair)
    else:
        L_lo = L_hi = None

    print('Loading data ...')
    raw_mwpm, raw_bp = load_data()
    for variant in VARIANT_ORDER:
        if variant not in raw_mwpm:
            continue
        n_pts = sum(len(rows) for rows in raw_mwpm[variant].values())
        n_shots = sum(r.get('shots', 0)
                      for rows in raw_mwpm[variant].values()
                      for r in rows)
        print(f'  {variant:<12} {n_pts:>3} points, {n_shots:>10,} total shots (per decoder)')

    print('\nConverting to per-round ...')
    processed = compute_per_round(raw_mwpm)
    processed_bp = compute_per_round(raw_bp) if raw_bp else None

    if L_lo is None:
        print(f'\nFinding crossings (auto: smallest/largest L per variant, '
              f'{args.bootstrap} bootstrap samples) ...')
    else:
        print(f'\nFinding crossings (L={L_lo}/L={L_hi}, {args.bootstrap} bootstrap samples) ...')
    fits = crossing_all_variants(processed, L_lo, L_hi,
                                  n_bootstrap=args.bootstrap, seed=args.seed)
    fits_bp = (crossing_all_variants(processed_bp, L_lo, L_hi,
                                     n_bootstrap=args.bootstrap, seed=args.seed)
               if processed_bp else None)

    fits_out = RESULTS_DIR / 'threshold_fits.pkl'
    with open(fits_out, 'wb') as f:
        pickle.dump({'mwpm': fits, 'bp': fits_bp}, f)
    print(f'  wrote {fits_out}')

    def print_fits_table(name, fit_dict):
        print(f"\n{name}")
        print(f"{'variant':<22} {'p_th (%)':>10} {'+/- SE':>10} {'pair':>9} {'n_bs':>6}")
        print('-' * 65)
        for v, f in fit_dict.items():
            if f['p_th'] is None:
                print(f"  {PRETTY_NAME[v]:<20}      —          —    "
                      f"L={f['crossing_pair'][0]}/L={f['crossing_pair'][1]}  "
                      f"WARNING: {f.get('warning','')}")
                continue
            se = f.get('p_th_se')
            se_str = f"{se*100:>5.3f}%" if se is not None else " n/a   "
            cp = f['crossing_pair']
            print(f"  {PRETTY_NAME[v]:<20} {f['p_th']*100:>9.3f}% {se_str:>10}  "
                  f"L={cp[0]}/L={cp[1]}  {f['n_bootstrap_success']:>6}")
            if f.get('warning'):
                print(f"    warning: {f['warning']}")

    print_fits_table('MWPM crossings:', fits)
    if fits_bp is not None:
        print_fits_table('BP+matching crossings:', fits_bp)

    print('\nMaking plot ...')
    plot_separated(processed, fits,
                   out_path=PLOTS_DIR / 'threshold.png',
                   processed_bp=processed_bp, fits_bp=fits_bp, xmax_pct=0.7)
    print('\nDone.')


if __name__ == '__main__':
    main()