#!/usr/bin/env python3
"""Draw the four result figures of the paper. fig_laundering (a), fig_witness and fig_retention read stored result
files and check every plotted value against the stored estimates. fig_laundering (b) to (d) and fig_streams plot
the numbers of the paper's tables, which are held as constants below. No result is recomputed.

fig_laundering (figure*, \\textwidth), four panels:
    (a) Platform percentile of the laundering targets before the deletions and after exact retraining (U2), per
        network: cohort-weighted means from results/analysis/laundering_extra.json (<ds>.cohort_weighted.pct_before,
        pct_after_u2); dots are the individual targets (mean over the five seeds) from
        results/laundering/probe4_<ds>_s<seed>.json (pct_before, direct.ops.exact_retrain.pct_after), calibration
        targets excluded as in keeper/analysis/laundering_extra.py, coloured after the deletions by whether the
        target holds an endorsement (the stratum of the stratified draw). The dots' plain mean and their
        cohort-weighted mean are asserted equal to the stored values.
    (b) Gain of the targets under the five unlearning operators (mean and sd over seeds) and under indirect
        deletion with U2 (OPERATOR_GAIN, INDIRECT_GAIN).
    (c) Share of the full gain that is reached when only some of a target's two or three warnings are deleted,
        with 95% intervals (PARTIAL_SHARE). The networks are shifted sideways by at most 0.018 so that their
        intervals stay apart; hollow markers are the targets with two warnings.
    (d) Erasure requests that pass every request-level check (PASS_ALL).

fig_streams (figure*, \\textwidth), two blocks:
    (a) Fakes that one attack edge makes credible: the bound of Property 1, the best attack found with and without
        the cap, and transitive endorsement, which admits every fake (FAKES_BOUND, FAKES_BEST, FAKES_BEST_NO_CAP).
    (b) Release policies of the escrow engine (POLICIES): the durable share after one snapshot, the
        revoked share of the ratings from senders flagged later (hollow: the subset first warned later) and from
        other senders, and the revoked share of a committed rating under a late attack.

fig_witness (figure, \\columnwidth):
    Fake-witness attack on the SCEM escrow (E1b): k fake accounts warn about an honest rater one snapshot
    after that rater's warning. Mean revoked share of the honest warning caused by the attack (attack minus
    clean, results/analysis/e1.json e1b.<ds>[].total.mean) against k, per network, with lag 1, ring, no
    farming; without credibility weighting (M-B off, g = 0; g = 1, 2 asserted within 1e-5) and with it
    (M-B on) for g = 0, 1, 2 attack edges. Each cell is asserted to exist exactly once. Symmetric-log y-axis,
    linear below 0.1%.

fig_retention (figure*, \\textwidth):
    Retention on the confirmatory runs. (a) Direct laundering on ten fresh seeds: the per-seed mean gain under
    Plain Erasure and the control-adjusted gain under CPPR (results/analysis/cppr_confirm.json, H1 component
    b_cut: by_seed_denominator, by_seed_numerator). (b) Fakes that rate a target and delete themselves:
    control-adjusted whitewash gain without attack edges under a single pool and under CPPR (secondary.
    attack_minus_control.btpc, ten seeds) and with one attack edge under KEEPER's tiers and under transitive
    endorsement (H3b, each seed's gain in the cell that is best on the other seeds; five seeds). (c) Insider
    linkage: within-pool AUC of three attackers with 95% intervals (results/analysis/e6.json, cppr_btp). Bars are
    means over seeds, asserted equal to the stored estimates; dots are the seeds.

Every figure is drawn at its printed width (IEEEtran [10pt,journal,compsoc]: \\textwidth = 517.935 pt,
\\columnwidth = 252.945 pt, measured with pdflatex), so matplotlib font sizes are printed sizes. The
script asserts that every text is at least 7 pt, lies inside the canvas, and overlaps no other text.

Usage (repository root, CPU only): python -m keeper.paperkit.make_figures
"""
import itertools
import json
import os
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PF = os.path.join(ROOT, "paper")
FIG_DIR = os.path.join(PF, "figures")

# ---------------------------------------------------------------------------------------------
# Printed sizes (inches) and style
# ---------------------------------------------------------------------------------------------
PT_PER_IN = 72.27
TEXTWIDTH_IN = 517.935 / PT_PER_IN       # 7.167 in
COLUMNWIDTH_IN = 252.945 / PT_PER_IN     # 3.500 in
MIN_PT = 7.0

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 7.5,
    'axes.labelsize': 7.5,
    'axes.titlesize': 8.0,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 7.0,
    'ytick.labelsize': 7.0,
    'legend.fontsize': 7.0,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.35,
    'legend.handlelength': 1.6,
    'legend.handletextpad': 0.45,
    'legend.columnspacing': 1.0,
    'legend.borderpad': 0.3,
    'legend.labelspacing': 0.25,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

C_NAVY, C_BLUE, C_TEAL, C_CRIMSON = "#1B365D", "#1F4E79", "#0B6E4F", "#9E1B1B"
C_AMBER, C_SLATE, C_PURPLE, C_GRAY = "#8C4A08", "#2C3E50", "#5B2C6F", "#7F8C8D"
C_EDGE = "#1C2833"
C_PALE = "#C5CBCC"     # C_GRAY at 45% on white: a fill that carries dark text
MINUS, AT_LEAST = "\u2212", "\u2265"    # minus sign and greater-than-or-equal sign in labels

DS3 = ['bitcoinalpha', 'bitcoinotc', 'wiki-rfa']
DS4 = DS3 + ['epinions-dated']
DS_NAME = {'bitcoinalpha': 'Bitcoin-Alpha', 'bitcoinotc': 'Bitcoin-OTC', 'wiki-rfa': 'Wiki-RfA',
           'epinions-dated': 'Epinions-Dated'}


def load(path):
    with open(path) as fh:
        return json.load(fh)


laund_extra = load(os.path.join(ROOT, "results", "analysis", "laundering_extra.json"))
e1 = load(os.path.join(ROOT, "results", "analysis", "e1.json"))

# ---------------------------------------------------------------------------------------------
# Numbers of fig_laundering (b) to (d) and of fig_streams: copied from the paper's tables, not recomputed
# ---------------------------------------------------------------------------------------------
NET_SHORT = {'bitcoinalpha': 'Alpha', 'bitcoinotc': 'OTC', 'wiki-rfa': 'Wiki-RfA', 'epinions-dated': 'Epinions'}
# One colour, marker and line style per network wherever a panel tells the networks apart.
NET_STYLE = {'bitcoinalpha': (C_BLUE, 'o', '-'), 'bitcoinotc': (C_AMBER, 's', '--'),
             'wiki-rfa': (C_PURPLE, '^', '-.'), 'epinions-dated': (C_TEAL, 'D', ':')}

# Former supplement table tab_strata_cutoff, column Delta r(T): gain under U1 to U5 as (mean, sd over seeds).
OPERATOR_GAIN = {
    'bitcoinalpha': ((0.3490, 0.0444), (0.3516, 0.0465), (0.3544, 0.0462), (0.3498, 0.0448), (0.3500, 0.0450)),
    'bitcoinotc': ((0.3176, 0.0366), (0.3169, 0.0371), (0.3239, 0.0378), (0.3168, 0.0360), (0.3157, 0.0352)),
    'wiki-rfa': ((0.5792, 0.0432), (0.5808, 0.0425), (0.6063, 0.0437), (0.5801, 0.0442), (0.5804, 0.0343)),
    'epinions-dated': ((0.3483, 0.0308), (0.3472, 0.0349), (0.3691, 0.0322), None, (0.3524, 0.0322)),   # U4 not run
}
# Former "at a glance" table, row of the indirect deletion (U2): gain of the targets.
INDIRECT_GAIN = {'bitcoinalpha': 0.018, 'bitcoinotc': 0.002, 'wiki-rfa': -0.002, 'epinions-dated': 0.001}
# Former supplement table s_partial_deletion (share table): share of the full gain (%) as (value, low, high), 95%.
PARTIAL_SHARE = {
    'bitcoinalpha': {'1 of 2': (21.875, 14.200, 30.297), '1 of 3': (18.462, 12.636, 24.287),
                     '2 of 3': (39.158, 29.906, 48.737)},
    'bitcoinotc': {'1 of 2': (36.700, 27.437, 46.429), '1 of 3': (20.398, 15.811, 25.096),
                   '2 of 3': (37.257, 30.897, 43.118)},
    'wiki-rfa': {'1 of 2': (18.582, 11.747, 25.003), '1 of 3': (9.786, 7.226, 12.225),
                 '2 of 3': (24.123, 18.168, 30.094)},
    'epinions-dated': {'1 of 2': (27.486, 19.506, 37.259), '1 of 3': (22.763, 15.319, 32.061),
                       '2 of 3': (45.770, 31.713, 58.706)},
}
# Former supplement table tab_placement, column Pass-All: requests passing every request-level check (%), DS4 order.
PASS_ALL = {'laundering': (75.9, 82.4, 84.9, 100.0), 'benign': (95.3, 94.7, 92.0, 96.0)}
# Former Table IV (credibility layer): fakes made credible by one attack edge, DS3 order.
FAKES_BOUND = (24, 26, 28)             # bound of Property 1
FAKES_BEST = (21, 23, 25)              # best attack found, with the cap
FAKES_BEST_NO_CAP = (350, 750, 200)    # best attack found without the cap: lower bounds ("at least")
#                                        transitive endorsement admits every fake: no number


class Policy(NamedTuple):
    """One row of the former Table V; the three values of a tuple follow DS3."""
    name: str
    durable: str                 # durable after one snapshot (%), as printed in the table
    flagged: tuple               # revoked from senders flagged later (%)
    first_warned: tuple          # the same for the subset first warned later
    others: tuple                # revoked from other senders (%)
    late: tuple | None           # late attack on a committed rating (% revoked); None: nothing is revocable


# Former Table V (release policies at the same mean delay).
POLICIES = (
    Policy('On arrival (no escrow)', '100', (0, 0, 0), (0, 0, 0), (0, 0, 0), None),
    Policy('One stage', '22.1', (23.7, 23.3, 37.6), (16.6, 12.8, 29.3), (0.7, 1.4, 8.4), (0, 0, 0)),
    Policy('Two stages (SCEM)', '9.0', (25.4, 24.7, 41.3), (16.8, 11.5, 31.9), (0.7, 1.6, 8.8), (0, 0, 0)),
    Policy('Three stages', '4.1', (26.0, 24.9, 42.4), (16.6, 10.4, 32.4), (0.6, 1.6, 8.8), (0, 0, 0)),
    Policy('Five stages', '0.9', (26.4, 24.9, 42.6), (16.2, 9.1, 31.6), (0.6, 1.7, 8.5), (0, 0, 0)),
    Policy('Quarantine, then commit', '0', (26.0, 23.9, 41.9), (15.7, 6.5, 30.1), (0.6, 1.6, 8.0), (0, 0, 0)),
    Policy('Never commit', '0', (73.4, 75.8, 81.9), (69.7, 68.4, 79.5), (4.3, 4.2, 20.5), (34, 34, 55)),
)
SCEM_POLICY = 'Two stages (SCEM)'      # the policy of the paper: bold label, shaded row

VERIFICATION = []   # (check, result) lines printed at the end


def same(x, ref, nd):
    """True if x and ref agree when both are printed with nd decimals."""
    return f"{x:.{nd}f}" == f"{ref:.{nd}f}"


def require(ok, msg):
    if not ok:
        raise SystemExit(f"STOP: {msg}")
    VERIFICATION.append(msg)


# ---------------------------------------------------------------------------------------------
# Layout checks: font size, canvas bounds, text overlaps
# ---------------------------------------------------------------------------------------------
def drawn_text_elements(fig):
    """(owner, bbox, label) for every drawn text and legend frame in the figure."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    out = []

    def add(owner, artist, label=None):
        if artist is None or not artist.get_visible():
            return
        if isinstance(artist, mtext.Text) and not artist.get_text().strip():
            return
        out.append((owner, artist.get_window_extent(rend), label or getattr(artist, 'get_text', lambda: '')()))

    for ax in fig.axes:
        for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label):
            add(id(ax), t)
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis._update_ticks():        # the ticks matplotlib actually draws (inside the view)
                for t in (tick.label1, tick.label2):
                    add(id(ax), t)
        for t in ax.texts:
            add(id(t), t)
        leg = ax.get_legend()
        if leg is not None:
            add(id(leg), leg.get_frame(), 'legend frame')
            for t in leg.get_texts():
                add(id(leg), t)
    for t in fig.texts:
        add(id(t), t)
    for leg in fig.legends:
        add(id(leg), leg.get_frame(), 'legend frame')
        for t in leg.get_texts():
            add(id(leg), t)
    return out


def check_layout(fig, stem):
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(mtext.Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-9]
    require(not small, f"{stem}: every text >= {MIN_PT} pt (smallest "
            f"{min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip()):.1f} pt)"
            if not small else f"{stem}: texts below {MIN_PT} pt: {small}")
    elems = drawn_text_elements(fig)
    W, H = fig.bbox.width, fig.bbox.height
    tol = 0.5
    outside = [lab for _, bb, lab in elems
               if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > W + tol or bb.y1 > H + tol]
    require(not outside, f"{stem}: all {len(elems)} text elements inside the canvas"
            if not outside else f"{stem}: clipped texts: {outside}")
    clashes = []
    for (o1, b1, l1), (o2, b2, l2) in itertools.combinations(elems, 2):
        if o1 == o2 and ('legend frame' in (l1, l2)):
            continue
        w = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        h = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if w > 0.5 and h > 0.5:
            clashes.append((l1, l2))
    require(not clashes, f"{stem}: no overlapping text elements"
            if not clashes else f"{stem}: overlapping texts: {clashes}")


def save(fig, stem, width_in):
    w, h = fig.get_size_inches()
    require(abs(w - width_in) < 1e-6, f"{stem}: canvas width {w:.3f} in = printed width {width_in:.3f} in "
                                      f"(height {h:.2f} in, scale 1.00)")
    try:
        check_layout(fig, stem)
    except SystemExit:
        debug_dir = os.environ.get("FIG_DEBUG_DIR")   # optional: keep the failing layout for inspection
        if debug_dir:
            fig.savefig(os.path.join(debug_dir, f"{stem}_FAILED.png"), dpi=200)
        raise
    path = os.path.join(FIG_DIR, f"{stem}.pdf")
    fig.savefig(path)                       # no tight bbox: the page is exactly the printed size
    fig.savefig(path[:-4] + ".png", dpi=300)
    plt.close(fig)
    print("saved", path)


def style_axes(ax, grid_axis='y'):
    ax.grid(True, axis=grid_axis, linestyle=':')
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------------------------
# Hand-placed panels (fig_laundering, fig_streams): frames in inches, all titles on one line at the top of the
# canvas, each legend in the rows under its title
# ---------------------------------------------------------------------------------------------
HEAD_TOP_IN = 0.03       # canvas top to the top of the titles
HEAD_TITLE_IN = 0.135    # one title line
HEAD_ROW_IN = 0.115      # one legend row (a row with a subscript is taller than plain text)


def inch_axes(fig: plt.Figure, left: float, bottom: float, width: float, height: float, **kwargs) -> plt.Axes:
    """Axes whose frame is given in inches from the lower left corner of the canvas."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h], **kwargs)


def by_rows(items: list, ncol: int) -> list:
    """Reorder a list given row by row for a legend, which matplotlib fills column by column."""
    return [items[i] for c in range(ncol) for i in range(c, len(items), ncol)]


def panel_head(fig: plt.Figure, ax: plt.Axes, x: float, title: str, handles: list, labels: list, ncol: int,
               drop: float = 0.0, **legend_kw) -> None:
    """Bold title at the top of the canvas, flush left at x inches, and the legend of ax under it (`drop` inches
    lower than the first legend row)."""
    top = fig.get_size_inches()[1] - HEAD_TOP_IN
    fig.text(x, top, title, transform=fig.dpi_scale_trans, ha='left', va='top',
             fontsize=plt.rcParams['axes.titlesize'], fontweight=plt.rcParams['axes.titleweight'])
    ax.legend(by_rows(handles, ncol), by_rows(labels, ncol), ncol=ncol, loc='upper left',
              bbox_to_anchor=(x, top - HEAD_TITLE_IN - drop), bbox_transform=fig.dpi_scale_trans, frameon=False,
              borderpad=0.0, borderaxespad=0.0, **legend_kw)


# ---------------------------------------------------------------------------------------------
# fig_laundering: what a laundering attack gains (rank, operators, partial deletion, request filters)
# ---------------------------------------------------------------------------------------------
BOTTOM_PCT = 10.0     # the shaded band: lowest 10% of the platform, the level of percentile.escape_bottom10
LAUND_HEIGHT_IN = 1.95
TICK_ROW_PT = 12.0      # pad of an x tick label that sits one row below its neighbours
OPERATOR_COLOURS = ("#D0DAE8", "#A3B1C5", "#7688A3", "#485F80", C_NAVY)     # U1 to U5, pale to dark


def percentile_targets(ds):
    """Per-target platform percentiles (%) before the deletions and after U2, each the mean over the five
    seeds, from the probe4 records with the calibration targets excluded (keeper/analysis/laundering_extra.py)."""
    lau = os.path.join(ROOT, "results", "laundering")
    cal = set(load(os.path.join(lau, f"calibration_{ds}_s0.json"))["calibration_targets"])
    before, after, stratum = {}, {}, {}
    for s in range(laund_extra[ds]['n_seeds']):
        for t in load(os.path.join(lau, f"probe4_{ds}_s{s}.json"))["targets"]:
            if t["target"] not in cal:
                before.setdefault(t["target"], []).append(t["pct_before"])
                after.setdefault(t["target"], []).append(t["direct"]["ops"]["exact_retrain"]["pct_after"])
                stratum[t["target"]] = t["stratum"]
    n_seeds = laund_extra[ds]['n_seeds']
    require(all(len(v) == n_seeds for v in before.values()),
            f"fig_laundering (a): {DS_NAME[ds]} {len(before)} targets, each evaluated in all {n_seeds} seeds")
    keys = sorted(before)
    b = 100.0 * np.array([np.mean(before[k]) for k in keys])
    a = 100.0 * np.array([np.mean(after[k]) for k in keys])
    p = laund_extra[ds]['percentile']
    require(same(b.mean(), 100.0 * p['before_mean'], 6) and same(a.mean(), 100.0 * p['after_mean_u2'], 6),
            f"fig_laundering (a): {DS_NAME[ds]} mean of the per-target dots {b.mean():.4f}% / {a.mean():.4f}% = "
            f"laundering_extra.json percentile.before_mean / after_mean_u2")
    pos = np.array([stratum[k] == "pos" for k in keys])        # targets with at least one endorsement
    cw = laund_extra[ds]['cohort_weighted']
    w = cw['cohort_share_pos']
    wb, wa = (w * x[pos].mean() + (1.0 - w) * x[~pos].mean() for x in (b, a))
    require(same(wb, 100.0 * cw['pct_before'], 6) and same(wa, 100.0 * cw['pct_after_u2'], 6),
            f"fig_laundering (a): {DS_NAME[ds]} cohort-weighted mean of the dots {wb:.4f}% / {wa:.4f}% = "
            f"laundering_extra.json cohort_weighted.pct_before / pct_after_u2 (endorsed share {w:.3f})")
    ev_b = np.concatenate([before[k] for k in keys])
    ev_a = np.concatenate([after[k] for k in keys])
    low = ev_b < BOTTOM_PCT / 100.0
    print(f"  {DS_NAME[ds]}: {len(keys)} targets x {n_seeds} seeds; U2 leaves the lowest {BOTTOM_PCT:.0f}% for "
          f"{int((ev_a[low] >= BOTTOM_PCT / 100.0).sum())} of the {int(low.sum())} evaluations that start in it")
    return b, a, pos


def panel_percentile(fig: plt.Figure, ax: plt.Axes, x: float) -> None:
    """(a) Platform percentile of the laundering targets before the deletions and after exact retraining (U2)."""
    print("fig_laundering (a): plotted values (percent; source results/analysis/laundering_extra.json)")
    off, jit = 0.27, 0.08                       # dot strips above (before) and below (after) each arrow
    rng = np.random.default_rng(0)              # vertical jitter of the dots only; it changes no value
    ax.axvspan(0, BOTTOM_PCT, color=C_CRIMSON, alpha=0.10, lw=0, zorder=0)
    ax.text(0.8, -0.78, f"lowest {BOTTOM_PCT:.0f}% of the platform", ha='left', va='center', fontsize=7.0,
            color=C_CRIMSON)
    for i, ds in enumerate(DS4):
        b, a, pos = percentile_targets(ds)
        p = laund_extra[ds]['cohort_weighted']
        mb, ma = 100.0 * p['pct_before'], 100.0 * p['pct_after_u2']     # cohort-weighted means
        ax.scatter(b, i - off + rng.uniform(-jit, jit, len(b)), s=3.5, color=C_GRAY, alpha=0.55, lw=0, zorder=2)
        ya = i + off + rng.uniform(-jit, jit, len(a))
        ax.scatter(a[~pos], ya[~pos], s=3.5, color=C_CRIMSON, alpha=0.55, lw=0, zorder=2)
        ax.scatter(a[pos], ya[pos], s=3.5, color=C_BLUE, alpha=0.55, lw=0, zorder=2)
        ax.annotate('', xy=(ma, i), xytext=(mb, i), zorder=3,
                    arrowprops=dict(arrowstyle='-|>', color=C_EDGE, lw=0.9, mutation_scale=7, shrinkA=2.5,
                                    shrinkB=2.5))
        ax.plot(mb, i, 'o', ms=4.0, mfc='white', mec=C_EDGE, mew=0.9, zorder=4)
        ax.plot(ma, i, 'o', ms=4.0, color=C_CRIMSON, mec=C_CRIMSON, zorder=4)
        ax.annotate(f'{mb:.1f} $\\to$ {ma:.1f}', (ma, i), xytext=(5, 0), textcoords='offset points', ha='left',
                    va='center', fontsize=7.0, color=C_EDGE, zorder=5)
        print(f"  {DS_NAME[ds]}: cohort-weighted before {mb:.2f} (ci95 {100 * p['ci95']['pct_before'][0]:.2f} to "
              f"{100 * p['ci95']['pct_before'][1]:.2f}), after U2 {ma:.2f} (ci95 "
              f"{100 * p['ci95']['pct_after_u2'][0]:.2f} to {100 * p['ci95']['pct_after_u2'][1]:.2f}); "
              f"per-target before {b.min():.1f} to {b.max():.1f}, after {a.min():.1f} to {a.max():.1f}; after U2 "
              f"without an endorsement {a[~pos].mean():.1f} ({int((~pos).sum())} targets), with one "
              f"{a[pos].mean():.1f} ({int(pos.sum())} targets)")
    ax.set_yticks(np.arange(len(DS4)))
    ax.set_yticklabels([DS_NAME[ds] for ds in DS4])
    ax.set_ylim(len(DS4) - 0.5, -1.0)
    ax.set_xlim(-1.5, 101.5)                    # padded so that dots at 0% and 100% are not clipped
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel(r'Platform percentile of the target $T$ (%)', labelpad=2)
    ax.tick_params(axis='y', length=0)
    style_axes(ax, grid_axis='x')
    dot = dict(marker='o', ls='none', ms=2.2, alpha=0.7)
    handles = [(Line2D([], [], color=C_GRAY, **dot), Line2D([], [], marker='o', ls='none', ms=4.0, mfc='white',
                                                            mec=C_EDGE, mew=0.9)),
               Line2D([], [], marker='o', ls='none', ms=4.0, color=C_CRIMSON),
               Line2D([], [], color=C_CRIMSON, **dot), Line2D([], [], color=C_BLUE, **dot)]
    panel_head(fig, ax, x, '(a) Rank before and after', handles,
               ['before the deletions', r'after $\mathcal{U}_2$ (weighted mean)', 'after: no endorsement',
                'after: with an endorsement'], ncol=2, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
               handlelength=1.5, handletextpad=0.4, columnspacing=0.8)


def panel_operators(fig: plt.Figure, ax: plt.Axes, x: float) -> None:
    """(b) Gain of the targets under the five unlearning operators and under indirect deletion."""
    w, gap = 0.125, 0.05                        # five adjoining operator bars, a gap, the indirect bar
    xs = np.arange(len(DS4), dtype=float) - (6 * w + gap) / 2 + w / 2      # centre of the first bar per network
    handles = []
    for j, col in enumerate(OPERATOR_COLOURS):
        run = [(xi + j * w, *OPERATOR_GAIN[ds][j]) for xi, ds in zip(xs, DS4) if OPERATOR_GAIN[ds][j] is not None]
        xj, mean, sd = zip(*run)
        handles.append(ax.bar(xj, mean, w, yerr=sd, color=col, edgecolor=C_EDGE, linewidth=0.4, zorder=2,
                              error_kw=dict(elinewidth=0.6, ecolor=C_EDGE, capsize=1.0, capthick=0.6)))
    x_ind = xs + 5 * w + gap
    indirect = [INDIRECT_GAIN[ds] for ds in DS4]
    handles.append(ax.bar(x_ind, indirect, w, color='white', edgecolor=C_GRAY, hatch='//////', hatch_linewidth=0.5,
                          linewidth=0.5, zorder=2))
    for xi, v in zip(x_ind, indirect):          # the indirect bars are too short to see: print their values
        ax.annotate(f'{v:.3f}'.replace('-', MINUS), (xi, max(v, 0.0)), xytext=(1, 2), textcoords='offset points',
                    rotation=90, ha='center', va='bottom', fontsize=7.0, color=C_SLATE)
    for xi, ds in zip(xs, DS4):                 # an operator that was not run leaves its slot empty
        for j, cell in enumerate(OPERATOR_GAIN[ds]):
            if cell is None:
                reach = max(m + s for m, s in filter(None, OPERATOR_GAIN[ds]))
                ax.annotate('n/a', (xi + j * w, 0.0), xytext=(xi + j * w, reach + 0.03), ha='center', va='bottom',
                            fontsize=7.0, color=C_SLATE,
                            arrowprops=dict(arrowstyle='-', lw=0.5, ls=':', color=C_SLATE, shrinkA=0, shrinkB=0))
    ax.set_xticks(np.arange(len(DS4)))
    ax.set_xticklabels([NET_SHORT[ds] for ds in DS4])
    for tick in ax.xaxis.get_major_ticks()[1::2]:      # every second label one row lower: side by side they collide
        tick.set_pad(TICK_ROW_PT)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.5, len(DS4) - 0.5)
    ax.set_ylim(0.0, 0.7)
    ax.set_yticks(np.arange(0.0, 0.61, 0.2))
    ax.set_ylabel(r'Reputation gain $\Delta r(T)$', labelpad=2)
    style_axes(ax)
    panel_head(fig, ax, x, '(b) Gain by unlearning operator', handles,
               [rf'$\mathcal{{U}}_{j}$' for j in range(1, 6)] + [r'indirect deletion ($\mathcal{U}_2$)'], ncol=3,
               handlelength=1.0, handletextpad=0.35, columnspacing=0.7)


def panel_partial(fig: plt.Figure, ax: plt.Axes, x: float) -> None:
    """(c) Share of the full gain that is reached when only some of a target's warnings are deleted."""
    ax.plot([0, 1], [0, 100], color=C_GRAY, ls='--', lw=0.8, zorder=1)
    ax.text(0.59, 62.5, 'proportional', rotation=np.degrees(np.arctan2(100, 1)), rotation_mode='anchor',
            transform_rotates_text=True, ha='center', va='bottom', fontsize=7.0, color=C_SLATE)
    handles = []
    for shift, ds in zip(np.linspace(-0.018, 0.018, len(DS4)), DS4):       # sideways, so the intervals stay apart
        col, marker, ls = NET_STYLE[ds]
        share = PARTIAL_SHARE[ds]
        for key, xk in (('1 of 3', 1 / 3), ('1 of 2', 1 / 2), ('2 of 3', 2 / 3)):
            v, low, high = share[key]
            ax.errorbar(xk + shift, v, yerr=[[v - low], [high - v]], fmt='none', ecolor=col, elinewidth=0.6,
                        capsize=1.2, capthick=0.6, zorder=2)
        handles += ax.plot(np.array([1 / 3, 2 / 3, 1.0]) + shift, [share['1 of 3'][0], share['2 of 3'][0], 100.0],
                           color=col, ls=ls, lw=1.0, marker=marker, ms=3.2, zorder=3)
        ax.plot(1 / 2 + shift, share['1 of 2'][0], ls='none', marker=marker, ms=3.2, mfc='white', mec=col, mew=0.8,
                zorder=4)                       # targets with two warnings: hollow and not on the line
    ax.set_xlim(0.295, 1.04)
    ax.set_xticks([1 / 3, 1 / 2, 2 / 3, 1.0])
    ax.set_xticklabels(['1 of 3', '1 of 2', '2 of 3', 'all'])
    ax.set_xlabel('Deleted warnings of the target', labelpad=2)
    ax.set_ylim(0.0, 104.0)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.set_ylabel('Share of the full gain (%)', labelpad=2)
    style_axes(ax)
    panel_head(fig, ax, x, '(c) Gain under partial deletion', handles, [NET_SHORT[ds] for ds in DS4], ncol=2,
               handlelength=2.2)


def panel_pass_all(fig: plt.Figure, ax: plt.Axes, x: float) -> None:
    """(d) Erasure requests that pass every request-level check, laundering against benign."""
    rows = np.arange(len(DS4), dtype=float)
    h = 0.38
    handles = []
    for k, (key, face, ink) in enumerate((('laundering', C_CRIMSON, 'white'), ('benign', C_PALE, C_EDGE))):
        y = rows + (k - 0.5) * h
        handles.append(ax.barh(y, PASS_ALL[key], h, color=face, edgecolor=C_EDGE, linewidth=0.4, zorder=2))
        for yi, v in zip(y, PASS_ALL[key]):
            ax.annotate(f'{v:.1f}', (v, yi), xytext=(-2.5, 0), textcoords='offset points', ha='right', va='center',
                        fontsize=7.0, color=ink, zorder=3)
    ax.set_yticks(rows)
    ax.set_yticklabels([NET_SHORT[ds] for ds in DS4])
    ax.set_ylim(len(DS4) - 0.5, -0.5)           # first network on top, as in panel (a)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks([0, 50, 100])
    frame, canvas = ax.get_position(), fig.get_size_inches()[0]
    centre = (x + canvas - 0.02) / 2            # the label sits under the middle of the panel, not of the axes
    ax.set_xlabel('Passing every check (%)', labelpad=2, x=(centre / canvas - frame.x0) / frame.width)
    style_axes(ax, grid_axis='x')
    panel_head(fig, ax, x, '(d) Requests that pass\nevery filter', handles, list(PASS_ALL), ncol=2, drop=HEAD_ROW_IN,
               handlelength=1.0, handletextpad=0.35, columnspacing=0.8)


def fig_laundering() -> None:
    require(all(len(gains) == len(OPERATOR_COLOURS) for gains in OPERATOR_GAIN.values()),
            f"fig_laundering (b): {len(OPERATOR_COLOURS)} operator slots in each of the {len(OPERATOR_GAIN)} networks")
    require(all(low <= v <= high for share in PARTIAL_SHARE.values() for v, low, high in share.values()),
            "fig_laundering (c): every share lies inside its 95% interval")
    fig = plt.figure(figsize=(TEXTWIDTH_IN, LAUND_HEIGHT_IN))
    bottom = 0.35
    height = LAUND_HEIGHT_IN - HEAD_TOP_IN - HEAD_TITLE_IN - 2 * HEAD_ROW_IN - 0.03 - bottom
    # (panel, left edge of the panel, left edge of its axes, width of its axes), in inches
    for panel, x, left, width in ((panel_percentile, 0.02, 0.69, 1.71), (panel_operators, 2.49, 2.84, 1.32),
                                  (panel_partial, 4.25, 4.60, 1.28), (panel_pass_all, 5.955, 6.405, 0.655)):
        panel(fig, inch_axes(fig, left, bottom, width, height), x)
    save(fig, "fig_laundering", TEXTWIDTH_IN)


# ---------------------------------------------------------------------------------------------
# fig_streams: what one attack edge buys in the credibility layer, and the release policies of the escrow
# ---------------------------------------------------------------------------------------------
STREAMS_HEIGHT_IN = 1.95
STREAMS_HEAD_IN = 0.465       # under the title line: four legend rows in (a); a legend row and column titles in (b)
FAKES_AXIS_TOP = 3000.0       # the log axis of panel (a) ends here; the bar of transitive endorsement runs past it
POLICY_LABELS_IN = 1.03       # width of the policy names to the left of the first column
ZERO_PAD_IN = 4.0 / 72        # a dot column starts this far left of zero, so that a marker at zero clears the spine
# (column title, end of the x axis, ticks) of the four columns of panel (b)
POLICY_COLUMNS = (('Durable after\n1 snapshot (%)', 100, (0, 50, 100)),
                  ('Revoked from senders\nflagged later (%)', 90, (0, 30, 60, 90)),
                  ('Revoked from\nother senders (%)', 22, (0, 10, 20)),
                  ('Late attack on a committed\nrating (% revoked)', 60, (0, 20, 40, 60)))


def panel_fakes(fig: plt.Figure, ax: plt.Axes, x: float) -> None:
    """(a) Fakes that one attack edge makes credible, on a log axis; the label of each bar is its value."""
    xs = np.arange(len(DS3), dtype=float)
    w = 0.21
    # (legend label, values, face, edge, colour and format of the value inside the bar)
    series = (('bound of Property 1', FAKES_BOUND, 'white', C_SLATE, C_EDGE, '{}'),
              ('best attack found (with the cap)', FAKES_BEST, C_TEAL, C_EDGE, 'white', '{}'),
              ('best attack found, without the cap', FAKES_BEST_NO_CAP, C_AMBER, C_EDGE, 'white', AT_LEAST + '{}'),
              ('transitive endorsement', (10 * FAKES_AXIS_TOP,) * len(DS3), C_CRIMSON, C_EDGE, 'white', 'all'))
    handles = []
    for j, (_, values, face, edge, ink, fmt) in enumerate(series):
        xj = xs + (j - 1.5) * w
        handles.append(ax.bar(xj, np.array(values) - 1.0, w, bottom=1.0, color=face, edgecolor=edge, linewidth=0.6,
                              zorder=2))
        for xi, v in zip(xj, values):           # the value at the top of the bar, or under the break of a cut bar
            ax.annotate(fmt.format(v), (xi, min(v, 0.37 * FAKES_AXIS_TOP)), xytext=(0, -2.5),
                        textcoords='offset points', rotation=90, ha='center', va='top', fontsize=7.0, color=ink,
                        zorder=4)
    for xi, v in zip(xs + 0.5 * w, FAKES_BEST_NO_CAP):      # a lower bound: an arrow on top of the bar
        ax.annotate('', (xi, v), xytext=(0, 8), textcoords='offset points',
                    arrowprops=dict(arrowstyle='<|-', color=C_AMBER, lw=0.9, mutation_scale=6, shrinkA=0, shrinkB=0))
    for xi in xs + 1.5 * w:                                 # no bound: the bar leaves the axis, with a break on it
        ax.plot([xi - 0.62 * w, xi + 0.62 * w], [0.45 * FAKES_AXIS_TOP, 0.65 * FAKES_AXIS_TOP], color='white',
                lw=1.6, solid_capstyle='butt', zorder=3)
    ax.set_yscale('log')
    ax.set_ylim(1.0, FAKES_AXIS_TOP)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(['1', '10', '100', '1000'])
    ax.set_ylabel('Fakes (log scale)', labelpad=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([NET_SHORT[ds] for ds in DS3])
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.55, len(DS3) - 0.45)
    style_axes(ax)
    panel_head(fig, ax, x, '(a) Fakes made credible by one attack edge', handles, [s[0] for s in series], ncol=1,
               handlelength=1.3, handletextpad=0.4, labelspacing=0.15)


def panel_policies(fig: plt.Figure, axes: list, x: float) -> None:
    """(b) Release policies of the escrow engine: one row per policy, one column per measure."""
    first = axes[0]
    rows = np.arange(len(POLICIES), dtype=float)
    first.barh(rows, [float(p.durable) for p in POLICIES], 0.62, color=C_SLATE, edgecolor=C_EDGE, linewidth=0.4,
               zorder=2)
    for yi, p in zip(rows, POLICIES):           # the value at the end of the bar; inside it when the bar is long
        inside = float(p.durable) > 80.0
        first.annotate(p.durable, (float(p.durable), yi), xytext=(-2.5 if inside else 2.5, 0),
                       textcoords='offset points', ha='right' if inside else 'left', va='center', fontsize=7.0,
                       color='white' if inside else C_EDGE, zorder=3)
    handles = []
    for k, (dy, ds) in enumerate(zip((-0.3, 0.0, 0.3), DS3)):       # the networks sit one above the other in a row
        col, marker, _ = NET_STYLE[ds]
        full = dict(ls='none', marker=marker, ms=3.2, color=col, zorder=4)
        ring = dict(ls='none', marker=marker, ms=3.2, mfc='white', mec=col, mew=0.8, zorder=3)
        handles += axes[1].plot([p.flagged[k] for p in POLICIES], rows + dy, **full)
        axes[1].plot([p.first_warned[k] for p in POLICIES], rows + dy, **ring)
        axes[1].hlines(rows + dy, [p.first_warned[k] for p in POLICIES], [p.flagged[k] for p in POLICIES],
                       color=col, lw=0.6, zorder=2)     # the subset and the whole of one network belong together
        axes[2].plot([p.others[k] for p in POLICIES], rows + dy, **full)
        axes[3].plot(*zip(*[(p.late[k], yi + dy) for yi, p in zip(rows, POLICIES) if p.late is not None]), **full)
    for yi, p in zip(rows, POLICIES):
        if p.late is None:
            axes[3].text(1.5, yi, 'n/a', ha='left', va='center', fontsize=7.0, color=C_SLATE)
    canvas = fig.get_size_inches()[0]
    for ax, (title, end, ticks) in zip(axes, POLICY_COLUMNS):
        width = ax.get_position().width * canvas
        ax.set_title(title, loc='left', pad=3, fontsize=7.5, fontweight='normal')
        ax.set_xlim(0.0 if ax is first else -end * ZERO_PAD_IN / (width - ZERO_PAD_IN), end)
        ax.set_xticks(ticks)
        ax.tick_params(axis='y', length=0, labelleft=ax is first)
        ax.patch.set_visible(False)             # the shaded row of the first column shows through
        style_axes(ax, grid_axis='both')
    first.set_ylim(len(POLICIES) - 0.42, -0.58)        # first policy on top
    first.set_yticks(rows)
    first.set_yticklabels([p.name for p in POLICIES])
    scem = [p.name for p in POLICIES].index(SCEM_POLICY)
    first.get_yticklabels()[scem].set_fontweight('bold')
    frame, last = first.get_position(), axes[-1].get_position()
    first.axhspan(scem - 0.5, scem + 0.5, xmin=-POLICY_LABELS_IN / (frame.width * canvas),
                  xmax=(last.x1 - frame.x0) / frame.width, color=C_TEAL, alpha=0.10, lw=0, zorder=0, clip_on=False)
    neutral = [Line2D([], [], ls='none', marker='o', ms=3.2, color=C_GRAY),
               Line2D([], [], ls='none', marker='o', ms=3.2, mfc='white', mec=C_GRAY, mew=0.8)]
    panel_head(fig, first, x, '(b) Release policies of the escrow engine', handles + neutral,
               [NET_SHORT[ds] for ds in DS3] + ['flagged later', 'first warned later'], ncol=5, drop=0.05,
               handlelength=1.0, handletextpad=0.3, columnspacing=1.3)


def fig_streams() -> None:
    fig = plt.figure(figsize=(TEXTWIDTH_IN, STREAMS_HEIGHT_IN))
    bottom = 0.21
    height = STREAMS_HEIGHT_IN - HEAD_TOP_IN - HEAD_TITLE_IN - STREAMS_HEAD_IN - bottom
    panel_fakes(fig, inch_axes(fig, 0.44, bottom, 1.42, height), 0.02)
    first = inch_axes(fig, 3.01, bottom, 0.70, height)
    panel_policies(fig, [first] + [inch_axes(fig, left, bottom, width, height, sharey=first)
                                   for left, width in ((3.84, 1.06), (5.03, 0.795), (5.955, 1.13))], 3.01)
    save(fig, "fig_streams", TEXTWIDTH_IN)


# ---------------------------------------------------------------------------------------------
# fig_witness: fake-witness attack on the escrow (E1b), with and without credibility weighting
# ---------------------------------------------------------------------------------------------
WITNESS_K = [1, 5, 10, 50]
# (label, M-B, g, colour, line style, marker); without weighting g changes the mean by < 1e-5 (checked), so g = 0.
WITNESS_LINES = [('no credibility weighting', 'off', 0, C_CRIMSON, '-', 'o'),
                 ('credibility weighting, no attack edge', 'on', 0, C_TEAL, '-.', 's'),
                 ('credibility weighting, 1 attack edge', 'on', 1, C_BLUE, '--', '^'),
                 ('credibility weighting, 2 attack edges', 'on', 2, C_PURPLE, ':', 'D')]
WITNESS_LINTHRESH = 0.1   # percent: the symmetric-log y-axis is linear on [-0.1, 0.1] and logarithmic above


def witness_cells(ds):
    """{(mb, g, k): cell} of E1b with lag 1, ring, no farming; every plotted cell must exist exactly once."""
    cells = {}
    for c in e1['e1b'][ds]:
        if c['lag'] == 1 and c['ring'] is True and c['farming'] is False:
            cells.setdefault((c['mb'], c['g'], c['k']), []).append(c)
    want = [(mb, g, k) for mb in ('off', 'on') for g in (0, 1, 2) for k in WITNESS_K]
    bad = [key for key in want if len(cells.get(key, [])) != 1]
    require(not bad, f"fig_witness: {DS_NAME[ds]} each of the {len(want)} (M-B, g, k) cells (lag 1, ring, no "
                     f"farming) exists exactly once in e1b" if not bad else
            f"fig_witness: {DS_NAME[ds]} cells missing or duplicated: {bad}")
    cells = {key: cells[key][0] for key in want}
    # Without weighting the attack edges barely matter (differences in the 7th decimal): plotting g = 0 is exact
    # to the drawn precision.
    dev = max(abs(cells[('off', g, k)]['total']['mean'] - cells[('off', 0, k)]['total']['mean'])
              for g in (1, 2) for k in WITNESS_K)
    require(dev < 1e-5, f"fig_witness: {DS_NAME[ds]} without weighting the means for g = 1, 2 differ from g = 0 "
                        f"by at most {100 * dev:.5f} percentage points")
    require(all(abs(c['total']['mean'] - (c['mean_attack'] - c['mean_clean'])) < 1e-12 for c in cells.values()),
            f"fig_witness: {DS_NAME[ds]} total.mean = mean_attack - mean_clean in every cell")
    return cells


def fig_witness():
    print("fig_witness: plotted values (mean revoked share of the honest warning, attack minus clean, %; "
          "source results/analysis/e1.json e1b, lag 1, ring, no farming)")
    fig, axes = plt.subplots(1, 3, figsize=(COLUMNWIDTH_IN, 2.0), layout="constrained", sharey=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03)
    for ax, ds in zip(axes, DS3):
        cells = witness_cells(ds)
        print(f"  {DS_NAME[ds]} (n = {cells[('off', 0, 1)]['total']['n']} honest warnings per cell)")
        for lbl, mb, g, col, ls, mk in WITNESS_LINES:
            y = [100.0 * cells[(mb, g, k)]['total']['mean'] for k in WITNESS_K]
            ax.plot(WITNESS_K, y, color=col, ls=ls, lw=1.1, marker=mk, ms=3.2, mfc=col, mec=col, label=lbl,
                    zorder=4 if g == 0 and mb == 'on' else 3)
            print(f"    {lbl:<38s} " + ", ".join(f"k={k}: {v:+.3f}" for k, v in zip(WITNESS_K, y)))
        ax.set_xscale('log')
        ax.set_xticks(WITNESS_K)
        ax.set_xticklabels([str(k) for k in WITNESS_K])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0.7, 72)
        ax.set_title(DS_NAME[ds], pad=3)
        style_axes(ax, grid_axis='both')
    ax0 = axes[0]
    ax0.set_yscale('symlog', linthresh=WITNESS_LINTHRESH, linscale=0.6)
    ax0.set_yticks([0, 0.1, 1, 10])
    ax0.set_yticklabels(['0', '0.1', '1', '10'])
    ax0.yaxis.set_minor_locator(NullLocator())
    ax0.set_ylim(-0.03, 40)
    ax0.set_ylabel('Revoked share of the honest\nwarning (%), symmetric log')
    fig.supxlabel(r'Number of fake witnesses $k$ (log scale)', fontsize=7.5)
    fig.legend(*ax0.get_legend_handles_labels(), loc='outside upper center', ncol=2, frameon=False,
               handlelength=2.2, columnspacing=1.0)
    save(fig, "fig_witness", COLUMNWIDTH_IN)


# ---------------------------------------------------------------------------------------------
# fig_retention: what retention removes, what fake accounts still gain, what an insider links (E3, E4, E6)
# ---------------------------------------------------------------------------------------------
RET_TICKS = ['Alpha', 'OTC', 'Wiki-RfA']
RET_ATTACKERS = [('hand-built', 'handcrafted', C_GRAY), ('degree only', 'degree', C_AMBER),
                 ('learned', 'learned', C_CRIMSON)]
LINK_CRITERION = 0.60     # the registered linkage criterion (H7): AUC below 0.60
LINK_CHANCE = 0.50


def seed_bars(ax, xs, width, by_seed, color, label, hatch=None):
    """One bar per network at the mean of its per-seed values, with the seeds as dots; returns the means."""
    means = []
    for i, (xi, vals) in enumerate(zip(xs, by_seed)):
        vals = np.asarray(vals, dtype=float)
        means.append(float(vals.mean()))
        ax.bar(xi, means[-1], width, color=color, edgecolor=C_EDGE, linewidth=0.5, hatch=hatch, zorder=2,
               label=label if i == 0 else None)
        ax.scatter(xi + np.linspace(-0.32, 0.32, len(vals)) * width, vals, s=4.0, color='white', edgecolor=C_EDGE,
                   linewidth=0.4, zorder=4)
    return means


def fig_retention():
    cc = load(os.path.join(ROOT, "results", "analysis", "cppr_confirm.json"))
    e6 = load(os.path.join(ROOT, "results", "analysis", "e6.json"))
    print("fig_retention: plotted values (sources results/analysis/cppr_confirm.json and e6.json)")
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 1.72), layout="constrained",
                                           gridspec_kw=dict(width_ratios=[0.85, 1.3, 1.0]))
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.04)
    x = np.arange(len(DS3), dtype=float)

    # (a) direct laundering: Plain Erasure against CPPR (control-adjusted), ten fresh seeds
    cut = {ds: cc['hypotheses']['H1']['networks'][ds]['components']['b_cut'] for ds in DS3}
    w = 0.36
    pe = seed_bars(ax_a, x - w / 2, w, [list(cut[ds]['by_seed_denominator'].values()) for ds in DS3], C_CRIMSON,
                   'plain erasure')
    cp = seed_bars(ax_a, x + w / 2, w, [list(cut[ds]['by_seed_numerator'].values()) for ds in DS3], C_TEAL, 'CPPR')
    for ds, a, b in zip(DS3, pe, cp):
        require(abs(a - cut[ds]['denominator_mean']) < 1e-12 and abs(b - cut[ds]['numerator_mean']) < 1e-12,
                f"fig_retention (a): {DS_NAME[ds]} bars equal the stored means of H1's cut "
                f"(plain erasure {a:+.4f}, CPPR control-adjusted {b:+.4f}; {cut[ds]['n_seeds']} seeds)")
    ax_a.set_ylim(-0.12, 0.50)
    ax_a.set_ylabel("Gain of the target's reputation")
    ax_a.set_title('(a) Direct laundering', loc='left', pad=3)

    # (b) fakes that rate a target and delete themselves: no attack edge (E3) and one attack edge (E4, H3b)
    amc = {ds: cc['secondary'][ds]['attack_minus_control']['btpc'] for ds in DS3}
    h3b = {ds: cc['hypotheses']['H3b']['networks'][ds]['components']['difference'] for ds in DS3}
    w = 0.2
    series = [('single pool, $g = 0$', C_GRAY, None,
               [list(amc[ds]['whitewash_vpe']['by_seed'].values()) for ds in DS3]),
              ('CPPR, $g = 0$', C_TEAL, None, [list(amc[ds]['whitewash_cppr']['by_seed'].values()) for ds in DS3]),
              ("CPPR, $g = 1$, KEEPER's tiers", C_BLUE, None,
               [[s['contribution']['btpc'] for s in h3b[ds]['by_seed'].values()] for ds in DS3]),
              ('CPPR, $g = 1$, transitive', 'white', '/////',
               [[s['contribution']['bfs'] for s in h3b[ds]['by_seed'].values()] for ds in DS3])]
    means = [seed_bars(ax_b, x + (j - 1.5) * w, w, vals, col, lbl, hatch) for j, (lbl, col, hatch, vals)
             in enumerate(series)]
    for i, ds in enumerate(DS3):
        require(abs(means[0][i] - amc[ds]['whitewash_vpe']['estimate']) < 1e-12
                and abs(means[1][i] - amc[ds]['whitewash_cppr']['estimate']) < 1e-12
                and abs((means[3][i] - means[2][i]) - h3b[ds]['estimate']) < 1e-12,
                f"fig_retention (b): {DS_NAME[ds]} bars equal the stored estimates (single pool {means[0][i]:+.4f}, "
                f"CPPR {means[1][i]:+.4f}; one attack edge: KEEPER's tiers {means[2][i]:+.4f}, transitive "
                f"{means[3][i]:+.4f}, difference {h3b[ds]['estimate']:+.4f})")
    ax_b.set_ylim(-0.12, 0.66)
    ax_b.set_ylabel('Whitewash gain, control-adjusted')
    ax_b.set_title('(b) Fake accounts that rate and delete', loc='left', pad=3)

    # (c) insider linkage of retained ratings (E6), CPPR with KEEPER's tiers
    w = 0.26
    for j, (lbl, key, col) in enumerate(RET_ATTACKERS):
        rows = [e6['networks'][ds]['cppr_btp']['attackers'][key] for ds in DS3]
        est = np.array([r['estimate'] for r in rows])
        lo, hi = np.array([r['ci95'][0] for r in rows]), np.array([r['ci95'][1] for r in rows])
        xs = x + (j - 1) * w
        ax_c.bar(xs, est - LINK_CHANCE, w, bottom=LINK_CHANCE, color=col, edgecolor=C_EDGE, linewidth=0.5, zorder=2,
                 label=lbl)
        ax_c.errorbar(xs, est, yerr=[est - lo, hi - est], fmt='none', ecolor=C_EDGE, elinewidth=0.7, capsize=1.6,
                      zorder=3)
        print(f"  (c) {lbl:<12s} " + ", ".join(f"{DS_NAME[ds]} {e:.3f} [{a:.3f}, {b:.3f}]"
                                               for ds, e, a, b in zip(DS3, est, lo, hi)))
    ax_c.axhline(LINK_CRITERION, color=C_EDGE, lw=0.8, ls='--', zorder=1, label=f'criterion ({LINK_CRITERION:.2f})')
    ax_c.set_ylim(LINK_CHANCE, 0.99)
    ax_c.set_ylabel('Linkage AUC within the pool')
    ax_c.set_title('(c) Insider linkage', loc='left', pad=3)

    for ax in (ax_a, ax_b, ax_c):
        ax.set_xticks(x)
        ax.set_xticklabels(RET_TICKS)
        ax.tick_params(axis='x', length=0)
        ax.axhline(0.0, color=C_EDGE, lw=0.6, zorder=1)
        style_axes(ax)
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(handles)), key=lambda i: isinstance(handles[i], Line2D))   # bars first, lines last
        ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center', ncol=2, frameon=False,
                  handlelength=1.3, columnspacing=0.9, borderaxespad=0.2)
    save(fig, "fig_retention", TEXTWIDTH_IN)


if __name__ == "__main__":
    fig_laundering()
    fig_streams()
    fig_witness()
    fig_retention()
    print("\nVERIFICATION")
    for line in VERIFICATION:
        print(" ok:", line)
