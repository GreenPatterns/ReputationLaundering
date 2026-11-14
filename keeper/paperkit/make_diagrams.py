"""Fig. 1 (fig_overview): the laundering problem and KEEPER over the lifecycle of one rating;
fig_mechanisms: the three mechanisms of KEEPER side by side (BTP, SCEM, CPPR).

Both figures span \\textwidth and are drawn at TEXT_WIDTH_IN of diagram_helpers_archive.py, so a
font size set here is the printed font size to within 0.4% (that constant is 516 bp wide, and the
class sets \\textwidth to 516 pt). They carry no measured numbers. The numbers they
show are method constants (tier weights w_0 = 0.01 and w_3 = 1.0, stage counts and rates) and two
analytic release densities, g_{1,phi}(x) = phi e^{-phi x} and g_{2,phi}(x) = phi^2 x e^{-phi x}.
Positions on the rank meters and the fake counts of the BTP panel are illustrative.

Run ``python -m keeper.paperkit.make_diagrams``; the exit status is 1 if check_layout finds a font
below the floor, a label outside the canvas or two overlapping labels.
"""
import sys
from collections.abc import Sequence
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon
from matplotlib.text import Text

from keeper.paperkit.diagram_helpers_archive import (
    C_AMB_LT, C_AMBER, C_BLUE, C_BLUE_LT, C_CRIM_LT, C_CRIMSON, C_GRAY, C_INK, C_NAVY, C_SLAT_LT,
    C_SLATE, C_TEAL, C_TEAL_LT, TEXT_WIDTH_IN, arrow, box, card, check_layout, cross, edge_between,
    label, new_canvas, node, save_both, stage_badge)

Point = tuple[float, float]
HAlign = Literal['left', 'center', 'right']
ThumbMode = Literal['warn', 'erase', 'gone', 'pooled']

TITLE_H = 0.20
DASH = (0, (3, 1.5))
C_FILL = "#BFE0D2"                               # released (durable) share of a rating
C_FAKE = "#EDB4B4"                               # fake accounts: darker than honest ones in grayscale too
BASE = 0.024                                     # baseline below the centre line of 7 to 7.5 pt text
PILL_BASE = 0.03                                 # the same inside a tag, where capitals are centred
ALIGN = {'left': 0.0, 'center': 0.5, 'right': 1.0}


# ----------------------------------------------------------------- shared pieces
def _width(t: Text) -> float:
    """Width of a text in inches, the data unit of these canvases. Measured with the Agg renderer
    that the helpers module selects; the PDF sets the same text up to about 1.5% narrower."""
    fig = t.figure
    return t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi


def _pills(ax: Axes, x: float, y: float, strings: Sequence[str], col: str, fc: str, text_col: str,
           ha: HAlign = 'left', bold: bool = False, width: float | None = None, gap: float = 0.06,
           pad: float = 0.045, h: float = 0.15) -> float:
    """Row of rounded tags of equal height, one per string, each as wide as its text plus pad or
    of the given width; ha says which point of the row is at x. Returns the right edge of the row."""
    texts = [label(ax, 0.0, y - PILL_BASE, s, fs=7.0, col=text_col, bold=bold, va='baseline')
             for s in strings]
    widths = [width or _width(t) + 2 * pad for t in texts]
    cur = x - ALIGN[ha] * (sum(widths) + gap * (len(texts) - 1))
    for t, w in zip(texts, widths):
        box(ax, cur, y - h / 2, w, h, col, fc, lw=0.6, radius=0.03, z=7)
        t.set_x(cur + w / 2)
        cur += w + gap
    return cur - gap


def _lines(ax: Axes, x: float, y: float, lines: Sequence[str], pitch: float = 0.125, **kw) -> None:
    """Left-aligned lines of text on a common baseline grid, the first one centred at y."""
    for i, s in enumerate(lines):
        label(ax, x, y - BASE - i * pitch, s, ha='left', va='baseline', **kw)


# =========================================================================
# FIGURE 1: problem and solution over one lifecycle (figure*, \textwidth)
# =========================================================================
# The two rows share three steps: (1) the rating arrives, (2) deletion request, (3) what the model reads.
COLS = (0.34, 2.87, 5.36)                         # left edge of the three step blocks
X_ARRIVE, X_COMMIT, X_DELETE, X_AFTER = 0.34, 2.56, 3.89, 6.16   # marks of the time axis in row (b)
R_RATER, R_T, R_POOL, THUMB_DX = 0.055, 0.12, 0.13, 0.50
METER = (("risky", 0.30, C_CRIMSON, C_CRIM_LT), ("", 0.24, C_GRAY, "#FFFFFF"),
         ("trusted", 0.36, C_TEAL, C_TEAL_LT))    # zones of the rank meter: name, width, edge, fill
RANK_RISKY, RANK_NEWCOMER = 0.15, 0.42            # pointer positions, in inches from the risky end


def _thumb(ax: Axes, x: float, cy: float, mode: ThumbMode) -> None:
    """Thumbnail graph: three raters at x and account T at x + THUMB_DX. 'warn': -1 warnings;
    'erase': the warnings crossed out; 'gone': no warning left; 'pooled': one pool node replaces
    the authors and keeps the three warnings (CPPR)."""
    pT = (x + THUMB_DX, cy)
    if mode == 'pooled':
        node(ax, x, cy, R_POOL, r"$P^{(t)}_{c}$", col=C_TEAL, fc=C_TEAL_LT, lw=1.1, fs=7.5)
        for dy in (0.065, 0.0, -0.065):
            arrow(ax, x + (R_POOL ** 2 - dy ** 2) ** 0.5 + 0.02, cy + dy,
                  pT[0] - (R_T ** 2 - dy ** 2) ** 0.5 - 0.02, cy + dy, col=C_CRIMSON, lw=1.0, ms=6)
    else:
        for y in (cy + 0.17, cy, cy - 0.17):
            node(ax, x, y, R_RATER, "", col=C_BLUE, fc=C_BLUE_LT)
            if mode == 'gone':
                continue
            a, b = edge_between(ax, (x, y), pT, R_RATER, R_T, col=C_CRIMSON, lw=1.0,
                                ls='-' if mode == 'warn' else DASH, ms=6)
            if mode == 'erase':
                cross(ax, 0.6 * a[0] + 0.4 * b[0], 0.6 * a[1] + 0.4 * b[1], size=0.033)
    if mode in ('warn', 'pooled'):
        label(ax, x + 0.255, cy + (0.20 if mode == 'warn' else 0.165), r"$-1$", fs=7.0, col=C_CRIMSON)
    node(ax, *pT, R_T, r"$T$", col=C_CRIMSON, fc=C_CRIM_LT, lw=1.2, fs=8.5)


def _meter(ax: Axes, cx: float, cy: float, rank: float, caption: str, col: str, bold: bool = False,
           h: float = 0.125) -> None:
    """Rank meter from 'risky' to 'trusted' with a pointer at the model's rank of T (rank is the
    distance from the risky end) and a caption under the bar. A newcomer ranks in the middle zone."""
    x0, y = cx - sum(z[1] for z in METER) / 2, cy - 0.085
    x = x0
    for name, w, edge, fill in METER:
        box(ax, x, y, w, h, edge, fill, lw=0.7, radius=0.01)
        label(ax, x + w / 2, y + h / 2, name, fs=7.0, col=edge)
        x += w
    xp = x0 + rank
    ax.add_patch(Polygon([(xp, y + h + 0.012), (xp - 0.045, y + h + 0.082), (xp + 0.045, y + h + 0.082)],
                         closed=True, fc=C_INK, ec="none", zorder=7))
    label(ax, xp, y + h + 0.14, r"$T$", fs=7.5)
    label(ax, cx, y - 0.085, caption, fs=7.0, col=col, bold=bold)


def _row_problem(ax: Axes, x0: float, y0: float, w: float, h: float) -> None:
    """Row (a): raters warn about T, erase their warnings, and the model reads a graph without them."""
    card(ax, x0, y0, w, h, "(a) Problem: lawful erasure launders a distrusted account", C_CRIMSON,
         title_h=TITLE_H)
    yh, cy = y0 + h - TITLE_H - 0.115, y0 + h - TITLE_H - 0.435
    heads = (r"Raters warn about risky account $T$", "Raters erase their warnings",
             r"$T$ looks like a newcomer")
    for i, (x, head, mode) in enumerate(zip(COLS, heads, ("warn", "erase", "gone")), start=1):
        stage_badge(ax, x, yh, i, r=0.065)
        label(ax, x + 0.11, yh, head, fs=7.5, bold=True, ha='left')
        _thumb(ax, x, cy, mode)
    _meter(ax, COLS[0] + 1.25, cy, RANK_RISKY, r"model ranks $T$ as risky", C_SLATE)
    _lines(ax, COLS[1] + 0.80, cy + 0.06, ("authentic requests,", "exact unlearning"), fs=7.0, col=C_SLATE)
    _meter(ax, COLS[2] + 1.22, cy, RANK_NEWCOMER, "no longer ranked risky", C_CRIMSON)
    for x in (COLS[0] + 1.81, COLS[1] + 1.77):               # step arrows of equal length
        arrow(ax, x, cy, x + 0.56, cy, col=C_SLATE, lw=1.4, ms=9)
    label(ax, x0 + w / 2, y0 + 0.10, r"Nothing is forged, and $T$ never changed.", fs=7.5,
          col=C_CRIMSON, bold=True)


def _guarantee(ax: Axes, x: float, y: float, num: int, text: str, ha: HAlign = 'left') -> None:
    """Green guarantee tag: a filled badge 'P<num>' and the property in a few words."""
    badge_w, gap = 0.20, 0.05
    t = label(ax, 0.0, y - PILL_BASE, text, fs=7.0, col=C_TEAL, ha='left', va='baseline')
    x0 = x - ALIGN[ha] * (badge_w + gap + _width(t))
    t.set_x(_pills(ax, x0, y, (f"P{num}",), C_TEAL, C_TEAL, "white", bold=True, width=badge_w) + gap)


def _adversaries(ax: Axes, x: float, y: float, tags: Sequence[tuple[int, str]]) -> None:
    """Red adversary tags centred at x; tags are (number, name) pairs."""
    _pills(ax, x, y, [rf"$\mathbf{{A{n}}}$ {name}" for n, name in tags], C_CRIMSON, C_CRIM_LT,
           C_CRIMSON, ha='center')


def _credibility_band(ax: Axes, x: float, y: float, w: float, h: float,
                      feeds: Sequence[tuple[float, float, str]]) -> None:
    """The credibility layer as one band; each feed is (x, y of the box it enters, annotation)."""
    box(ax, x, y, w, h, C_TEAL, C_TEAL_LT, lw=1.0)
    label(ax, x + 0.08, y + h / 2, r"Credibility layer (BTP): tier $c(u)$ of every account", fs=7.5,
          col=C_TEAL, bold=True, ha='left')
    _guarantee(ax, x + w - 0.08, y + h / 2, 1,
               r"one honest endorsement admits at most $\lfloor 2K \rfloor$ fakes", ha='right')
    for xf, yf, note in feeds:
        arrow(ax, xf, y, xf, yf, col=C_TEAL, lw=0.9, ms=6)
        label(ax, xf + 0.06, (y + yf) / 2 + 0.01, note, fs=7.0, col=C_SLATE, ha='left')


def _mechanism_box(ax: Axes, x: float, y: float, w: float, h: float, title: str,
                   lines: Sequence[str]) -> None:
    """Mechanism box of row (b): a bold title line and plain lines."""
    box(ax, x, y, w, h, C_TEAL, "#FFFFFF", lw=1.0)
    label(ax, x + 0.07, y + h - 0.09, title, fs=7.5, col=C_TEAL, bold=True, ha='left')
    _lines(ax, x + 0.07, y + h - 0.225, lines, fs=7.5)


def _escrow_window(ax: Axes, y: float, h: float) -> None:
    """Escrow window [t_k, t_k + A) on the time axis: the durable share of an unrevoked rating
    grows from zero at arrival (two release stages) and is final at the commitment age A."""
    z = np.linspace(0.0, 10.0, 2001)                       # z = phi * age
    z = z[np.exp(-z) * (1.0 + z) >= 0.01]                  # until about 1% is left, which commits
    xs = X_ARRIVE + z / z[-1] * (X_COMMIT - X_ARRIVE)
    ax.fill_between(xs, y, y + (1.0 - np.exp(-z) * (1.0 + z)) * h, color=C_FILL, lw=0, zorder=4)
    box(ax, X_ARRIVE, y, X_COMMIT - X_ARRIVE, h, C_TEAL, "none", lw=0.8, radius=0.0, z=4.5)
    label(ax, X_COMMIT - 0.07, y + h / 2 - 0.005, "durable share", fs=7.0, col=C_TEAL, ha='right')


def _lifecycle_axis(ax: Axes, ya: float, x_end: float) -> None:
    """Time axis of one rating: the three shared steps as badges, the commitment tick, the labels
    of the marks and, under them, the adversaries at the stage where they act."""
    arrow(ax, 0.12, ya, x_end, ya, col=C_SLATE, lw=1.1, ms=8)
    ax.plot([X_COMMIT] * 2, [ya - 0.045, ya + 0.045], color=C_SLATE, lw=1.1, zorder=6)
    for num, x in ((1, X_ARRIVE), (2, X_DELETE), (3, X_AFTER)):
        stage_badge(ax, x, ya, num, r=0.065)
    x_window = (X_ARRIVE + X_COMMIT) / 2
    for x, s in ((X_ARRIVE, r"$t_k$ arrival"), (x_window, r"escrow window $[t_k,\,t_k{+}A)$"),
                 (X_COMMIT, r"$t_k{+}A$ commitment"),
                 (X_DELETE, r"$t_{\mathrm{del}}$ deletion request"), (X_AFTER, "after retention")):
        label(ax, x, ya - 0.125, s, fs=7.0, col=C_SLATE)
    for x, tags in ((x_window, ((2, "silent sender"), (3, "fake witness"))),
                    (X_DELETE, ((1, "deleter"), (4, "inject-then-erase"))),
                    (X_AFTER, ((5, "insider linker"),))):
        _adversaries(ax, x, ya - 0.29, tags)


def _row_keeper(ax: Axes, x0: float, y0: float, w: float, h: float) -> None:
    """Row (b): the same lifecycle under KEEPER. SCEM acts in the escrow window, CPPR at the
    deletion request, and the credibility layer feeds both."""
    card(ax, x0, y0, w, h, "(b) KEEPER: govern the evidence over a rating\u2019s lifecycle", C_TEAL,
         title_h=TITLE_H)
    top, ya = y0 + h - TITLE_H, y0 + 0.42
    yb0, yb1 = ya + 0.17, top - 0.41                          # the two mechanism boxes
    xc0, xc1 = X_DELETE - 1.02, X_DELETE + 1.02               # CPPR box, centred on the deletion request
    _credibility_band(ax, x0 + 0.12, top - 0.24, w - 0.24, 0.19,
                      ((X_ARRIVE + 0.30, yb1, "weights the warnings"),
                       (xc0 + 0.30, yb1, "sets the weight a retained rating keeps")))
    _mechanism_box(ax, X_ARRIVE, yb0, X_COMMIT - X_ARRIVE, yb1 - yb0, "SCEM: released gradually",
                   ("credible warnings about the sender", "revoke the unreleased part"))
    _guarantee(ax, X_ARRIVE + 0.07, yb0 + 0.11, 2, "nothing durable at arrival")
    _guarantee(ax, X_COMMIT - 0.07, yb0 + 0.11, 3, r"final after $A$", ha='right')
    _mechanism_box(ax, xc0, yb0, xc1 - xc0, yb1 - yb0, "CPPR: keep the verdict, drop the author",
                   ("pool node $P^{(t)}_{c}$ of the rater\u2019s tier", "(where the law allows retention)"))
    _guarantee(ax, xc0 + 0.07, yb0 + 0.11, 4, "first-layer evidence unchanged")
    _escrow_window(ax, ya, 0.15)
    arrow(ax, X_DELETE, ya + 0.075, X_DELETE, yb0, col=C_SLATE, lw=0.9, ms=6)
    _lifecycle_axis(ax, ya, x0 + w - 0.10)
    cy = (yb0 + yb1) / 2
    arrow(ax, xc1 + 0.02, cy, COLS[2] - R_POOL - 0.03, cy, col=C_TEAL, lw=1.4, ms=9)
    _thumb(ax, COLS[2], cy, 'pooled')
    _meter(ax, COLS[2] + 1.22, cy, RANK_RISKY, r"$T$ stays risky", C_TEAL, bold=True)


def make_fig_overview() -> dict:
    """Fig. 1: row (a) the problem, row (b) KEEPER, over the same three steps. Returns the
    report of check_layout."""
    W = TEXT_WIDTH_IN
    h_problem, h_keeper, gap = 1.06, 1.82, 0.06
    H = 0.03 + h_keeper + gap + h_problem + 0.02
    fig, ax = new_canvas(W, H)
    _row_keeper(ax, 0.02, 0.03, W - 0.05, h_keeper)
    _row_problem(ax, 0.02, 0.03 + h_keeper + gap, W - 0.05, h_problem)
    report = check_layout(fig, TEXT_WIDTH_IN, "fig_overview")
    save_both(fig, "fig_overview")
    return report


# =========================================================================
# fig_mechanisms: BTP, SCEM and CPPR side by side (figure*, \textwidth)
# =========================================================================
R_ACC, R_FAKE, N_FAKES, N_CREDIBLE = 0.05, 0.045, 6, 2    # fakes in the ring, credible under BTP
RING_RX, RING_RY = 0.40, 0.20


def _valve(ax: Axes, x: float, y: float, s: float = 0.045) -> None:
    """Valve symbol on a horizontal pipe: two triangles that meet at their tips."""
    for sgn in (-1, 1):
        ax.add_patch(Polygon([(x, y), (x + sgn * s, y + s), (x + sgn * s, y - s)], closed=True,
                             fc="#FFFFFF", ec=C_SLATE, lw=0.8, zorder=8))


def _trusted_accounts(ax: Axes, bx: float, cy: float) -> Point:
    """Two seeds endorse three honest accounts; returns the account the attack edge leaves from."""
    seeds = [(bx + 0.13, cy + 0.12), (bx + 0.13, cy - 0.12)]
    honest = [(bx + 0.42, cy + 0.23), (bx + 0.42, cy), (bx + 0.42, cy - 0.23)]
    for p in seeds:
        node(ax, *p, R_ACC, "", col=C_NAVY, fc=C_NAVY)
    for p in honest:
        node(ax, *p, R_ACC, "", col=C_BLUE, fc=C_BLUE_LT)
    for i, j in ((0, 0), (0, 1), (1, 1), (1, 2)):
        edge_between(ax, seeds[i], honest[j], R_ACC, R_ACC, col=C_SLATE, lw=0.9, ms=5)
    return honest[1]


def _fake_ring(ax: Axes, cx: float, cy: float, budgeted: bool) -> Point:
    """Ring of fakes that endorse one another, entered on its left; returns the entry fake.
    Transitive endorsement makes every fake credible; under BTP the trust thins out along the
    ring and only the first fakes reach q_min."""
    box(ax, cx - RING_RX - 0.10, cy - RING_RY - 0.09, 2 * RING_RX + 0.20, 2 * RING_RY + 0.18,
        C_CRIMSON, "#FFFFFF", lw=0.8, ls=DASH)
    ang = np.pi - 2 * np.pi / N_FAKES * np.arange(N_FAKES)
    ring = [(cx + RING_RX * np.cos(t), cy + RING_RY * np.sin(t)) for t in ang]
    for k, p in enumerate(ring):
        credible = not budgeted or k < N_CREDIBLE
        ax.add_patch(Circle(p, R_FAKE, fc=C_FAKE if credible else "#FFFFFF",
                            ec=C_CRIMSON if credible else C_GRAY, lw=0.9,
                            ls='-' if credible else (0, (2, 1)), zorder=6))
        if budgeted:
            col, lw = (C_CRIMSON, 1.2 - 0.4 * k / max(N_CREDIBLE - 1, 1)) if credible else (C_GRAY, 0.6)
        else:
            col, lw = C_CRIMSON, 0.9
        edge_between(ax, p, ring[(k + 1) % N_FAKES], R_FAKE, R_FAKE, gap=0.015, col=col, lw=lw,
                     ms=5, rad=-0.25)
    return ring[0]


def _btp_row(ax: Axes, bx: float, by: float, w: float, h: float, budgeted: bool) -> None:
    """One rule of panel (a): seeds, honest accounts, one attack edge and the ring of fakes."""
    col = C_TEAL if budgeted else C_CRIMSON
    box(ax, bx, by, w, h, col, "#FFFFFF", lw=0.8, z=3)
    head = (r"BTP: at most $\lfloor 2Kg \rfloor$ fakes credible" if budgeted
            else "Transitive endorsement: all fakes credible")
    label(ax, bx + 0.07, by + h - 0.09, head, fs=7.5, col=col, bold=True, ha='left')
    cx, cy = bx + w - RING_RX - 0.14, by + h - 0.17 - RING_RY - 0.09
    a, b = edge_between(ax, _trusted_accounts(ax, bx, cy), _fake_ring(ax, cx, cy, budgeted), R_ACC,
                        R_FAKE, col=C_CRIMSON, lw=1.3, ms=7)
    xm = (a[0] + b[0]) / 2
    label(ax, xm, cy + 0.10, r"$g = 1$", fs=7.0, col=C_CRIMSON)
    rule = ("credible if", r"trust $\geq q_{\min}$") if budgeted else ("credible if", "endorsed")
    for s, dy in zip(rule, (0.06, -0.06)):
        label(ax, cx, cy + dy, s, fs=7.0, col=C_SLATE)
    if budgeted:
        _valve(ax, xm, cy)
        _valve(ax, bx + 0.12, by + 0.09)
        label(ax, bx + 0.21, by + 0.09, r"at most $q_{\mathrm{cap}}$ per rating and step", fs=7.0,
              col=C_SLATE, ha='left')


def _btp_legend(ax: Axes, x: float, y: float) -> None:
    """Two-row legend of panel (a): the kinds of account, then the kinds of edge."""
    cur = x
    for name, col, fc in (("seed", C_NAVY, C_NAVY), ("honest", C_BLUE, C_BLUE_LT),
                          ("fake", C_CRIMSON, C_FAKE)):
        node(ax, cur, y, R_ACC, "", col=col, fc=fc, lw=0.9)
        cur += 0.09 + _width(label(ax, cur + 0.09, y, name, fs=7.0, ha='left')) + 0.22
    cur = x
    for name, col, lw in ((r"$+1$ rating", C_SLATE, 0.9), ("attack edge", C_CRIMSON, 1.3)):
        arrow(ax, cur - 0.05, y - 0.15, cur + 0.19, y - 0.15, col=col, lw=lw, ms=6)
        cur += 0.25 + _width(label(ax, cur + 0.25, y - 0.15, name, fs=7.0, ha='left')) + 0.22


def _panel_btp(ax: Axes, x0: float, y0: float, w: float, h: float) -> None:
    """Panel (a): transitive endorsement (top) against budgeted trust propagation (bottom)."""
    card(ax, x0, y0, w, h, "(a) Credibility layer (BTP)", C_SLATE, title_h=TITLE_H)
    top, pad, h1, h_legend = y0 + h - TITLE_H, 0.06, 0.79, 0.43
    h2 = top - y0 - 2 * pad - h1 - h_legend
    _btp_row(ax, x0 + pad, top - pad - h1, w - 2 * pad, h1, budgeted=False)
    _btp_row(ax, x0 + pad, top - 2 * pad - h1 - h2, w - 2 * pad, h2, budgeted=True)
    _btp_legend(ax, x0 + 0.16, y0 + 0.26)


def g_exp(x: np.ndarray | float, phi: float) -> np.ndarray | float:
    """Release density of exponential decay, N = 1."""
    return phi * np.exp(-phi * x)


def g_erlang2(x: np.ndarray | float, phi: float) -> np.ndarray | float:
    """Release density of the order-2 Erlang cascade, N = 2."""
    return phi ** 2 * x * np.exp(-phi * x)


def _escrow_flow(ax: Axes, x0: float, top: float, w: float) -> float:
    """Unit mass of one rating: escrow stage 0 -> stage 1 -> durable memory, the two release steps
    at rate phi; the sender hazard moves mass from both stages to the revoked sink. The compartments
    carry the symbols of the mass identity omega_k + pi_k + R_k = 1. Returns the y under the drawing."""
    ys, sh = top - 0.44, 0.30                               # stage row: centre, box height
    xe, xs0, xs1, xd = x0 + 0.07, x0 + 0.75, x0 + 1.35, x0 + 1.95
    we, sw, wd = 0.50, 0.42, x0 + w - 0.07 - xd
    box(ax, xe, ys - sh / 2, we, sh, C_SLATE, C_SLAT_LT)
    label(ax, xe + we / 2, ys + 0.065, r"rating $e_k$", fs=7.5, col=C_SLATE, bold=True)
    label(ax, xe + we / 2, ys - 0.075, "mass 1", fs=7.0, col=C_SLATE)
    for xs, name in ((xs0, "stage 0"), (xs1, "stage 1")):
        box(ax, xs, ys - sh / 2, sw, sh, C_BLUE, C_BLUE_LT)
        label(ax, xs + sw / 2, ys, name, fs=7.5, col=C_BLUE, bold=True)
    box(ax, xd, ys - sh / 2, wd, sh, C_TEAL, C_TEAL_LT, lw=1.1)
    label(ax, xd + wd / 2, ys + 0.065, r"durable $\omega_{k}$", fs=7.5, col=C_TEAL, bold=True)
    label(ax, xd + wd / 2, ys - 0.075, r"memory of $v_k$", fs=7.0, col=C_TEAL)
    for xa, xb, s, col in ((xe + we, xs0, r"$t_k$", C_SLATE), (xs0 + sw, xs1, r"$\phi$", C_BLUE),
                           (xs1 + sw, xd, r"$\phi$", C_BLUE)):
        arrow(ax, xa, ys, xb, ys, col=col, lw=1.1, ms=7)
        label(ax, (xa + xb) / 2, ys + 0.09, s, fs=7.5, col=col)
    # the escrowed remainder commits at age A
    yt = ys + sh / 2
    ax.plot([xs0 + 0.04, xs0 + 0.04, xs1 + sw - 0.04, xs1 + sw - 0.04],
            [yt + 0.03, yt + 0.08, yt + 0.08, yt + 0.03], color=C_TEAL, lw=0.9, zorder=5)
    ax.plot([xs1 + sw - 0.04, xd + wd / 2], [yt + 0.08] * 2, color=C_TEAL, lw=1.0, ls=DASH, zorder=5)
    arrow(ax, xd + wd / 2, yt + 0.08, xd + wd / 2, yt, col=C_TEAL, lw=1.0, ms=7)
    label(ax, x0 + w / 2, yt + 0.19, r"in escrow $\pi_{k}$; at age $A$ the remainder commits",
          fs=7.0, col=C_TEAL)
    # revoked sink under both stages; the drains run at the sender hazard
    yr, rh = ys - sh / 2 - 0.41, 0.20
    box(ax, xs0, yr, xs1 + sw - xs0, rh, C_CRIMSON, C_CRIM_LT, lw=1.0)
    label(ax, (xs0 + xs1 + sw) / 2, yr + rh / 2, r"revoked $R_{k}$", fs=7.5, col=C_CRIMSON, bold=True)
    for xs in (xs0, xs1):
        arrow(ax, xs + sw / 2, ys - sh / 2, xs + sw / 2, yr + rh, col=C_CRIMSON, lw=1.1, ms=7)
        label(ax, xs + sw / 2 + 0.05, (ys - sh / 2 + yr + rh) / 2, r"$\lambda_{u_k}(t)$", fs=8.0,
              col=C_CRIMSON, ha='left')
    y_note, pitch = yr - 0.115, 0.152                       # two lines with subscripts need this pitch
    _lines(ax, x0 + 0.10, y_note, (r"warnings about $u_k$, weighted by witness",
                                   r"credibility $w_c$, raise the hazard $\lambda_{u_k}(t)$"), fs=7.5,
           pitch=pitch)
    return y_note - pitch - 0.125


def _release_density(fig: Figure, rect: tuple[float, float, float, float]) -> None:
    """Analytic release densities of one and two stages at the same mean delay N / phi, in an
    axes at rect = (x, y, width, height) in inches. The dots mark the densities at arrival."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([rect[0] / W, rect[1] / H, rect[2] / W, rect[3] / H], facecolor="none")
    x = np.linspace(0, 16, 400)
    curves = ((1, 0.25, g_exp, "one stage", dict(color=C_AMBER, lw=1.2, ls=(0, (4, 1.5)))),
              (2, 0.5, g_erlang2, "two stages", dict(color=C_BLUE, lw=1.4)))
    delay = curves[0][0] / curves[0][1]
    for n, phi, density, name, style in curves:
        assert n / phi == delay, "the two release rules must have the same mean delay"
        ax.plot(x, density(x, phi), label=rf"{name} ($N{{=}}{n}$), $\phi{{=}}{phi:g}$", **style)
        ax.plot([0], [density(0.0, phi)], marker='o', ms=3.2, color=style["color"], clip_on=False,
                zorder=6)
    ax.axvline(delay, color=C_GRAY, lw=0.7, ls=':', zorder=0)
    ax.text(delay + 0.3, 0.25, f"mean delay {delay:g}", fontsize=7.0, color=C_GRAY, va='center')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 0.27)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_yticks([0, 0.1, 0.2])
    ax.set_xlabel(r"record age $x$ (snapshots)", labelpad=1.5)
    ax.set_ylabel("release\ndensity", labelpad=2)
    ax.tick_params(pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", frameon=False, handlelength=1.8, borderaxespad=0.0,
              labelspacing=0.25, bbox_to_anchor=(1.0, 0.88))


def _panel_scem(fig: Figure, ax: Axes, x0: float, y0: float, w: float, h: float) -> None:
    """Panel (b): the compartment flow of one rating (top) and the release densities (bottom)."""
    card(ax, x0, y0, w, h, "(b) Escrow memory (SCEM)", C_SLATE, title_h=TITLE_H)
    y_plot = _escrow_flow(ax, x0, y0 + h - TITLE_H, w)
    _release_density(fig, (x0 + 0.52, y0 + 0.31, w - 0.62, y_plot - y0 - 0.31))


def _rater(ax: Axes, p: Point, name: str, tier: Sequence[str], col: str, fc: str,
           fs: float = 8.0) -> None:
    """Deleted rater with its credibility tier and tier weight written to its left."""
    node(ax, *p, 0.14, name, col=col, fc=fc, fs=fs)
    for s, dy in zip(tier, (0.06, -0.065)):
        label(ax, p[0] - 0.19, p[1] + dy, s, fs=7.0, col=col, ha='right')


def _cppr_half(ax: Axes, bx: float, by: float, w: float, h: float, tiered: bool) -> None:
    """One retention rule of panel (c): deleted raters on the left, pool nodes, target T on the right."""
    col = C_TEAL if tiered else C_CRIMSON
    box(ax, bx, by, w, h, col, "#FFFFFF", lw=0.8, z=3)
    head = (r"CPPR: tier-matched pools $P^{(t)}_{c}$" if tiered
            else r"Single pool $P^{(t)}$ per snapshot")
    label(ax, bx + 0.07, by + h - 0.10, head, fs=7.5, col=col, bold=True, ha='left')
    dy = 0.21 if tiered else 0.18
    cy = by + 0.20 + dy + 0.14
    ph, pf, pT = (bx + 0.70, cy + dy), (bx + 0.70, cy - dy), (bx + w - 0.21, cy)
    _rater(ax, ph, r"$u_h$", ("Tier 3", r"$w_3{=}1.0$"), C_BLUE, C_BLUE_LT)
    _rater(ax, pf, "fakes", ("Tier 0", r"$w_0{=}0.01$"), C_CRIMSON, C_FAKE, fs=7.0)
    node(ax, *pT, 0.13, r"$T$", col=C_CRIMSON, fc=C_CRIM_LT, lw=1.2, fs=8.5)
    xp, rp = bx + 1.26, 0.16
    if tiered:
        p3, p0 = (xp, cy + dy), (xp, cy - dy)
        node(ax, *p3, rp, r"$P^{(t)}_{3}$", col=C_TEAL, fc=C_TEAL_LT, lw=1.2, fs=8.0)
        node(ax, *p0, rp, r"$P^{(t)}_{0}$", col=C_SLATE, fc=C_SLAT_LT, lw=1.2, fs=8.0)
        edge_between(ax, ph, p3, 0.14, rp, gap=0.03, col=C_BLUE, lw=1.1, ms=7)
        edge_between(ax, pf, p0, 0.14, rp, gap=0.03, col=C_CRIMSON, lw=1.1, ls=DASH, ms=7)
        edge_between(ax, p3, pT, rp, 0.13, gap=0.03, col=C_TEAL, lw=1.3, ms=8)
        edge_between(ax, p0, pT, rp, 0.13, gap=0.03, col=C_SLATE, lw=1.1, ls=DASH, ms=7)
        label(ax, (xp + pT[0]) / 2 + 0.04, cy + dy / 2 + 0.13, r"$w_3$", fs=7.5, col=C_TEAL)
        label(ax, (xp + pT[0]) / 2 + 0.04, cy - dy / 2 - 0.13, r"$w_0$", fs=7.5, col=C_SLATE)
        note = r"tier weight kept: fakes stay at $w_0$"
    else:
        pP = (xp, cy)
        node(ax, *pP, rp, r"$P^{(t)}$", col=C_AMBER, fc=C_AMB_LT, lw=1.2, fs=8.5)
        edge_between(ax, ph, pP, 0.14, rp, gap=0.03, col=C_BLUE, lw=1.1, ms=7)
        edge_between(ax, pf, pP, 0.14, rp, gap=0.03, col=C_CRIMSON, lw=1.1, ls=DASH, ms=7)
        edge_between(ax, pP, pT, rp, 0.13, gap=0.03, col=C_CRIMSON, lw=1.3, ms=8)
        label(ax, (xp + pT[0]) / 2, cy + 0.24, "mean weight", fs=7.0, col=C_CRIMSON)
        note = "fakes inherit the mean weight"
    label(ax, bx + w / 2, by + 0.10, note, fs=7.5, col=col, bold=True)


def _panel_cppr(ax: Axes, x0: float, y0: float, w: float, h: float) -> None:
    """Panel (c): one pool per snapshot (top) against one pool per snapshot and tier (bottom)."""
    card(ax, x0, y0, w, h, "(c) Retention at deletion (CPPR)", C_SLATE, title_h=TITLE_H)
    top, pad = y0 + h - TITLE_H, 0.06
    h1 = 1.00
    h2 = top - y0 - 3 * pad - h1
    _cppr_half(ax, x0 + pad, top - pad - h1, w - 2 * pad, h1, tiered=False)
    _cppr_half(ax, x0 + pad, y0 + pad, w - 2 * pad, h2, tiered=True)


def make_fig_mechanisms() -> dict:
    """The three mechanisms as panels of equal height: (a) BTP, (b) SCEM, (c) CPPR. Returns the
    report of check_layout."""
    W, H = TEXT_WIDTH_IN, 2.52
    gap, wa, wc = 0.06, 2.25, 2.13
    wb = W - 0.035 - wa - wc - 2 * gap
    fig, ax = new_canvas(W, H)
    _panel_btp(ax, 0.02, 0.03, wa, H - 0.05)
    _panel_scem(fig, ax, 0.02 + wa + gap, 0.03, wb, H - 0.05)
    _panel_cppr(ax, 0.02 + wa + wb + 2 * gap, 0.03, wc, H - 0.05)
    report = check_layout(fig, TEXT_WIDTH_IN, "fig_mechanisms")
    save_both(fig, "fig_mechanisms")
    return report


if __name__ == '__main__':
    reports = (make_fig_overview(), make_fig_mechanisms())
    sys.exit(1 if any(r["below_floor"] or r["outside"] or r["overlaps"] for r in reports) else 0)
