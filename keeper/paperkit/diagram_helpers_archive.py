#!/usr/bin/env python3
"""Shared drawing helpers for the four schematic diagrams (Figs. 1 to 4).

Every diagram is drawn at its printed size: the figure size equals the width at
which LaTeX includes it (\\textwidth = 7.17 in, \\columnwidth = 3.50 in for the
10pt journal,compsoc IEEEtran class), and the axes use inches as data units.
A font size set here is therefore the printed font size. ``check_layout``
verifies the 7 pt floor, that no label leaves the canvas, and that no two
labels overlap. The colour palette and the card / node / arrow vocabulary are
kept from the user's updated/generate_all_figures.py.
"""
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Text

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper", "figures")

TEXT_WIDTH_IN = 517.935 / 72.27     # \textwidth of the paper, measured with pdflatex
COLUMN_WIDTH_IN = 252.945 / 72.27   # \columnwidth of the paper
MIN_FONT_PT = 7.0                   # legibility floor at the printed size

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8.0,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 7.0,
    'ytick.labelsize': 7.0,
    'legend.fontsize': 7.0,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

C_NAVY    = "#1B365D"
C_BLUE    = "#1F4E79"
C_BLUE_LT = "#EDF4FA"
C_TEAL    = "#0B6E4F"
C_TEAL_LT = "#E9F6F1"
C_CRIMSON = "#9E1B1B"
C_CRIM_LT = "#FDF0F0"
C_AMBER   = "#8C4A08"
C_AMB_LT  = "#FDF6EC"
C_SLATE   = "#2C3E50"
C_SLAT_LT = "#F4F6F8"
C_GRAY    = "#6C7A89"
C_GRAY_LT = "#E5E8EB"
C_INK     = "#1A252F"


# ----------------------------------------------------------------- canvas
def new_canvas(width_in, height_in):
    """Figure whose single axes covers the canvas, with inches as data units."""
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_in)
    ax.set_ylim(0, height_in)
    ax.axis('off')
    return fig, ax


def save_both(fig, stem):
    """Save at the exact canvas size (no tight bbox), so printed scale is 1.0."""
    path = os.path.join(FIG_DIR, f"{stem}.pdf")
    fig.savefig(path)
    fig.savefig(path[:-4] + ".png", dpi=300)
    plt.close(fig)
    print("saved", path)


# ----------------------------------------------------------------- checks
def _visible_texts(fig):
    out = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_alpha() != 0:
            out.append(t)
    return out


def check_layout(fig, printed_width_in, name, allow_overlap=()):
    """Report font floor, canvas overflow and pairwise label overlap.

    ``allow_overlap`` lists label strings that may overlap by design. Returns
    a dict that the caller logs; raises nothing so all problems are reported.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = printed_width_in / fig.get_figwidth()
    fig_bb = fig.bbox
    texts = _visible_texts(fig)
    sizes = [t.get_fontsize() * scale for t in texts]
    small = [(t.get_text(), round(s, 2)) for t, s in zip(texts, sizes) if s < MIN_FONT_PT - 1e-6]
    outside, boxes = [], []
    for t in texts:
        bb = t.get_window_extent(renderer)
        boxes.append((t, bb))
        if (bb.x0 < fig_bb.x0 - 0.5 or bb.y0 < fig_bb.y0 - 0.5
                or bb.x1 > fig_bb.x1 + 0.5 or bb.y1 > fig_bb.y1 + 0.5):
            outside.append(t.get_text())
    overlaps = []
    for (ta, a), (tb, b) in itertools.combinations(boxes, 2):
        if ta.get_text() in allow_overlap or tb.get_text() in allow_overlap:
            continue
        if a.overlaps(b):
            w = min(a.x1, b.x1) - max(a.x0, b.x0)
            h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > 0.5 and h > 0.5:
                overlaps.append((ta.get_text(), tb.get_text()))
    report = dict(name=name, printed_width_in=round(printed_width_in, 3),
                  native_width_in=round(fig.get_figwidth(), 3),
                  native_height_in=round(fig.get_figheight(), 3),
                  printed_height_in=round(fig.get_figheight() * scale, 3),
                  scale=round(scale, 4), n_labels=len(texts),
                  min_font_pt=round(min(sizes), 2), max_font_pt=round(max(sizes), 2),
                  below_floor=small, outside=outside, overlaps=overlaps)
    print(f"[check] {name}: printed {report['printed_width_in']} x "
          f"{report['printed_height_in']} in, scale {report['scale']}, "
          f"{len(texts)} labels, font {report['min_font_pt']} to {report['max_font_pt']} pt")
    for key in ("below_floor", "outside", "overlaps"):
        if report[key]:
            print(f"   {key}: {report[key]}")
    return report


# ----------------------------------------------------------------- shapes
def label(ax, x, y, s, fs=7.5, col=C_INK, bold=False, ha='center', va='center',
          z=8, bg=None, ec="none", pad=0.12, **kw):
    bbox = None
    if bg is not None:
        bbox = dict(boxstyle=f"round,pad={pad}", fc=bg, ec=ec, lw=0.5)
    return ax.text(x, y, s, ha=ha, va=va, color=col, fontsize=fs,
                   fontweight='bold' if bold else 'normal', zorder=z, bbox=bbox, **kw)


def card(ax, x, y, w, h, title, col, body="#FFFFFF", title_h=0.2, title_fs=8.0,
         lw=0.9, radius=0.04, title_ha='center'):
    """Rounded panel with a coloured title bar (the paper's card style)."""
    ax.add_patch(FancyBboxPatch((x + 0.012, y - 0.014), w, h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc="#DCE2E8", ec="none", zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=body, ec=col, lw=lw, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - title_h), w, title_h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=col, ec=col, lw=lw, zorder=3))
    ax.add_patch(Rectangle((x + 0.005, y + h - title_h), w - 0.01, title_h / 2,
                           fc=col, ec="none", zorder=3))
    tx = x + w / 2 if title_ha == 'center' else x + 0.07
    label(ax, tx, y + h - title_h / 2, title, fs=title_fs, col='white', bold=True,
          ha=title_ha, z=5)


def box(ax, x, y, w, h, col, fc, lw=0.9, radius=0.035, ls='-', z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=fc, ec=col, lw=lw, ls=ls, zorder=z))


def node(ax, x, y, r, name, sub=None, col=C_NAVY, fc="#FFFFFF", lw=1.0,
         fs=8.5, sub_fs=7.0):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=col, lw=lw, zorder=6))
    if sub:
        label(ax, x, y + r * 0.32, name, fs=fs, col=col, bold=True, z=7)
        label(ax, x, y - r * 0.40, sub, fs=sub_fs, col=col, z=7)
    else:
        label(ax, x, y, name, fs=fs, col=col, bold=True, z=7)


def arrow(ax, x1, y1, x2, y2, col=C_SLATE, lw=1.0, ls='-', rad=0.0, ms=7,
          style='-|>', z=5, shrink=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle=style, mutation_scale=ms, color=col, lw=lw,
                                 linestyle=ls, zorder=z, shrinkA=shrink, shrinkB=shrink))


def edge_between(ax, p, q, rp, rq, gap=0.02, **kw):
    """Arrow from circle p (radius rp) to circle q (radius rq), boundary to boundary."""
    (x1, y1), (x2, y2) = p, q
    dx, dy = x2 - x1, y2 - y1
    d = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / d, dy / d
    a = (x1 + ux * (rp + gap), y1 + uy * (rp + gap))
    b = (x2 - ux * (rq + gap), y2 - uy * (rq + gap))
    arrow(ax, a[0], a[1], b[0], b[1], **kw)
    return a, b


def cross(ax, x, y, col=C_CRIMSON, size=0.045, lw=1.3):
    ax.plot([x - size, x + size], [y - size, y + size], color=col, lw=lw, zorder=9,
            solid_capstyle='round')
    ax.plot([x - size, x + size], [y + size, y - size], color=col, lw=lw, zorder=9,
            solid_capstyle='round')


def stage_badge(ax, x, y, num, col=C_SLATE, r=0.075, fs=7.0):
    """Filled circle with a white stage number, centred on a timeline."""
    ax.add_patch(Circle((x, y), r, fc=col, ec="white", lw=0.8, zorder=9))
    label(ax, x, y - 0.004, str(num), fs=fs, col='white', bold=True, z=10)
