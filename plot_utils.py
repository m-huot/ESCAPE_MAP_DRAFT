import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import to_rgb
from cycler import cycler
import numpy as np

# --- constants ---
_MM_TO_IN = 1.0 / 25.4
# Nature common column widths (mm): single=89, one-and-a-half≈114, double=183
_NATURE_WIDTHS_MM = {"single": 89, "onehalf": 114, "double": 183}


def nature_rc():
    """
    Minimal, journal-ready rcParams tuned for Nature-style figures.
    Use together with nature_figsize() when creating figures.
    """
    # Subtle, colorblind-friendly cycle (Tableau 10), slightly lightened
    base_cycle = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#59A14F",
        "#76B7B2",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]
    light_cycle = [lighten_color(c, amount=0.18) for c in base_cycle]

    mpl.rcParams.update(
        {
            # --- output quality / fonts ---
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.transparent": True,
            "pdf.fonttype": 42,  # embed TrueType, keep text selectable
            "ps.fonttype": 42,
            "svg.fonttype": "none",  # keep text as text
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "mathtext.fontset": "stixsans",
            "axes.unicode_minus": False,
            # --- sizes (8pt labels, 7pt ticks is Nature-friendly) ---
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            # --- lines / ticks / spines ---
            "axes.prop_cycle": cycler(color=light_cycle),
            "axes.linewidth": 0.5,
            "lines.linewidth": 0.8,
            "lines.markersize": 3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "errorbar.capsize": 2,
            # --- layout ---
            "figure.constrained_layout.use": True,
            "figure.autolayout": False,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
        }
    )


def nature_figsize(width="single", ratio=0.62):
    """
    Return a (w, h) tuple in inches for a Nature-style figure.
    width: "single" (89 mm), "onehalf" (~114 mm), or "double" (183 mm)
    ratio: height/width; ~0.62 (golden-ish) is a good default
    """
    if width not in _NATURE_WIDTHS_MM:
        raise ValueError(f"width must be one of {list(_NATURE_WIDTHS_MM)}")
    w_in = _NATURE_WIDTHS_MM[width] * _MM_TO_IN
    h_in = max(1e-3, ratio * w_in)
    return (w_in, h_in)


def apply_yearly_date_axis(ax):
    """Major ticks: yearly labels; minor ticks: quarterly (no labels)."""
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", which="major", pad=2)
    ax.minorticks_on()


def finalize_axes(ax, minor=True, trim_spines=False):
    """
    Small finishing touches: enable minor ticks, optionally trim top/right spines.
    """
    if minor:
        ax.minorticks_on()
    if trim_spines:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def add_panel_labels(axes, labels=None, x=-0.02, y=1.02, weight="bold"):
    """
    Add 'a', 'b', 'c'… panel labels. Works with a single Axes or an iterable.
    """
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    if labels is None:
        labels = [chr(97 + i) for i in range(len(axes))]  # a, b, c, ...
    for ax, lab in zip(axes, labels):
        ax.text(
            x,
            y,
            lab,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            fontweight=weight,
        )


def lighten_color(color, amount=0.35):
    """
    Lighten an RGB/hex color by mixing toward white.
    amount in [0,1]: 0 = original, 1 = white.
    """
    r, g, b = to_rgb(color)
    return (
        (1 - amount) * r + amount,
        (1 - amount) * g + amount,
        (1 - amount) * b + amount,
    )


def get_lightened_cycle(n, amount=0.35):
    """
    Use current Matplotlib prop cycle hues, but lighten them.
    """
    base_cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", [])
    while len(base_cycle) < n:
        base_cycle = base_cycle + base_cycle
    return [lighten_color(c, amount=amount) for c in base_cycle[:n]]
