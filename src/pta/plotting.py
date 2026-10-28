"""One plotting style for every notebook: two colour maps, one line-colour cycle, figures saved at 300 dpi.

    from pta.plotting import use_style, save_figure, show_field
    use_style()                                   # once, after the imports
    fig, ax = plt.subplots()
    show_field(ax, J_par, dx, dy, centre=0, label=r"$J_\\parallel$")   # signed field: RdBu_r, symmetric limits
    show_field(ax, J_par**2, dx, dy, label=r"$J_\\parallel^2$")         # magnitude: inferno
    save_figure(fig, RESULTS, "J_par_t200")                         # RESULTS/J_par_t200.png at 300 dpi

Fields are expected with x along axis 0 (as pta.Run.load returns them); they are drawn with x
horizontal, y vertical and the axes in d_p.
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
from cycler import cycler

MAGNITUDE_CMAP = "inferno"     # quantities that are >= 0: |J|^2, |B_perp|^2, density, ...
SIGNED_CMAP = "RdBu_r"         # quantities with a sign (or around a reference value): J_z, B_z, omega, Delta T, A_p
MASK_CMAP = "gray"             # binary masks (detected sheets, selected regions)
LINE_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]   # matplotlib's standard ten
FIGURE_DPI = 300
COLOUR_LIMIT_PERCENTILE = 99   # automatic colour limits ignore the most extreme 1% of pixels


def use_style():
    """Apply the shared style: colour maps, line colours and 300 dpi saving for every figure."""
    mpl.rcParams.update({
        "image.cmap": MAGNITUDE_CMAP,
        "axes.prop_cycle": cycler(color=LINE_COLOURS),
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
    })


def save_figure(fig, folder, name, file_type="png"):
    """Save fig as <folder>/<name>.<file_type> at 300 dpi and return the path."""
    path = Path(folder) / f"{name}.{file_type}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    return path


def colour_limits(field, centre=None, percentile=COLOUR_LIMIT_PERCENTILE):
    """Automatic colour limits: symmetric about `centre` for signed fields, from the minimum for magnitudes."""
    if centre is None:
        return float(np.min(field)), float(np.percentile(field, percentile))
    half_range = float(np.percentile(np.abs(field - centre), percentile))
    return centre - half_range, centre + half_range


def show_field(ax, field, dx, dy, centre=None, limits=None, label=None, colorbar=True, axis_labels=True, mask=False):
    """Draw a 2D field (x along axis 0) on ax with x horizontal and the axes in d_p.

    centre : None for a magnitude (MAGNITUDE_CMAP); a value (0 for signed fields, 1 for A_p) for a field
             that varies around it (SIGNED_CMAP, limits symmetric about it)
    limits : (vmin, vmax) to fix the colour range; by default taken from the data (see colour_limits)
    mask   : True for a binary mask (MASK_CMAP, range 0-1)
    """
    nx, ny = field.shape
    if mask:
        cmap, (vmin, vmax) = MASK_CMAP, (limits if limits is not None else (0, 1))
    else:
        cmap = MAGNITUDE_CMAP if centre is None else SIGNED_CMAP
        vmin, vmax = limits if limits is not None else colour_limits(field, centre)
    image = ax.imshow(field.T, origin="lower", extent=[0, nx * dx, 0, ny * dy], cmap=cmap, vmin=vmin, vmax=vmax)
    if axis_labels:
        ax.set_xlabel(r"$x/d_p$")
        ax.set_ylabel(r"$y/d_p$")
    if colorbar:
        ax.figure.colorbar(image, ax=ax, label=label)
    return image


def add_slope_guide(ax, k, spectrum, slope, k_min, k_max, offset=2.0, label=None):
    """Dashed k^slope line over [k_min, k_max], starting `offset` times above the spectrum at k_min."""
    inside = (k >= k_min) & (k <= k_max)
    k_line = k[inside]
    if k_line.size < 2:
        return None
    anchor = spectrum[inside][0] * offset
    line = anchor * (k_line / k_line[0]) ** slope
    ax.loglog(k_line, line, "k--", lw=1)
    if label is not None:
        ax.text(k_line[len(k_line) // 2], line[len(k_line) // 2] * offset, label, fontsize=11)
    return k_line, line


def outline_mask(ax, mask, dx, dy, color="yellow", linewidth=1):
    """Outline the regions of a mask (x along axis 0) on a map drawn by show_field (axes in d_p)."""
    nx, ny = mask.shape
    x, y = (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy
    return ax.contour(x, y, np.asarray(mask, dtype=float).T, levels=[0.5], colors=color, linewidths=linewidth)


def mark_cells(ax, cells, dx, dy, marker="x", color="yellow", markersize=10, markeredgewidth=1.5):
    """Mark grid cells, given as (i, j) indices, on a map drawn by show_field (axes in d_p)."""
    for i, j in cells:
        ax.plot((i + 0.5) * dx, (j + 0.5) * dy, marker=marker, color=color, markersize=markersize,
                markeredgewidth=markeredgewidth, linestyle="none")
