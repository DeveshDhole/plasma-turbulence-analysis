"""
Labelled versions of the turbulent clock and the rings, from the saved composites
(composite_wheel_Jz.npy, composite_rings_Jz.npy) - no simulation data is read.

Each picture gets: x/d_p and y/d_p axes starting from 0 at the bottom-left corner,
a J_z colour bar, large fonts, black background. No time labels.
Colours are identical to the unlabelled version if PALETTE / GAMMA_J / CLIP_PCT match.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import FuncNorm, LinearSegmentedColormap

# shared code and paths (see README.md and shared/)
import sys
from pathlib import Path
REPO = next(p for p in Path(__file__).resolve().parents if (p / "shared" / "paths.py").exists())
sys.path.insert(0, str(REPO))
from shared.paths import results_dir
RESULTS = results_dir("dirac_images")

# ------------------------------------------------------------------ settings
OUT_DIR = str(RESULTS / "clock_rings")            # folder with composite_wheel_Jz.npy / composite_rings_Jz.npy
dx = 0.0625                        # cell size [d_p] -> box size = 4096 * 0.0625 = 256 d_p
PALETTE = "neon"
GAMMA_J = 0.6                      # same scaling as the unlabelled version
CLIP_PCT = 99.5
FONT = 20                          # tick labels; axis labels and the colour-bar label are larger

PALETTES = {
    "neon":          ["#b8f6ff", "#00a8ff", "#08306b", "#000000", "#6b1d08", "#ff6a00", "#ffe39a"],
    "neon_hot":      ["#ffffff", "#9ff3ff", "#00b8ff", "#003a8c", "#000000", "#7a1f00", "#ff6a00",
                      "#ffd38a", "#ffffff"],
    "neon_soft":     ["#45d4ff", "#0a8fd6", "#06305e", "#000000", "#5a1e05", "#d65a00", "#ff9a3c"],
    "neon_ice_rose": ["#f0fbff", "#7fdcff", "#1e7bff", "#0b1f4d", "#000000", "#4d0b26", "#ff2d6f",
                      "#ff9fbf", "#fff0f5"],
}
# ------------------------------------------------------------------------------


def labelled(v, path):
    n = v.shape[0]
    L = n * dx
    vmax = np.nanpercentile(np.abs(v), CLIP_PCT)
    f = lambda x: np.sign(x) * np.abs(x) ** GAMMA_J
    finv = lambda x: np.sign(x) * np.abs(x) ** (1 / GAMMA_J)
    norm = FuncNorm((f, finv), vmin=-vmax, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list(PALETTE, PALETTES[PALETTE])
    rgb = cmap(norm(np.nan_to_num(v)))[..., :3]
    rgb[np.isnan(v)] = 0                                   # empty hub of the clock stays black

    fig = plt.figure(figsize=(12.5, 10.5), facecolor="black")
    ax = fig.add_axes([0.12, 0.11, 0.70, 0.84], facecolor="black")
    ax.imshow(rgb, origin="lower", extent=[0, L, 0, L], interpolation="bilinear")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel(r"$x\,/\,d_p$", color="white", fontsize=FONT + 4)
    ax.set_ylabel(r"$y\,/\,d_p$", color="white", fontsize=FONT + 4)
    ax.tick_params(colors="white", labelsize=FONT, width=1.5, length=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("white")
        sp.set_linewidth(1.5)
    cax = fig.add_axes([0.85, 0.11, 0.025, 0.84])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"$J_z$", color="white", fontsize=FONT + 8, labelpad=12)
    cb.ax.tick_params(colors="white", labelsize=FONT, width=1.5, length=6)
    cb.outline.set_edgecolor("white")
    fig.savefig(path, dpi=300, facecolor="black", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def main():
    for geom, name in (("wheel", "clock"), ("rings", "rings")):
        fn = os.path.join(OUT_DIR, f"composite_{geom}_Jz.npy")
        if not os.path.exists(fn):
            print(f"missing {fn} - skipped")
            continue
        labelled(np.load(fn).astype(np.float64), os.path.join(OUT_DIR, f"{name}_Jz_{PALETTE}_labelled.png"))


if __name__ == "__main__":
    main()