"""
Palette explorer for the time pictures made by gallery.py (wheel / spiral / rings).

gallery.py saves the raw data as composite_<geometry>_<Jz|Bperp>.npy. This script
re-colours them with many colour maps WITHOUT re-reading the simulation, and writes

  palettes_<geometry>_<quantity>.png   contact sheet: the same image in every colour map
  final_<geometry>_<quantity>_<cmap>.png   full-size versions of the maps listed in FINAL

quantity: "absJz" (current strength), "Jz" (signed current, diverging maps), "Bperp".
Workflow: run once, look at the contact sheets, put your favourites in FINAL, run again.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from PIL import Image

# shared code and paths (see README.md and shared/)
import sys
from pathlib import Path
REPO = next(p for p in Path(__file__).resolve().parents if (p / "shared" / "paths.py").exists())
sys.path.insert(0, str(REPO))
from shared.paths import results_dir
RESULTS = results_dir("dirac_images")

# ------------------------------------------------------------------ settings
OUT_DIR = str(RESULTS / "gallery_bp0.0625")          # where gallery.py wrote composite_*.npy
GEOMS = ["spiral", "wheel", "rings"]  # which pictures to make contact sheets for
QUANTITIES = ["absJz", "Jz", "Bperp"]
THUMB_PX = 700                        # thumbnail size in the contact sheets
GAMMA = {"absJz": 0.5, "Jz": 0.6, "Bperp": 0.5}   # power-law scaling (1 = linear)
CLIP_PCT = 99.5
FINAL = []                            # e.g. [("spiral", "absJz", "nebula"), ("wheel", "Jz", "fire_ice")]
OUT_PX = 3000
# ------------------------------------------------------------------------------


def cmap(name, colours):
    return LinearSegmentedColormap.from_list(name, colours)


# sequential maps (dark -> bright) for |J_z| and |B_perp|
CUSTOM_SEQ = {
    "nebula":    ["#000000", "#1a0033", "#4a0a77", "#a01a7d", "#ec4067", "#ffd0c0"],
    "aurora":    ["#000000", "#002b36", "#00695c", "#00c853", "#b2ff59", "#f8ffd0"],
    "glacier":   ["#000000", "#0b1d51", "#1565c0", "#29b6f6", "#b3e5fc", "#ffffff"],
    "lava":      ["#000000", "#3b0000", "#9b0000", "#ff3d00", "#ff9100", "#fff0b0"],
    "synthwave": ["#0d0221", "#261447", "#6b1e8f", "#ff2a6d", "#ff9e64", "#fffb96"],
    "cyberpunk": ["#000000", "#001f3f", "#00b3ff", "#c400ff", "#ff00a0", "#ffffff"],
    "deep_sea":  ["#000000", "#001233", "#023e7d", "#0077b6", "#48cae4", "#e0fbff"],
    "emerald":   ["#000000", "#002614", "#00552e", "#00a86b", "#7fffd4", "#f0fff0"],
    "gold":      ["#000000", "#2b1d00", "#6b4e00", "#c9a227", "#ffe066", "#fffbe6"],
    "ultraviolet": ["#000000", "#12002b", "#3d00a8", "#7b2fff", "#c77dff", "#f3e8ff"],
}
BUILTIN_SEQ = ["magma", "inferno", "plasma", "viridis", "cividis", "cubehelix", "afmhot",
               "gist_heat", "CMRmap", "gnuplot2", "turbo", "bone"]

# diverging maps (negative <- dark centre -> positive) for signed J_z
CUSTOM_DIV = {
    "neon":        ["#b8f6ff", "#00a8ff", "#08306b", "#000000", "#6b1d08", "#ff6a00", "#ffe39a"],
    "fire_ice":    ["#e0ffff", "#00bfff", "#0040ff", "#000000", "#ff2000", "#ff9000", "#ffff80"],
    "aurora_split": ["#d4ffea", "#00e676", "#00523a", "#000000", "#4a0057", "#e040fb", "#ffd6ff"],
    "synth_div":   ["#c0fff8", "#00e5ff", "#0d3b66", "#000000", "#5a0a3d", "#ff2a9d", "#ffe0f0"],
    "gold_violet": ["#f3e5ff", "#9b4dff", "#3a0ca3", "#000000", "#7a4b00", "#ffb300", "#fff3c4"],
    "emerald_ruby": ["#e0fff4", "#00c896", "#004d3a", "#000000", "#5c0011", "#ff3355", "#ffe0e6"],
}
BUILTIN_DIV = ["berlin", "managua", "vanimo", "RdBu_r", "coolwarm", "PuOr_r", "seismic",
               "Spectral_r", "twilight_shifted"]


def available(builtin, custom):
    maps = {}
    for name in builtin:
        try:
            maps[name] = plt.get_cmap(name)
        except ValueError:      # e.g. berlin/managua/vanimo need matplotlib >= 3.10
            pass
    maps.update({k: cmap(k, v) for k, v in custom.items()})
    return maps


def load(geom, quantity):
    fn = os.path.join(OUT_DIR, f"composite_{geom}_{'Bperp' if quantity == 'Bperp' else 'Jz'}.npy")
    if not os.path.exists(fn):
        return None
    return np.load(fn, mmap_mode="r")


def colour(v, quantity, cm):
    """v: 2D data (NaN = empty hub) -> RGB in [0,1], row 0 = bottom."""
    nan = np.isnan(v)
    if quantity == "Jz":
        vmax = np.nanpercentile(np.abs(v), CLIP_PCT)
        s = np.sign(v) * np.clip(np.abs(np.nan_to_num(v)) / vmax, 0, 1) ** GAMMA["Jz"]
        rgb = cm(Normalize(-1, 1)(s))[..., :3]
    else:
        a = np.abs(v) if quantity == "absJz" else v
        vmax = np.nanpercentile(a, CLIP_PCT)
        rgb = cm(PowerNorm(GAMMA[quantity], 0, vmax, clip=True)(np.nan_to_num(a)).filled(0))[..., :3]
    rgb[nan] = 0.0
    return rgb


def save_full(rgb, path):
    img = Image.fromarray((np.clip(rgb[::-1], 0, 1) * 255).astype(np.uint8), "RGB")
    if OUT_PX and img.width != OUT_PX:
        img = img.resize((OUT_PX, round(OUT_PX * img.height / img.width)), Image.LANCZOS)
    img.save(path, dpi=(300, 300))
    if os.path.getsize(path) > 20e6:
        os.remove(path)
        path = path[:-4] + ".jpg"
        img.save(path, quality=95, dpi=(300, 300))
    print(f"  saved {path}")


def main():
    seq = available(BUILTIN_SEQ, CUSTOM_SEQ)
    div = available(BUILTIN_DIV, CUSTOM_DIV)

    for geom in GEOMS:
        for q in QUANTITIES:
            data = load(geom, q)
            if data is None:
                print(f"no composite for {geom}/{q} in {OUT_DIR}: run gallery.py with SAVE_COMPOSITES = True")
                continue
            step = max(1, data.shape[0] // THUMB_PX)
            thumb = np.array(data[::step, ::step], dtype=np.float64)
            maps = div if q == "Jz" else seq
            cols = 5
            rows = int(np.ceil(len(maps) / cols))
            fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.45 * rows), facecolor="white")
            for ax in np.ravel(axs):
                ax.axis("off")
            for ax, (name, cm) in zip(np.ravel(axs), maps.items()):
                ax.imshow(colour(thumb, q, cm), origin="lower", interpolation="none")
                ax.set_title(name, fontsize=12)
            fig.suptitle(f"{geom} - {q}", fontsize=15)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            path = os.path.join(OUT_DIR, f"palettes_{geom}_{q}.png")
            fig.savefig(path, dpi=130)
            plt.close(fig)
            print(f"  saved {path} ({len(maps)} colour maps)")

    for geom, q, name in FINAL:
        data = load(geom, q)
        cm = {**seq, **div}.get(name)
        if data is None or cm is None:
            print(f"skipping {geom}/{q}/{name}: composite or colour map not found")
            continue
        save_full(colour(np.array(data, dtype=np.float64), q, cm),
                  os.path.join(OUT_DIR, f"final_{geom}_{q}_{name}.png"))


if __name__ == "__main__":
    main()