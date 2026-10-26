"""
Final neon renders of the J_z wheel, spiral and rings (from gallery.py's composite_*_Jz.npy).

Adds three finishing touches on top of the colour map:
  * glow ("bloom"): bright current sheets get a soft halo, like real neon tubes
  * contrast: GAMMA closer to 1 keeps the gaps between sheets black, so the sheets glow
  * vignette: corners slightly darkened to pull the eye to the centre

Outputs (in OUT_DIR)
  neon_compare_<geom>.png        every palette, plain (top row) vs finished (bottom row)
  neon_<geom>_<palette>_<style>_<N>px.png   full-size images for every palette in
                                  FINAL_PALETTES and every style in FINAL_STYLES
                                  (a .jpg copy is added if the PNG is over 20 MB)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.ndimage import gaussian_filter

# shared code and paths (see README.md and shared/)
import sys
from pathlib import Path
REPO = next(p for p in Path(__file__).resolve().parents if (p / "shared" / "paths.py").exists())
sys.path.insert(0, str(REPO))
from shared.paths import results_dir
RESULTS = results_dir("dirac_images")

# ------------------------------------------------------------------ settings
OUT_DIR = str(RESULTS / "gallery_bp0.0625")            # where gallery.py wrote composite_<geom>_Jz.npy
GEOMS = ["wheel", "spiral", "rings"]
FINAL_PALETTES = ["neon_hot", "neon_ice_rose"]   # palettes for the full-size images
FINAL_STYLES = ["plain"]                # "plain" (gamma 0.6, no glow, no vignette) and/or "finished"
COMPARE = False                         # also make the comparison sheets

GAMMA = 0.8                             # 0.6 = everything lit, 1.0 = only strong sheets glow
CLIP_PCT = 99.5                         # colour range = this percentile of |J_z|
BLOOM = 0.7                             # glow strength (0 = off)
BLOOM_SIGMA = 0.0025                    # glow width as a fraction of the image size
BLOOM_THRESHOLD = 0.3                   # only parts brighter than this glow
VIGNETTE = 0.3                          # corner darkening (0 = off)
OUT_PX = None                           # None = native resolution (4096 px, best for zooming in)
THUMB_PX = 600

# neon family: negative current -> black centre -> positive current
PALETTES = {
    "neon":            ["#b8f6ff", "#00a8ff", "#08306b", "#000000", "#6b1d08", "#ff6a00", "#ffe39a"],
    "neon_hot":        ["#ffffff", "#9ff3ff", "#00b8ff", "#003a8c", "#000000", "#7a1f00", "#ff6a00",
                        "#ffd38a", "#ffffff"],
    "neon_synth":      ["#e6ffff", "#3ff0ff", "#0077ff", "#0a1a4a", "#000000", "#4a0a40", "#ff1fbf",
                        "#ff9fe6", "#fff0fb"],
    "neon_teal_gold":  ["#e8fff9", "#2ef2c7", "#008f7a", "#00332b", "#000000", "#3d2a00", "#e0a100",
                        "#ffd95e", "#fff7d6"],
    "neon_violet_amber": ["#f4e9ff", "#c18cff", "#7b2dff", "#250a4d", "#000000", "#4d2600", "#ff9a1f",
                          "#ffd59e", "#fff6e8"],
    "neon_ice_rose":   ["#f0fbff", "#7fdcff", "#1e7bff", "#0b1f4d", "#000000", "#4d0b26", "#ff2d6f",
                        "#ff9fbf", "#fff0f5"],
}
# ------------------------------------------------------------------------------


def render(v, cmap, gamma, bloom, sigma_px, vignette):
    """v: J_z composite (NaN = empty hub). Returns RGB float32, row 0 = bottom."""
    nan = np.isnan(v)
    vmax = np.nanpercentile(np.abs(v), CLIP_PCT)
    s = np.sign(v) * np.clip(np.abs(np.nan_to_num(v)) / vmax, 0, 1) ** gamma   # in [-1, 1]
    rgb = cmap(0.5 + 0.5 * s)[..., :3].astype(np.float32)
    rgb[nan] = 0
    if bloom > 0:
        lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        bright = rgb * np.clip((lum - BLOOM_THRESHOLD) / (1 - BLOOM_THRESHOLD), 0, 1)[..., None]
        halo = np.stack([gaussian_filter(bright[..., c], sigma_px, mode="wrap") for c in range(3)], -1)
        rgb = 1 - (1 - rgb) * (1 - np.clip(bloom * 2.5 * halo, 0, 1))      # "screen" blend
    if vignette > 0:
        ny, nx = v.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        r = np.hypot((xx - nx / 2) / (nx / 2), (yy - ny / 2) / (ny / 2)) / np.sqrt(2)
        rgb *= (1 - vignette * r ** 2)[..., None]
    return np.clip(rgb, 0, 1)


def save(rgb, path):
    """PNG at full quality (for zooming); plus a .jpg copy if the PNG is over the 20 MB limit."""
    img = Image.fromarray((rgb[::-1] * 255).astype(np.uint8), "RGB")
    if OUT_PX and img.width != OUT_PX:
        img = img.resize((OUT_PX, round(OUT_PX * img.height / img.width)), Image.LANCZOS)
    img.save(path, dpi=(300, 300))
    mb = os.path.getsize(path) / 1e6
    note = ""
    if mb > 20:
        img.save(path[:-4] + ".jpg", quality=95, dpi=(300, 300))
        note = f"; over 20 MB, so also saved {os.path.basename(path[:-4])}.jpg " \
               f"({os.path.getsize(path[:-4] + '.jpg') / 1e6:.1f} MB) for submission"
    print(f"  saved {path} ({img.width} px, {mb:.1f} MB{note})")


def main():
    cmaps = {k: LinearSegmentedColormap.from_list(k, v) for k, v in PALETTES.items()}
    for geom in GEOMS:
        fn = os.path.join(OUT_DIR, f"composite_{geom}_Jz.npy")
        if not os.path.exists(fn):
            print(f"missing {fn}: run gallery.py with SAVE_COMPOSITES = True")
            continue
        data = np.load(fn).astype(np.float32)
        n = data.shape[0]

        if COMPARE:
            step = max(1, n // (2 * THUMB_PX))          # render at 2x thumbnail size, then shrink
            th = data[::step, ::step]
            sig = BLOOM_SIGMA * th.shape[0]

            def small(rgb):
                img = Image.fromarray((rgb[::-1] * 255).astype(np.uint8), "RGB")
                return np.asarray(img.resize((THUMB_PX, THUMB_PX), Image.LANCZOS))
            cols = len(cmaps)
            fig, axs = plt.subplots(2, cols, figsize=(3.3 * cols, 7.2), facecolor="black")
            for j, (name, cm) in enumerate(cmaps.items()):
                axs[0, j].imshow(small(render(th, cm, 0.6, 0, sig, 0)))
                axs[1, j].imshow(small(render(th, cm, GAMMA, BLOOM, sig, VIGNETTE)))
                axs[0, j].set_title(name, color="w", fontsize=12)
                for i in range(2):
                    axs[i, j].axis("off")
            axs[0, 0].text(-0.08, 0.5, "plain", color="w", rotation=90, va="center",
                           transform=axs[0, 0].transAxes, fontsize=12)
            axs[1, 0].text(-0.08, 0.5, "finished", color="w", rotation=90, va="center",
                           transform=axs[1, 0].transAxes, fontsize=12)
            fig.suptitle(f"{geom}: plain (gamma 0.6, no glow)  vs  finished (gamma {GAMMA}, glow "
                         f"{BLOOM}, vignette {VIGNETTE})", color="w", fontsize=13)
            fig.tight_layout(rect=(0.02, 0, 1, 0.95))
            path = os.path.join(OUT_DIR, f"neon_compare_{geom}.png")
            fig.savefig(path, dpi=120, facecolor="black")
            plt.close(fig)
            print(f"  saved {path}")

        styles = {"plain": (0.6, 0.0, 0.0), "finished": (GAMMA, BLOOM, VIGNETTE)}
        for pal in FINAL_PALETTES:
            for style in FINAL_STYLES:
                gamma, bloom, vignette = styles[style]
                rgb = render(data, cmaps[pal], gamma, bloom, BLOOM_SIGMA * n, vignette)
                px = OUT_PX or n
                save(rgb, os.path.join(OUT_DIR, f"neon_{geom}_{pal}_{style}_{px}px.png"))


if __name__ == "__main__":
    main()