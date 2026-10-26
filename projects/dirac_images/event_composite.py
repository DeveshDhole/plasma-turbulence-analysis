"""
Composite pictures of one reconnection event, two layouts.

--layout column (default)                     --layout corners
   +----------------------+  +-----+           +----------------------------+
   |                      |  | t1  |           | +-----+            +-----+ |
   |  whole simulation    |  +-----+           | | t1  |            | t2  | |
   |  box at T_MAIN,      |  | t2  |           | +-----+    [ ]     +-----+ |
   |  event boxed  [ ]    |  +-----+           | +-----+            +-----+ |
   |                      |  | t3  |           | | t3  |            | t4  | |
   +----------------------+  +-----+           | +-----+            +-----+ |
                                               +----------------------------+
column : three zoomed panels of the event box on the right (earliest at the top)
corners: four zoomed panels in the corners of the whole box, in reading order
         (top-left, top-right, bottom-left, bottom-right; --order clockwise for
         TL, TR, BR, BL). The whole box is shifted periodically to put the event in the
         centre, so the corner panels never cover it.

  python event_composite.py --center 87.98 237.96 --tmain 77 --times 72 77 82 --filter if --kcut 10
  python event_composite.py --center 87.98 237.96 --tmain 77 --layout corners --times 72 77 82 87 --filter if --kcut 10

No zoom lines by default; --connect draws them in either layout (corners: one line from
each corner of the event box to the panel in that corner).
Neon: add --cmap neon (or neon_hot, neon_ice_rose) - black background, white frames,
lines and labels.
Same filtering and colour options as event_videos_v2.py.

Outputs (in OUT_DIR): a clean version (competition) and a _labelled version (axes from 0 at
the bottom-left corner, colour bar; --true-coords for the original simulation coordinates).
--fontsize sets all text sizes (default 16).
"""
import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import FuncNorm, LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image

# shared code and paths (see README.md and shared/)
import sys
from pathlib import Path
REPO = next(p for p in Path(__file__).resolve().parents if (p / "shared" / "paths.py").exists())
sys.path.insert(0, str(REPO))
from shared.paths import NAS_ROOT, results_dir
RESULTS = results_dir("dirac_images")

# ------------------------------------------------------------------ settings
RUN = f"{NAS_ROOT}/DATA/Brms_0.24/beta_p_1/beta_e_1/"
OUT_DIR = str(RESULTS / "composites")
dx = 0.0625
SMOOTH_PSI = 2.0
SMOOTH_J = 2.0                 # default Gaussian filter of J_z [cells]
GAMMA_J = 0.6
CLIP_PCT = 99.5
N_CONTOURS = 18                # field lines across a zoom panel
FONT = 16                      # base font size (ticks, panel times); axis labels and titles +2
IF_VERBOSE = False

NEON = {
    "neon":              ["#b8f6ff", "#00a8ff", "#08306b", "#000000", "#6b1d08", "#ff6a00", "#ffe39a"],
    "neon_soft":         ["#45d4ff", "#0a8fd6", "#06305e", "#000000", "#5a1e05", "#d65a00", "#ff9a3c"],
    "neon_hot":          ["#ffffff", "#9ff3ff", "#00b8ff", "#003a8c", "#000000", "#7a1f00", "#ff6a00",
                          "#ffd38a", "#ffffff"],
    "neon_ice_rose":     ["#f0fbff", "#7fdcff", "#1e7bff", "#0b1f4d", "#000000", "#4d0b26", "#ff2d6f",
                          "#ff9fbf", "#fff0f5"],
    "neon_synth":        ["#e6ffff", "#3ff0ff", "#0077ff", "#0a1a4a", "#000000", "#4a0a40", "#ff1fbf",
                          "#ff9fe6", "#fff0fb"],
    "neon_teal_gold":    ["#e8fff9", "#2ef2c7", "#008f7a", "#00332b", "#000000", "#3d2a00", "#e0a100",
                          "#ffd95e", "#fff7d6"],
    "neon_violet_amber": ["#f4e9ff", "#c18cff", "#7b2dff", "#250a4d", "#000000", "#4d2600", "#ff9a1f",
                          "#ffd59e", "#fff6e8"],
}
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------ your iterative filtering
def k_to_mask_length(k, dx=0.0625, angular=True, prefactor=1.0, min_len=3):
    """FIF mask length (grid points) for a cut-off wavenumber k (angular: wavelength = 2 pi / k)."""
    wavelength = (2.0 * np.pi / k) if angular else (1.0 / k)
    L = prefactor * wavelength / dx
    return max(min_len, int(round(L)))


def iterative_filtering1DWithKCutOff(field, Xi=1.6, k=10):
    import fifpy
    fif = fifpy.MIF(NIMFs=1, alpha=30, Xi=Xi)
    mask_length = k_to_mask_length(k)
    fif.run(field, M=[mask_length])
    if IF_VERBOSE:
        print(fif.IMC.shape)
    return np.sum(fif.IMC[1:], axis=0)     # large-scale part; IMC[0] (finest IMF) removed


# ------------------------------------------------------------------ fields
def read(tag, t):
    fn = os.path.join(RUN, f"{tag}_ApJ_t{t}.h5")
    if not os.path.exists(fn):
        return None
    with h5py.File(fn, "r") as f:
        return f["DS1"][:].T.astype(np.float64)


def fields(t, a, transpose):
    """psi and filtered J_z at time t (x along columns, y along rows)."""
    bx, by = read("Bx", t), read("By", t)
    if bx is None or by is None:
        raise SystemExit(f"no Bx/By for t = {t} in {RUN}")
    if a.filter == "if":
        bx = np.asarray(iterative_filtering1DWithKCutOff(bx, Xi=a.xi, k=a.kcut), dtype=np.float64)
        by = np.asarray(iterative_filtering1DWithKCutOff(by, Xi=a.xi, k=a.kcut), dtype=np.float64)
    if transpose:
        bx, by = bx.T, by.T
    n = bx.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = k[None, :], k[:, None]
    k2 = kx ** 2 + ky ** 2
    bxk, byk = np.fft.fft2(bx), np.fft.fft2(by)
    pk = (-1j * ky * bxk + 1j * kx * byk) / np.where(k2 == 0, 1, k2)
    pk[0, 0] = 0
    psi = np.real(np.fft.ifft2(pk * np.exp(-0.5 * (SMOOTH_PSI * dx) ** 2 * k2)))
    filt = np.exp(-0.5 * (a.smooth * dx) ** 2 * k2)
    if a.highpass > 0:
        filt = filt - np.exp(-0.5 * a.highpass ** 2 * k2)
    jz = np.real(np.fft.ifft2((1j * kx * byk - 1j * ky * bxk) * filt))
    b_rms = float(np.sqrt(np.mean(bx ** 2 + by ** 2)))
    return psi, jz, b_rms


def orientation(t):
    bx, by = read("Bx", t), read("By", t)
    n = bx.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = k[None, :], k[:, None]
    dc = np.std(np.real(np.fft.ifft2(1j * kx * np.fft.fft2(bx) + 1j * ky * np.fft.fft2(by))))
    dr = np.std(np.real(np.fft.ifft2(1j * ky * np.fft.fft2(bx) + 1j * kx * np.fft.fft2(by))))
    return dr < dc


# ------------------------------------------------------------------ drawing
def palette(name):
    return LinearSegmentedColormap.from_list(name, NEON[name]) if name in NEON else plt.get_cmap(name)


def show(ax, data, extent, a, vmax):
    if a.cmap in NEON:
        s = np.sign(data) * np.clip(np.abs(data) / vmax, 0, 1) ** a.gamma
        ax.imshow(palette(a.cmap)(0.5 + 0.5 * s)[..., :3], origin="lower", extent=extent, interpolation="bilinear")
        return None
    f = lambda v: np.sign(v) * np.abs(v) ** a.gamma
    finv = lambda v: np.sign(v) * np.abs(v) ** (1 / a.gamma)
    return ax.imshow(data, origin="lower", extent=extent, cmap=palette(a.cmap), interpolation="bilinear",
                     norm=FuncNorm((f, finv), vmin=-vmax, vmax=vmax))


def window(arr, x, y, zoom, n):
    m = int(round(zoom / dx))
    c0, r0 = int(round(x / dx)) - m // 2, int(round(y / dx)) - m // 2
    rr, cc = np.arange(r0, r0 + m) % n, np.arange(c0, c0 + m) % n
    lx_ = (c0 + np.arange(m)) * dx - x
    ly_ = (r0 + np.arange(m)) * dx - y
    return arr[np.ix_(rr, cc)], lx_, ly_


def zoom_panel(p_, t, a, data, neon, fg, bg, n):
    """One zoomed panel of the event box at time t."""
    x, y, zoom = a.center[0], a.center[1], a.zoom
    psi, jz = data[t]
    jw, lx_, ly_ = window(jz, x, y, zoom, n)
    pw, _, _ = window(psi, x, y, zoom, n)
    show(p_, jw, [lx_[0], lx_[-1] + dx, ly_[0], ly_[-1] + dx], a, data["vmax_zoom"])
    if a.lines:
        dpsi = data["b_rms"] * zoom / N_CONTOURS
        lo, hi = np.floor(pw.min() / dpsi), np.ceil(pw.max() / dpsi)
        st = max(1, len(lx_) // 384)
        if hi - lo < 400:
            X, Y = np.meshgrid(lx_[::st], ly_[::st])
            p_.contour(X, Y, pw[::st, ::st], levels=np.arange(lo, hi + 1) * dpsi,
                       colors="white" if neon else "k", linewidths=0.5,
                       alpha=0.35 if neon else 0.6, linestyles="solid")
    p_.set_xlim(-zoom / 2, zoom / 2)
    p_.set_ylim(-zoom / 2, zoom / 2)
    p_.set_xticks([])
    p_.set_yticks([])
    for sp in p_.spines.values():
        sp.set_edgecolor(fg)
        sp.set_linewidth(1.6)                   # same frame for every time
    p_.text(0.04, 0.95, f"t = {t}", transform=p_.transAxes, va="top", fontsize=a.fontsize, color=fg,
            bbox=dict(fc=bg, ec="none", alpha=0.6, pad=2))


def whole_box(ax, a, data, labelled, fg):
    """Whole-box J_z at T_MAIN (periodically shifted by data['shift']) with the event boxed."""
    _, jz_main, L, n, shift = data["main"]
    x, y, zoom = a.center[0], a.center[1], a.zoom
    jm = np.roll(jz_main, (int(round(shift[1] / dx)), int(round(shift[0] / dx))), axis=(0, 1))
    im = show(ax, jm, [0, L, 0, L], a, data["vmax"])
    xs, ys = (x + shift[0]) % L, (y + shift[1]) % L
    ax.add_patch(Rectangle((xs - zoom / 2, ys - zoom / 2), zoom, zoom, fill=False, ec=fg, lw=2))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    if labelled:
        if a.true_coords:   # original simulation coordinates (wrap around where the box was shifted)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p_: f"{(v - shift[0]) % L:.0f}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p_: f"{(v - shift[1]) % L:.0f}"))
        ax.set_xlabel(r"$x/d_p$", color=fg, fontsize=a.fontsize + 2)
        ax.set_ylabel(r"$y/d_p$", color=fg, fontsize=a.fontsize + 2)
        ax.tick_params(colors=fg, labelsize=a.fontsize)
        ax.set_title(f"t = {a.tmain}", color=fg, fontsize=a.fontsize + 2)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(fg)
    return im, xs, ys


def colour_bar(fig, im, a, data, rect, fg):
    if im is None:   # neon images are drawn as RGB: build the matching colour bar
        f = lambda v: np.sign(v) * np.abs(v) ** a.gamma
        finv = lambda v: np.sign(v) * np.abs(v) ** (1 / a.gamma)
        im = plt.cm.ScalarMappable(norm=FuncNorm((f, finv), vmin=-data["vmax"], vmax=data["vmax"]),
                                   cmap=palette(a.cmap))
    cb = fig.colorbar(im, cax=fig.add_axes(rect))
    cb.set_label(r"$J_z$", color=fg, fontsize=a.fontsize + 2)
    cb.ax.tick_params(colors=fg, labelsize=a.fontsize)


def save(fig, path, labelled, bg):
    if labelled:   # include everything (axis labels, colour-bar label) with a small margin
        fig.savefig(path, dpi=350, facecolor=bg, bbox_inches="tight", pad_inches=0.2)
    else:
        fig.savefig(path, dpi=350, facecolor=bg)
    plt.close(fig)
    mb = os.path.getsize(path) / 1e6
    msg = f"  saved {path} ({mb:.1f} MB)"
    if mb > 20 and not labelled:
        Image.open(path).convert("RGB").save(path[:-4] + ".jpg", quality=95, dpi=(350, 350))
        msg += f"; also {os.path.basename(path[:-4])}.jpg for the 20 MB limit"
    print(msg)


def make_column(a, data, labelled, path):
    neon = a.cmap in NEON
    fg, bg = ("white", "black") if neon else ("black", "white")
    n = data["main"][3]
    S, g = 8.0, 0.6 if not a.connect else 1.2       # main size, gap to the column [inches]
    gap = 0.12
    ph = (S - 2 * gap) / 3
    mx, my = (1.3, 1.0) if labelled else (0.15, 0.15)
    extra = 2.0 if labelled else 0.15
    W, H = mx + S + g + ph + extra, my * 2 + S
    fig = plt.figure(figsize=(W, H), facecolor=bg)
    ax = fig.add_axes([mx / W, my / H, S / W, S / H])
    im, xs, ys = whole_box(ax, a, data, labelled, fg)
    pax = []
    for i, t in enumerate(a.times):
        p_ = fig.add_axes([(mx + S + g) / W, (my + S - (i + 1) * ph - i * gap) / H, ph / W, ph / H])
        zoom_panel(p_, t, a, data, neon, fg, bg, n)
        pax.append(p_)
    if a.connect:
        z = a.zoom
        for (cx, cy), (pa, py_) in (((xs + z / 2, ys + z / 2), (pax[0], 1.0)),
                                    ((xs + z / 2, ys - z / 2), (pax[-1], 0.0))):
            fig.add_artist(ConnectionPatch(xyA=(cx, cy), coordsA=ax.transData, xyB=(0, py_),
                                           coordsB=pa.transAxes, color=fg, lw=1.2, alpha=0.8))
    if labelled:
        colour_bar(fig, im, a, data, [(mx + S + g + ph + 0.35) / W, my / H, 0.22 / W, S / H], fg)
    save(fig, path, labelled, bg)


def make_corners(a, data, labelled, path):
    neon = a.cmap in NEON
    fg, bg = ("white", "black") if neon else ("black", "white")
    n = data["main"][3]
    S = 10.0
    mx, my = (1.3, 1.0) if labelled else (0.0, 0.0)
    extra = 2.0 if labelled else 0.0
    W, H = mx + S + extra, my * 2 + S
    fig = plt.figure(figsize=(W, H), facecolor=bg)
    ax = fig.add_axes([mx / W, my / H, S / W, S / H])
    im, xs, ys = whole_box(ax, a, data, labelled, fg)
    f, m, z = a.inset, 0.015, a.zoom
    # corner -> (panel position, the event-box corner facing it, the panel corner facing the box)
    corner = {"tl": ((m, 1 - m - f), (xs - z / 2, ys + z / 2), (1, 0)),
              "tr": ((1 - m - f, 1 - m - f), (xs + z / 2, ys + z / 2), (0, 0)),
              "bl": ((m, m), (xs - z / 2, ys - z / 2), (1, 1)),
              "br": ((1 - m - f, m), (xs + z / 2, ys - z / 2), (0, 1))}
    order = ["tl", "tr", "br", "bl"] if a.order == "clockwise" else ["tl", "tr", "bl", "br"]
    for t, c in zip(a.times, order):
        (px_, py_), box_corner, panel_corner = corner[c]
        p_ = ax.inset_axes([px_, py_, f, f])
        zoom_panel(p_, t, a, data, neon, fg, bg, n)
        if a.connect:
            fig.add_artist(ConnectionPatch(xyA=box_corner, coordsA=ax.transData, xyB=panel_corner,
                                           coordsB=p_.transAxes, color=fg, lw=1.3, alpha=0.85))
    if labelled:
        colour_bar(fig, im, a, data, [(mx + S + 0.35) / W, my / H, 0.22 / W, S / H], fg)
    save(fig, path, labelled, bg)


def main():
    ap = argparse.ArgumentParser(description="Whole box + three zoomed panels of one event.")
    ap.add_argument("--center", nargs=2, type=float, required=True, metavar=("X", "Y"), help="event [d_p]")
    ap.add_argument("--tmain", type=int, required=True, help="time of the whole-box panel")
    ap.add_argument("--layout", choices=["column", "corners"], default="column")
    ap.add_argument("--times", nargs="+", type=int,
                    help="panel times: 3 for column (default tmain-5 tmain tmain+5), "
                         "4 for corners (default tmain-5 tmain tmain+5 tmain+10)")
    ap.add_argument("--connect", action="store_true", help="zoom lines from the event box to the panels")
    ap.add_argument("--order", choices=["reading", "clockwise"], default="reading",
                    help="corners: reading order (TL TR BL BR) or clockwise (TL TR BR BL)")
    ap.add_argument("--inset", type=float, default=0.3, help="corners: panel size as a fraction of the box")
    ap.add_argument("--fontsize", type=float, default=FONT, help=f"base font size (default {FONT})")
    ap.add_argument("--true-coords", action="store_true",
                    help="labelled version: original simulation coordinates instead of plain 0..L axes")
    ap.add_argument("--zoom", type=float, default=48.0, help="event box size [d_p]")
    ap.add_argument("--cmap", default="RdBu_r", help=f"RdBu_r (default), {', '.join(NEON)} or any matplotlib map")
    ap.add_argument("--filter", choices=["gauss", "if"], default="gauss")
    ap.add_argument("--kcut", type=float, help="angular cut-off [rad/d_p] for --filter if")
    ap.add_argument("--xi", type=float, default=1.6)
    ap.add_argument("--smooth", type=float, default=None, help=f"Gaussian [cells] (default {SMOOTH_J:g}, 0 with if)")
    ap.add_argument("--highpass", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=GAMMA_J,
                    help="colour scaling: 0.6 lights up weak currents, 0.9-1 keeps them dark (less glowing)")
    ap.add_argument("--clip", type=float, default=CLIP_PCT,
                    help="colour range = this percentile of |J_z|; 99.9 = fewer saturated bright pixels")
    ap.add_argument("--lines", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--zoom-scale", choices=["own", "main"], default="own",
                    help="colour range of the panels: their own (default, more contrast) or the main panel's")
    a = ap.parse_args()
    if a.smooth is None:
        a.smooth = 0.0 if a.filter == "if" else SMOOTH_J
    if a.filter == "if" and a.kcut is None:
        raise SystemExit("--filter if needs --kcut")
    need = 3 if a.layout == "column" else 4
    if a.times is None:
        a.times = [a.tmain - 5, a.tmain, a.tmain + 5] + ([a.tmain + 10] if need == 4 else [])
    if len(a.times) != need:
        raise SystemExit(f"--layout {a.layout} needs {need} --times, got {len(a.times)}")
    a.times = sorted(a.times)
    os.makedirs(OUT_DIR, exist_ok=True)

    transpose = orientation(a.tmain)
    data = {}
    for t in sorted(set(a.times + [a.tmain])):
        print(f"t = {t}: reading and filtering ...", flush=True)
        psi, jz, b_rms = fields(t, a, transpose)
        data[t] = (psi, jz)
        if t == a.tmain:
            n = jz.shape[0]
            L = n * dx
            data["b_rms"] = b_rms
            data["vmax"] = float(np.percentile(np.abs(jz), a.clip))
    psi_m, jz_m = data[a.tmain]
    x, y, z = a.center[0], a.center[1], a.zoom
    shift = [0.0, 0.0]
    if a.layout == "corners":                # event in the centre, clear of the corner panels
        shift = [L / 2 - x, L / 2 - y]
        if a.inset + 0.015 > 0.5 - z / (2 * L):
            print(f"warning: --inset {a.inset:g} is large enough for the corner panels to overlap the event box")
    else:
        for i, c in enumerate((x, y)):       # periodic shift so the event box is complete
            if c - z / 2 < 1 or c + z / 2 > L - 1:
                shift[i] = L / 2 - c
    data["main"] = (psi_m, jz_m, L, n, shift)
    if a.zoom_scale == "main":
        data["vmax_zoom"] = data["vmax"]
    else:   # one scale shared by the three panels (so they can be compared)
        data["vmax_zoom"] = max(float(np.percentile(np.abs(window(data[t][1], x, y, z, n)[0]), a.clip))
                                for t in a.times)

    style = f"{a.cmap}" + (f"_if{a.kcut:g}" if a.filter == "if" else "") \
        + (f"_s{a.smooth:g}" if a.smooth != (0.0 if a.filter == "if" else SMOOTH_J) else "") \
        + (f"_hp{a.highpass:g}" if a.highpass else "") + ("" if a.lines else "_nolines") \
        + (f"_g{a.gamma:g}" if a.gamma != GAMMA_J else "") + (f"_c{a.clip:g}" if a.clip != CLIP_PCT else "")
    layout = a.layout + ("_clockwise" if a.layout == "corners" and a.order == "clockwise" else "") \
        + ("_connect" if a.connect else "")
    base = os.path.join(OUT_DIR, f"composite_{layout}_x{x:05.1f}_y{y:05.1f}_t{a.tmain:03d}_z{z:g}_"
                                 f"t{'-'.join(str(t) for t in a.times)}_{style}")
    make = make_column if a.layout == "column" else make_corners
    make(a, data, False, base + ".png")
    make(a, data, True, base + "_labelled.png")


if __name__ == "__main__":
    main()