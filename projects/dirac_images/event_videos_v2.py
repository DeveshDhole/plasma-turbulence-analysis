"""
Videos (and stills) of reconnection events: a box fixed in space, one frame per output time.

Modes
  default            one video per event in SELECTED (IDs from scan_events.py)
  --ids t190_e3 ...  just these events
  --per site         one video per site (sites.csv from rerender_events.py)
  --center X Y       a box centred anywhere (e.g. an X-point found with find_xpoint.py)

Look
  --cmap neon        neon family (neon, neon_hot, neon_synth, neon_teal_gold,
                     neon_violet_amber, neon_ice_rose) -> dark style, light field lines;
                     any matplotlib name (e.g. RdBu_r) -> inspection style, black field lines
  --smooth 3         Gaussian filter of J_z, sigma in grid cells (noise removal; 0 = none)
  --filter if --kcut K   your iterative filtering (iterative_filtering1DWithKCutOff, defined
                     below, needs fifpy) applied to Bx and By before J_z and the field lines
                     are computed. K is the angular cut-off wavenumber in rad/d_p, as in your
                     compute_radial_spectrum (K = 10 -> mask of 2*pi/10 d_p = 10 cells).
                     No Gaussian on top unless you also give --smooth. --xi sets Xi (1.6).
  --highpass 4       also remove structure larger than ~this many d_p (sheets stand out)
  --glow 0.6         neon glow around bright sheets (0 = off)
  --no-lines         hide the field lines
  --axes             x/y coordinates around the frame
Stills
  --still 77         instead of a video: one clean high-resolution PNG at t = 77
                     (--still-px, default 3000) plus a labelled version for checking

Two stages for videos: frames (parallel over times, each output read once), then MP4
encoding (parallel over videos). Resumable per video and per time.

Example (your event):
  python event_videos_v2.py --center 87.98 237.96 --zoom 48 --cmap neon --smooth 3 --glow 0.6
  python event_videos_v2.py --center 87.98 237.96 --zoom 48 --cmap neon --smooth 3 --glow 0.6 --still 77
"""
import argparse
import csv
import glob
import os
import re

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import FuncNorm, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter

# shared code and paths (see README.md and shared/)
import sys
from pathlib import Path
REPO = next(p for p in Path(__file__).resolve().parents if (p / "shared" / "paths.py").exists())
sys.path.insert(0, str(REPO))
from shared.paths import NAS_ROOT, results_dir
RESULTS = results_dir("dirac_images")

# ------------------------------------------------------------------ settings
RUN = f"{NAS_ROOT}/DATA/Brms_0.24/beta_p_1/beta_e_1/"
SCAN_DIR = str(RESULTS / "events_bp1_be1")            # scan_events.py output (catalog.csv)
SELECTED_DIR = "selected_zoom48"       # rerender_events.py output inside SCAN_DIR (sites.csv)
T_FIRST, T_LAST = 0, 300
dx = 0.0625
ZOOM = 48.0                            # box size [d_p]
FPS = 15
FRAME_PX = 720
REF_T = 200                            # time that fixes the colour scale and field-line spacing
N_CONTOURS = 24
SMOOTH_PSI = 2.0
SMOOTH_J = 2.0                         # default Gaussian filter of J_z [cells]
GAMMA_J = 0.6
CMAP = "RdBu_r"                        # default palette (use --cmap neon for the neon look)
CLIP_PCT = 99.5
IF_VERBOSE = False                     # True: print fif.IMC.shape on every filtering call
KEEP_FRAMES = False
N_WORKERS = "auto"
GB_PER_WORKER = 4.0

NEON = {   # the palettes from the turbulent clock (negative J_z -> black -> positive J_z)
    "neon":              ["#b8f6ff", "#00a8ff", "#08306b", "#000000", "#6b1d08", "#ff6a00", "#ffe39a"],
    "neon_soft":         ["#45d4ff", "#0a8fd6", "#06305e", "#000000", "#5a1e05", "#d65a00", "#ff9a3c"],
    "neon_hot":          ["#ffffff", "#9ff3ff", "#00b8ff", "#003a8c", "#000000", "#7a1f00", "#ff6a00",
                          "#ffd38a", "#ffffff"],
    "neon_synth":        ["#e6ffff", "#3ff0ff", "#0077ff", "#0a1a4a", "#000000", "#4a0a40", "#ff1fbf",
                          "#ff9fe6", "#fff0fb"],
    "neon_teal_gold":    ["#e8fff9", "#2ef2c7", "#008f7a", "#00332b", "#000000", "#3d2a00", "#e0a100",
                          "#ffd95e", "#fff7d6"],
    "neon_violet_amber": ["#f4e9ff", "#c18cff", "#7b2dff", "#250a4d", "#000000", "#4d2600", "#ff9a1f",
                          "#ffd59e", "#fff6e8"],
    "neon_ice_rose":     ["#f0fbff", "#7fdcff", "#1e7bff", "#0b1f4d", "#000000", "#4d0b26", "#ff2d6f",
                          "#ff9fbf", "#fff0f5"],
}

SELECTED = """
t040_e1 t041_e1 t042_e2 t043_e1 t044_e2 t044_e3 t045_e1 t045_e3 t046_e1 t046_e2 t046_e3 t047_e2 t047_e3
t047_e4 t048_e4 t048_e5 t049_e1 t049_e2 t049_e3 t049_e4 t050_e1 t050_e2 t050_e4 t051_e1 t051_e4 t051_e5
t052_e1 t052_e4 t053_e2 t054_e5 t055_e5 t056_e1 t060_e1 t061_e1 t061_e3 t061_e4 t061_e5 t062_e1 t062_e2
t062_e5 t063_e2 t063_e4 t063_e5 t064_e3 t065_e2 t065_e5 t066_e2 t066_e4 t068_e2 t068_e3 t069_e1 t069_e2
t070_e2 t071_e3 t071_e4 t071_e5 t072_e2 t072_e5 t073_e2 t074_e1 t075_e2 t076_e1 t076_e5 t077_e3 t078_e3
t079_e1 t079_e2 t079_e3 t079_e4 t079_e5 t080_e1 t080_e2 t080_e3 t080_e4 t080_e5 t081_e1 t081_e2 t081_e3
t081_e4 t081_e5 t082_e1 t082_e2 t082_e3 t082_e4 t082_e5 t083_e1 t083_e2 t083_e3 t083_e4 t083_e5 t084_e1
t084_e2 t084_e3 t084_e4 t084_e5 t085_e1 t085_e2 t085_e3 t085_e4 t085_e5 t086_e1 t086_e2 t086_e3 t086_e4
t086_e5 t087_e1 t087_e2 t087_e3 t087_e4 t087_e5 t088_e1 t088_e2 t088_e3 t088_e4 t088_e5 t089_e1 t089_e2
t089_e3 t089_e4 t089_e5 t090_e1 t090_e2 t090_e3 t090_e4 t090_e5 t091_e1 t091_e2 t091_e3 t091_e4 t091_e5
t092_e1 t093_e5 t094_e1 t094_e2 t102_e5 t103_e4 t117_e3 t118_e4 t145_e4 t168_e2 t170_e2 t172_e2 t175_e5
t179_e4 t180_e5 t181_e2 t182_e1 t183_e3 t184_e3 t186_e3 t186_e5 t187_e2 t188_e3 t188_e5 t189_e1 t190_e3
t190_e4 t190_e5 t191_e2 t196_e4 t196_e5 t197_e1 t201_e4 t205_e2 t213_e1 t213_e5 t214_e3 t225_e2 t227_e2
t231_e3 t235_e3 t251_e5 t254_e5 t259_e5 t265_e4 t282_e3 t288_e5 t296_e3 t298_e3 t299_e3
"""
# ------------------------------------------------------------------------------


def read(tag, t):
    fn = os.path.join(RUN, f"{tag}_ApJ_t{t}.h5")
    if not os.path.exists(fn):
        return None
    with h5py.File(fn, "r") as f:
        return f["DS1"][:].T.astype(np.float64)


# ------------------------------------------------------------------ your iterative filtering
def k_to_mask_length(k, dx=0.0625, angular=True, prefactor=1.0, min_len=3):
    """
    FIF mask length (in grid points) for a desired cutoff wavenumber k.
    Scales finer than ~this length get pushed into the discarded fine IMF.

    k         : cutoff wavenumber, in the SAME convention/units as your
                compute_radial_spectrum output.
    dx        : grid spacing you passed to compute_radial_spectrum (0.0625 here).
    angular   : True  -> k is angular       (rad/length,   wavelength = 2*pi/k)
                False -> k is ordinary freq  (cycles/length, wavelength = 1/k)
    prefactor : order-1 calibration. 1.0 = geometric estimate; tune it if you
                want the spectral rolloff to land exactly on k.
    min_len   : floor so a tiny mask stays valid.
    """
    wavelength = (2.0 * np.pi / k) if angular else (1.0 / k)   # in the length units of dx
    L = prefactor * wavelength / dx
    return max(min_len, int(round(L)))


def iterative_filtering1DWithKCutOff(field, Xi=1.6, k=10):
    import fifpy                          # imported here, so the Gaussian mode works without fifpy
    fif = fifpy.MIF(NIMFs=1, alpha=30, Xi=Xi)
    mask_length = k_to_mask_length(k)
    fif.run(field, M=[mask_length])       # one mask length per IMF (sequence, not scalar)
    if IF_VERBOSE:
        print(fif.IMC.shape)              # sanity-check the component layout
    return np.sum(fif.IMC[1:], axis=0)    # large-scale part; IMC[0] is the finest IMF removed


def kgrid(n):
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    return k[None, :], k[:, None]


def fields(t, transpose, smooth, highpass, method="gauss", kcut=None, xi=1.6):
    """psi and filtered J_z (x along columns).
    method "if": your iterative filtering on Bx and By first (as in your own scripts).
    Then in Fourier space: Gaussian low-pass (sigma = smooth cells; 0 = none) minus, if
    highpass > 0, the structure larger than ~highpass d_p."""
    bx, by = read("Bx", t), read("By", t)
    if bx is None or by is None:
        return None, None
    if method == "if":
        bx = np.asarray(iterative_filtering1DWithKCutOff(bx, Xi=xi, k=kcut), dtype=np.float64)
        by = np.asarray(iterative_filtering1DWithKCutOff(by, Xi=xi, k=kcut), dtype=np.float64)
    if transpose:
        bx, by = bx.T, by.T
    n = bx.shape[0]
    kx, ky = kgrid(n)
    k2 = kx ** 2 + ky ** 2
    bxk, byk = np.fft.fft2(bx), np.fft.fft2(by)
    pk = (-1j * ky * bxk + 1j * kx * byk) / np.where(k2 == 0, 1, k2)
    pk[0, 0] = 0
    psi = np.real(np.fft.ifft2(pk * np.exp(-0.5 * (SMOOTH_PSI * dx) ** 2 * k2)))
    filt = np.exp(-0.5 * (smooth * dx) ** 2 * k2)
    if highpass > 0:
        filt = filt - np.exp(-0.5 * highpass ** 2 * k2)
    jz = np.real(np.fft.ifft2((1j * kx * byk - 1j * ky * bxk) * filt))
    return psi, jz


def palette(name):
    return LinearSegmentedColormap.from_list(name, NEON[name]) if name in NEON else plt.get_cmap(name)


def colour(jw, vmax, p):
    """J_z window -> RGB (signed power-law scaling, optional neon glow)."""
    s = np.sign(jw) * np.clip(np.abs(jw) / vmax, 0, 1) ** p["gamma"]
    rgb = palette(p["cmap"])(0.5 + 0.5 * s)[..., :3]
    if p["glow"] > 0:
        lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        bright = rgb * np.clip((lum - 0.3) / 0.7, 0, 1)[..., None]
        sig = 0.004 * jw.shape[0]
        halo = np.stack([gaussian_filter(bright[..., c], sig) for c in range(3)], -1)
        rgb = 1 - (1 - rgb) * (1 - np.clip(p["glow"] * 2.5 * halo, 0, 1))
    return np.clip(rgb, 0, 1)


def draw(t, v, psi, jz, vmax, dpsi, p, path, px, clean=False):
    """One frame (or still) for video v at time t."""
    neon = p["cmap"] in NEON
    n = psi.shape[0]
    L = n * dx
    zoom = p["zoom"]
    m = int(round(zoom / dx))
    x, y = v["x"], v["y"]
    c0, r0 = int(round(x / dx)) - m // 2, int(round(y / dx)) - m // 2
    rr, cc = np.arange(r0, r0 + m) % n, np.arange(c0, c0 + m) % n
    jw, pw = jz[np.ix_(rr, cc)], psi[np.ix_(rr, cc)]
    lx_ = (c0 + np.arange(m)) * dx - x
    ly_ = (r0 + np.arange(m)) * dx - y
    ext = [lx_[0], lx_[-1] + dx, ly_[0], ly_[-1] + dx]

    axes_on = p["axes"] and not clean
    timeline = not clean
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100, facecolor="black")
    if axes_on:
        ax = fig.add_axes([0.09, 0.14, 0.88, 0.84])
    elif timeline:
        ax = fig.add_axes([0, 0.06, 1, 0.94])
    else:
        ax = fig.add_axes([0, 0, 1, 1])
    if neon or p["glow"] > 0:
        ax.imshow(colour(jw, vmax, p), origin="lower", extent=ext, interpolation="bilinear")
    else:
        f = lambda a: np.sign(a) * np.abs(a) ** p["gamma"]
        finv = lambda a: np.sign(a) * np.abs(a) ** (1 / p["gamma"])
        ax.imshow(jw, origin="lower", extent=ext, cmap=palette(p["cmap"]), interpolation="bilinear",
                  norm=FuncNorm((f, finv), vmin=-vmax, vmax=vmax))
    if p["lines"]:
        st = max(1, m // 384)
        lo, hi = np.floor(pw.min() / dpsi), np.ceil(pw.max() / dpsi)
        if hi - lo < 400:
            X, Y = np.meshgrid(lx_[::st], ly_[::st])
            ax.contour(X, Y, pw[::st, ::st], levels=np.arange(lo, hi + 1) * dpsi,
                       colors="white" if neon else "k", linewidths=0.5 * px / 720,
                       alpha=0.3 if neon else 0.6, linestyles="solid")
    if not clean:
        ax.plot(0, 0, marker="+", color="lime", ms=14, mew=2, alpha=0.9)
    ax.set_xlim(-zoom / 2, zoom / 2)
    ax.set_ylim(-zoom / 2, zoom / 2)
    if axes_on:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos, c=x: f"{(c + val) % L:.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos, c=y: f"{(c + val) % L:.0f}"))
        ax.tick_params(colors="white", labelsize=10)
        ax.set_xlabel(r"$x/d_p$", color="white")
        ax.set_ylabel(r"$y/d_p$", color="white")
        ax.grid(color="white", alpha=0.2, lw=0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("white")
    else:
        ax.axis("off")
    if not clean:
        here = "   <- event time" if t in v["t_marks"] else ""
        ax.text(0.015, 0.975, f"{v['name']}    t = {t}{here}", transform=ax.transAxes, va="top",
                fontsize=11, color="white" if neon else "k",
                bbox=dict(fc="black" if neon else "white", ec="none", alpha=0.6, pad=2))
        tl = fig.add_axes([0.03, 0.015, 0.94, 0.03], facecolor="black")
        tl.plot([p["t_first"], p["t_last"]], [0, 0], color="0.6", lw=2)
        for te in v["t_marks"]:
            tl.plot([te, te], [-1, 1], color="lime", lw=1.5)
        tl.plot([t], [0], marker="o", color="white", ms=7)
        tl.set_xlim(p["t_first"] - 2, p["t_last"] + 2)
        tl.set_ylim(-1.2, 1.2)
        tl.axis("off")
    kw = {"pil_kwargs": {"quality": 90}} if path.endswith(".jpg") else {}
    fig.savefig(path, dpi=100, facecolor="black", **kw)
    plt.close(fig)


def frames_for_time(t, videos, transpose, vmax, dpsi, frame_dir, p):
    psi, jz = fields(t, transpose, p["smooth"], p["highpass"], p["method"], p["kcut"], p["xi"])
    if psi is None:
        return t, 0
    for v in videos:
        draw(t, v, psi, jz, vmax, dpsi, p, os.path.join(frame_dir, v["name"], f"f{t:03d}.jpg"), p["px"])
    return t, len(videos)


def choose_workers(n_tasks):
    if N_WORKERS != "auto":
        return max(1, min(int(N_WORKERS), n_tasks))
    cores = os.cpu_count() or 1
    mem_gb = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    mem_gb = int(line.split()[1]) / 1e6
                    break
    except OSError:
        pass
    by_mem = int(mem_gb // GB_PER_WORKER) if mem_gb else cores
    return max(1, min(cores - 2, by_mem, n_tasks))


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def encode(name, frame_dir, out_dir, exe, fps, keep):
    out = os.path.join(out_dir, f"{name}.mp4")
    files = sorted(glob.glob(os.path.join(frame_dir, name, "f*.jpg")))
    if not files:
        return name, "no frames"
    lst = os.path.join(frame_dir, name, "frames.txt")
    with open(lst, "w") as fh:
        for f in files:
            fh.write(f"file '{os.path.abspath(f)}'\nduration {1 / fps:.6f}\n")
        fh.write(f"file '{os.path.abspath(files[-1])}'\n")
    cmd = [exe, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-r", str(max(fps, 10)), out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return name, f"ffmpeg failed: {r.stderr.strip()[:300]}"
    if not keep:
        shutil.rmtree(os.path.join(frame_dir, name), ignore_errors=True)
    return name, "ok"


def main():
    ap = argparse.ArgumentParser(description="Videos/stills of reconnection events (see top of file).")
    ap.add_argument("--ids", nargs="+", help="event IDs (default: the SELECTED list)")
    ap.add_argument("--per", choices=["event", "site"], default="event")
    ap.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"), help="box centred here [d_p]")
    ap.add_argument("--name", help="name for --center (default box_x..._y..._z...)")
    ap.add_argument("--zoom", type=float, default=ZOOM)
    ap.add_argument("--px", type=int, default=FRAME_PX)
    ap.add_argument("--contours", type=int, default=N_CONTOURS)
    ap.add_argument("--lines", action=argparse.BooleanOptionalAction, default=True, help="field lines")
    ap.add_argument("--axes", action="store_true")
    ap.add_argument("--cmap", default=CMAP, help=f"palette: {', '.join(NEON)} or any matplotlib name")
    ap.add_argument("--smooth", type=float, default=None,
                    help=f"Gaussian filter of J_z [cells] (default {SMOOTH_J:g}, or 0 with --filter if)")
    ap.add_argument("--filter", choices=["gauss", "if"], default="gauss",
                    help="gauss (default) or if = your iterative filtering of Bx, By")
    ap.add_argument("--kcut", type=float, help="angular cut-off wavenumber [rad/d_p] for --filter if")
    ap.add_argument("--xi", type=float, default=1.6, help="Xi of the iterative filtering (default 1.6)")
    ap.add_argument("--highpass", type=float, default=0.0, help="remove structure larger than this [d_p]")
    ap.add_argument("--glow", type=float, default=0.0, help="neon glow strength (0 = off)")
    ap.add_argument("--gamma", type=float, default=GAMMA_J,
                    help="colour scaling: 0.6 lights up weak currents, 0.9-1 keeps them dark (less glowing)")
    ap.add_argument("--clip", type=float, default=CLIP_PCT,
                    help="colour range = this percentile of |J_z|; 99.9 = fewer saturated bright pixels")
    ap.add_argument("--fps", type=float, default=FPS, help="output times per second (e.g. 1)")
    ap.add_argument("--trange", nargs=2, type=int, metavar=("T0", "T1"),
                    help=f"only film t = T0..T1 (default {T_FIRST}..{T_LAST})")
    ap.add_argument("--still", type=int, help="make one high-resolution still at this time instead")
    ap.add_argument("--still-px", type=int, default=3000)
    a = ap.parse_args()
    if a.smooth is None:
        a.smooth = 0.0 if a.filter == "if" else SMOOTH_J
    if a.filter == "if":
        if a.kcut is None:
            raise SystemExit("--filter if needs --kcut (the k you use in your own code)")
        try:
            import fifpy  # noqa: F401
        except ImportError as err:
            raise SystemExit(f"--filter if needs fifpy, which could not be imported ({err}). "
                             "Run from the environment where your own filtering scripts work.")
        ml = k_to_mask_length(a.kcut)
        print(f"iterative filtering: k = {a.kcut:g} rad/d_p -> wavelength {2 * np.pi / a.kcut:.3g} d_p "
              f"-> mask length {ml} cells; Xi = {a.xi:g}")
    if a.cmap not in NEON:
        try:
            plt.get_cmap(a.cmap)
        except ValueError:
            raise SystemExit(f"unknown palette {a.cmap}; neon family: {', '.join(NEON)}")

    zoom = a.zoom
    t_first, t_last = a.trange if a.trange else (T_FIRST, T_LAST)
    style = ("" if a.cmap == CMAP else f"_{a.cmap}") \
        + (f"_if{a.kcut:g}" + (f"_xi{a.xi:g}" if a.xi != 1.6 else "") if a.filter == "if" else "") \
        + (f"_s{a.smooth:g}" if a.smooth != (0.0 if a.filter == "if" else SMOOTH_J) else "") \
        + (f"_hp{a.highpass:g}" if a.highpass > 0 else "") + (f"_glow{a.glow:g}" if a.glow > 0 else "") \
        + ("_nolines" if not a.lines else "") + (f"_fps{a.fps:g}" if a.fps != FPS else "") \
        + (f"_g{a.gamma:g}" if a.gamma != GAMMA_J else "") + (f"_c{a.clip:g}" if a.clip != CLIP_PCT else "") \
        + (f"_t{t_first:03d}-{t_last:03d}" if a.trange else "")
    out_dir = os.path.join(SCAN_DIR, f"videos_zoom{zoom:g}")
    frame_dir = os.path.join(out_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    p = dict(zoom=zoom, px=a.px, axes=a.axes, t_first=t_first, t_last=t_last, cmap=a.cmap,
             smooth=a.smooth, highpass=a.highpass, glow=a.glow, gamma=a.gamma, lines=a.lines,
             method=a.filter, kcut=a.kcut, xi=a.xi)
    t0 = time.time()

    # ---------------- what to film
    cat = {}
    cat_path = os.path.join(SCAN_DIR, "catalog.csv")
    if os.path.exists(cat_path):
        with open(cat_path) as fh:
            for row in csv.DictReader(fh):
                cat[row["id"]] = row
    if a.center:
        cx, cy = a.center
        base = a.name or f"box_x{cx:05.1f}_y{cy:05.1f}_z{zoom:g}"
        videos = [dict(name=base + style, x=cx, y=cy, t_marks=[a.still] if a.still is not None else [])]
    elif a.per == "site":
        videos = []
        with open(os.path.join(SCAN_DIR, SELECTED_DIR, "sites.csv")) as fh:
            for row in csv.DictReader(fh):
                tm = [int(cat[i]["t"]) for i in row["events"].split() if i in cat]
                videos.append(dict(name=row["folder"] + style, x=float(row["x_mean"]), y=float(row["y_mean"]),
                                   t_marks=tm))
    else:
        videos, seen = [], set()
        for mm in re.finditer(r"t(\d+)_e(\d+)", " ".join(a.ids) if a.ids else SELECTED):
            eid = f"t{int(mm.group(1)):03d}_e{int(mm.group(2))}"
            if eid in seen:
                continue
            seen.add(eid)
            if eid not in cat:
                print(f"  {eid} not in catalog - skipped")
                continue
            r = cat[eid]
            x, y = float(r["x"]), float(r["y"])
            videos.append(dict(name=f"{eid}_x{x:05.1f}_y{y:05.1f}_z{zoom:g}{style}", x=x, y=y,
                               t_marks=[int(r["t"])]))
    if not videos:
        raise SystemExit("nothing to film")

    # ---------------- fixed colour scale and field-line spacing (from REF_T, same filtering)
    bx, by = read("Bx", REF_T), read("By", REF_T)
    if bx is None:
        raise SystemExit(f"REF_T = {REF_T}: Bx not found")
    n = bx.shape[0]
    kx, ky = kgrid(n)
    dc = np.std(np.real(np.fft.ifft2(1j * kx * np.fft.fft2(bx) + 1j * ky * np.fft.fft2(by))))
    dr = np.std(np.real(np.fft.ifft2(1j * ky * np.fft.fft2(bx) + 1j * kx * np.fft.fft2(by))))
    transpose = dr < dc
    b_rms = float(np.sqrt(np.mean(bx ** 2 + by ** 2)))
    del bx, by
    _, jz_ref = fields(REF_T, transpose, a.smooth, a.highpass, a.filter, a.kcut, a.xi)
    vmax = float(np.percentile(np.abs(jz_ref), a.clip))
    dpsi = b_rms * zoom / a.contours
    del jz_ref
    filt_txt = f"iterative filtering k = {a.kcut:g}" + (f" + Gaussian {a.smooth:g} cells" if a.smooth else "") \
        if a.filter == "if" else f"Gaussian {a.smooth:g} cells"
    print(f"{len(videos)} item(s), box {zoom:g} d_p, palette {a.cmap}, {filt_txt}"
          + (f", high-pass {a.highpass:g} d_p" if a.highpass else "") + (f", glow {a.glow:g}" if a.glow else ""))

    # ---------------- stills
    if a.still is not None:
        psi, jz = fields(a.still, transpose, a.smooth, a.highpass, a.filter, a.kcut, a.xi)
        if psi is None:
            raise SystemExit(f"no Bx/By for t = {a.still}")
        stills = os.path.join(out_dir, "stills")
        os.makedirs(stills, exist_ok=True)
        for v in videos:
            clean = os.path.join(stills, f"{v['name']}_t{a.still:03d}.png")
            check = os.path.join(stills, f"{v['name']}_t{a.still:03d}_labelled.png")
            draw(a.still, v, psi, jz, vmax, dpsi, p, clean, a.still_px, clean=True)
            draw(a.still, v, psi, jz, vmax, dpsi, dict(p, axes=True), check, 1080)
            mb = os.path.getsize(clean) / 1e6
            print(f"  saved {clean} ({a.still_px} px, {mb:.1f} MB) and {os.path.basename(check)}")
        return

    # ---------------- stage 1: frames (resume per video and per time)
    pending = [v for v in videos if not os.path.exists(os.path.join(out_dir, f"{v['name']}.mp4"))]
    for v in pending:
        os.makedirs(os.path.join(frame_dir, v["name"]), exist_ok=True)
    work = {}
    for t in range(t_first, t_last + 1):
        if not os.path.exists(os.path.join(RUN, f"Bx_ApJ_t{t}.h5")):
            continue
        need = [v for v in pending if not os.path.exists(os.path.join(frame_dir, v["name"], f"f{t:03d}.jpg"))]
        if need:
            work[t] = need
    times = sorted(work)
    if times:
        w = choose_workers(len(times))
        print(f"stage 1: frames for {len(pending)} video(s) at {len(times)} times with {w} workers")
        with ProcessPoolExecutor(max_workers=w) as pool:
            futs = [pool.submit(frames_for_time, t, work[t], transpose, vmax, dpsi, frame_dir, p) for t in times]
            for k, fu in enumerate(futs, 1):
                try:
                    fu.result()
                except Exception as err:
                    print(f"  a time failed: {type(err).__name__}: {err}")
                    continue
                el = time.time() - t0
                if k % 10 == 0 or k == len(times):
                    print(f"  {k}/{len(times)} times  ({el:5.0f} s, ~{el / k * (len(times) - k):5.0f} s to go)",
                          flush=True)
    else:
        print("stage 1: nothing to draw (videos already made or all frames present)")

    # ---------------- stage 2: encode
    exe = ffmpeg_exe()
    if exe is None:
        print("\nffmpeg not found: pip install imageio-ffmpeg (or conda install -c conda-forge ffmpeg), "
              "then rerun - frames are kept.")
        return
    todo = [v["name"] for v in pending]
    made = 0
    if todo:
        with ProcessPoolExecutor(max_workers=max(1, min((os.cpu_count() or 2) - 2, len(todo), 32))) as pool:
            for name, status in pool.map(encode, todo, [frame_dir] * len(todo), [out_dir] * len(todo),
                                         [exe] * len(todo), [a.fps] * len(todo), [KEEP_FRAMES] * len(todo)):
                if status == "ok":
                    made += 1
                    print(f"  made {os.path.join(out_dir, name + '.mp4')}")
                else:
                    print(f"  {name}: {status}")
    print(f"\ndone: {made} new video(s), {len(videos) - len(pending)} already existed "
          f"-> {out_dir}/ ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()