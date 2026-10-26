"""
Image gallery for the DiRAC competition: ideas 1-5 in one go.

One pass over the output times builds every time-based picture at once
(wedges, spiral, rings, each for |B_perp| and J_z); one snapshot (T_SINGLE)
gives the relief images and the colour-scaling comparison.

  1  wheel_Jz_neon         wedges, J_z on a dark "neon" diverging map
  2  wheel_Bperp_power     wedges, |B_perp| with power-law (gamma) scaling
     scaling_comparison    linear vs gamma scaling at T_SINGLE (for choosing)
  3  relief_Jz, relief_Bperp   hillshaded "landscape" at T_SINGLE
  4  spiral_Jz_neon, spiral_Bperp   wedges twisted into spiral arms
  5  rings_Jz_neon, rings_Bperp     time runs outwards like tree rings
  +  *_absJz versions of 1, 3, 4, 5: |J_z| (current strength, sign ignored)
  6  3d_absJz, 3d_Bperp, 3d_Jz      3D landscapes at T_SINGLE (height = field strength)
     overview.png          thumbnails of everything, labelled

Output files can be named by time (Bx_ApJ_t200.h5, NAMING = "time") or by iteration
(Bx_ApJ_i15681.h5, NAMING = "iteration"): all Bx files are found, sorted in ascending
order and used in that order (every T_STEP-th one). The time loop runs in parallel.

Set T_STEP = 5 for a quick preview, then T_STEP = 1 for the final images.
Set TIME_IMAGES = False to make only the T_SINGLE images (relief, scaling, 3D): fast,
handy while tuning the 3D view.
"""
import glob
import os
import re
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize, PowerNorm, TwoSlopeNorm
from PIL import Image
from scipy.ndimage import gaussian_filter

# shared code and paths (see README.md and shared/)
import sys
from pathlib import Path
REPO = next(p for p in Path(__file__).resolve().parents if (p / "shared" / "paths.py").exists())
sys.path.insert(0, str(REPO))
from shared.paths import NAS_ROOT, results_dir
RESULTS = results_dir("dirac_images")

# ------------------------------------------------------------------ settings
RUN = f"{NAS_ROOT}/DATA/Brms_0.24/beta_p_0.0625/beta_e_1/"
OUT_DIR = str(RESULTS / "gallery_bp0.0625")
dx = 0.0625                       # cell size [d_p]

NAMING = "iteration"              # "time": Bx_ApJ_t200.h5    "iteration": Bx_ApJ_i15681.h5
T_FIRST, T_LAST = 1, 300          # NAMING = "time": range of times for the wheels
I_FIRST, I_LAST = None, None      # NAMING = "iteration": optional range of iterations (None = all found)
T_STEP = 1                        # use every T_STEP-th output in ascending order (5 = quick preview)
T_SINGLE = None                   # output for relief, scaling comparison and 3D: a time or an
                                  # iteration number (depending on NAMING); None = the last one found
N_WORKERS = "auto"                # parallel readers for the time loop ("auto" = cores and memory)
GB_PER_WORKER = 3.0
TIME_IMAGES = True                # False = skip the wheels/spiral/rings (no time loop)
SAVE_COMPOSITES = True            # save the wheel/spiral/rings data as .npy (for palette_explorer.py)

TWIST = 1.5                       # spiral: turns of twist from centre to edge
R_INNER = 0.04                    # empty hub for wedges/spiral (fraction of half-width)
GAMMA_B = 0.5                     # power-law scaling for |B_perp| (1 = linear)
GAMMA_J = 0.6                     # power-law scaling for J_z (1 = linear)
GAMMA_ABSJ = 0.5                  # power-law scaling for |J_z| (1 = linear)
ABSJ_CMAP = "magma"               # colour map for |J_z|; also try "inferno", "hot", "afmhot"
CLIP_PCT = 99.5                   # colour range = this percentile
SMOOTH_J = 2.0                    # Gaussian smoothing of J_z, sigma in cells (0 = off; 2 cells = 0.125 d_p)
SMOOTH_B = 0.0                    # same for |B_perp| (usually not needed)
VERT_EXAG = 3.0                   # relief strength (larger = deeper ridges)
OUT_PX = 3000                     # final image size (10 in at 300 dpi)

# 3D landscapes (height = field strength); each render takes ~0.5-1 min
WIN3D = (96.0, 96.0, 64.0)        # x0, y0, size in d_p of the 3D snapshot window (64 d_p = 1024 cells)
N3D = 512                         # surface points per side after downsampling (more = finer, slower)
HEIGHT = 0.3                      # vertical size relative to the horizontal size
ELEV, AZIM = 35, -60              # viewing angles in degrees (elevation, azimuth)

NEON = LinearSegmentedColormap.from_list(
    "neon", ["#b8f6ff", "#00a8ff", "#08306b", "#000000", "#6b1d08", "#ff6a00", "#ffe39a"])
# ------------------------------------------------------------------------------


def deriv(f, d, axis):
    n = f.shape[axis]
    k = 2 * np.pi * np.fft.fftfreq(n, d=d)
    shape = [1] * f.ndim
    shape[axis] = n
    return np.real(np.fft.ifft(1j * k.reshape(shape) * np.fft.fft(f, axis=axis), axis=axis))


PREFIX = "t" if NAMING == "time" else "i"


def fname(tag, t):
    return os.path.join(RUN, f"{tag}_ApJ_{PREFIX}{t}.h5")


def read(tag, t):
    with h5py.File(fname(tag, t), "r") as f:
        return f["DS1"][:].T.astype(np.float64)


def fields(t, transpose):
    bx, by = read("Bx", t), read("By", t)
    if transpose:
        bx, by = bx.T, by.T
    jz = deriv(by, dx, 1) - deriv(bx, dx, 0)       # x along columns, y along rows
    bperp = np.hypot(bx, by)
    if SMOOTH_J > 0:
        jz = gaussian_filter(jz, SMOOTH_J, mode="wrap")      # periodic box
    if SMOOTH_B > 0:
        bperp = gaussian_filter(bperp, SMOOTH_B, mode="wrap")
    return bperp, jz


def times_available():
    """All outputs that have both Bx and By files, in ascending order (every T_STEP-th one)."""
    found = []
    for fn in glob.glob(os.path.join(RUN, f"Bx_ApJ_{PREFIX}*.h5")):
        m = re.search(rf"_{PREFIX}(\d+)\.h5$", fn)
        if m:
            found.append(int(m.group(1)))
    found.sort()
    if NAMING == "time":
        found = [t for t in found if T_FIRST <= t <= T_LAST]
    else:
        found = [i for i in found if (I_FIRST is None or i >= I_FIRST) and (I_LAST is None or i <= I_LAST)]
    have_by = [t for t in found if os.path.exists(fname("By", t))]
    if len(have_by) < len(found):
        print(f"  {len(found) - len(have_by)} outputs have Bx but no By file - skipped")
    out = have_by[::T_STEP]
    label = "t" if NAMING == "time" else "iteration"
    if out:
        print(f"found {len(have_by)} outputs ({label} {have_by[0]} .. {have_by[-1]}); "
              f"using {len(out)} (every {T_STEP})")
    return out


def signed_power(j, vmax, gamma):
    return np.sign(j) * np.clip(np.abs(j) / vmax, 0, 1) ** gamma   # in [-1, 1]


def save(rgb, name, saved):
    """rgb: float array (ny, nx, 3) in [0,1], row 0 = bottom. Saves PNG (JPEG if > 20 MB)."""
    img = Image.fromarray((np.clip(rgb[::-1], 0, 1) * 255).astype(np.uint8), "RGB")
    if OUT_PX and img.width != OUT_PX:
        img = img.resize((OUT_PX, round(OUT_PX * img.height / img.width)), Image.LANCZOS)
    path = os.path.join(OUT_DIR, name + ".png")
    img.save(path, dpi=(300, 300))
    if os.path.getsize(path) > 20e6:
        os.remove(path)
        path = os.path.join(OUT_DIR, name + ".jpg")
        img.save(path, quality=95, dpi=(300, 300))
    saved.append((name, path))
    print(f"  saved {path}")


def colour_B(v, vmax):
    return plt.get_cmap("inferno")(PowerNorm(GAMMA_B, 0, vmax)(np.nan_to_num(v)))[..., :3]


def colour_J(v, vmax):
    return NEON(Normalize(-1, 1)(signed_power(np.nan_to_num(v), vmax, GAMMA_J)))[..., :3]


def colour_A(v, vmax):
    v = np.abs(np.nan_to_num(v))
    return plt.get_cmap(ABSJ_CMAP)(PowerNorm(GAMMA_ABSJ, 0, vmax)(v))[..., :3]


def block(arr, f, how):
    """Downsample by f x f blocks: 'max' keeps thin peaks (current sheets), 'mean' for signed fields."""
    if f <= 1:
        return arr
    n0, n1 = (arr.shape[0] // f) * f, (arr.shape[1] // f) * f
    a = arr[:n0, :n1].reshape(n0 // f, f, n1 // f, f)
    return a.max(axis=(1, 3)) if how == "max" else a.mean(axis=(1, 3))


def save_fig(fig, name, saved, crop=True):
    path = os.path.join(OUT_DIR, name + ".png")
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    if crop:  # trim the empty black border around the 3D surface, keep a small margin
        img = Image.open(path).convert("RGB")
        box = img.point(lambda v: 255 if v > 12 else 0).convert("L").getbbox()
        if box:
            pad = int(0.03 * max(img.size))
            box = (max(box[0] - pad, 0), max(box[1] - pad, 0),
                   min(box[2] + pad, img.width), min(box[3] + pad, img.height))
            img.crop(box).save(path, dpi=(300, 300))
    if os.path.getsize(path) > 20e6:
        img = Image.open(path).convert("RGB")
        os.remove(path)
        path = path[:-4] + ".jpg"
        img.save(path, quality=95, dpi=(300, 300))
    saved.append((name, path))
    print(f"  saved {path}")


def surface(z, cmap, norm, zlim, name, saved):
    """3D landscape: z (row 0 = bottom) is both the height and, via cmap/norm, the colour."""
    rgb = LightSource(azdeg=315, altdeg=40).shade(z, cmap=plt.get_cmap(cmap), norm=norm,
                                                  blend_mode="soft", vert_exag=VERT_EXAG)
    ny, nx = z.shape
    Y, X = np.mgrid[0:ny, 0:nx] / max(nx, ny)
    fig = plt.figure(figsize=(10, 10), facecolor="black")
    ax = fig.add_subplot(projection="3d", facecolor="black")
    ax.plot_surface(X, Y, z, facecolors=rgb, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, shade=False)
    ax.set_zlim(*zlim)
    try:
        ax.set_box_aspect((1, ny / nx, HEIGHT), zoom=1.05)
    except TypeError:  # matplotlib < 3.6
        ax.set_box_aspect((1, ny / nx, HEIGHT))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    save_fig(fig, name, saved)
    plt.close(fig)


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


def time_worker(k, t, transpose, idx):
    """Fields of one output; returns only the pixels this output contributes to each picture."""
    bperp, jz = fields(t, transpose)
    bperp, jz = bperp.ravel(), jz.ravel()
    return k, {g: (bperp[i].astype(np.float32), jz[i].astype(np.float32)) for g, i in idx.items()}


def groups_for(index_map, nt):
    flat = index_map.ravel()
    order = np.argsort(flat, kind="stable").astype(np.int64)
    start = np.searchsorted(flat[order], 0)
    counts = np.bincount(flat[flat >= 0], minlength=nt)
    return np.split(order[start:], np.cumsum(counts)[:-1])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    saved = []

    # ---------------- all outputs (ascending), and the single snapshot
    available = times_available()
    if not available:
        raise SystemExit(f"no Bx_ApJ_{PREFIX}*.h5 / By files found in {RUN}")
    ts = T_SINGLE if T_SINGLE is not None else available[-1]
    for tag in ("Bx", "By"):
        if not os.path.exists(fname(tag, ts)):
            raise SystemExit(f"T_SINGLE = {ts}: file not found: {fname(tag, ts)}")
    print(f"single snapshot for relief / scaling / 3D: {PREFIX}{ts}")
    bx, by = read("Bx", ts), read("By", ts)
    div_a = np.std(deriv(bx, dx, 0) + deriv(by, dx, 1))
    div_b = np.std(deriv(bx, dx, 1) + deriv(by, dx, 0))
    transpose = div_a < div_b
    del bx, by
    print(f"x along {'rows -> transposing' if transpose else 'columns'}; "
          f"smoothing sigma: J_z {SMOOTH_J:g} cells, |B_perp| {SMOOTH_B:g} cells")

    # ---------------- single snapshot: scaling comparison (2) and relief (3)
    bperp, jz = fields(ts, transpose)
    vb = np.percentile(bperp, CLIP_PCT)
    vj = np.percentile(np.abs(jz), CLIP_PCT)
    print(f"{PREFIX}{ts}: |B_perp| {CLIP_PCT}th pct = {vb:.3g}, |J_z| {CLIP_PCT}th pct = {vj:.3g}")

    s = slice(None, None, 2)
    fig, axs = plt.subplots(3, 2, figsize=(12, 18))
    axs[0, 0].imshow(bperp[s, s], origin="lower", cmap="inferno", vmin=0, vmax=vb)
    axs[0, 0].set_title(r"$|B_\perp|$ linear")
    axs[0, 1].imshow(colour_B(bperp[s, s], vb), origin="lower")
    axs[0, 1].set_title(rf"$|B_\perp|$ power law, $\gamma$ = {GAMMA_B}")
    axs[1, 0].imshow(jz[s, s], origin="lower", cmap=NEON, vmin=-vj, vmax=vj)
    axs[1, 0].set_title(r"$J_z$ linear")
    axs[1, 1].imshow(colour_J(jz[s, s], vj), origin="lower")
    axs[1, 1].set_title(rf"$J_z$ signed power, $\gamma$ = {GAMMA_J}")
    axs[2, 0].imshow(np.abs(jz[s, s]), origin="lower", cmap=ABSJ_CMAP, vmin=0, vmax=vj)
    axs[2, 0].set_title(r"$|J_z|$ linear")
    axs[2, 1].imshow(colour_A(jz[s, s], vj), origin="lower")
    axs[2, 1].set_title(rf"$|J_z|$ power law, $\gamma$ = {GAMMA_ABSJ}")
    for ax in axs.flat:
        ax.axis("off")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "scaling_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")

    ls = LightSource(azdeg=315, altdeg=35)
    jn = np.clip(jz / vj, -1, 1)
    rgb = ls.shade(jn, cmap=plt.get_cmap("RdBu_r"), norm=TwoSlopeNorm(0, -1, 1),
                   blend_mode="soft", vert_exag=VERT_EXAG)[..., :3]
    save(rgb, "3_relief_Jz", saved)
    an = np.clip(np.abs(jz) / vj, 0, 1)
    rgb = ls.shade(an, cmap=plt.get_cmap(ABSJ_CMAP), norm=PowerNorm(GAMMA_ABSJ, 0, 1),
                   blend_mode="soft", vert_exag=VERT_EXAG)[..., :3]
    save(rgb, "3b_relief_absJz", saved)
    del an
    bn = np.clip(bperp / vb, 0, 1)
    rgb = ls.shade(bn, cmap=plt.get_cmap("inferno"), norm=PowerNorm(GAMMA_B, 0, 1),
                   blend_mode="soft", vert_exag=VERT_EXAG)[..., :3]
    save(rgb, "3_relief_Bperp", saved)

    # 3D landscapes of a window at T_SINGLE (idea 6)
    n_full = bperp.shape[0]
    x0, y0, size = WIN3D
    m = int(round(size / dx))
    rows = (np.arange(m) + int(round(y0 / dx))) % n_full
    cols = (np.arange(m) + int(round(x0 / dx))) % n_full
    f = max(1, m // N3D)
    jw = jz[np.ix_(rows, cols)]
    bw = bperp[np.ix_(rows, cols)]
    print(f"3D window {size:g} d_p ({m} cells) -> {m // f} points per side; rendering ...", flush=True)
    za = np.clip(block(np.abs(jw), f, "max") / vj, 0, 1)
    surface(za, ABSJ_CMAP, PowerNorm(GAMMA_ABSJ, 0, 1), (0, 1), "6_3d_absJz", saved)
    zb = np.clip(block(bw, f, "max") / vb, 0, 1)
    surface(zb, "inferno", PowerNorm(GAMMA_B, 0, 1), (0, 1), "6_3d_Bperp", saved)
    zj = np.clip(block(jw, f, "mean") / vj, -1, 1)
    surface(zj, NEON, TwoSlopeNorm(0, -1, 1), (-1, 1), "6_3d_Jz", saved)
    del bperp, jz, jn, bn, rgb, jw, bw, za, zb, zj

    if TIME_IMAGES:
        time_images(transpose, saved, available)
    overview(saved)
    print(f"done in {time.time() - t0:.0f} s")


def time_images(transpose, saved, times):
    """Wheels, spirals and rings (ideas 1, 2, 4, 5): one pass over the outputs, in parallel."""
    # ---------------- geometry of the time pictures (whole square box)
    nt = len(times)
    if nt == 0:
        raise SystemExit("no Bx/By files found for the wheels")
    n = read("Bx", times[0]).shape[0]
    r = n // 2
    Y, X = np.mgrid[0:n, 0:n] + 0.5 - r
    rad = np.hypot(X, Y)
    theta = np.mod(np.arctan2(X, Y), 2 * np.pi)              # clockwise from 12 o'clock
    hub = rad < R_INNER * r
    wedge = np.minimum((theta / (2 * np.pi) * nt).astype(np.int32), nt - 1)
    spiral = np.minimum((np.mod(theta + 2 * np.pi * TWIST * rad / r, 2 * np.pi)
                         / (2 * np.pi) * nt).astype(np.int32), nt - 1)
    rings = np.minimum((rad / (r * np.sqrt(2)) * nt).astype(np.int32), nt - 1)
    wedge[hub] = -1
    spiral[hub] = -1
    maps = {"wheel": wedge, "spiral": spiral, "rings": rings}
    groups = {k: groups_for(v, nt) for k, v in maps.items()}
    del X, Y, rad, theta, wedge, spiral, rings
    values = {(g, q): np.full(n * n, np.nan, dtype=np.float32) for g in maps for q in ("B", "J")}

    workers = choose_workers(nt)
    print(f"{nt} outputs ({PREFIX}{times[0]} .. {PREFIX}{times[-1]}) -> {nt} wedges / rings; "
          f"{workers} parallel workers ...")
    t_loop = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(time_worker, k, t, transpose, {g: groups[g][k] for g in maps})
                for k, t in enumerate(times)]
        for done, fu in enumerate(futs, 1):
            k, vals = fu.result()
            for g, (vb, vj) in vals.items():
                idx = groups[g][k]
                values[(g, "B")][idx] = vb
                values[(g, "J")][idx] = vj
            if done % 50 == 0 or done == nt:
                el = time.time() - t_loop
                print(f"  {done}/{nt} outputs  {el:5.0f} s, ~{el / done * (nt - done):5.0f} s to go", flush=True)

    if SAVE_COMPOSITES:   # raw data of every time picture, so colours can be re-tried without re-reading
        for (g, q), v in values.items():
            np.save(os.path.join(OUT_DIR, f"composite_{g}_{'Jz' if q == 'J' else 'Bperp'}.npy"), v.reshape(n, n))
        print(f"  saved composite_*.npy in {OUT_DIR} (for palette_explorer.py)")

    # ---------------- render the time pictures (1, 2, 4, 5)
    names = {("wheel", "J"): "1_wheel_Jz_neon", ("wheel", "A"): "1b_wheel_absJz",
             ("wheel", "B"): "2_wheel_Bperp_power",
             ("spiral", "J"): "4_spiral_Jz_neon", ("spiral", "A"): "4b_spiral_absJz",
             ("spiral", "B"): "4_spiral_Bperp",
             ("rings", "J"): "5_rings_Jz_neon", ("rings", "A"): "5b_rings_absJz",
             ("rings", "B"): "5_rings_Bperp"}
    for (g, q), name in names.items():
        v = values[(g, "J" if q == "A" else q)].reshape(n, n)   # |J_z| reuses the J_z values
        if q == "B":
            rgb = colour_B(v, np.nanpercentile(v, CLIP_PCT))
        elif q == "A":
            rgb = colour_A(v, np.nanpercentile(np.abs(v), CLIP_PCT))
        else:
            rgb = colour_J(v, np.nanpercentile(np.abs(v), CLIP_PCT))
        rgb[np.isnan(v)] = 0.0                                  # hub -> black
        save(rgb, name, saved)



def overview(saved):
    cols = 3
    rows = int(np.ceil(len(saved) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4.3 * rows))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, (name, path) in zip(np.ravel(axs), saved):
        im = Image.open(path)
        im.thumbnail((800, 800))
        ax.imshow(im)
        ax.set_title(name, fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "overview.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()