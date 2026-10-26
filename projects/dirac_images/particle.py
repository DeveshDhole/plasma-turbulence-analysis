"""
Particle pictures combined over MANY rank files (_ApJ_n{rank}.h5) into single images.

Each rank file is read by a worker process and reduced straight away to small
histograms, so memory stays bounded; the main process adds them into one picture:

  <OUT>_flow.png      hue = flow direction, colour strength = flow speed, brightness = density
  <OUT>_flow_check.png  same with axes and a colour key
  <OUT>_vdf_xy.png, <OUT>_vdf_xz.png   mosaics of velocity distributions (one per box)
  <OUT>_stars.png     the most energetic particles glowing over a haze of all particles
  <OUT>_energy.png    energy distribution of all particles read

Choose the ranks either as an explicit list (RANKS) or as a rectangular block of patches
(BLOCK, needs RANK_ORDER). Consecutive ranks usually form a strip of patches, not a square.
The first thing printed is each rank's position range: check that the patches tile
the way you expect.

Run it in a batch job or on a big workstation: ~4 GB of memory per worker.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm, hsv_to_rgb
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
RUN = f"{NAS_ROOT}/DATA/Particle/"                 # directory with _ApJ_n*.h5
RANKS = list(range(0, 1024))         # explicit list of ranks ...
BLOCK = None                          # ... or (ix0, iy0, nx, ny): a block of patches, e.g. (20, 20, 10, 10)
NPROC_SIDE = 32                       # ranks per side (4096 cells / 128 cells per rank)
RANK_ORDER = "x-fastest"              # rank = ix + NPROC_SIDE*iy ("x-fastest") or iy + NPROC_SIDE*ix
N_WORKERS = 64                         # parallel readers (~4 GB memory each)
OUT = str(RESULTS / "particles_many")

MAX_PX = 4000                         # long side of the combined images in pixels
SMOOTH_PX = 1.0                       # flow image smoothing in pixels (0 = off)
VDF_BOX = 4.0                         # VDF mosaic box size, in the units of Xx1/Yy1
VDF_BINS = 48                         # velocity bins per axis in each small VDF
VDF_RANGE = None                      # velocity range +/- in v_A (None = automatic)
STAR_PCT = 99.9                       # star field: particles above this energy percentile
SAMPLE = 100_000                      # particles read per rank to locate its patch
# ------------------------------------------------------------------------------


def fname(rank):
    return os.path.join(RUN, f"_ApJ_n{rank}.h5")


def ranks_to_use():
    if BLOCK is None:
        return list(RANKS)
    ix0, iy0, nx, ny = BLOCK
    out = []
    for iy in range(iy0, iy0 + ny):
        for ix in range(ix0, ix0 + nx):
            out.append(ix + NPROC_SIDE * iy if RANK_ORDER == "x-fastest" else iy + NPROC_SIDE * ix)
    return out


def rank_patch_index(rank):
    a, b = rank % NPROC_SIDE, rank // NPROC_SIDE
    return (a, b) if RANK_ORDER == "x-fastest" else (b, a)


def worker(rank, g):
    """Read one rank and return its contributions (all small)."""
    with h5py.File(fname(rank), "r") as f:
        x, y, vx, vy, vz = (f[k].astype("f4")[:] for k in ("Xx1", "Yy1", "Vx1", "Vy1", "Vz1"))
    x += g["offset"][rank][0]
    y += g["offset"][rank][1]

    # flow image: only the pixel window covered by this rank
    ix = np.clip(((x - g["x0"]) / g["pix"]).astype(np.int64), 0, g["nx"] - 1)
    iy = np.clip(((y - g["y0"]) / g["pix"]).astype(np.int64), 0, g["ny"] - 1)
    wx0, wx1, wy0, wy1 = ix.min(), ix.max() + 1, iy.min(), iy.max() + 1
    w = wx1 - wx0
    pix = (iy - wy0) * w + (ix - wx0)
    size = w * (wy1 - wy0)
    cnt = np.bincount(pix, minlength=size)
    sux = np.bincount(pix, weights=vx, minlength=size)
    suy = np.bincount(pix, weights=vy, minlength=size)
    del ix, iy, pix

    # VDF mosaic boxes
    nbx, nby, nv, vr = g["nbx"], g["nby"], VDF_BINS, g["vrange"]
    bx = np.clip(((x - g["x0"]) / g["box"]).astype(np.int64), 0, nbx - 1)
    by = np.clip(((y - g["y0"]) / g["box"]).astype(np.int64), 0, nby - 1)
    box = by * nbx + bx
    del bx, by

    def vbin(v):
        return np.clip(((v + vr) / (2 * vr) * nv).astype(np.int64), 0, nv - 1)

    ivx = vbin(vx)
    ok = np.abs(vx) < vr
    m_xy = ok & (np.abs(vy) < vr)
    m_xz = ok & (np.abs(vz) < vr)
    nall = nbx * nby * nv * nv
    vdf_xy = np.bincount(((box * nv + vbin(vy)) * nv + ivx)[m_xy], minlength=nall)
    vdf_xz = np.bincount(((box * nv + vbin(vz)) * nv + ivx)[m_xz], minlength=nall)
    del ivx, ok, m_xy, m_xz, box

    # energies
    e = 0.5 * (vx * vx + vy * vy + vz * vz)
    e_hist = np.histogram(e, bins=g["e_edges"])[0]
    thr = np.percentile(e[::20], STAR_PCT)
    pick = e > thr
    stars = (x[pick], y[pick], e[pick])
    return dict(rank=rank, n=x.size, win=(wy0, wy1, wx0, wx1), cnt=cnt, sux=sux, suy=suy,
                vdf_xy=vdf_xy, vdf_xz=vdf_xz, e_hist=e_hist, stars=stars)


def save_png(rgb, path):
    img = Image.fromarray((np.clip(rgb[::-1], 0, 1) * 255).astype(np.uint8), "RGB")
    img.save(path, dpi=(300, 300))
    if os.path.getsize(path) > 20e6:
        os.remove(path)
        path = path[:-4] + ".jpg"
        img.save(path, quality=95, dpi=(300, 300))
    print(f"  saved {path} ({img.width} x {img.height} px)")


def main():
    t0 = time.time()
    ranks = ranks_to_use()
    print(f"{len(ranks)} ranks: {ranks[:5]} ... {ranks[-5:]}")

    # ---------------- locate each rank's patch from a small sample
    lo, hi, vs = {}, {}, []
    for r in ranks:
        with h5py.File(fname(r), "r") as f:
            xs, ys = f["Xx1"][:SAMPLE], f["Yy1"][:SAMPLE]
            vs.append(np.concatenate([f[k][:SAMPLE // 10] for k in ("Vx1", "Vy1", "Vz1")]))
        lo[r], hi[r] = (xs.min(), ys.min()), (xs.max(), ys.max())
    vsample = np.concatenate(vs)
    span = max(max(hi[r][0] - lo[r][0], hi[r][1] - lo[r][1]) for r in ranks)
    patch = float(np.round(span)) if span > 2 else float(span)
    mins = np.array([lo[r] for r in ranks])
    local = np.ptp(mins[:, 0]) < 0.5 * patch and np.ptp(mins[:, 1]) < 0.5 * patch and len(ranks) > 1
    print(f"patch size ~ {patch:g} (units of Xx1); coordinates look "
          f"{'LOCAL to each rank: placing patches with RANK_ORDER' if local else 'GLOBAL'}")
    for r in ranks[:4]:
        print(f"  rank {r}: x {lo[r][0]:.4g}..{hi[r][0]:.4g}, y {lo[r][1]:.4g}..{hi[r][1]:.4g}")

    offset, boxes = {}, {}
    for r in ranks:
        if local:
            px, py = rank_patch_index(r)
            offset[r] = (px * patch, py * patch)
        else:
            offset[r] = (0.0, 0.0)
        bx0 = np.floor((lo[r][0] + offset[r][0]) / patch + 1e-3) * patch
        by0 = np.floor((lo[r][1] + offset[r][1]) / patch + 1e-3) * patch
        boxes[r] = (bx0, by0)
    x0 = min(b[0] for b in boxes.values())
    y0 = min(b[1] for b in boxes.values())
    x1 = max(b[0] for b in boxes.values()) + patch
    y1 = max(b[1] for b in boxes.values()) + patch
    lx, ly = x1 - x0, y1 - y0
    pix = max(lx, ly) / MAX_PX
    nx, ny = int(np.ceil(lx / pix)), int(np.ceil(ly / pix))
    nbx, nby = max(1, int(round(lx / VDF_BOX))), max(1, int(round(ly / VDF_BOX)))
    vrange = VDF_RANGE or 1.1 * float(np.percentile(np.abs(vsample), 99.9))
    print(f"region x {x0:g}..{x1:g}, y {y0:g}..{y1:g} -> images {nx} x {ny} px; "
          f"VDF mosaic {nbx} x {nby} boxes, +/-{vrange:.3g} v_A")

    g = dict(offset=offset, x0=x0, y0=y0, pix=pix, nx=nx, ny=ny, box=lx / nbx, nbx=nbx, nby=nby,
             vrange=vrange, e_edges=np.logspace(-5, 3, 161))

    # ---------------- accumulate all ranks in parallel
    cnt = np.zeros((ny, nx))
    sux = np.zeros((ny, nx))
    suy = np.zeros((ny, nx))
    vdf_xy = np.zeros(nbx * nby * VDF_BINS ** 2)
    vdf_xz = np.zeros_like(vdf_xy)
    e_hist = np.zeros(160)
    stars, ntot = [], 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(worker, r, g) for r in ranks]
        for k, fu in enumerate(as_completed(futs), 1):
            res = fu.result()
            wy0, wy1, wx0, wx1 = res["win"]
            sh = (wy1 - wy0, wx1 - wx0)
            cnt[wy0:wy1, wx0:wx1] += res["cnt"].reshape(sh)
            sux[wy0:wy1, wx0:wx1] += res["sux"].reshape(sh)
            suy[wy0:wy1, wx0:wx1] += res["suy"].reshape(sh)
            vdf_xy += res["vdf_xy"]
            vdf_xz += res["vdf_xz"]
            e_hist += res["e_hist"]
            stars.append(res["stars"])
            ntot += res["n"]
            el = time.time() - t0
            print(f"  {k}/{len(ranks)} ranks (last {res['rank']}), {el:5.0f} s, "
                  f"~{el / k * (len(ranks) - k):5.0f} s to go", flush=True)
    print(f"{ntot:,} particles, ~{ntot / cnt.size:.0f} per pixel")
    extent = [x0, x1, y0, y1]

    # ---------------- flow image
    if SMOOTH_PX > 0:
        cnt_s, sux_s, suy_s = (gaussian_filter(a, SMOOTH_PX, mode="nearest") for a in (cnt, sux, suy))
    else:
        cnt_s, sux_s, suy_s = cnt, sux, suy
    ux, uy = sux_s / np.maximum(cnt_s, 1e-9), suy_s / np.maximum(cnt_s, 1e-9)
    speed = np.hypot(ux, uy)
    filled = cnt > 0
    smax = np.percentile(speed[filled], 99.5)
    hue = (np.arctan2(uy, ux) + np.pi) / (2 * np.pi)
    sat = np.clip(speed / smax, 0, 1)
    val = PowerNorm(0.6, 0, np.percentile(cnt_s[filled], 99.5), clip=True)(cnt_s).filled(0)
    val[~filled] = 0
    rgb = hsv_to_rgb(np.dstack([hue, sat, val]))
    save_png(rgb, f"{OUT}_flow.png")

    fig, ax = plt.subplots(figsize=(12, 12 * ly / lx + 1.5))
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="none")
    ax.set_xlabel("x (units of Xx1)")
    ax.set_ylabel("y (units of Yy1)")
    ax.set_title("hue = flow direction, saturation = flow speed, brightness = density", fontsize=10)
    key = ax.inset_axes([1.01, 0.75, 0.2, 0.2])
    gg = np.linspace(-1, 1, 201)
    KX, KY = np.meshgrid(gg, gg)
    kr = np.hypot(KX, KY)
    krgb = hsv_to_rgb(np.dstack([(np.arctan2(KY, KX) + np.pi) / (2 * np.pi), np.clip(kr, 0, 1), np.ones_like(kr)]))
    krgb[kr > 1] = 1.0
    key.imshow(krgb, origin="lower")
    key.set_title(f"edge = {smax:.2g} $v_A$", fontsize=8)
    key.axis("off")
    fig.savefig(f"{OUT}_flow_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT}_flow_check.png")

    # ---------------- VDF mosaics
    dens_bg = gaussian_filter(cnt, 2)
    nv = VDF_BINS
    for name, vdf, lab in (("xy", vdf_xy, r"$f(v_x,v_y)$"), ("xz", vdf_xz, r"$f(v_x,v_z)$")):
        v = vdf.reshape(nby, nbx, nv, nv)
        fig = plt.figure(figsize=(12, 12 * ly / lx), facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
        ax.imshow(dens_bg, origin="lower", extent=extent, cmap="gray",
                  norm=PowerNorm(0.6, 0, np.percentile(dens_bg[filled], 99.5)), alpha=0.35)
        wx, wy = lx / nbx, ly / nby
        for j in range(nby):
            for i in range(nbx):
                h = v[j, i]
                if h.max() <= 0:
                    continue
                im = plt.get_cmap("inferno")(LogNorm(h.max() * 1e-3, h.max(), clip=True)(np.maximum(h, 1e-30)))
                im[..., 3] = np.where(h > h.max() * 1e-3, 1.0, 0.0)
                cx, cy = x0 + (i + 0.5) * wx, y0 + (j + 0.5) * wy
                ax.imshow(im, origin="lower", interpolation="bilinear",
                          extent=[cx - 0.45 * wx, cx + 0.45 * wx, cy - 0.45 * wy, cy + 0.45 * wy])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.axis("off")
        ax.text(0.005, 0.01, f"{lab}, $\\pm${vrange:.2g} $v_A$ per box", color="w", fontsize=9,
                transform=ax.transAxes)
        fig.savefig(f"{OUT}_vdf_{name}.png", dpi=300, facecolor="black")
        plt.close(fig)
        print(f"  saved {OUT}_vdf_{name}.png")

    # ---------------- star field (raster: brightest energy per pixel, with a soft glow)
    sx = np.concatenate([s[0] for s in stars])
    sy = np.concatenate([s[1] for s in stars])
    se = np.concatenate([s[2] for s in stars])
    six = np.clip(((sx - x0) / pix).astype(np.int64), 0, nx - 1)
    siy = np.clip(((sy - y0) / pix).astype(np.int64), 0, ny - 1)
    emax = np.zeros(nx * ny)
    np.maximum.at(emax, siy * nx + six, se)
    emax = emax.reshape(ny, nx)
    lo_e, hi_e = np.percentile(se, [1, 99.9])
    level = np.where(emax > 0, LogNorm(lo_e, hi_e, clip=True)(np.maximum(emax, lo_e)).filled(0), 0)
    glow = np.clip(level + 0.8 * gaussian_filter(level, 1.5) * 4, 0, 1)
    star_rgb = plt.get_cmap("plasma")(level)[..., :3] * glow[..., None]
    haze = gaussian_filter(cnt, 1.5)
    haze_rgb = plt.get_cmap("bone")(PowerNorm(0.7, 0, np.percentile(haze[filled], 99.8), clip=True)(haze).filled(0))[..., :3]
    img = np.clip(0.3 * haze_rgb + star_rgb, 0, 1)
    img[~filled] = 0
    save_png(img, f"{OUT}_stars.png")

    # ---------------- energy distribution
    edges = g["e_edges"]
    centres = np.sqrt(edges[1:] * edges[:-1])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(centres, e_hist / np.diff(edges) / e_hist.sum(), "k-")
    ax.set_xlabel(r"$E$  [$m_p v_A^2$]")
    ax.set_ylabel(r"$f(E)$ (normalised)")
    ax.set_title(f"ion energy distribution, {len(ranks)} ranks")
    fig.tight_layout()
    fig.savefig(f"{OUT}_energy.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT}_energy.png\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()