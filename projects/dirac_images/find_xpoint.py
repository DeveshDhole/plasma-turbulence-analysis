"""
Find the exact coordinates of a reconnection X-point you have spotted by eye.

  1. In the video (made with --axes), pause where the event looks best and note the
     time t (top-left label) and the rough position (x, y) from the axes.
  2. Run
        python find_xpoint.py --t 150 --near 80 230
     It finds every X-point within --radius of that position at that time, ranks them by
     reconnection rate (|E_z| at the X-point), prints a numbered table and saves
        xpoints_t150_near_x080_y230.png   J_z + field lines, every X-point numbered
        xpoints_t150_near_x080_y230.csv   the same table
  3. Find your event's number in the picture, then
        python find_xpoint.py --t 150 --near 80 230 --pick 2
     draws that X-point's separatrix (to confirm it is the one you mean) and prints its
     exact coordinates, ready to paste into event_videos.py --center.

  --click   instead of 3: opens the picture in a window, click on the event and it snaps
            to the nearest X-point (needs a desktop session).
  --box 96  search a square of this size instead of a circle: with the video's centre and
            box size this gives exactly the video's view at that time.
  --top 20  only the 20 strongest X-points (keeps big areas readable).
"""
import argparse
import os
import sys

# shared code and paths (see README.md and shared/)
import sys
from pathlib import Path
REPO = next(p for p in Path(__file__).resolve().parents if (p / "shared" / "paths.py").exists())
sys.path.insert(0, str(REPO))
from shared.paths import NAS_ROOT, results_dir
RESULTS = results_dir("dirac_images")

ap = argparse.ArgumentParser(description="Locate X-points near a position (see top of file).")
ap.add_argument("--t", type=int, required=True, help="output time")
ap.add_argument("--near", nargs=2, type=float, required=True, metavar=("X", "Y"), help="rough position [d_p]")
ap.add_argument("--radius", type=float, default=15.0, help="search radius [d_p] (default 15)")
ap.add_argument("--box", type=float, help="search a square of this side [d_p] instead (e.g. the video's box)")
ap.add_argument("--top", type=int, help="keep only the N strongest X-points")
ap.add_argument("--pick", type=int, help="number of the X-point to highlight")
ap.add_argument("--click", action="store_true", help="click on the event in a window")
ap.add_argument("--run", default=f"{NAS_ROOT}/DATA/Brms_0.24/beta_p_1/beta_e_1/",
                help="simulation output directory")
ARGS = ap.parse_args()

import matplotlib
if not ARGS.click:
    matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import FuncNorm
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

dx = 0.0625
SMOOTH_PSI = 2.0
SMOOTH_J = 2.0
GAMMA_J = 0.6


def read(tag, t):
    fn = os.path.join(ARGS.run, f"{tag}_ApJ_t{t}.h5")
    if not os.path.exists(fn):
        return None
    with h5py.File(fn, "r") as f:
        return f["DS1"][:].T.astype(np.float64)


def corner_mean(a):
    return 0.25 * (a + np.roll(a, -1, 0) + np.roll(a, -1, 1) + np.roll(np.roll(a, -1, 0), -1, 1))


def crosses(g):
    c = (g, np.roll(g, -1, 0), np.roll(g, -1, 1), np.roll(np.roll(g, -1, 0), -1, 1))
    return (np.minimum.reduce(c) < 0) & (np.maximum.reduce(c) > 0)


def main():
    t, (x0, y0), R = ARGS.t, ARGS.near, ARGS.radius
    bx, by = read("Bx", t), read("By", t)
    if bx is None or by is None:
        sys.exit(f"no Bx/By for t = {t} in {ARGS.run}")
    n = bx.shape[0]
    L = n * dx
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = k[None, :], k[:, None]
    k2 = kx ** 2 + ky ** 2
    ifft = lambda a: np.real(np.fft.ifft2(a))

    # orientation: x along columns (div B = 0 decides), as in the other scripts
    dc = np.std(ifft(1j * kx * np.fft.fft2(bx) + 1j * ky * np.fft.fft2(by)))
    dr = np.std(ifft(1j * ky * np.fft.fft2(bx) + 1j * kx * np.fft.fft2(by)))
    transpose = dr < dc
    if transpose:
        bx, by = bx.T, by.T
    bxk, byk = np.fft.fft2(bx), np.fft.fft2(by)
    pk = (-1j * ky * bxk + 1j * kx * byk) / np.where(k2 == 0, 1, k2)
    pk[0, 0] = 0
    ps = pk * np.exp(-0.5 * (SMOOTH_PSI * dx) ** 2 * k2)
    psi = ifft(ps)
    gx, gy = ifft(1j * kx * ps), ifft(1j * ky * ps)
    hxx, hyy, hxy = ifft(-kx ** 2 * ps), ifft(-ky ** 2 * ps), ifft(-kx * ky * ps)
    jz = ifft((1j * kx * byk - 1j * ky * bxk) * np.exp(-0.5 * (SMOOTH_J * dx) ** 2 * k2))

    # X-points: both gradient components change sign in a cell and psi is a saddle there
    rows, cols = np.nonzero(crosses(gx) & crosses(gy) & (corner_mean(hxx * hyy - hxy ** 2) < 0))
    g1, g2 = corner_mean(gx)[rows, cols], corner_mean(gy)[rows, cols]
    a, b, c = corner_mean(hxx)[rows, cols], corner_mean(hyy)[rows, cols], corner_mean(hxy)[rows, cols]
    with np.errstate(divide="ignore", invalid="ignore"):
        ddx = -(b * g1 - c * g2) / (a * b - c * c)
        ddy = -(a * g2 - c * g1) / (a * b - c * c)
    ddx = np.where(np.isfinite(ddx), np.clip(ddx, -dx, dx), 0.0)
    ddy = np.where(np.isfinite(ddy), np.clip(ddy, -dx, dx), 0.0)
    xs = np.mod((cols + 0.5) * dx + ddx, L)
    ys = np.mod((rows + 0.5) * dx + ddy, L)
    xs[xs >= L] = 0.0
    ys[ys >= L] = 0.0

    ez = read("Ez", t)
    if ez is not None:
        rate_field, source = (ez.T if transpose else ez), "E_z"
    else:
        rate_field, source = jz, "J_z (no Ez file)"
    rate = corner_mean(rate_field)[rows, cols]

    # merge duplicate detections, keep those within R of the given position
    tree = cKDTree(np.c_[xs, ys], boxsize=L)
    keep = np.ones(xs.size, bool)
    for i in np.argsort(-np.abs(rate)):
        if keep[i]:
            for j in tree.query_ball_point([xs[i], ys[i]], r=4 * dx):
                if j != i:
                    keep[j] = False
    xs, ys, rate = xs[keep], ys[keep], rate[keep]
    ddx_ = (xs - x0 + L / 2) % L - L / 2                   # periodic offsets from the given position
    ddy_ = (ys - y0 + L / 2) % L - L / 2
    dist = np.hypot(ddx_, ddy_)
    if ARGS.box:
        inside = (np.abs(ddx_) <= ARGS.box / 2) & (np.abs(ddy_) <= ARGS.box / 2)
        region = f"in the {ARGS.box:g} d_p box around"
    else:
        inside = dist <= R
        region = f"within {R:g} d_p of"
    near = np.nonzero(inside)[0]
    near = near[np.argsort(-np.abs(rate[near]), kind="stable")]
    if near.size == 0:
        sys.exit(f"no X-points {region} ({x0:g}, {y0:g}) at t = {t}; try a larger --radius / --box")
    total = near.size
    if ARGS.top:
        near = near[:ARGS.top]

    tag = f"xpoints_t{t:03d}_near_x{x0:03.0f}_y{y0:03.0f}" + (f"_box{ARGS.box:g}" if ARGS.box else "")
    shown = f" (showing the {near.size} strongest)" if near.size < total else ""
    print(f"t = {t}: {total} X-points {region} ({x0:g}, {y0:g}){shown}; rate = {source} at the X-point")
    print(f"  {'#':>3} {'x [d_p]':>9} {'y [d_p]':>9} {'rate':>11} {'distance':>9}")
    with open(RESULTS / f"{tag}.csv", "w") as fh:
        fh.write("number,x_dp,y_dp,rate,distance_dp\n")
        for num, i in enumerate(near, 1):
            print(f"  {num:3d} {xs[i]:9.2f} {ys[i]:9.2f} {rate[i]:11.4g} {dist[i]:9.2f}")
            fh.write(f"{num},{xs[i]:.3f},{ys[i]:.3f},{rate[i]:.6g},{dist[i]:.3f}\n")

    # ---------------- picture: window around the given position, coordinates of the whole box
    half = ARGS.box / 2 if ARGS.box else max(R + 3, 12.0)
    m = int(round(2 * half / dx))
    c0, r0 = int(round(x0 / dx)) - m // 2, int(round(y0 / dx)) - m // 2
    rr, cc = np.arange(r0, r0 + m) % n, np.arange(c0, c0 + m) % n
    jw, pw = jz[np.ix_(rr, cc)], psi[np.ix_(rr, cc)]
    lx_ = (c0 + np.arange(m)) * dx - x0
    ly_ = (r0 + np.arange(m)) * dx - y0
    vmax = np.percentile(np.abs(jw), 99.5)
    f = lambda v: np.sign(v) * np.abs(v) ** GAMMA_J
    finv = lambda v: np.sign(v) * np.abs(v) ** (1 / GAMMA_J)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(jw, origin="lower", cmap="RdBu_r", norm=FuncNorm((f, finv), vmin=-vmax, vmax=vmax),
                   extent=[lx_[0], lx_[-1] + dx, ly_[0], ly_[-1] + dx], interpolation="bilinear")
    st = max(1, m // 400)
    X, Y = np.meshgrid(lx_[::st], ly_[::st])
    ax.contour(X, Y, pw[::st, ::st], levels=30, colors="k", linewidths=0.5, alpha=0.55, linestyles="solid")
    if ARGS.pick and 1 <= ARGS.pick <= near.size:
        i = near[ARGS.pick - 1]
        psix = map_coordinates(psi, [[ys[i] / dx], [xs[i] / dx]], order=3, mode="grid-wrap")[0]
        ax.contour(X, Y, pw[::st, ::st], levels=[psix], colors="lime", linewidths=2.2, linestyles="solid")
    for num, i in enumerate(near, 1):
        chosen = ARGS.pick == num
        ax.plot(ddx_[i], ddy_[i], marker="x", color="lime" if chosen else "k", ms=11 if chosen else 8, mew=2.5)
        ax.text(ddx_[i] + 0.4, ddy_[i] + 0.4, str(num), fontsize=12 if chosen else 10, fontweight="bold",
                color="k", bbox=dict(fc="lime" if chosen else "white", ec="none", alpha=0.85, pad=1))
    if not ARGS.box:
        ax.add_patch(plt.Circle((0, 0), R, fill=False, ec="gray", ls="--", lw=1))
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{(x0 + v) % L:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{(y0 + v) % L:.0f}"))
    ax.grid(alpha=0.3)
    ax.set_xlabel(r"$x/d_p$")
    ax.set_ylabel(r"$y/d_p$")
    title = f"t = {t}: X-points near ({x0:g}, {y0:g}), numbered by rate ({source})"
    if ARGS.pick:
        title += f"; #{ARGS.pick} with its separatrix (green)"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$J_z$")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{tag}.png", dpi=130)
    print(f"saved {RESULTS / tag}.png and .csv")

    chosen = None
    if ARGS.click:
        print("click on the event in the window ...")
        pts = plt.ginput(1, timeout=0)
        if pts:
            px, py = pts[0]
            j = int(np.argmin(np.hypot(ddx_[near] - px, ddy_[near] - py)))
            chosen = near[j]
            print(f"clicked at ({(x0 + px) % L:.2f}, {(y0 + py) % L:.2f}) -> nearest X-point #{j + 1}")
    elif ARGS.pick and 1 <= ARGS.pick <= near.size:
        chosen = near[ARGS.pick - 1]
    if chosen is not None:
        print(f"\nX-point at t = {t}:  x = {xs[chosen]:.2f} d_p,  y = {ys[chosen]:.2f} d_p,  "
              f"rate ({source}) = {rate[chosen]:.4g}")
        print(f"film it:  python event_videos.py --center {xs[chosen]:.2f} {ys[chosen]:.2f} --zoom 48 --axes")


if __name__ == "__main__":
    main()