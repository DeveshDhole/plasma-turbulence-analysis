"""
Reconnection close-ups over the turbulent background.

1. Computes the flux function psi from Bx, By (in 2D the in-plane field lines are
   contours of psi) and finds every X-point (saddle of psi).
2. Ranks the X-points by reconnection rate. In 2D the rate at an X-point is exactly
   E_z there (E_z = -d psi/dt), so the default uses the Ez output; if Ez is missing it
   falls back to d psi/dt from the next output, or to |J_z| at the X-point.
3. Picks the N best, well-separated events and draws them as zoomed insets
   (J_z, field lines, the separatrix, the X-point and, optionally, the ion flow in the
   X-point frame) on top of the whole-box J_z, with zoom lines to where they come from.

Outputs
  <OUT>.png          the picture (3000 px, 300 dpi)
  <OUT>_check.png    all X-points found, with the chosen ones numbered
  <OUT>_xpoints.csv  every X-point: x, y, rate, J_z (useful beyond the picture)
"""
import itertools
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial import cKDTree

# shared code and paths (see README.md and shared/)
import sys
from pathlib import Path
REPO = next(p for p in Path(__file__).resolve().parents if (p / "shared" / "paths.py").exists())
sys.path.insert(0, str(REPO))
from shared.paths import NAS_ROOT, results_dir
RESULTS = results_dir("dirac_images")

# ------------------------------------------------------------------ settings
RUN = f"{NAS_ROOT}/DATA/Brms_0.24/beta_p_1/beta_e_1/"
T = 200
dx = 0.0625
OUT = str(RESULTS / "reconnection_t200")

RATE_SOURCE = "Ez"        # "Ez" (best), "dpsidt" (needs Bx/By at T_NEXT) or "Jz"
T_NEXT = 201              # only for "dpsidt"
SMOOTH_PSI = 2.0          # Gaussian smoothing (cells) before locating X-points
SMOOTH_J = 2.0            # Gaussian smoothing (cells) of J_z for display

N_EVENTS = 4              # number of close-ups (1-4)
ZOOM = 12.0               # side of each zoomed window [d_p]
MIN_SEP = 30.0            # minimum distance between chosen events [d_p]
EDGE = 10.0               # ignore X-points closer than this to the box edge [d_p]
MANUAL_EVENTS = []        # e.g. [(120.5, 64.0), (30.0, 200.0)] to choose events yourself [d_p]

SHOW_FLOW = True          # ion-flow streamlines in the close-ups (needs Ux1, Uy1 files)
N_CONTOURS = 14           # field lines per close-up
BG_DIM = 0.7              # background brightness (1 = full) so the close-ups stand out
GAMMA_J = 0.6             # signed power-law colour scaling of J_z
INSET_SIZE = 0.31         # close-up size as a fraction of the picture width
CMAP = LinearSegmentedColormap.from_list(
    "fire_ice", ["#e0ffff", "#00bfff", "#0040ff", "#000000", "#ff2000", "#ff9000", "#ffff80"])
# ------------------------------------------------------------------------------


def read(tag, t):
    fn = os.path.join(RUN, f"{tag}_ApJ_t{t}.h5")
    if not os.path.exists(fn):
        return None
    with h5py.File(fn, "r") as f:
        return f["DS1"][:].T.astype(np.float64)


def kgrid(n):
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    return k[None, :], k[:, None]          # kx along columns (x), ky along rows (y)


def psi_from_b(bx, by, kx, ky):
    """Flux function with Bx = dpsi/dy, By = -dpsi/dx (Fourier space)."""
    k2 = kx ** 2 + ky ** 2
    k2s = np.where(k2 == 0, 1, k2)
    psik = (-1j * ky * np.fft.fft2(bx) + 1j * kx * np.fft.fft2(by)) / k2s
    psik[0, 0] = 0
    return psik, k2


def colour_j(j, vmax):
    s = np.sign(j) * np.clip(np.abs(j) / vmax, 0, 1) ** GAMMA_J
    return CMAP(Normalize(-1, 1)(s))[..., :3]


def crosses(g):
    """Cells (between grid points i..i+1, j..j+1) where g changes sign."""
    c = (g, np.roll(g, -1, 0), np.roll(g, -1, 1), np.roll(np.roll(g, -1, 0), -1, 1))
    return (np.minimum.reduce(c) < 0) & (np.maximum.reduce(c) > 0)


def corner_mean(a):
    return 0.25 * (a + np.roll(a, -1, 0) + np.roll(a, -1, 1) + np.roll(np.roll(a, -1, 0), -1, 1))


def main():
    bx, by = read("Bx", T), read("By", T)
    if bx is None or by is None:
        raise SystemExit(f"Bx/By for t={T} not found in {RUN}")
    n = bx.shape[0]
    L = n * dx

    # orientation: put x along columns, y along rows (div B = 0 decides)
    kx, ky = kgrid(n)
    div_cols = np.std(np.real(np.fft.ifft2(1j * kx * np.fft.fft2(bx) + 1j * ky * np.fft.fft2(by))))
    div_rows = np.std(np.real(np.fft.ifft2(1j * ky * np.fft.fft2(bx) + 1j * kx * np.fft.fft2(by))))
    transpose = div_rows < div_cols
    fix = (lambda a: a.T) if transpose else (lambda a: a)
    bx, by = fix(bx), fix(by)
    print(f"grid {n}^2, box {L:g} d_p; {'transposed' if transpose else 'kept'} array axes")

    psik, k2 = psi_from_b(bx, by, kx, ky)
    gauss_p = np.exp(-0.5 * (SMOOTH_PSI * dx) ** 2 * k2)
    gauss_j = np.exp(-0.5 * (SMOOTH_J * dx) ** 2 * k2)
    ps = psik * gauss_p
    ifft = lambda a: np.real(np.fft.ifft2(a))
    psi = ifft(ps)
    gx, gy = ifft(1j * kx * ps), ifft(1j * ky * ps)
    det = corner_mean(ifft(-kx ** 2 * ps) * ifft(-ky ** 2 * ps) - ifft(-kx * ky * ps) ** 2)
    jz = ifft((1j * kx * np.fft.fft2(by) - 1j * ky * np.fft.fft2(bx)) * gauss_j)

    # ---------------- X-points: both gradient components change sign, saddle (det < 0)
    xm = crosses(gx) & crosses(gy) & (det < 0)
    rows, cols = np.nonzero(xm)
    xs, ys = (cols + 0.5) * dx, (rows + 0.5) * dx
    del gx, gy, det

    # ---------------- reconnection rate at each X-point
    source = RATE_SOURCE
    rate_field = None
    if source == "Ez":
        ez = read("Ez", T)
        if ez is None:
            print("Ez file not found -> trying d psi/dt")
            source = "dpsidt"
        else:
            rate_field = fix(ez)
    if source == "dpsidt":
        bx2, by2 = read("Bx", T_NEXT), read("By", T_NEXT)
        if bx2 is None:
            print("next output not found -> ranking by |J_z| at the X-point")
            source = "Jz"
        else:
            psik2, _ = psi_from_b(fix(bx2), fix(by2), kx, ky)
            rate_field = ifft((psik2 - psik) * gauss_p) / (T_NEXT - T)
    if source == "Jz":
        rate_field = jz
    rate = corner_mean(rate_field)[rows, cols]
    jx = corner_mean(jz)[rows, cols]

    # merge duplicates (neighbouring cells flagging the same X-point): keep the strongest
    order = np.argsort(-np.abs(rate))
    tree = cKDTree(np.c_[xs, ys], boxsize=L)
    keep = np.ones(xs.size, bool)
    for i in order:
        if keep[i]:
            for j in tree.query_ball_point([xs[i], ys[i]], r=4 * dx):
                if j != i and abs(rate[j]) <= abs(rate[i]):
                    keep[j] = False
    xs, ys, rate, jx = xs[keep], ys[keep], rate[keep], jx[keep]
    print(f"{xs.size} X-points; rate from {source}: |rate| max {np.abs(rate).max():.3g}, "
          f"median {np.median(np.abs(rate)):.3g}")
    np.savetxt(f"{OUT}_xpoints.csv", np.c_[xs, ys, rate, jx], delimiter=",",
               header=f"x_dp,y_dp,rate_{source},Jz_at_X", comments="", fmt="%.6g")

    # close-ups sit in the picture corners; events are chosen outside those zones
    s = INSET_SIZE
    corners = [(0.015, 1 - s - 0.015), (1 - s - 0.015, 1 - s - 0.015),
               (0.015, 0.015), (1 - s - 0.015, 0.015)]
    half = ZOOM / 2 / L

    def covers(c, e):
        return (c[0] - half < e[0] / L < c[0] + s + half) and (c[1] - half < e[1] / L < c[1] + s + half)

    # ---------------- choose events
    if MANUAL_EVENTS:
        events = []
        for (ex, ey) in MANUAL_EVENTS:
            d = np.hypot((xs - ex + L / 2) % L - L / 2, (ys - ey + L / 2) % L - L / 2)
            k = int(np.argmin(d))
            events.append((xs[k], ys[k], rate[k]) if d[k] < ZOOM / 2 else (ex, ey, np.nan))
    else:
        ok = (xs > EDGE) & (xs < L - EDGE) & (ys > EDGE) & (ys < L - EDGE)
        ok &= ~np.array([any(covers(c, (x_, y_)) for c in corners) for x_, y_ in zip(xs, ys)])
        events = []
        for i in np.argsort(-np.abs(rate)):
            if not ok[i]:
                continue
            if all(np.hypot((xs[i] - e[0] + L / 2) % L - L / 2, (ys[i] - e[1] + L / 2) % L - L / 2) > MIN_SEP
                   for e in events):
                events.append((xs[i], ys[i], rate[i]))
            if len(events) == N_EVENTS:
                break
    for k, (ex, ey, er) in enumerate(events, 1):
        print(f"  event {k}: X-point at ({ex:.2f}, {ey:.2f}) d_p, rate {er:.3g}")

    # ---------------- flow (optional)
    ux = uy = None
    if SHOW_FLOW:
        ux, uy = read("Ux1", T), read("Uy1", T)
        if ux is None or uy is None:
            print("Ux1/Uy1 not found -> no flow streamlines")
            ux = uy = None
        else:
            ux, uy = fix(ux), fix(uy)

    # ---------------- picture
    vbg = np.percentile(np.abs(jz), 99.5)
    bg = colour_j(jz, vbg) * BG_DIM
    fig = plt.figure(figsize=(10, 10), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(bg, origin="lower", extent=[0, L, 0, L], interpolation="bilinear")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.axis("off")

    # assign close-ups to corners: shortest zoom lines, never covering an event
    best, best_cost = None, np.inf
    for perm in itertools.permutations(range(4), len(events)):
        if any(covers(corners[c], e) for c in perm for e in events):
            continue
        cost = sum(np.hypot(corners[c][0] + s / 2 - e[0] / L, corners[c][1] + s / 2 - e[1] / L)
                   for c, e in zip(perm, events))
        if cost < best_cost:
            best, best_cost = perm, cost
    if best is None:  # no layout avoids every event: accept the shortest lines
        best = min(itertools.permutations(range(4), len(events)),
                   key=lambda p: sum(np.hypot(corners[c][0] + s / 2 - e[0] / L, corners[c][1] + s / 2 - e[1] / L)
                                     for c, e in zip(p, events)))

    m = int(round(ZOOM / dx))
    for k, ((ex, ey, er), ci) in enumerate(zip(events, best), 1):
        c = corners[ci]
        c0, r0 = int(round(ex / dx)) - m // 2, int(round(ey / dx)) - m // 2
        rr, cc = np.arange(r0, r0 + m) % n, np.arange(c0, c0 + m) % n
        x0, y0 = c0 * dx, r0 * dx
        ext = [x0, x0 + m * dx, y0, y0 + m * dx]
        jzz = jz[np.ix_(rr, cc)]
        pz = psi[np.ix_(rr, cc)]
        iax = ax.inset_axes([c[0], c[1], s, s])
        iax.imshow(colour_j(jzz, np.percentile(np.abs(jzz), 99.5)), origin="lower", extent=ext,
                   interpolation="bilinear")
        X, Y = np.meshgrid(x0 + (np.arange(m) + 0.5) * dx, y0 + (np.arange(m) + 0.5) * dx)
        iax.contour(X, Y, pz, levels=N_CONTOURS, colors="white", linewidths=0.45, alpha=0.55)
        psi_x = pz[m // 2, m // 2]
        iax.contour(X, Y, pz, levels=[psi_x], colors="white", linewidths=1.3, alpha=0.95)
        if ux is not None:
            st = max(1, m // 64)
            uxz, uyz = ux[np.ix_(rr, cc)][::st, ::st], uy[np.ix_(rr, cc)][::st, ::st]
            uxz, uyz = uxz - uxz[uxz.shape[0] // 2, uxz.shape[1] // 2], uyz - uyz[uyz.shape[0] // 2, uyz.shape[1] // 2]
            spd = np.hypot(uxz, uyz)
            iax.streamplot(X[::st, ::st][0], Y[::st, ::st][:, 0], uxz, uyz, color="white",
                           density=0.9, linewidth=0.3 + 1.2 * spd / max(spd.max(), 1e-12),
                           arrowsize=0.6, broken_streamlines=True)
        iax.plot(ex, ey, marker="x", color="white", ms=9, mew=2.2)
        iax.set_xlim(ext[0], ext[1])
        iax.set_ylim(ext[2], ext[3])
        iax.set_xticks([])
        iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_edgecolor("white")
            sp.set_linewidth(1.5)
        iax.text(0.04, 0.93, f"{k}", color="white", fontsize=13, fontweight="bold",
                 transform=iax.transAxes)
        ax.indicate_inset_zoom(iax, edgecolor="white", alpha=0.9, linewidth=1.2)
    fig.savefig(f"{OUT}.png", dpi=300, facecolor="black")
    plt.close(fig)
    print(f"saved {OUT}.png")

    # ---------------- check plot: every X-point, chosen ones numbered
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(jz, origin="lower", extent=[0, L, 0, L], cmap="RdBu_r", vmin=-vbg, vmax=vbg)
    sc = ax.scatter(xs, ys, s=4 + 40 * np.abs(rate) / np.abs(rate).max(), c="k", alpha=0.6)
    for k, (ex, ey, er) in enumerate(events, 1):
        ax.add_patch(plt.Rectangle((ex - ZOOM / 2, ey - ZOOM / 2), ZOOM, ZOOM, fill=False, ec="lime", lw=1.5))
        ax.text(ex + ZOOM / 2, ey + ZOOM / 2, f"{k}: {er:.3g}", color="lime", fontsize=10)
    ax.set_xlabel(r"$x/d_p$")
    ax.set_ylabel(r"$y/d_p$")
    ax.set_title(f"t = {T}: {xs.size} X-points (dot size ~ rate from {source}); chosen events in green")
    fig.savefig(f"{OUT}_check.png", dpi=130)
    plt.close(fig)
    print(f"saved {OUT}_check.png and {OUT}_xpoints.csv")


if __name__ == "__main__":
    main()