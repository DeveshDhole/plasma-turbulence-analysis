"""
Scan every output time for reconnection events and save pictures to choose from by eye.

At each time: find all X-points (saddles of the flux function psi), rank them by the
reconnection rate |E_z| at the X-point (in 2D, E_z = -d psi/dt there), keep the
N_PER_TIME strongest well-separated ones, and save

  <OUT_ROOT>/overview/t150.png
        whole box at t=150, each event boxed and labelled e1, e2, ...
  <OUT_ROOT>/events/t150_e2_x123.4_y045.6.png
        left: whole box with the event boxed; right: close-up with field lines,
        the separatrix through the X-point (thick) and the X-point (x)
  <OUT_ROOT>/best/07_t150_e2.png
        copies of the N_BEST strongest events over ALL times, ranked 01, 02, ...
  <OUT_ROOT>/catalog.csv      every saved event (id, t, x, y, rate, ...)

Event IDs are "t<time>_e<rank at that time>", e.g. t150_e2 = second-strongest event at
t = 150. That ID is all you need to tell me which event you mean.
"""
import csv
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # each worker process uses one core; parallelism comes from workers

import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import FuncNorm
from matplotlib.patches import Rectangle
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
RUN_LABEL = "bp1_be1"                 # goes into the output folder name
T_FIRST, T_LAST, T_STEP = 0, 300, 1
dx = 0.0625

N_PER_TIME = 5                        # events kept per time
N_BEST = 30                           # strongest events over all times copied to best/
MIN_SEP = 20.0                        # minimum distance between events at one time [d_p]
ZOOM = 16.0                           # close-up window [d_p]
RATE_SOURCE = "Ez"                    # "Ez", "dpsidt" (needs next output) or "Jz"

SMOOTH_PSI = 2.0                      # Gaussian smoothing (cells) before finding X-points
SMOOTH_J = 2.0                        # Gaussian smoothing (cells) of J_z for display
GAMMA_J = 0.6                         # colour scaling: <1 shows weaker sheets too
CMAP = "RdBu_r"                       # clear for inspection; field lines drawn in black
CLIP_PCT = 99.5                       # colour range = this percentile of |J_z| (per picture)
N_CONTOURS = 18                       # field lines in the close-up
FULL_PX = 1024                        # resolution of the whole-box panels
N_WORKERS = 64                    # times processed in parallel; "auto" = limited by cores and memory
GB_PER_WORKER = 3.5                   # memory one worker needs at 4096^2 (used by "auto")
OUT_ROOT = str(RESULTS / f"events_{RUN_LABEL}")
# ------------------------------------------------------------------------------


def read(tag, t):
    fn = os.path.join(RUN, f"{tag}_ApJ_t{t}.h5")
    if not os.path.exists(fn):
        return None
    with h5py.File(fn, "r") as f:
        return f["DS1"][:].T.astype(np.float64)


def kgrid(n):
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    return k[None, :], k[:, None]


def psi_k(bx, by, kx, ky):
    k2 = kx ** 2 + ky ** 2
    pk = (-1j * ky * np.fft.fft2(bx) + 1j * kx * np.fft.fft2(by)) / np.where(k2 == 0, 1, k2)
    pk[0, 0] = 0
    return pk, k2


def crosses(g):
    c = (g, np.roll(g, -1, 0), np.roll(g, -1, 1), np.roll(np.roll(g, -1, 0), -1, 1))
    return (np.minimum.reduce(c) < 0) & (np.maximum.reduce(c) > 0)


def corner_mean(a):
    return 0.25 * (a + np.roll(a, -1, 0) + np.roll(a, -1, 1) + np.roll(np.roll(a, -1, 0), -1, 1))


def jnorm(vmax):
    f = lambda x: np.sign(x) * np.abs(x) ** GAMMA_J
    finv = lambda y: np.sign(y) * np.abs(y) ** (1 / GAMMA_J)
    return FuncNorm((f, finv), vmin=-vmax, vmax=vmax)


def pdist(a, b, L):
    d = np.abs(np.asarray(a) - np.asarray(b))
    d = np.minimum(d, L - d)
    return np.hypot(d[0], d[1])


def choose_workers(n_tasks):
    """Number of parallel workers: N_WORKERS, or 'auto' = min(cores - 2, free memory / GB_PER_WORKER)."""
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
    w = max(1, min(cores - 2, by_mem, n_tasks))
    print(f"workers: {w}  (cores {cores}, free memory {mem_gb:.0f} GB -> {by_mem} by memory)"
          if mem_gb else f"workers: {w}  (cores {cores})")
    return w


def process_time(t, transpose):
    bx, by = read("Bx", t), read("By", t)
    if bx is None or by is None:
        return []
    fix = (lambda a: a.T) if transpose else (lambda a: a)
    bx, by = fix(bx), fix(by)
    n = bx.shape[0]
    L = n * dx
    kx, ky = kgrid(n)
    ifft = lambda a: np.real(np.fft.ifft2(a))

    pk, k2 = psi_k(bx, by, kx, ky)
    ps = pk * np.exp(-0.5 * (SMOOTH_PSI * dx) ** 2 * k2)
    psi = ifft(ps)
    gx, gy = ifft(1j * kx * ps), ifft(1j * ky * ps)
    hxx, hyy, hxy = ifft(-kx ** 2 * ps), ifft(-ky ** 2 * ps), ifft(-kx * ky * ps)
    det = corner_mean(hxx * hyy - hxy ** 2)
    jz = ifft((1j * kx * np.fft.fft2(by) - 1j * ky * np.fft.fft2(bx)) * np.exp(-0.5 * (SMOOTH_J * dx) ** 2 * k2))
    rows, cols = np.nonzero(crosses(gx) & crosses(gy) & (det < 0))
    if rows.size == 0:
        return []
    # refine each X-point inside its cell with one Newton step on grad(psi) = 0,
    # and get psi exactly at the X-point (the level of the separatrix)
    g1, g2 = corner_mean(gx)[rows, cols], corner_mean(gy)[rows, cols]
    a, b, c = corner_mean(hxx)[rows, cols], corner_mean(hyy)[rows, cols], corner_mean(hxy)[rows, cols]
    dd = a * b - c * c
    with np.errstate(divide="ignore", invalid="ignore"):
        ddx = -(b * g1 - c * g2) / dd
        ddy = -(a * g2 - c * g1) / dd
    ddx = np.where(np.isfinite(ddx), np.clip(ddx, -dx, dx), 0.0)
    ddy = np.where(np.isfinite(ddy), np.clip(ddy, -dx, dx), 0.0)
    xs, ys = (cols + 0.5) * dx + ddx, (rows + 0.5) * dx + ddy
    xs, ys = np.mod(xs, L), np.mod(ys, L)          # periodic box: keep positions in [0, L)
    xs[xs >= L] = 0.0
    ys[ys >= L] = 0.0
    psix = corner_mean(psi)[rows, cols] + g1 * ddx + g2 * ddy + 0.5 * (a * ddx ** 2 + 2 * c * ddx * ddy + b * ddy ** 2)
    del gx, gy, hxx, hyy, hxy, det

    source, field = RATE_SOURCE, None
    if source == "Ez":
        ez = read("Ez", t)
        if ez is not None:
            field = fix(ez)
        else:
            source = "dpsidt"
    if source == "dpsidt":
        b2x, b2y = read("Bx", t + T_STEP), read("By", t + T_STEP)
        if b2x is not None and b2y is not None:
            pk2, _ = psi_k(fix(b2x), fix(b2y), kx, ky)
            field = ifft((pk2 - pk) * np.exp(-0.5 * (SMOOTH_PSI * dx) ** 2 * k2)) / T_STEP
        else:
            source = "Jz"
    if source == "Jz":
        field = jz
    rate = corner_mean(field)[rows, cols]
    jzx = corner_mean(jz)[rows, cols]

    # merge neighbouring detections of one X-point, then pick strong, well-separated ones
    tree = cKDTree(np.c_[xs, ys], boxsize=L)
    keep = np.ones(xs.size, bool)
    for i in np.argsort(-np.abs(rate)):
        if keep[i]:
            for j in tree.query_ball_point([xs[i], ys[i]], r=4 * dx):
                if j != i:
                    keep[j] = False
    xs, ys, rate, jzx, psix = xs[keep], ys[keep], rate[keep], jzx[keep], psix[keep]
    chosen = []
    for i in np.argsort(-np.abs(rate)):
        if all(pdist((xs[i], ys[i]), (xs[c], ys[c]), L) > MIN_SEP for c in chosen):
            chosen.append(i)
        if len(chosen) == N_PER_TIME:
            break

    # ---------------- pictures
    vmax = np.percentile(np.abs(jz), CLIP_PCT)
    f = max(1, n // FULL_PX)
    jsmall = jz[: n // f * f, : n // f * f].reshape(n // f, f, n // f, f).mean(axis=(1, 3))
    events = []

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(jsmall, origin="lower", extent=[0, L, 0, L], cmap=CMAP, norm=jnorm(vmax))
    for k, i in enumerate(chosen, 1):
        ax.add_patch(Rectangle((xs[i] - ZOOM / 2, ys[i] - ZOOM / 2), ZOOM, ZOOM, fill=False, ec="lime", lw=2))
        ax.text(xs[i] + ZOOM / 2 + 1, ys[i] + ZOOM / 2 + 1, f"e{k}", color="k", fontsize=13, fontweight="bold",
                bbox=dict(fc="lime", ec="none", pad=1.5))
    ax.set_xlabel(r"$x/d_p$")
    ax.set_ylabel(r"$y/d_p$")
    ax.set_title(f"t = {t}: strongest {len(chosen)} reconnection events (rate from {source})")
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$J_z$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "overview", f"t{t:03d}.png"), dpi=110)
    plt.close(fig)

    m = int(round(ZOOM / dx))
    for k, i in enumerate(chosen, 1):
        eid = f"t{t:03d}_e{k}"
        fname = f"{eid}_x{xs[i]:05.1f}_y{ys[i]:05.1f}.png"
        c0, r0 = int(round(xs[i] / dx)) - m // 2, int(round(ys[i] / dx)) - m // 2
        rr, cc = np.arange(r0, r0 + m) % n, np.arange(c0, c0 + m) % n
        jzz, pz = jz[np.ix_(rr, cc)], psi[np.ix_(rr, cc)]
        lx_ = (c0 + np.arange(m)) * dx - xs[i]                 # coordinates relative to the X-point
        ly_ = (r0 + np.arange(m)) * dx - ys[i]
        X, Y = np.meshgrid(lx_, ly_)
        ext = [lx_[0] - dx / 2, lx_[-1] + dx / 2, ly_[0] - dx / 2, ly_[-1] + dx / 2]

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
        a1.imshow(jsmall, origin="lower", extent=[0, L, 0, L], cmap=CMAP, norm=jnorm(vmax))
        a1.add_patch(Rectangle((xs[i] - ZOOM / 2, ys[i] - ZOOM / 2), ZOOM, ZOOM, fill=False, ec="lime", lw=2.5))
        a1.set_xlabel(r"$x/d_p$")
        a1.set_ylabel(r"$y/d_p$")
        a1.set_title(f"whole box, t = {t}")
        vz = np.percentile(np.abs(jzz), CLIP_PCT)
        im = a2.imshow(jzz, origin="lower", extent=ext, cmap=CMAP, norm=jnorm(vz), interpolation="bilinear")
        a2.contour(X, Y, pz, levels=N_CONTOURS, colors="k", linewidths=0.6, alpha=0.7, linestyles="solid")
        a2.contour(X, Y, pz, levels=[psix[i]], colors="k", linewidths=2.0, linestyles="solid")
        a2.plot(0, 0, marker="x", color="lime", ms=12, mew=3)
        a2.set_xlim(-ZOOM / 2, ZOOM / 2)
        a2.set_ylim(-ZOOM / 2, ZOOM / 2)
        a2.set_xlabel(r"$(x - x_X)/d_p$")
        a2.set_ylabel(r"$(y - y_X)/d_p$")
        a2.set_title(f"close-up: field lines, separatrix (thick), X-point (x)")
        fig.colorbar(im, ax=a2, label=r"$J_z$")
        fig.suptitle(f"{eid}    X-point at ({xs[i]:.1f}, {ys[i]:.1f}) $d_p$    rate ({source}) = {rate[i]:.3g}"
                     f"    $J_z$ at X = {jzx[i]:.3g}", fontsize=14)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        path = os.path.join(OUT_ROOT, "events", fname)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        events.append(dict(id=eid, t=t, rank_at_t=k, x=round(xs[i], 2), y=round(ys[i], 2),
                           rate=float(rate[i]), abs_rate=abs(float(rate[i])), Jz_at_X=float(jzx[i]),
                           rate_source=source, file=os.path.join("events", fname)))
    return events


def main():
    t0 = time.time()
    for sub in ("overview", "events", "best", "parts"):
        os.makedirs(os.path.join(OUT_ROOT, sub), exist_ok=True)
    times = [t for t in range(T_FIRST, T_LAST + 1, T_STEP)
             if os.path.exists(os.path.join(RUN, f"Bx_ApJ_t{t}.h5"))]
    if not times:
        raise SystemExit(f"no Bx files for t = {T_FIRST}..{T_LAST} in {RUN}")
    print(f"{len(times)} times: t = {times[0]} .. {times[-1]}")

    # orientation once, from the first time
    bx, by = read("Bx", times[0]), read("By", times[0])
    kx, ky = kgrid(bx.shape[0])
    dc = np.std(np.real(np.fft.ifft2(1j * kx * np.fft.fft2(bx) + 1j * ky * np.fft.fft2(by))))
    dr = np.std(np.real(np.fft.ifft2(1j * ky * np.fft.fft2(bx) + 1j * kx * np.fft.fft2(by))))
    transpose = dr < dc
    del bx, by

    keys = ["id", "t", "rank_at_t", "x", "y", "rate", "abs_rate", "Jz_at_X", "rate_source", "file"]
    part = lambda t: os.path.join(OUT_ROOT, "parts", f"t{t:03d}.csv")
    todo = [t for t in times if not os.path.exists(part(t))]
    if len(todo) < len(times):
        print(f"resuming: {len(times) - len(todo)} times already done, {len(todo)} to go")
    failed = []
    with ProcessPoolExecutor(max_workers=choose_workers(len(todo))) as pool:
        futures = {pool.submit(process_time, t, transpose): t for t in todo}
        for k, fu in enumerate(futures, 1):
            t = futures[fu]
            try:
                ev = fu.result()
            except Exception as err:                       # one bad time must not stop the scan
                failed.append(t)
                print(f"  t={t:3d}: FAILED ({type(err).__name__}: {err})", flush=True)
                continue
            with open(part(t), "w", newline="") as fh:     # marks this time as done
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(ev)
            el = time.time() - t0
            print(f"  t={t:3d}: {len(ev)} events   ({k}/{len(todo)}, {el:5.0f} s, "
                  f"~{el / k * (len(todo) - k):5.0f} s to go)", flush=True)
    if failed:
        print(f"failed times: {failed} (rerun the script to retry just these)")

    catalog = []
    for t in times:
        if os.path.exists(part(t)):
            with open(part(t)) as fh:
                for row in csv.DictReader(fh):
                    for key in ("t", "rank_at_t"):
                        row[key] = int(row[key])
                    for key in ("x", "y", "rate", "abs_rate", "Jz_at_X"):
                        row[key] = float(row[key])
                    catalog.append(row)
    catalog.sort(key=lambda e: (e["t"], e["rank_at_t"]))
    with open(os.path.join(OUT_ROOT, "catalog.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(catalog)

    # rank over all times; E_z and d psi/dt are the same physical rate, J_z is not comparable
    physical = [e for e in catalog if e["rate_source"] in ("Ez", "dpsidt")]
    fallback = sorted({e["t"] for e in catalog if e["rate_source"] == "Jz"})
    if fallback:
        print(f"note: no Ez / next output at t = {fallback[:10]}{' ...' if len(fallback) > 10 else ''}: "
              f"those times were ranked by J_z and are left out of best/")
    best = sorted(physical or catalog, key=lambda e: -e["abs_rate"])[:N_BEST]
    for r, e in enumerate(best, 1):
        shutil.copy(os.path.join(OUT_ROOT, e["file"]), os.path.join(OUT_ROOT, "best", f"{r:02d}_{e['id']}.png"))
    print(f"\n{len(catalog)} events saved in {OUT_ROOT}/ ({time.time() - t0:.0f} s). Strongest:")
    for r, e in enumerate(best[:10], 1):
        print(f"  {r:2d}. {e['id']:10s} x = {e['x']:6.1f}  y = {e['y']:6.1f}  rate = {e['rate']:.3g}")


if __name__ == "__main__":
    main()