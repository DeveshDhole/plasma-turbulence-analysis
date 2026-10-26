"""
Re-render chosen events from scan_events.py with a bigger window (default 3x: 48 d_p).

Give the events as file names or IDs (anything containing "t040_e1"-style IDs) in SELECTED.
Positions and rates come from <SCAN_DIR>/catalog.csv. Each output time is read once,
times run in parallel.

Outputs in <SCAN_DIR>/selected_zoom48/
  t040_e1_x221.5_y252.9_z48.png     same ID as before, now with the 48 d_p close-up
  sites.csv                         events grouped into reconnection SITES: the same
                                    structure seen at several times (close in space and time)
  sites/S03_x149_y172_t061-090/     one folder per site, its events in time order (links)

You can refer to an event by its ID (t072_e2) or to a whole site by its name (S03).
"""
import csv
import os
import re

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
from scipy.ndimage import map_coordinates

# shared code and paths (see README.md and shared/)
import sys
from pathlib import Path
REPO = next(p for p in Path(__file__).resolve().parents if (p / "shared" / "paths.py").exists())
sys.path.insert(0, str(REPO))
from shared.paths import NAS_ROOT, results_dir
RESULTS = results_dir("dirac_images")

# ------------------------------------------------------------------ settings
RUN = f"{NAS_ROOT}/DATA/Brms_0.24/beta_p_1/beta_e_1/"
SCAN_DIR = str(RESULTS / "events_bp1_be1")            # folder made by scan_events.py (contains catalog.csv)
dx = 0.0625
ZOOM = 48.0                            # close-up window [d_p] (scan used 16 -> 3x)
N_CONTOURS = 30                        # field lines in the close-up (bigger window -> more)
SMOOTH_PSI = 2.0
SMOOTH_J = 2.0
GAMMA_J = 0.6
CMAP = "RdBu_r"
CLIP_PCT = 99.5
FULL_PX = 1024
LINK_DIST = 8.0                        # sites: events closer than this [d_p] ...
MAX_GAP = 10                           # ... and at most this many time units apart are one site
N_WORKERS = 64
GB_PER_WORKER = 3.5

SELECTED = """
t040_e1_x221.5_y252.9.png	t049_e3_x224.3_y251.6.png	t061_e5_x031.2_y112.1.png	t071_e4_x195.5_y137.9.png	t080_e3_x247.8_y131.4.png	t084_e1_x092.6_y057.2.png	t087_e4_x060.8_y131.1.png	t091_e2_x070.5_y132.2.png	t180_e5_x032.3_y054.3.png	t201_e4_x156.8_y179.7.png
t041_e1_x224.4_y251.6.png	t049_e4_x150.4_y093.9.png	t062_e1_x249.1_y117.1.png	t071_e5_x023.6_y241.3.png	t080_e4_x142.2_y082.1.png	t084_e2_x026.6_y104.0.png	t087_e5_x153.3_y091.7.png	t091_e3_x102.3_y060.2.png	t181_e2_x028.2_y051.2.png	t205_e2_x156.0_y179.8.png
t042_e2_x224.4_y251.6.png	t050_e1_x193.5_y248.3.png	t062_e2_x001.8_y142.3.png	t072_e2_x148.8_y172.5.png	t080_e5_x148.3_y171.8.png	t084_e3_x091.1_y234.6.png	t088_e1_x073.5_y125.1.png	t091_e4_x175.0_y122.1.png	t182_e1_x028.0_y051.0.png	t213_e1_x023.8_y045.3.png
t043_e1_x224.4_y251.6.png	t050_e2_x224.4_y251.5.png	t062_e5_x010.0_y243.3.png	t072_e5_x023.2_y241.0.png	t081_e1_x069.0_y103.1.png	t084_e4_x234.5_y224.8.png	t088_e2_x153.3_y091.4.png	t091_e5_x207.6_y002.4.png	t183_e3_x028.0_y050.8.png	t213_e5_x161.8_y032.2.png
t044_e2_x224.3_y251.6.png	t050_e4_x149.7_y230.7.png	t063_e2_x145.2_y178.0.png	t073_e2_x148.7_y172.5.png	t081_e2_x241.9_y225.1.png	t084_e5_x070.7_y101.7.png	t088_e3_x248.9_y117.5.png	t092_e1_x152.1_y086.3.png	t184_e3_x027.7_y050.5.png	t214_e3_x023.7_y045.1.png
t044_e3_x150.1_y093.8.png	t051_e1_x150.5_y093.8.png	t063_e4_x183.0_y022.0.png	t074_e1_x148.7_y172.3.png	t081_e3_x005.7_y030.6.png	t085_e1_x151.1_y087.5.png	t088_e4_x148.1_y170.8.png	t093_e5_x153.5_y090.9.png	t186_e3_x020.2_y175.2.png	t225_e2_x195.4_y119.5.png
t045_e1_x224.3_y251.6.png	t051_e4_x224.3_y251.5.png	t063_e5_x031.1_y112.0.png	t075_e2_x148.7_y172.3.png	t081_e4_x148.3_y171.6.png	t085_e2_x148.1_y171.1.png	t088_e5_x004.1_y029.7.png	t094_e1_x067.2_y101.9.png	t186_e5_x144.4_y150.0.png	t227_e2_x194.5_y125.1.png
t045_e3_x150.1_y093.8.png	t051_e5_x193.6_y248.3.png	t064_e3_x031.0_y112.0.png	t076_e1_x148.6_y172.2.png	t081_e5_x084.5_y241.9.png	t085_e3_x010.8_y089.5.png	t089_e1_x151.1_y086.0.png	t094_e2_x100.1_y058.2.png	t187_e2_x144.6_y150.0.png	t231_e3_x194.7_y121.6.png
t046_e1_x150.2_y093.8.png	t052_e1_x193.7_y248.2.png	t065_e2_x010.0_y243.8.png	t076_e5_x082.6_y243.0.png	t082_e1_x068.0_y103.4.png	t085_e4_x073.2_y125.7.png	t089_e2_x102.8_y061.2.png	t102_e5_x032.5_y059.2.png	t188_e3_x013.3_y178.5.png	t235_e3_x254.4_y121.3.png
t046_e2_x224.3_y251.6.png	t052_e4_x061.5_y069.0.png	t065_e5_x145.2_y177.8.png	t077_e3_x148.5_y172.1.png	t082_e2_x010.6_y089.8.png	t085_e5_x241.0_y224.9.png	t089_e3_x248.5_y117.5.png	t103_e4_x035.7_y061.5.png	t188_e5_x149.5_y001.6.png	t251_e5_x187.8_y053.9.png
t046_e3_x149.4_y230.8.png	t053_e2_x150.5_y093.8.png	t066_e2_x145.1_y178.0.png	t078_e3_x148.5_y172.0.png	t082_e3_x148.3_y171.3.png	t086_e1_x020.2_y091.4.png	t089_e4_x003.6_y141.4.png	t117_e3_x031.0_y057.0.png	t189_e1_x144.8_y150.2.png	t254_e5_x187.1_y053.8.png
t047_e2_x150.3_y093.8.png	t054_e5_x194.0_y248.2.png	t066_e4_x182.7_y022.1.png	t079_e1_x148.5_y171.7.png	t082_e4_x241.5_y225.1.png	t086_e2_x086.2_y244.0.png	t089_e5_x086.9_y241.0.png	t118_e4_x031.0_y056.9.png	t190_e3_x066.8_y244.3.png	t259_e5_x186.2_y053.8.png
t047_e3_x149.5_y230.8.png	t055_e5_x150.7_y093.8.png	t068_e2_x190.8_y138.4.png	t079_e2_x010.6_y090.3.png	t082_e5_x059.0_y074.2.png	t086_e3_x060.7_y130.9.png	t090_e1_x153.3_y091.1.png	t145_e4_x222.4_y148.0.png	t190_e4_x087.4_y023.7.png	t265_e4_x042.7_y079.7.png
t047_e4_x224.6_y251.5.png	t056_e1_x194.2_y248.2.png	t068_e3_x145.0_y178.0.png	t079_e3_x070.5_y102.3.png	t083_e1_x064.2_y073.5.png	t086_e4_x153.2_y091.8.png	t090_e2_x071.2_y101.1.png	t168_e2_x046.4_y029.4.png	t190_e5_x158.5_y180.1.png	t282_e3_x234.0_y216.6.png
t048_e4_x150.3_y093.8.png	t060_e1_x253.0_y118.0.png	t069_e1_x144.8_y178.2.png	t079_e4_x150.5_y088.3.png	t083_e2_x222.7_y158.8.png	t086_e5_x004.5_y030.0.png	t090_e3_x147.9_y170.6.png	t170_e2_x047.2_y022.2.png	t191_e2_x144.8_y150.2.png	t288_e5_x238.3_y222.7.png
t048_e5_x224.4_y251.6.png	t061_e1_x249.3_y117.2.png	t069_e2_x150.6_y090.6.png	t079_e5_x005.1_y145.5.png	t083_e3_x248.1_y132.0.png	t087_e1_x086.4_y244.3.png	t090_e4_x026.3_y102.9.png	t172_e2_x255.3_y182.2.png	t196_e4_x144.9_y150.3.png	t296_e3_x042.7_y143.8.png
t049_e1_x193.2_y248.2.png	t061_e3_x000.2_y141.7.png	t070_e2_x144.8_y178.1.png	t080_e1_x068.7_y103.4.png	t083_e4_x010.7_y089.7.png	t087_e2_x240.4_y224.8.png	t090_e5_x222.6_y158.0.png	t175_e5_x161.2_y180.0.png	t196_e5_x151.3_y002.2.png	t298_e3_x154.3_y019.6.png
t049_e2_x030.4_y114.5.png	t061_e4_x148.3_y174.5.png	t071_e3_x148.7_y172.8.png	t080_e2_x010.5_y090.1.png	t083_e5_x241.3_y225.1.png	t087_e3_x099.8_y058.8.png	t091_e1_x153.3_y091.0.png	t179_e4_x028.5_y051.7.png	t197_e1_x157.7_y179.6.png	t299_e3_x154.6_y018.5.png
"""
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


def jnorm(vmax):
    f = lambda x: np.sign(x) * np.abs(x) ** GAMMA_J
    finv = lambda y: np.sign(y) * np.abs(y) ** (1 / GAMMA_J)
    return FuncNorm((f, finv), vmin=-vmax, vmax=vmax)


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


def render_time(t, events, transpose, out_dir):
    bx, by = read("Bx", t), read("By", t)
    if bx is None or by is None:
        return []
    if transpose:
        bx, by = bx.T, by.T
    n = bx.shape[0]
    L = n * dx
    kx, ky = kgrid(n)
    k2 = kx ** 2 + ky ** 2
    bxk, byk = np.fft.fft2(bx), np.fft.fft2(by)
    pk = (-1j * ky * bxk + 1j * kx * byk) / np.where(k2 == 0, 1, k2)
    pk[0, 0] = 0
    psi = np.real(np.fft.ifft2(pk * np.exp(-0.5 * (SMOOTH_PSI * dx) ** 2 * k2)))
    jz = np.real(np.fft.ifft2((1j * kx * byk - 1j * ky * bxk) * np.exp(-0.5 * (SMOOTH_J * dx) ** 2 * k2)))
    vmax = np.percentile(np.abs(jz), CLIP_PCT)
    f = max(1, n // FULL_PX)
    jsmall = jz[: n // f * f, : n // f * f].reshape(n // f, f, n // f, f).mean(axis=(1, 3))
    m = int(round(ZOOM / dx))
    done = []
    for e in events:
        x, y = e["x"], e["y"]
        psix = map_coordinates(psi, [[y / dx], [x / dx]], order=3, mode="grid-wrap")[0]
        c0, r0 = int(round(x / dx)) - m // 2, int(round(y / dx)) - m // 2
        rr, cc = np.arange(r0, r0 + m) % n, np.arange(c0, c0 + m) % n
        jzz, pz = jz[np.ix_(rr, cc)], psi[np.ix_(rr, cc)]
        lx_ = (c0 + np.arange(m)) * dx - x
        ly_ = (r0 + np.arange(m)) * dx - y
        X, Y = np.meshgrid(lx_, ly_)
        ext = [lx_[0] - dx / 2, lx_[-1] + dx / 2, ly_[0] - dx / 2, ly_[-1] + dx / 2]

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
        a1.imshow(jsmall, origin="lower", extent=[0, L, 0, L], cmap=CMAP, norm=jnorm(vmax))
        a1.add_patch(Rectangle((x - ZOOM / 2, y - ZOOM / 2), ZOOM, ZOOM, fill=False, ec="lime", lw=2.5))
        a1.set_xlim(0, L)
        a1.set_ylim(0, L)
        a1.set_xlabel(r"$x/d_p$")
        a1.set_ylabel(r"$y/d_p$")
        a1.set_title(f"whole box, t = {t}")
        vz = np.percentile(np.abs(jzz), CLIP_PCT)
        im = a2.imshow(jzz, origin="lower", extent=ext, cmap=CMAP, norm=jnorm(vz), interpolation="bilinear")
        a2.contour(X, Y, pz, levels=N_CONTOURS, colors="k", linewidths=0.5, alpha=0.6, linestyles="solid")
        a2.contour(X, Y, pz, levels=[psix], colors="k", linewidths=1.8, linestyles="solid")
        a2.plot(0, 0, marker="x", color="lime", ms=12, mew=3)
        a2.set_xlim(-ZOOM / 2, ZOOM / 2)
        a2.set_ylim(-ZOOM / 2, ZOOM / 2)
        a2.set_xlabel(r"$(x - x_X)/d_p$")
        a2.set_ylabel(r"$(y - y_X)/d_p$")
        a2.set_title(f"{ZOOM:g} $d_p$ window: field lines, separatrix (thick), X-point (x)")
        fig.colorbar(im, ax=a2, label=r"$J_z$")
        rate = f"   rate = {e['rate']:.3g}" if e.get("rate") is not None else ""
        fig.suptitle(f"{e['id']}    X-point at ({x:.1f}, {y:.1f}) $d_p${rate}", fontsize=14)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        name = f"{e['id']}_x{x:05.1f}_y{y:05.1f}_z{int(ZOOM)}.png"
        fig.savefig(os.path.join(out_dir, name), dpi=110)
        plt.close(fig)
        done.append(dict(e, file=name))
    return done


def periodic_dist(a, b, L):
    d = np.abs(np.array(a) - np.array(b))
    d = np.minimum(d, L - d)
    return float(np.hypot(*d))


def main():
    t0 = time.time()
    out_dir = os.path.join(SCAN_DIR, f"selected_zoom{int(ZOOM)}")
    os.makedirs(out_dir, exist_ok=True)

    # which events, and where (catalog first, file name as fallback)
    cat = {}
    cat_path = os.path.join(SCAN_DIR, "catalog.csv")
    if os.path.exists(cat_path):
        with open(cat_path) as fh:
            for row in csv.DictReader(fh):
                cat[row["id"]] = row
    events, seen = [], set()
    for m in re.finditer(r"t(\d+)_e(\d+)(?:_x(\d+\.\d+)_y(\d+\.\d+))?", SELECTED):
        eid = f"t{int(m.group(1)):03d}_e{int(m.group(2))}"
        if eid in seen:
            continue
        seen.add(eid)
        if eid in cat:
            r = cat[eid]
            events.append(dict(id=eid, t=int(r["t"]), x=float(r["x"]), y=float(r["y"]), rate=float(r["rate"])))
        elif m.group(3):
            events.append(dict(id=eid, t=int(m.group(1)), x=float(m.group(3)), y=float(m.group(4)), rate=None))
        else:
            print(f"  {eid}: not in catalog and no position in the name - skipped")
    by_time = {}
    for e in events:
        by_time.setdefault(e["t"], []).append(e)
    print(f"{len(events)} events at {len(by_time)} times; window {ZOOM:g} d_p")

    # orientation from the first time (same test as the scan)
    t_first = next((t for t in sorted(by_time) if read("Bx", t) is not None), None)
    if t_first is None:
        raise SystemExit(f"none of the selected times has Bx/By files in {RUN}")
    bx, by = read("Bx", t_first), read("By", t_first)
    n = bx.shape[0]
    L = n * dx
    kx, ky = kgrid(n)
    dc = np.std(np.real(np.fft.ifft2(1j * kx * np.fft.fft2(bx) + 1j * ky * np.fft.fft2(by))))
    dr = np.std(np.real(np.fft.ifft2(1j * ky * np.fft.fft2(bx) + 1j * kx * np.fft.fft2(by))))
    transpose = dr < dc
    del bx, by

    done = []
    workers = choose_workers(len(by_time))
    print(f"workers: {workers}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(render_time, t, ev, transpose, out_dir): t for t, ev in sorted(by_time.items())}
        for k, fu in enumerate(futures, 1):
            t = futures[fu]
            try:
                res = fu.result()
            except Exception as err:
                print(f"  t={t}: FAILED ({type(err).__name__}: {err})")
                continue
            done += res
            note = "" if res else "  (no Bx/By files for this time)"
            print(f"  t={t:3d}: {len(res)} events  ({k}/{len(by_time)}, {time.time() - t0:4.0f} s){note}", flush=True)

    # ---------------- group into sites: same place (within LINK_DIST), nearby in time (MAX_GAP)
    done.sort(key=lambda e: (e["t"], e["id"]))
    parent = list(range(len(done)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(done)):
        for j in range(i + 1, len(done)):
            if abs(done[j]["t"] - done[i]["t"]) > MAX_GAP:
                continue
            if periodic_dist((done[i]["x"], done[i]["y"]), (done[j]["x"], done[j]["y"]), L) < LINK_DIST:
                parent[root(j)] = root(i)
    groups = {}
    for i, e in enumerate(done):
        groups.setdefault(root(i), []).append(e)
    sites = sorted(groups.values(), key=lambda g: (g[0]["t"], g[0]["x"]))

    sites_dir = os.path.join(out_dir, "sites")
    shutil.rmtree(sites_dir, ignore_errors=True)
    os.makedirs(sites_dir)
    rows = []
    for k, g in enumerate(sites, 1):
        xs = np.array([e["x"] for e in g])
        ys = np.array([e["y"] for e in g])
        name = f"S{k:02d}_x{xs.mean():03.0f}_y{ys.mean():03.0f}_t{g[0]['t']:03d}-{g[-1]['t']:03d}"
        os.makedirs(os.path.join(sites_dir, name))
        for e in g:
            src = os.path.join("..", "..", e["file"])
            dst = os.path.join(sites_dir, name, e["file"])
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy(os.path.join(out_dir, e["file"]), dst)
        rates = [abs(e["rate"]) for e in g if e["rate"] is not None]
        rows.append(dict(site=f"S{k:02d}", folder=name, n_events=len(g), t_first=g[0]["t"], t_last=g[-1]["t"],
                         x_mean=round(xs.mean(), 1), y_mean=round(ys.mean(), 1),
                         max_abs_rate=max(rates) if rates else "", events=" ".join(e["id"] for e in g)))
    with open(os.path.join(out_dir, "sites.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(done)} images in {out_dir}/ ({time.time() - t0:.0f} s)")
    print(f"{len(sites)} sites (sites.csv, sites/), longest-lived first:")
    for r in sorted(rows, key=lambda r: -r["n_events"])[:12]:
        print(f"  {r['site']}: {r['n_events']:2d} events, t = {r['t_first']}-{r['t_last']}, "
              f"around ({r['x_mean']}, {r['y_mean']})")


if __name__ == "__main__":
    main()