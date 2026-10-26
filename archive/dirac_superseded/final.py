"""
The two J_z neon pictures only:

  clock_Jz_neon.png   turbulent clock: one wedge per output, clockwise from 12 o'clock
  rings_Jz_neon.png   one ring per output, time running outwards like tree rings

All Bx files are found, sorted in ascending order (iteration or time number) and used in
that order. The clock and the rings can use different strides (e.g. every file for the
clock, every 10th for the rings so the rings are visible bands); each file needed by
either picture is read once, in parallel.

The raw pictures are also saved as composite_wheel_Jz.npy / composite_rings_Jz.npy, so
neon_final.py / palette_explorer.py can re-colour them without reading the data again.
"""
import glob
import os
import re
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# ------------------------------------------------------------------ settings
RUN = "/home/UNN/w24021992/NAS/DATA/high_res/more_res"
OUT_DIR = "clock_rings"
NAMING = "iteration"            # "iteration": Bx_ApJ_i15681.h5    "time": Bx_ApJ_t200.h5
FIRST, LAST = None, None        # optional range of iteration/time numbers (None = all found)
CLOCK_STEP = 10                  # clock: every file (2000 files -> 2000 thin wedges)
RINGS_STEP = 10                 # rings: every 10th file (-> visible bands); 1 = every file
dx = 0.0625

PALETTE = "neon"                # neon, neon_hot, neon_soft, neon_ice_rose (all J_z, black centre)
GAMMA_J = 0.6                   # signed power-law scaling (0.9-1 = darker gaps, less glowing)
CLIP_PCT = 99.5                 # colour range = this percentile of |J_z| (99.9 = fewer saturated pixels)
SMOOTH_J = 2.0                  # Gaussian smoothing of J_z [cells] (0 = none)
R_INNER = 0.04                  # clock: empty hub (fraction of the half-width)
OUT_PX = None                   # None = native resolution (4096 px); e.g. 3000
SAVE_COMPOSITES = True
N_WORKERS = 64
GB_PER_WORKER = 3.0

PALETTES = {
    "neon":          ["#b8f6ff", "#00a8ff", "#08306b", "#000000", "#6b1d08", "#ff6a00", "#ffe39a"],
    "neon_hot":      ["#ffffff", "#9ff3ff", "#00b8ff", "#003a8c", "#000000", "#7a1f00", "#ff6a00",
                      "#ffd38a", "#ffffff"],
    "neon_soft":     ["#45d4ff", "#0a8fd6", "#06305e", "#000000", "#5a1e05", "#d65a00", "#ff9a3c"],
    "neon_ice_rose": ["#f0fbff", "#7fdcff", "#1e7bff", "#0b1f4d", "#000000", "#4d0b26", "#ff2d6f",
                      "#ff9fbf", "#fff0f5"],
}
# ------------------------------------------------------------------------------
PREFIX = "i" if NAMING == "iteration" else "t"


def fname(tag, num):
    return os.path.join(RUN, f"{tag}_ApJ_{PREFIX}{num}.h5")


def read(tag, num):
    with h5py.File(fname(tag, num), "r") as f:
        return f["DS1"][:].T.astype(np.float64)


def outputs():
    """All output numbers with both Bx and By, ascending."""
    nums = []
    for fn in glob.glob(os.path.join(RUN, f"Bx_ApJ_{PREFIX}*.h5")):
        m = re.search(rf"_{PREFIX}(\d+)\.h5$", fn)
        if m:
            nums.append(int(m.group(1)))
    nums = sorted(n for n in nums if (FIRST is None or n >= FIRST) and (LAST is None or n <= LAST))
    ok = [n for n in nums if os.path.exists(fname("By", n))]
    if len(ok) < len(nums):
        print(f"  {len(nums) - len(ok)} outputs have Bx but no By - skipped")
    return ok


def x_along_columns(bx, by):
    """True if x already runs along columns (div B = 0 decides)."""
    n = bx.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = k[None, :], k[:, None]
    dc = np.std(np.real(np.fft.ifft2(1j * kx * np.fft.fft2(bx) + 1j * ky * np.fft.fft2(by))))
    dr = np.std(np.real(np.fft.ifft2(1j * ky * np.fft.fft2(bx) + 1j * kx * np.fft.fft2(by))))
    return dc <= dr


def jz_of(num, transpose):
    bx, by = read("Bx", num), read("By", num)
    if transpose:
        bx, by = bx.T, by.T
    n = bx.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = k[None, :], k[:, None]
    filt = np.exp(-0.5 * (SMOOTH_J * dx) ** 2 * (kx ** 2 + ky ** 2))
    return np.real(np.fft.ifft2((1j * kx * np.fft.fft2(by) - 1j * ky * np.fft.fft2(bx)) * filt))


def worker(num, transpose, idx):
    """J_z of one output, only at the pixels it contributes to (clock wedge and/or ring)."""
    jz = jz_of(num, transpose).ravel()
    return num, {g: jz[i].astype(np.float32) for g, i in idx.items()}


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


def groups(index_map, n_slices):
    """Pixel indices belonging to each slice (wedge or ring)."""
    flat = index_map.ravel()
    order = np.argsort(flat, kind="stable")
    start = np.searchsorted(flat[order], 0)
    counts = np.bincount(flat[flat >= 0], minlength=n_slices)
    return np.split(order[start:], np.cumsum(counts)[:-1])


def render(v, path):
    """Signed power-law neon colouring; NaN (hub) -> black; native or OUT_PX size."""
    cmap = LinearSegmentedColormap.from_list(PALETTE, PALETTES[PALETTE])
    vmax = np.nanpercentile(np.abs(v), CLIP_PCT)
    s = np.sign(v) * np.clip(np.abs(np.nan_to_num(v)) / vmax, 0, 1) ** GAMMA_J
    rgb = cmap(0.5 + 0.5 * s)[..., :3]
    rgb[np.isnan(v)] = 0
    img = Image.fromarray((rgb[::-1] * 255).astype(np.uint8), "RGB")
    if OUT_PX and img.width != OUT_PX:
        img = img.resize((OUT_PX, OUT_PX), Image.LANCZOS)
    img.save(path, dpi=(300, 300))
    mb = os.path.getsize(path) / 1e6
    note = ""
    if mb > 20:
        img.save(path[:-4] + ".jpg", quality=95, dpi=(300, 300))
        note = f"; over 20 MB, so also {os.path.basename(path[:-4])}.jpg for submission"
    print(f"  saved {path} ({img.width} px, {mb:.1f} MB{note})")


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    nums = outputs()
    if not nums:
        raise SystemExit(f"no Bx_ApJ_{PREFIX}*.h5 / By files in {RUN}")
    clock, rings = nums[::CLOCK_STEP], nums[::RINGS_STEP]
    print(f"found {len(nums)} outputs ({PREFIX}{nums[0]} .. {PREFIX}{nums[-1]}): "
          f"clock {len(clock)} wedges, rings {len(rings)} rings")

    bx, by = read("Bx", nums[0]), read("By", nums[0])
    transpose = not x_along_columns(bx, by)
    n = bx.shape[0]
    del bx, by

    # geometry: whole square box, centre of the box
    r = n // 2
    Y, X = np.mgrid[0:n, 0:n] + 0.5 - r
    rad = np.hypot(X, Y)
    theta = np.mod(np.arctan2(X, Y), 2 * np.pi)                    # clockwise from 12 o'clock
    wedge = np.minimum((theta / (2 * np.pi) * len(clock)).astype(np.int32), len(clock) - 1)
    wedge[rad < R_INNER * r] = -1
    ring = np.minimum((rad / (r * np.sqrt(2)) * len(rings)).astype(np.int32), len(rings) - 1)
    g_clock, g_rings = groups(wedge, len(clock)), groups(ring, len(rings))
    del X, Y, rad, theta, wedge, ring
    k_clock = {num: k for k, num in enumerate(clock)}
    k_rings = {num: k for k, num in enumerate(rings)}
    values = {"wheel": np.full(n * n, np.nan, np.float32), "rings": np.full(n * n, np.nan, np.float32)}

    needed = sorted(set(clock) | set(rings))
    workers = choose_workers(len(needed))
    print(f"reading {len(needed)} outputs with {workers} parallel workers ...")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = []
        for num in needed:
            idx = {}
            if num in k_clock:
                idx["wheel"] = g_clock[k_clock[num]]
            if num in k_rings:
                idx["rings"] = g_rings[k_rings[num]]
            futs.append(pool.submit(worker, num, transpose, idx))
        for done, fu in enumerate(futs, 1):
            num, vals = fu.result()
            if "wheel" in vals:
                values["wheel"][g_clock[k_clock[num]]] = vals["wheel"]
            if "rings" in vals:
                values["rings"][g_rings[k_rings[num]]] = vals["rings"]
            if done % 50 == 0 or done == len(needed):
                el = time.time() - t0
                print(f"  {done}/{len(needed)}  {el:5.0f} s, ~{el / done * (len(needed) - done):5.0f} s to go",
                      flush=True)

    for g, v in values.items():
        v = v.reshape(n, n)
        if SAVE_COMPOSITES:
            np.save(os.path.join(OUT_DIR, f"composite_{g}_Jz.npy"), v)
        name = "clock" if g == "wheel" else "rings"
        render(v, os.path.join(OUT_DIR, f"{name}_Jz_{PALETTE}.png"))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()