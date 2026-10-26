"""
Videos of the selected events: each event's box (ZOOM x ZOOM d_p, fixed in space)
from T_FIRST to T_LAST, one frame per output time.

Stage 1 (parallel over times): every output time is read ONCE and a frame is drawn for
          every video -> frames/<video name>/f<t>.jpg
Stage 2 (parallel over videos): frames are joined into MP4 with ffmpeg -> videos/

Each frame: J_z (fixed colour scale for the whole film, so brightness is comparable in
time), field lines with a fixed spacing (so they move smoothly instead of jumping),
the event's X-point position (small +), the time, and a timeline at the bottom with the
event's own time marked in green.

VIDEO_PER = "event": one video per selected event (180), named by event ID.
VIDEO_PER = "site":  one video per reconnection site from rerender_events.py
                     (sites.csv), far fewer and no near-duplicate films.

Resumable: rerun after an interruption and finished times / videos are skipped.
ffmpeg: uses the system ffmpeg, or the one from `pip install imageio-ffmpeg`.

Command line (overrides the settings below), e.g.
  python event_videos.py --ids t190_e3 --zoom 96 --axes        one event, box twice as big,
                                                               with x/y coordinates on the frame
  python event_videos.py --center 80 230 --zoom 48 --axes      a box centred anywhere you like
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
from matplotlib.colors import FuncNorm
from matplotlib.ticker import FuncFormatter

# ------------------------------------------------------------------ settings
RUN = "/home/UNN/w24021992/NAS/DATA/Brms_0.24/beta_p_1/beta_e_1/"
SCAN_DIR = "events_bp1_be1"            # scan_events.py output (catalog.csv)
SELECTED_DIR = "selected_zoom48"       # rerender_events.py output inside SCAN_DIR (sites.csv)
VIDEO_PER = "event"                    # "event" (one per selected event) or "site"
T_FIRST, T_LAST = 0, 300
dx = 0.0625
ZOOM = 48.0                            # box size [d_p]
FPS = 15                               # frames (= output times) per second: 301 frames -> 20 s
FRAME_PX = 720                         # video size in pixels
REF_T = 200                            # time used to fix the colour scale and field-line spacing
N_CONTOURS = 24                        # about this many field lines across a box
SMOOTH_PSI = 2.0
SMOOTH_J = 2.0
GAMMA_J = 0.6
CMAP = "RdBu_r"
KEEP_FRAMES = False                    # delete the JPEG frames once a video is made
SHOW_AXES = False                      # x/y coordinates (whole-box d_p) around each frame
N_WORKERS = "auto"
GB_PER_WORKER = 4.0

# the same 180 events as in rerender_events.py (IDs are enough; positions come from catalog.csv)
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


def kgrid(n):
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    return k[None, :], k[:, None]


def fields(t, transpose):
    """Smoothed flux function psi and J_z (x along columns, y along rows)."""
    bx, by = read("Bx", t), read("By", t)
    if bx is None or by is None:
        return None, None
    if transpose:
        bx, by = bx.T, by.T
    n = bx.shape[0]
    kx, ky = kgrid(n)
    k2 = kx ** 2 + ky ** 2
    bxk, byk = np.fft.fft2(bx), np.fft.fft2(by)
    pk = (-1j * ky * bxk + 1j * kx * byk) / np.where(k2 == 0, 1, k2)
    pk[0, 0] = 0
    psi = np.real(np.fft.ifft2(pk * np.exp(-0.5 * (SMOOTH_PSI * dx) ** 2 * k2)))
    jz = np.real(np.fft.ifft2((1j * kx * byk - 1j * ky * bxk) * np.exp(-0.5 * (SMOOTH_J * dx) ** 2 * k2)))
    return psi, jz


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


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def frames_for_time(t, videos, transpose, vmax, dpsi, frame_dir, p):
    zoom, px, axes_on = p["zoom"], p["px"], p["axes"]
    psi, jz = fields(t, transpose)
    if psi is None:
        return t, 0
    n = psi.shape[0]
    L = n * dx
    m = int(round(zoom / dx))
    st = max(1, m // 384)                          # contour on a coarser grid (fast, looks the same)
    norm = jnorm(vmax)
    for v in videos:
        x, y = v["x"], v["y"]
        c0, r0 = int(round(x / dx)) - m // 2, int(round(y / dx)) - m // 2
        rr, cc = np.arange(r0, r0 + m) % n, np.arange(c0, c0 + m) % n
        jw, pw = jz[np.ix_(rr, cc)], psi[np.ix_(rr, cc)]
        lx_ = (c0 + np.arange(m)) * dx - x
        ly_ = (r0 + np.arange(m)) * dx - y

        fig = plt.figure(figsize=(px / 100, px / 100), dpi=100, facecolor="black")
        ax = fig.add_axes([0.09, 0.14, 0.88, 0.84] if axes_on else [0, 0.06, 1, 0.94])
        ax.imshow(jw, origin="lower", cmap=CMAP, norm=norm, interpolation="bilinear",
                  extent=[lx_[0], lx_[-1] + dx, ly_[0], ly_[-1] + dx])
        lo, hi = np.floor(pw.min() / dpsi), np.ceil(pw.max() / dpsi)
        if hi - lo < 400:
            X, Y = np.meshgrid(lx_[::st], ly_[::st])
            ax.contour(X, Y, pw[::st, ::st], levels=np.arange(lo, hi + 1) * dpsi, colors="k",
                       linewidths=0.5, alpha=0.6, linestyles="solid")
        ax.plot(0, 0, marker="+", color="lime", ms=14, mew=2, alpha=0.9)
        ax.set_xlim(-zoom / 2, zoom / 2)
        ax.set_ylim(-zoom / 2, zoom / 2)
        if axes_on:   # label the ticks with whole-box coordinates, so positions can be read off
            ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos, c=x: f"{(c + val) % L:.0f}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos, c=y: f"{(c + val) % L:.0f}"))
            ax.tick_params(colors="white", labelsize=10)
            ax.set_xlabel(r"$x/d_p$", color="white")
            ax.set_ylabel(r"$y/d_p$", color="white")
            ax.grid(color="white", alpha=0.25, lw=0.5)
            for sp in ax.spines.values():
                sp.set_edgecolor("white")
        else:
            ax.axis("off")
        here = "   <- event time" if t in v["t_marks"] else ""
        ax.text(0.015, 0.975, f"{v['name']}    t = {t}{here}", transform=ax.transAxes, va="top",
                fontsize=11, color="k", bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))
        tl = fig.add_axes([0.03, 0.015, 0.94, 0.03], facecolor="black")   # timeline
        tl.plot([p["t_first"], p["t_last"]], [0, 0], color="0.6", lw=2)
        for te in v["t_marks"]:
            tl.plot([te, te], [-1, 1], color="lime", lw=1.5)
        tl.plot([t], [0], marker="o", color="white", ms=7)
        tl.set_xlim(p["t_first"] - 2, p["t_last"] + 2)
        tl.set_ylim(-1.2, 1.2)
        tl.axis("off")
        fig.savefig(os.path.join(frame_dir, v["name"], f"f{t:03d}.jpg"), dpi=100,
                    facecolor="black", pil_kwargs={"quality": 90})
        plt.close(fig)
    return t, len(videos)


def encode(name, frame_dir, out_dir, exe, fps=None, keep=None):
    fps = fps or FPS
    keep = KEEP_FRAMES if keep is None else keep
    out = os.path.join(out_dir, f"{name}.mp4")
    files = sorted(glob.glob(os.path.join(frame_dir, name, "f*.jpg")))
    if not files:
        return name, "no frames"
    lst = os.path.join(frame_dir, name, "frames.txt")          # concat list: robust to gaps in time
    with open(lst, "w") as fh:
        for f in files:
            fh.write(f"file '{os.path.abspath(f)}'\nduration {1 / fps:.6f}\n")
        fh.write(f"file '{os.path.abspath(files[-1])}'\n")
    cmd = [exe, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", "-r", str(fps), out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return name, f"ffmpeg failed: {r.stderr.strip()[:300]}"
    if not keep:
        shutil.rmtree(os.path.join(frame_dir, name), ignore_errors=True)
    return name, "ok"


def main():
    ap = argparse.ArgumentParser(description="Videos of reconnection events (see the top of this file).")
    ap.add_argument("--ids", nargs="+", help="event IDs, e.g. t190_e3 t072_e2 (default: SELECTED list)")
    ap.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"), help="film a box centred here [d_p]")
    ap.add_argument("--name", help="video name for --center (default box_x..._y..._z...)")
    ap.add_argument("--zoom", type=float, default=ZOOM, help=f"box size in d_p (default {ZOOM:g})")
    ap.add_argument("--px", type=int, default=FRAME_PX, help=f"video size in pixels (default {FRAME_PX})")
    ap.add_argument("--contours", type=int, default=N_CONTOURS, help="field lines across a box")
    ap.add_argument("--axes", action="store_true", default=SHOW_AXES, help="show x/y coordinates")
    ap.add_argument("--per", choices=["event", "site"], default=VIDEO_PER)
    ap.add_argument("--fps", type=int, default=FPS)
    a = ap.parse_args()
    zoom = a.zoom
    out_dir = os.path.join(SCAN_DIR, f"videos_zoom{zoom:g}")
    p = dict(zoom=zoom, px=a.px, axes=a.axes, t_first=T_FIRST, t_last=T_LAST)

    t0 = time.time()
    frame_dir = os.path.join(out_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    # ---------------- what to film
    cat = {}
    with open(os.path.join(SCAN_DIR, "catalog.csv")) as fh:
        for row in csv.DictReader(fh):
            cat[row["id"]] = row
    selected = " ".join(a.ids) if a.ids else SELECTED
    if a.center:
        cx, cy = a.center
        videos = [dict(name=a.name or f"box_x{cx:05.1f}_y{cy:05.1f}_z{zoom:g}", x=cx, y=cy, t_event=-1, t_marks=[])]
    elif a.per == "site":
        videos = []
        with open(os.path.join(SCAN_DIR, SELECTED_DIR, "sites.csv")) as fh:
            for row in csv.DictReader(fh):
                ids = row["events"].split()
                tm = [int(cat[i]["t"]) for i in ids if i in cat]
                videos.append(dict(name=row["folder"], x=float(row["x_mean"]), y=float(row["y_mean"]),
                                   t_event=tm[0] if tm else -1, t_marks=tm))
    else:
        videos, seen = [], set()
        for m in re.finditer(r"t(\d+)_e(\d+)", selected):
            eid = f"t{int(m.group(1)):03d}_e{int(m.group(2))}"
            if eid in seen or eid not in cat:
                if eid not in cat:
                    print(f"  {eid} not in catalog - skipped")
                continue
            seen.add(eid)
            r = cat[eid]
            x, y = float(r["x"]), float(r["y"])
            videos.append(dict(name=f"{eid}_x{x:05.1f}_y{y:05.1f}_z{zoom:g}", x=x, y=y,
                               t_event=int(r["t"]), t_marks=[int(r["t"])]))
    for v in videos:
        os.makedirs(os.path.join(frame_dir, v["name"]), exist_ok=True)
    kind = "centred box" if a.center else a.per
    print(f"{len(videos)} video(s) ({kind}), box {zoom:g} d_p, {a.px} px, t = {T_FIRST}..{T_LAST} -> {out_dir}/")

    # ---------------- fixed colour scale and field-line spacing (from REF_T)
    bx = read("Bx", REF_T)
    by = read("By", REF_T)
    if bx is None:
        raise SystemExit(f"REF_T = {REF_T}: Bx not found")
    n = bx.shape[0]
    kx, ky = kgrid(n)
    dc = np.std(np.real(np.fft.ifft2(1j * kx * np.fft.fft2(bx) + 1j * ky * np.fft.fft2(by))))
    dr = np.std(np.real(np.fft.ifft2(1j * ky * np.fft.fft2(bx) + 1j * kx * np.fft.fft2(by))))
    transpose = dr < dc
    b_rms = float(np.sqrt(np.mean(bx ** 2 + by ** 2)))
    del bx, by
    _, jz_ref = fields(REF_T, transpose)
    vmax = float(np.percentile(np.abs(jz_ref), 99.5))
    dpsi = b_rms * zoom / a.contours
    del jz_ref
    print(f"colour scale |J_z| <= {vmax:.3g} (t = {REF_T}); field-line spacing {dpsi:.3g}")

    # ---------------- stage 1: frames, parallel over times
    # resume per video AND per time: a frame is drawn only if that video has no MP4 yet
    # and that frame does not exist yet (so new videos in an existing folder get their frames)
    pending = [v for v in videos if not os.path.exists(os.path.join(out_dir, f"{v['name']}.mp4"))]
    work = {}
    for t in range(T_FIRST, T_LAST + 1):
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
                    t, nv = fu.result()
                except Exception as err:
                    print(f"  a time failed: {type(err).__name__}: {err}")
                    continue
                el = time.time() - t0
                if k % 10 == 0 or k == len(times):
                    print(f"  {k}/{len(times)} times  ({el:5.0f} s, ~{el / k * (len(times) - k):5.0f} s to go)",
                          flush=True)
    else:
        print("stage 1: nothing to draw (videos already made or all frames present)")

    # ---------------- stage 2: encode, parallel over videos
    exe = ffmpeg_exe()
    if exe is None:
        print("\nffmpeg not found. Install one of:\n  pip install imageio-ffmpeg\n"
              "  conda install -c conda-forge ffmpeg\nthen rerun: frames are kept, only encoding runs.")
        return
    todo = [v["name"] for v in pending]
    print(f"stage 2: encoding {len(todo)} video(s)")
    made, problems = 0, 0
    if todo:
        with ProcessPoolExecutor(max_workers=max(1, min((os.cpu_count() or 2) - 2, len(todo), 32))) as pool:
            for name, status in pool.map(encode, todo, [frame_dir] * len(todo), [out_dir] * len(todo),
                                         [exe] * len(todo), [a.fps] * len(todo), [KEEP_FRAMES] * len(todo)):
                if status == "ok":
                    made += 1
                    print(f"  made {os.path.join(out_dir, name + '.mp4')}")
                else:
                    problems += 1
                    print(f"  {name}: {status}")
    already = len(videos) - len(pending)
    print(f"\ndone: {made} new video(s), {already} already existed, {problems} problem(s) "
          f"-> {out_dir}/ ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()