"""
1D cuts at an irrational angle through 2D periodic HDF5 fields.
Everything (cuts for all times, geometry, simulation parameters) goes into ONE h5 file.
Requires cut1d.py in the same folder.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # no window; just saves PNGs. Remove if you want plt.show()
import matplotlib.pyplot as plt
import h5py
from cut1d import irrational_cut, GOLDEN
from shared.fields import partial_x_fft, partial_y_fft

plt.rcParams.update({'font.size': 15, 'axes.titlesize': 15, 'axes.labelsize': 15,
                     'xtick.labelsize': 15, 'ytick.labelsize': 15,
                     'legend.fontsize': 15, 'figure.titlesize': 15})

# =====================================================================
# SETTINGS
# =====================================================================
times     = [200]                          # one or many; all end up in the same h5 file
base_path = "/home/UNN/w24021992/NAS/DATA/Brms_0.24/beta_p_1/beta_e_1/"
dx = dy   = 0.0625

raw_tags  = ["Bx", "By", "Bz", "Ex", "Ey", "Ez", "Dn1", "Ux1", "Uy1", "Uz1", "Ppar1", "Pper1"]             # fields read straight from files: <tag>_ApJ_t<time>.h5

# Anything you want cut: raw tags AND/OR derived names (keys returned by derived() below)
cut_quantities = ["Bx", "By", "Bz", "Dn1"]  # names of quantities to cut along the irrational line

# Vectors -> adds <name>_par, <name>_perp, <name>_z along the cut. Components can be raw or derived.
vectors   = {}                             # e.g. {"B": ("Bx", "By", "Bz"), "J": ("Jx", "Jy", "Jz")}

# What to show as the 2D background map on the left of the plot (raw or derived name)
plot_field = "Jz"

# Cut geometry
slope     = 1 / GOLDEN                     # irrational slope dy/dx (theta = arctan(slope))
x0, y0    = 0.0, 0.0
n_points  = 20000                    # cap on number of samples
ds        = dx                             # step along the cut
order     = 1                              # 1 = bilinear, 3 = cubic spline (both periodic)
swap_axes = False                          # True if arrays come out as [iy, ix]

# Output
out_h5    = "cuts.h5"
save_png  = True
traj_size = 0.1                          # trajectory marker size on the map (smaller = thinner)

# Simulation parameters stored in the h5 file (/parameters). Just add "name": value lines.
PARAMS = {
    # "dt": 0.01,
    # "n_particles": 1.0e8,
    # "beta_p": 1.0,
    # "beta_e": 1.0,
    "Brms": 0.24
}


# =====================================================================
# DERIVED QUANTITIES  (you fill this in)
# `f` is a dict of the raw 2D arrays {tag: array[ix, iy]} for this time.
# Return a dict {name: 2D array}. Names can then be used in cut_quantities,
# vectors and plot_field above. dx, dy are available as globals.
# =====================================================================
def derived(f):
    d = {}
    d["Jx"] =  partial_y_fft(f["Bz"], dx)
    d["Jy"] =  -partial_x_fft(f["Bz"], dx)
    d["Jz"] =  partial_x_fft(f["By"], dx) - partial_y_fft(f["Bx"], dx)
    d["Tper1"] = f["Pper1"] / f["Dn1"]
    d["Tpar1"] = f["Ppar1"] / f["Dn1"]
    return d


# =====================================================================
# LOAD
# =====================================================================
def load(tag, t):
    with h5py.File(f"{base_path}{tag}_ApJ_t{t}.h5", "r") as f:
        return f["DS1"][:].T


def get_all_fields(t):
    raw = {tag: load(tag, t) for tag in raw_tags}
    if swap_axes:
        raw = {k: v.T for k, v in raw.items()}
    allf = dict(raw)
    allf.update(derived(raw))              # derived can be used right next to raw ones
    return allf


# =====================================================================
# PLOT
# =====================================================================
def make_plot(t, fields, s, x, y, out, Lx, Ly):
    names = list(out.keys())
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(len(names), 2, width_ratios=[1, 1.2])

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(fields[plot_field].T, origin="lower", cmap="seismic",
              extent=[0, Lx, 0, Ly], aspect="equal", vmin=-0.5, vmax=0.5)
    sc = ax.scatter(x, y, c=s, s=traj_size, cmap="grey_r", alpha=0.8)
    ax.plot(x[0], y[0], "go", label="start")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"{plot_field}, t={t}")
    fig.colorbar(sc, ax=ax, label="s along cut", fraction=0.046, pad=0.04)
    ax.legend(loc="upper right")

    for i, name in enumerate(names):
        a = fig.add_subplot(gs[i, 1])
        a.plot(s, out[name], lw=0.8)
        a.set_ylabel(name)
        if i < len(names) - 1:
            a.tick_params(labelbottom=False)
    a.set_xlabel("s")
    plt.tight_layout()
    plt.savefig(f"cut_t{t}.png", dpi=150)
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================
def main():
    with h5py.File(out_h5, "w") as h:
        # ---- parameters (same for every time) ----
        gp = h.create_group("parameters")
        for k, v in PARAMS.items():
            gp.attrs[k] = v

        # ---- geometry / settings ----
        gg = h.create_group("geometry")
        for k, v in dict(dx=dx, dy=dy, slope=slope, x0=x0, y0=y0, ds=ds,
                         n_points=n_points, order=order).items():
            gg.attrs[k] = v

        h.create_dataset("times", data=np.array(times))
        wrote_geom = False

        for t in times:
            fields = get_all_fields(t)
            Nx, Ny = fields[plot_field].shape
            Lx, Ly = Nx * dx, Ny * dy

            to_cut = {k: fields[k] for k in cut_quantities}
            s, x, y, out = irrational_cut(to_cut, Lx, Ly, slope=slope, x0=x0, y0=y0,
                                          length=n_points * ds, ds=ds, order=order,
                                          vectors=vectors)

            if not wrote_geom:             # s, x, y are identical for every time
                for name, arr in (("s", s), ("x", x), ("y", y)):
                    h.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
                gg.attrs["Lx"], gg.attrs["Ly"] = Lx, Ly
                wrote_geom = True

            gt = h.create_group(f"t{t}")
            for name, arr in out.items():
                gt.create_dataset(name, data=arr, compression="gzip", compression_opts=4)

            if save_png:
                make_plot(t, fields, s, x, y, out, Lx, Ly)
            print(f"t={t}: wrote {len(out)} cuts")

    print(f"Saved -> {out_h5}")


if __name__ == "__main__":
    main()