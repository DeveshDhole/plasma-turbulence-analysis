#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage import morphology
from skimage import measure
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
from scipy.stats import gaussian_kde
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates
import seaborn as sns
from scipy.ndimage import maximum_filter, label
from scipy.stats import skewnorm
from scipy.stats import lognorm
import fifpy
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label, gaussian_filter
from scipy.stats import lognorm, gaussian_kde

# shared code and paths (see README.md and shared/)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repository root
from shared.fields import partial_x_fft, partial_y_fft
from shared.filtering import iterative_filtering1D, iterative_filtering3D
from shared.detection import detect_current_sheets             # 5 x J2_rms global, 0.5 x peak local
from shared.thickness import gaussian
from shared.stats import lognorm_mode_with_error, lognorm_mode
from shared.paths import DATA_ROOT, results_dir, cache_dir as shared_cache_dir
RESULTS = results_dir("cs_identification", "ElectronBetaScan")


# In[2]:


#Defining all the required functions



#Fourier Transform Functions




#Defining constants
dx = 0.0625
dy = 0.0625








# In[3]:


# =====================================================
# IMPORTS
# =====================================================
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from scipy.ndimage import label, gaussian_filter
from scipy.ndimage import distance_transform_edt as _edt
import scipy.ndimage as ndi
from scipy.optimize import curve_fit
from scipy.stats import lognorm
from skimage import morphology, measure
from skimage.segmentation import find_boundaries

# -----------------------------------------------------
# EXTERNAL DEPENDENCIES — these must be importable/defined.
# If they lived in earlier notebook cells, paste them above,
# or import them from your module, e.g.:
#   from my_turbulence_tools import (iterative_filtering3D,
#                                    partial_x_fft, partial_y_fft,
#                                    detect_current_sheets)
# -----------------------------------------------------
# iterative_filtering3D(Bx, By, Bz, n_iter, Xi)  -> filtered Bx, By, Bz
# partial_x_fft(field, d)                         -> d/dx via FFT
# partial_y_fft(field, d)                         -> d/dy via FFT
# detect_current_sheets(J2)                       -> list of sheets (lists of (i,j))


# =====================================================
# USER SETTINGS
# =====================================================
beta_es      = [0.0625, 1, 16]   # electron beta values to scan
beta_p       = 1                 # proton beta, held FIXED
snapshots    = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
t_ref        = 40                # reference snapshot for the image rows

# grid spacing — SET THESE to your simulation's values
dx           = 1.0
dy           = 1.0

# conversion from pixels to ion inertial lengths (cells per d_i)
cells_per_di = 16.0

# data location (absolute path — unaffected by moving the notebook)
data_root    = str(DATA_ROOT)

# output / cache locations (relative — created in the notebook's current dir)
cache_dir          = str(shared_cache_dir("cs_identification", "ElectronBetaScan", "filtered_cache"))
parent_save_folder = str(RESULTS / "gaussian_fits_J2_NR")

# M2/M3 analysis params
max_len      = 60
sigma_smooth = 3
step_size    = 1
max_steps    = 1000
factor       = 200.0    # arrow length in the image rows

# detection / cleaning
morph_disk_radius = 5
min_object_size   = 500

# performance: only save this many example Gaussian-fit PNGs per (beta_e, t)
n_example_fits = 10

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(parent_save_folder, exist_ok=True)


# =====================================================
# HELPER FUNCTIONS
# =====================================================



def get_max_coords(cleaned_mask, J2):
    """Per-region argmax of |J2|. Returns list of (x, y) and the labelled mask."""
    labeled_mask, num_features = label(cleaned_mask)
    max_coords = []
    for i in range(1, num_features + 1):
        region = (labeled_mask == i)
        region_vals = J2[region]
        if region_vals.size == 0:
            continue
        max_pos = np.argmax(np.abs(region_vals))
        xs_r, ys_r = np.where(region)
        max_coords.append((xs_r[max_pos], ys_r[max_pos]))
    return max_coords, labeled_mask





# =====================================================
# STAGE 1: load fields + detect sheets for all (beta_e, t)
# =====================================================
J2_all   = {}    # keyed by (beta_e, t)
mask_all = {}    # keyed by (beta_e, t)

# quick existence check before the slow loop
print("Checking data directories...")
for beta_e in beta_es:
    p = f'{data_root}/beta_p_{beta_p}/beta_e_{beta_e}/'
    print(f"  beta_e={beta_e}: exists={os.path.isdir(p)}  ({p})")

for beta_e in beta_es:
    # beta_p is fixed at 1, so the filter parameters are fixed too.
    n_iter, Xi = 2, 2.0

    for t in snapshots:
        print(f"\n--- beta_e = {beta_e}, t = {t} ---")
        base_path = f'{data_root}/beta_p_{beta_p}/beta_e_{beta_e}/'
        File_Bx = base_path + f'Bx_ApJ_t{t}.h5'
        File_By = base_path + f'By_ApJ_t{t}.h5'
        File_Bz = base_path + f'Bz_ApJ_t{t}.h5'

        cache_file = (f"{cache_dir}/filtered_betae_{beta_e}_betap_{beta_p}"
                      f"_t{t}_iter{n_iter}_Xi{Xi}.npz")

        if os.path.exists(cache_file):
            print("  loading cached filtered fields...")
            data = np.load(cache_file)
            data_Bx_f = data['Bx']
            data_By_f = data['By']
            data_Bz_f = data['Bz']
        else:
            print("  computing filtering (slow)...")
            with h5py.File(File_Bx, 'r') as fBx, \
                 h5py.File(File_By, 'r') as fBy, \
                 h5py.File(File_Bz, 'r') as fBz:
                data_Bx = fBx['DS1'][:].T
                data_By = fBy['DS1'][:].T
                data_Bz = fBz['DS1'][:].T
            data_Bx_f, data_By_f, data_Bz_f = iterative_filtering3D(
                data_Bx, data_By, data_Bz, n_iter, Xi=Xi
            )
            np.savez_compressed(cache_file,
                                Bx=data_Bx_f, By=data_By_f, Bz=data_Bz_f)

        # current density
        Jx = partial_y_fft(data_Bz_f, dx)
        Jy = -partial_x_fft(data_Bz_f, dy)
        Jz = partial_x_fft(data_By_f, dy) - partial_y_fft(data_Bx_f, dx)
        J2 = Jx**2 + Jy**2 + Jz**2

        # detection + morphological cleaning  (kept as bool — see note)
        sheets = detect_current_sheets(J2)
        mask = np.zeros_like(J2)
        for sheet in sheets:
            for (i, j) in sheet:
                mask[i, j] = 1
        struct = morphology.disk(morph_disk_radius)
        mask = morphology.binary_opening(mask, struct)
        mask = morphology.binary_closing(mask, struct)
        mask = morphology.remove_small_objects(
            mask.astype(bool), min_size=min_object_size
        )   # leave as bool — safe for both indexing and morphology

        J2_all[(beta_e, t)]   = J2
        mask_all[(beta_e, t)] = mask

# global colour scale across ALL snapshots and betas
all_values  = np.concatenate([J2_all[k].ravel() for k in J2_all])
vmax_global = np.percentile(all_values, 99)


# =====================================================
# STAGE 2: pooled M1 / M2 / M3 analysis + figure
# =====================================================
fig, axes = plt.subplots(5, len(beta_es), figsize=(4.5*len(beta_es), 22.5))
if len(beta_es) == 1:
    axes = axes.reshape(5, 1)

# collect summary numbers for an end-of-run table
summary = {}

for idx, beta_e in enumerate(beta_es):
    ax_img_m1  = axes[0, idx]
    ax_hist_m1 = axes[1, idx]
    ax_img_m2  = axes[2, idx]
    ax_hist_m2 = axes[3, idx]
    ax_hist_m3 = axes[4, idx]

    # ----------------------------------------
    # IMAGE ROWS — reference snapshot only
    # ----------------------------------------
    cleaned_mask_ref = mask_all[(beta_e, t_ref)]
    J2_ref           = J2_all[(beta_e, t_ref)]
    max_coords_ref, labeled_mask_ref = get_max_coords(cleaned_mask_ref, J2_ref)
    boundaries_ref = find_boundaries(labeled_mask_ref)

    # Row 0 — field + maxima
    im_m1 = ax_img_m1.imshow(J2_ref.T, cmap='Reds', origin='lower',
                             vmin=0, vmax=vmax_global)
    ax_img_m1.contour(boundaries_ref.T, levels=[0.5], colors='black', linewidths=1)
    for x, y in max_coords_ref:
        ax_img_m1.plot(x, y, marker='x', markersize=10,
                       markeredgewidth=1.5, color='blue')
    ax_img_m1.set_title(rf'$\beta_e = {beta_e}$ (t={t_ref})')
    ax_img_m1.set_xlabel(r'$x/d_p$')
    ax_img_m1.set_ylabel(r'$y/d_p$')
    cbar_m1 = fig.colorbar(im_m1, ax=ax_img_m1, fraction=0.046, pad=0.04)
    cbar_m1.set_label(r'$J^2$')

    # descent directions for the reference frame (arrows only)
    J2_smooth_ref      = gaussian_filter(J2_ref, sigma=sigma_smooth)
    dJdx_ref, dJdy_ref = np.gradient(J2_smooth_ref)
    descent_dirs_ref = []
    for (x, y) in max_coords_ref:
        grad = np.array([dJdx_ref[x, y], dJdy_ref[x, y]])
        if np.allclose(grad, 0):
            descent_dirs_ref.append(np.array([0.0, 0.0]))
        else:
            descent_dirs_ref.append(-grad / np.linalg.norm(grad))

    # Row 2 — field + descent arrows
    im_m2 = ax_img_m2.imshow(J2_ref.T, cmap='Reds', origin='lower',
                             vmin=0, vmax=vmax_global, aspect='equal', zorder=0)
    ax_img_m2.contour(boundaries_ref.T, levels=[0.5], colors='black',
                      linewidths=1, zorder=2)
    for (x, y), (dx_, dy_) in zip(max_coords_ref, descent_dirs_ref):
        start = (x - 0.5*factor*dx_, y - 0.5*factor*dy_)
        end   = (x + 0.5*factor*dx_, y + 0.5*factor*dy_)
        arrow = FancyArrowPatch(start, end, arrowstyle="<->",
                                color="blue", mutation_scale=10,
                                linewidth=2, zorder=10)
        ax_img_m2.add_patch(arrow)
    ax_img_m2.set_title(rf'$\beta_e = {beta_e}$ (t={t_ref}, descent dir.)')
    ax_img_m2.set_xlabel(r'$x/d_p$')
    ax_img_m2.set_ylabel(r'$y/d_p$')
    cbar_m2 = fig.colorbar(im_m2, ax=ax_img_m2, fraction=0.046, pad=0.04)
    cbar_m2.set_label(r'$J^2$')

    # ----------------------------------------
    # POOLED THICKNESSES across all snapshots
    # ----------------------------------------
    thicknesses_m1_pooled = []
    thicknesses_m2_pooled = []
    thicknesses_m3_pooled = []

    for t in snapshots:
        cleaned_mask = mask_all[(beta_e, t)]
        J2           = J2_all[(beta_e, t)]
        max_coords, _ = get_max_coords(cleaned_mask, J2)

        # ---- M1: inscribed-circle diameter per region ----
        labeled_mask_m1 = measure.label(cleaned_mask, connectivity=2)
        for label_id in range(1, labeled_mask_m1.max() + 1):
            region = (labeled_mask_m1 == label_id)
            dist = ndi.distance_transform_edt(region)
            thicknesses_m1_pooled.append(2 * dist.max())

        # ---- descent directions for this (beta_e, t) ----
        J2_smooth = gaussian_filter(J2, sigma=sigma_smooth)
        dJdx, dJdy = np.gradient(J2_smooth)
        descent_dirs = []
        for (x, y) in max_coords:
            grad = np.array([dJdx[x, y], dJdy[x, y]])
            if np.allclose(grad, 0):
                descent_dirs.append(np.array([0.0, 0.0]))
            else:
                descent_dirs.append(-grad / np.linalg.norm(grad))

        # ---- M2: walk along descent direction ----
        for (x0, y0), (dx_, dy_) in zip(max_coords, descent_dirs):
            x0f, y0f = float(x0), float(y0)

            xf, yf, steps = x0f, y0f, 0
            while steps < max_steps:
                xf_new = xf + dx_*step_size
                yf_new = yf + dy_*step_size
                xi, yi = int(round(xf_new)), int(round(yf_new))
                if (xi < 0 or yi < 0 or
                    xi >= cleaned_mask.shape[0] or yi >= cleaned_mask.shape[1]):
                    break
                if not cleaned_mask[xi, yi]:
                    break
                xf, yf = xf_new, yf_new
                steps += 1
            dist_forward = np.hypot(xf - x0f, yf - y0f)

            xb, yb, steps = x0f, y0f, 0
            while steps < max_steps:
                xb_new = xb - dx_*step_size
                yb_new = yb - dy_*step_size
                xi, yi = int(round(xb_new)), int(round(yb_new))
                if (xi < 0 or yi < 0 or
                    xi >= cleaned_mask.shape[0] or yi >= cleaned_mask.shape[1]):
                    break
                if not cleaned_mask[xi, yi]:
                    break
                xb, yb = xb_new, yb_new
                steps += 1
            dist_backward = np.hypot(xb - x0f, yb - y0f)

            total = dist_forward + dist_backward
            if total > 0:
                thicknesses_m2_pooled.append(total)

        # ---- M3: Gaussian-fit profile (save a few example fits) ----
        beta_t_save_folder = os.path.join(
            parent_save_folder, f"betae_{beta_e}", f"t{t}"
        )
        os.makedirs(beta_t_save_folder, exist_ok=True)

        for i, ((x, y), (dx_, dy_)) in enumerate(zip(max_coords, descent_dirs)):
            if dx_ == 0 and dy_ == 0:
                continue

            xs, vals = [], []
            for k in range(-max_len, max_len + 1):
                xc = x + k*dx_
                yc = y + k*dy_
                xi, yi = int(round(xc)), int(round(yc))
                if 0 <= xi < J2.shape[0] and 0 <= yi < J2.shape[1]:
                    xs.append(k)
                    vals.append(J2[xi, yi])
            xs = np.array(xs); vals = np.array(vals)
            if len(vals) < 3:
                continue

            A0 = vals.max() - vals.min()
            mu0 = xs[np.argmax(vals)]
            sigma0 = max_len / 4
            B0 = vals.min()
            try:
                popt, pcov = curve_fit(gaussian, xs, vals, p0=[A0, mu0, sigma0, B0])
            except RuntimeError:
                continue

            A, mu, sigma_fit, B = popt
            perr = np.sqrt(np.diag(pcov))
            A_err, mu_err, sigma_err, B_err = perr

            residuals = vals - gaussian(xs, *popt)
            dof = len(vals) - 4
            chi2_red = np.sum(residuals**2) / dof if dof > 0 else np.nan

            k_fwhm   = 2*np.sqrt(2*np.log(2))
            fwhm     = k_fwhm * np.abs(sigma_fit)
            fwhm_err = k_fwhm * sigma_err

            if 0 < fwhm <= 100:
                thicknesses_m3_pooled.append(fwhm)

            # only render a handful of example fits per (beta_e, t)
            if i < n_example_fits:
                x_dense   = np.linspace(xs.min(), xs.max(), 500)
                fit_curve = A*np.exp(-(x_dense - mu)**2 / (2*sigma_fit**2)) + B
                upper = A*np.exp(-(x_dense - mu)**2 / (2*(sigma_fit+sigma_err)**2)) + B
                lower = A*np.exp(-(x_dense - mu)**2 /
                                 (2*max(sigma_fit-sigma_err, 1e-9)**2)) + B

                fig_fit, ax_fit = plt.subplots()
                ax_fit.plot(xs, vals, 'o', label="sampled profile")
                ax_fit.plot(x_dense, fit_curve, label="Gaussian fit")
                ax_fit.fill_between(x_dense, lower, upper, alpha=0.2,
                                    color='tab:orange', label=r"$\pm 1\sigma$ on width")
                half_max = B + A/2
                ax_fit.axhline(half_max, color='grey', linestyle=':', linewidth=1)
                ax_fit.axvspan(mu - fwhm/2, mu + fwhm/2, alpha=0.08, color='tab:green')
                ax_fit.set_title(
                    f"beta_e={beta_e}, t={t}, sample {i}\n"
                    f"FWHM = {fwhm:.2f} ± {fwhm_err:.2f} px"
                    f"   |   $\\sigma$ = {sigma_fit:.2f} ± {sigma_err:.2f}"
                    f"   |   $\\chi^2_\\nu$ = {chi2_red:.2f}"
                )
                ax_fit.set_xlabel("t (pixels)")
                ax_fit.set_ylabel("Intensity")
                ax_fit.legend(fontsize=8)
                fig_fit.tight_layout()
                fig_fit.savefig(os.path.join(beta_t_save_folder, f"fit_{i:03d}.png"))
                plt.close(fig_fit)

    # ---- pooled stats (printed) ----
    for name, pooled in [("M1", thicknesses_m1_pooled),
                         ("M2", thicknesses_m2_pooled),
                         ("M3", thicknesses_m3_pooled)]:
        if pooled:
            a = np.array(pooled)
            print(f"[beta_e={beta_e}] {name} pooled (N={len(a)}) "
                  f"mean={a.mean():.3f}, median={np.median(a):.3f}, "
                  f"min={a.min():.3f}, max={a.max():.3f}")

    # ---- PDF rows ----
    def plot_pdf(ax, pooled, method_name):
        x_di = np.array(pooled)
        x_di = x_di[x_di > 0] / cells_per_di
        result = None
        if x_di.size > 0:
            shape, loc, scale = lognorm.fit(x_di, floc=0)
            xs_fit = np.linspace(0, x_di.max(), 300)
            ax.hist(x_di, bins=20, density=True, alpha=0.4, label="Histogram")
            ax.plot(xs_fit, lognorm.pdf(xs_fit, shape, loc, scale),
                    label="Log-normal fit")
            mode, mode_err = lognorm_mode_with_error(x_di)
            ax.axvline(mode, color='k', linestyle='--', linewidth=1,
                       label=f"Mode = {mode:.3f} ± {mode_err:.3f}")
            ax.axvspan(mode - mode_err, mode + mode_err, color='k', alpha=0.12)
            ax.legend()
            result = (mode, mode_err)
        ax.set_xlabel(r"$d_i$")
        ax.set_ylabel("Probability density")
        ax.set_title(rf'{method_name} PDF ($\beta_e = {beta_e}$, pooled)')
        return result

    m1_res = plot_pdf(ax_hist_m1, thicknesses_m1_pooled, "M1")
    m2_res = plot_pdf(ax_hist_m2, thicknesses_m2_pooled, "M2")
    m3_res = plot_pdf(ax_hist_m3, thicknesses_m3_pooled, "M3")

    summary[beta_e] = dict(M1=m1_res, M2=m2_res, M3=m3_res)

plt.tight_layout()
plt.savefig(RESULTS / "J2_NR_M1_M2_M3_pooled_betae_scan.png", dpi=300)
plt.show()

# ---- final summary table ----
print("\n=== Mode (d_i) ± fitting error, by beta_e ===")
for beta_e in beta_es:
    s = summary[beta_e]
    line = f"beta_e={beta_e:>7}: "
    for m in ("M1", "M2", "M3"):
        if s[m] is not None:
            line += f"{m}={s[m][0]:.3f}±{s[m][1]:.3f}  "
    print(line)


# In[ ]:




