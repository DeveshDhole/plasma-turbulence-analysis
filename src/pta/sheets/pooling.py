"""Pooled current-sheet thickness: one call from a run and a list of snapshots to a figure.

    from pta.sheets import pooled_thickness
    pooled = pooled_thickness(beta_p=1, beta_e=1, times=range(200, 250, 5), k=14.12/3)
    pooled.figure                      # histogram + lognormal fit, one panel per method
    pooled.thickness["m3"]             # every sheet's thickness in d_p, pooled over the snapshots
    pooled.statistics                  # one row per method: n, lognormal median and sigma, bootstrap CI

Each snapshot is filtered (iterative filtering with a k cutoff), its sheets are detected, and every sheet is
measured three ways: M1 the diameter of the largest inscribed circle, M2 the width of the sheet along the
steepest descent of |J_par|, M3 the FWHM of a bell curve fitted to |J_par| along that line. The filtered
fields are cached, so running again with the same k and Xi only measures. This is the pipeline of
CSDetectionMaxTurbulenceFullPipelineWithKCutoff, as a function.
"""
from dataclasses import dataclass, field

import numpy as np

from .. import config
from ..fields import partial_x_fft, partial_y_fft
from ..parallel import parallel_map
from ..plotting import save_figure
from ..statistics import bootstrap_lognorm_median, fit_and_plot_pdf
from .detection import detect_current_sheets
from .length import sheet_length
from .thickness import _m3_profile_and_fit, _walk_m2

METHOD_NAMES = {"m1": "M1: distance transform", "m2": "M2: steepest descent", "m3": "M3: Gaussian FWHM"}


@dataclass
class PooledThickness:
    """What pooled_thickness returns."""
    run: object                       # the pta.Run the sheets come from
    times: list                       # the snapshots that were measured
    k: float                          # the filter's cutoff wavenumber (None: no filtering)
    Xi: float
    thickness: dict                   # method -> every sheet's thickness in d_p, pooled
    per_time: dict                    # method -> {time: thicknesses of that snapshot, in d_p}
    sheets_per_time: dict             # time -> number of sheets detected
    statistics: list = field(default_factory=list)    # one row per method
    figure: object = None

    def save(self, folder, name="pooled_thickness"):
        """Save the figure (300 dpi) and the pooled thicknesses (.npz) in a folder; returns both paths."""
        data = {f"{method}_d_p": values for method, values in self.thickness.items()}
        array_path = folder / f"{name}.npz"
        np.savez_compressed(array_path, times=np.array(self.times), k=self.k or np.nan, Xi=self.Xi, **data)
        figure_path = save_figure(self.figure, folder, name) if self.figure is not None else None
        return figure_path, array_path


def filtered_field(run, time, components=("Bx", "By"), k=None, Xi=1.6, n_iter=None, cache=True):
    """The components of one snapshot, filtered and cached.

    k       : cutoff wavenumber, iterative filtering one component at a time (the Paper 1 scheme)
    n_iter  : instead, the older scheme: iterative_filtering3D of Bx, By and Bz from this IMF on
    Neither: the fields as they are. The filtered fields are cached under
    <cache>/filtered/<run>/t<time>_<filter>.npz, so the same filter is only applied once."""
    if k is None and n_iter is None:
        return run.load_many(components, time)
    cache_file = None
    if cache:
        folder = config.cache_dir("filtered", f"{run.parent.name}_{run.name}")
        setting = f"k{k:g}" if k is not None else f"iter{n_iter:g}"
        cache_file = folder / f"t{time}_{setting}_Xi{Xi:g}.npz"
        if cache_file.exists():
            stored = np.load(cache_file)
            if all(component in stored for component in components):
                return tuple(stored[component] for component in components)
    if n_iter is not None:                                        # the three components together
        from ..filtering import iterative_filtering3D             # needs fifpy; imported on use
        filtered = dict(zip(("Bx", "By", "Bz"),
                            iterative_filtering3D(*run.load_many(["Bx", "By", "Bz"], time), n_iter, Xi=Xi)))
        fields = tuple(filtered[component] for component in components)
    else:
        from ..filtering import iterative_filtering1DWithKCutOff
        filtered = dict(zip(components, (iterative_filtering1DWithKCutOff(field, Xi=Xi, k=k, dx=run.dx)
                                         for field in run.load_many(components, time))))
        fields = tuple(filtered[component] for component in components)
    if cache_file is not None:
        np.savez_compressed(cache_file, **filtered)
    return fields


def parallel_current(run, time, k=None, Xi=1.6, n_iter=None, cache=True):
    """|J_par| = |dBy/dx - dBx/dy| of one snapshot, from the filtered Bx and By."""
    Bx, By = filtered_field(run, time, ("Bx", "By"), k, Xi, n_iter, cache)
    return np.abs(partial_x_fft(By, run.dx) - partial_y_fft(Bx, run.dy))


def label_sheets(current, opening_radius=1, min_area=500, connectivity=2, **detection):
    """(cleaned mask, labelled mask, number of sheets) of a non-negative field: the sheets from
    detect_current_sheets, opened and closed with a disk, with regions below min_area cells dropped."""
    from skimage import measure, morphology              # imported here so the rest of pta works without it

    mask = np.zeros_like(current)
    for sheet in detect_current_sheets(current, **detection):
        for i, j in sheet:
            mask[i, j] = 1
    disk = morphology.disk(opening_radius)
    mask = morphology.closing(morphology.opening(mask, disk), disk)
    try:                                                  # scikit-image >= 0.26 renamed the size argument
        cleaned = morphology.remove_small_objects(mask.astype(bool), max_size=min_area - 1)
    except TypeError:
        cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=min_area)
    labelled = measure.label(cleaned, connectivity=connectivity)
    return cleaned, labelled, int(labelled.max())


def snapshot_thickness(run, time, k=None, Xi=1.6, n_iter=None, cache=True, method=("m1", "m2", "m3"),
                       smooth_sigma=3, max_fwhm_cells=100, detection=None, **mask_options):
    """Every sheet's thickness in one snapshot, in d_p: {"m1": array, ...} plus "n_sheets".

    M1 is the diameter of the largest circle inside the sheet, M2 its width along the steepest descent of
    |J_par| from the sheet's brightest cell, M3 the FWHM of a bell curve fitted to |J_par| along that line.
    Sheets with no width (M2) or no usable fit (M3) drop out of those methods, as in the pipeline.
    """
    from scipy import ndimage as ndi

    methods = (method,) if isinstance(method, str) else tuple(method)
    current = parallel_current(run, time, k, Xi, n_iter, cache)
    cleaned, labelled, n_sheets = label_sheets(current, **(detection or {}), **mask_options)
    gradient_x, gradient_y = np.gradient(ndi.gaussian_filter(current, sigma=smooth_sigma))

    measured = {method: [] for method in methods}
    for index in range(1, n_sheets + 1):
        region = labelled == index
        rows, columns = np.where(region)
        brightest = int(np.argmax(current[region]))
        i, j = int(rows[brightest]), int(columns[brightest])

        if "m1" in measured:
            measured["m1"].append(2 * float(ndi.distance_transform_edt(region).max()))
        if "m2" in measured or "m3" in measured:
            gradient = np.array([gradient_x[i, j], gradient_y[i, j]])
            if np.allclose(gradient, 0):
                direction = (0.0, 0.0)
            else:
                unit = -gradient / np.linalg.norm(gradient)
                direction = (float(unit[0]), float(unit[1]))
            if "m2" in measured:
                measured["m2"].append(_walk_m2(cleaned, i, j, *direction)[0]
                                      + _walk_m2(cleaned, i, j, -direction[0], -direction[1])[0])
            if "m3" in measured:
                measured["m3"].append(_m3_profile_and_fit(current, cleaned, i, j, *direction)[3])

    thickness = {}
    for method, values in measured.items():
        values = np.asarray(values, dtype=float)
        if method == "m2":
            values = values[values > 0]
        elif method == "m3":
            values = values[np.isfinite(values)]
            values = values[(values > 0) & (values <= max_fwhm_cells)]
        thickness[method] = values * run.dx                     # cells -> d_p
    thickness["n_sheets"] = n_sheets
    return thickness


def pooled_thickness(beta_p=None, beta_e=1, brms=None, run=None, times=None, k=None, Xi=1.6, n_iter=None,
                     method=("m1", "m2", "m3"), layout="beta_scan", cache=True, workers=None,
                     plot=True, bins=30, xmax=None, bootstrap=1000, seed=0, title=None, **snapshot_options):
    """Pool the thickness of every current sheet over several snapshots of one run, and fit a lognormal.

    beta_p, beta_e, brms : which run, through the layout in pta.toml (or pass run=<pta.Run> instead)
    times                : the snapshots to pool (default: every output time of the run)
    k, Xi, n_iter        : the filter: a cutoff wavenumber k, or the older scheme with n_iter IMFs
                           (iterative_filtering3D); neither leaves the fields unfiltered
    method               : which thickness method(s) to measure: "m3", or several as ("m1", "m2", "m3")
                           M1 the diameter of the largest inscribed circle, M2 the width of the sheet along
                           the steepest descent of |J_par|, M3 the FWHM of a bell curve fitted across it
    workers              : snapshots measured in parallel (default: one per core)
    plot                 : draw the histogram and the lognormal fit, one panel per method
    bootstrap            : resamples for the error on the lognormal median (0: skip)

    Returns a PooledThickness; see its docstring. Other snapshot_options (min_area, smooth_sigma,
    opening_radius, connectivity, detection=dict(global_threshold=..., local_fraction=...)) go to
    snapshot_thickness.
    """
    import matplotlib.pyplot as plt

    methods = (method,) if isinstance(method, str) else tuple(method)
    unknown = set(methods) - set(METHOD_NAMES)
    if unknown:
        raise ValueError(f"unknown thickness method(s) {sorted(unknown)}; choose from {sorted(METHOD_NAMES)}")
    if run is None:
        if beta_p is None:
            raise ValueError("give beta_p (and optionally beta_e, brms), or run=<pta.Run>")
        parameters = dict(beta_p=beta_p, beta_e=beta_e)
        if brms is not None:
            parameters["brms"] = brms
        run = config.run(layout, **parameters)
    times = list(run.times if times is None else times)

    def measure(time):
        return snapshot_thickness(run, time, k=k, Xi=Xi, n_iter=n_iter, cache=cache, method=methods, **snapshot_options)

    snapshots = parallel_map(measure, times, workers=workers)
    per_time = {method: {time: snapshot[method] for time, snapshot in zip(times, snapshots)} for method in methods}
    thickness = {method: np.concatenate([snapshot[method] for snapshot in snapshots]) for method in methods}
    pooled = PooledThickness(run=run, times=times, k=k, Xi=Xi, thickness=thickness, per_time=per_time,
                             sheets_per_time={time: snapshot["n_sheets"] for time, snapshot in zip(times, snapshots)})

    rng = np.random.default_rng(seed)
    if plot:
        pooled.figure, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5), squeeze=False)
        axes = axes.ravel()
    for number, method in enumerate(methods):
        values = thickness[method]
        if plot:
            statistics = fit_and_plot_pdf(axes[number], values, METHOD_NAMES.get(method, method),
                                          bins=bins, ylabel=(number == 0), xmax=xmax)
        else:
            statistics = fit_and_plot_pdf(plt.figure().gca(), values, METHOD_NAMES.get(method, method), bins=bins)
            plt.close("all")
        row = dict(method=method, n_snapshots=len(times), **statistics)
        if bootstrap and values.size:
            row["median_se"], row["ci_lo"], row["ci_hi"] = bootstrap_lognorm_median(values, bootstrap, rng)
        pooled.statistics.append(row)
    if plot:
        if title is None:
            title = (rf"Pooled current-sheet thickness   $\beta_p = {beta_p}$, $\beta_e = {beta_e}$"
                     if beta_p is not None else f"Pooled current-sheet thickness   {run.name}")
            if k is not None:
                title += rf"   ($k = {k:g}$, $\Xi = {Xi:g}$, "
            elif n_iter is not None:
                title += rf"   ({n_iter:g} IMFs, $\Xi = {Xi:g}$, "
            else:
                title += "   (unfiltered, "
            title += f"{len(times)} snapshots, t = {times[0]}-{times[-1]})"
        pooled.figure.suptitle(title, y=1.02, fontsize=13)
        pooled.figure.tight_layout()
    return pooled


def snapshot_shape(run, time, k=None, Xi=1.6, n_iter=None, cache=True, smooth_sigma=3, max_fwhm_cells=100,
                   max_hole_area=20, detection=None, **mask_options):
    """(length, M3 thickness) in d_p of every sheet in one snapshot, as an (n, 2) array.

    A sheet drops out if it is a ring round a plasmoid, if its middle line is a single pixel, or if no bell
    curve fits its profile. Returns (rows, number of sheets whose length was measurable), so the fraction
    lost to the fit can be reported."""
    from scipy import ndimage as ndi

    current = parallel_current(run, time, k, Xi, n_iter, cache)
    cleaned, labelled, n_sheets = label_sheets(current, **(detection or {}), **mask_options)
    gradient_x, gradient_y = np.gradient(ndi.gaussian_filter(current, sigma=smooth_sigma))

    rows, measurable = [], 0
    for index in range(1, n_sheets + 1):
        region = labelled == index
        length = sheet_length(region, step=run.dx, max_hole_area=max_hole_area)
        if length is None:
            continue
        measurable += 1
        cells_i, cells_j = np.where(region)
        brightest = int(np.argmax(current[region]))
        i, j = int(cells_i[brightest]), int(cells_j[brightest])
        gradient = np.array([gradient_x[i, j], gradient_y[i, j]])
        if np.allclose(gradient, 0):
            continue
        unit = -gradient / np.linalg.norm(gradient)
        fwhm = _m3_profile_and_fit(current, cleaned, i, j, float(unit[0]), float(unit[1]))[3]
        if not np.isfinite(fwhm) or fwhm <= 0 or fwhm > max_fwhm_cells:
            continue
        rows.append((length, fwhm * run.dx))
    return np.array(rows, dtype=float).reshape(-1, 2), measurable


@dataclass
class PooledShape:
    """What pooled_shape returns: length, thickness and aspect ratio of every sheet, in d_p."""
    run: object
    times: list
    length: np.ndarray
    thickness: np.ndarray
    measurable: int                   # sheets whose length was measurable (the rest had no usable M3 fit)
    figure: object = None

    @property
    def aspect(self):
        return self.length / self.thickness

    def save(self, folder, name="shape"):
        """Save the figure (300 dpi) and the per-sheet lengths and thicknesses (.npz); returns both paths."""
        array_path = folder / f"{name}.npz"
        np.savez_compressed(array_path, times=np.array(self.times), length_d_p=self.length,
                            thickness_d_p=self.thickness)
        figure_path = save_figure(self.figure, folder, name) if self.figure is not None else None
        return figure_path, array_path


def pooled_shape(beta_p=None, beta_e=1, brms=None, run=None, times=None, k=None, Xi=1.6, n_iter=None,
                 layout="beta_scan", cache=True, workers=None, plot=True, title=None, **snapshot_options):
    """Length, M3 thickness and aspect ratio of every current sheet, pooled over several snapshots.

    The arguments are those of pooled_thickness. With plot, draws the four panels of the Paper 1 shape figure:
    the aspect-ratio PDF, aspect against length, aspect against thickness, and length against thickness."""
    import matplotlib.pyplot as plt

    if run is None:
        if beta_p is None:
            raise ValueError("give beta_p (and optionally beta_e, brms), or run=<pta.Run>")
        parameters = dict(beta_p=beta_p, beta_e=beta_e)
        if brms is not None:
            parameters["brms"] = brms
        run = config.run(layout, **parameters)
    times = list(run.times if times is None else times)

    def measure(time):
        return snapshot_shape(run, time, k=k, Xi=Xi, n_iter=n_iter, cache=cache, **snapshot_options)

    snapshots = parallel_map(measure, times, workers=workers)
    data = np.concatenate([rows for rows, _ in snapshots]) if snapshots else np.empty((0, 2))
    measurable = sum(count for _, count in snapshots)
    pooled = PooledShape(run=run, times=times, length=data[:, 0], thickness=data[:, 1], measurable=measurable)

    if plot and len(data):
        aspect = pooled.aspect
        pooled.figure, axes = plt.subplots(2, 2, figsize=(15, 12.5))
        axes[0, 0].hist(aspect, bins=np.logspace(np.log10(aspect.min()), np.log10(aspect.max()), 35),
                        density=True, alpha=0.5, edgecolor="black", linewidth=0.4)
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_xlabel(r"aspect ratio $L/\lambda$")
        axes[0, 0].set_ylabel("PDF")
        axes[0, 0].set_title(f"aspect ratio (N={len(aspect)})")
        for ax, (x, y, xlabel, ylabel, heading) in zip(
                axes.ravel()[1:],
                [(pooled.length, aspect, r"length $L$ $(d_p)$", r"aspect ratio $L/\lambda$", "aspect vs length"),
                 (pooled.thickness, aspect, r"thickness $\lambda$ $(d_p)$", r"aspect ratio $L/\lambda$", "aspect vs thickness"),
                 (pooled.thickness, pooled.length, r"thickness $\lambda$ $(d_p)$", r"length $L$ $(d_p)$", "length vs thickness")]):
            ax.scatter(x, y, s=8, alpha=0.25)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(heading)
        if title is None:
            title = (rf"Current-sheet shape   $\beta_p = {beta_p}$, $\beta_e = {beta_e}$"
                     if beta_p is not None else f"Current-sheet shape   {run.name}")
            title += f"   ({len(times)} snapshots, N = {len(aspect)})"
        pooled.figure.suptitle(title, y=1.00)
        pooled.figure.tight_layout()
    return pooled
