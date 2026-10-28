"""Cuts across a current sheet: the direction across it, profiles along a line, widths and fitted profiles.

    direction = descent_direction(J2, cell)              # steepest descent of J^2 at the sheet's maximum
    s, values = line_profile(J2, cell, direction, 70)   # J^2 interpolated along the cut, s in cells
    width, forward, backward = width_along(mask, cell, direction)    # across a mask (thickness methods 2, 4, 5)
    region = grow_region(J2, cell, threshold)            # the connected region above a threshold around the cell
    k, values, parameters, fwhm = fwhm_nearest_samples(J2, cell, direction)   # Gaussian FWHM (method 3)
    fit = fit_harris(s, B_profile)                        # Harris sheet B0 tanh((s - s0) / delta) + C

Cells are (i, j) indices with x along axis 0; lengths are in cells (times dx for d_p).
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

from .thickness import _walk_m2, gaussian


def descent_direction(field, cell):
    """Unit vector of steepest descent, -grad(field) / |grad(field)|, at a cell (the differences np.gradient
    uses: central inside the box, one-sided at its edges)."""
    i, j = cell
    i0, j0 = max(i - 1, 0), max(j - 1, 0)
    gradient_x, gradient_y = np.gradient(field[i0:i + 2, j0:j + 2])
    direction = -np.array([gradient_x[i - i0, j - j0], gradient_y[i - i0, j - j0]])
    return direction / np.linalg.norm(direction)


def line_profile(field, cell, direction, half_length, samples=400, order=1):
    """(s, values): the field interpolated at `samples` points on the line through `cell` along `direction`,
    s from -half_length to +half_length cells (0 outside the box)."""
    s = np.linspace(-half_length, half_length, samples)
    values = ndi.map_coordinates(field, [cell[0] + s * direction[0], cell[1] + s * direction[1]], order=order)
    return s, values


def width_along(mask, cell, direction, step=1, max_steps=1000):
    """(width, forward, backward): how far one can walk from `cell` along +direction and -direction, in steps of
    `step` cells, before the next step leaves the mask (or the box)."""
    forward = _walk_m2(mask, cell[0], cell[1], direction[0], direction[1], step, max_steps)[0]
    backward = _walk_m2(mask, cell[0], cell[1], -direction[0], -direction[1], step, max_steps)[0]
    return forward + backward, forward, backward


def grow_region(field, cell, threshold, smoothing_radius=None):
    """The 4-connected region where field >= threshold that contains `cell` (empty if the cell itself is below
    the threshold): a flood fill from the cell. smoothing_radius: open and close it with a disk of that many cells."""
    labels, _ = ndi.label(field >= threshold)
    region = labels == labels[cell] if labels[cell] else np.zeros(field.shape, dtype=bool)
    if smoothing_radius:
        from skimage import morphology           # imported here so the rest of pta works without scikit-image
        disk = morphology.disk(smoothing_radius)
        region = morphology.closing(morphology.opening(region, disk), disk)
    return region


def fwhm_nearest_samples(field, cell, direction, half_length=60):
    """Gaussian FWHM across a sheet from the field at whole-cell steps k = -half_length..half_length along the
    direction (the nearest cell; steps outside the box are skipped), fitted with gaussian() from
    [max - min, k of the max, half_length / 4, min]. Returns (k, values, fit parameters, FWHM in cells); the
    parameters and FWHM are NaN if the fit fails. (The Paper 1 pipeline uses thickness._m3_profile_and_fit.)"""
    k, values = [], []
    for step in range(-half_length, half_length + 1):
        i, j = int(round(cell[0] + step * direction[0])), int(round(cell[1] + step * direction[1]))
        if 0 <= i < field.shape[0] and 0 <= j < field.shape[1]:
            k.append(step)
            values.append(field[i, j])
    k, values = np.array(k), np.array(values)
    start = [values.max() - values.min(), k[np.argmax(values)], half_length / 4, values.min()]
    try:
        parameters, _ = curve_fit(gaussian, k, values, p0=start)
    except RuntimeError:
        return k, values, np.full(4, np.nan), np.nan
    return k, values, parameters, 2 * np.sqrt(2 * np.log(2)) * abs(parameters[2])


def harris_profile(s, B0, s0, delta, C):
    """A Harris current sheet: B0 tanh((s - s0) / delta) + C."""
    return B0 * np.tanh((s - s0) / delta) + C


def fit_harris(s, profile, magnitude=False, centred=False, delta_guess=2.0, maxfev=100000):
    """Fit a Harris sheet to a profile across a sheet: B0 tanh((s - s0) / delta) + C, or B0 |tanh(...)| + C for a
    magnitude such as |B| (magnitude=True); centred=True keeps s0 = 0. Returns a dict with B0, s0, delta (> 0),
    C, their errors (B0_error, ...) and the fitted curve at s."""
    shape = np.abs if magnitude else (lambda value: value)
    amplitude, offset = profile.max() - profile.min(), profile.min()
    if centred:
        def model(s, B0, delta, C):
            return B0 * shape(np.tanh(s / delta)) + C
        start = [amplitude, delta_guess, offset]
        names = ["B0", "delta", "C"]
    else:
        def model(s, B0, s0, delta, C):
            return B0 * shape(np.tanh((s - s0) / delta)) + C
        centre_guess = s[np.argmin(profile)] if magnitude else s[np.argmin(np.abs(profile - profile.mean()))]
        start = [amplitude, centre_guess, delta_guess, offset]
        names = ["B0", "s0", "delta", "C"]
    parameters, covariance = curve_fit(model, s, profile, p0=start, maxfev=maxfev)
    errors = np.sqrt(np.diag(covariance))
    fit = dict(zip(names, parameters), **{f"{name}_error": error for name, error in zip(names, errors)})
    fit.setdefault("s0", 0.0)
    fit["delta"] = abs(fit["delta"])
    fit["curve"] = model(s, *parameters)
    return fit
