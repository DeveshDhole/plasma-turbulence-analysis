"""Thickness-method helpers (production versions from Paper 1).

_walk_m2, gaussian and bell were copied verbatim from the Paper 1 pipeline notebook
(CSDetectionMaxTurbulenceFullPipelineWithKCutoff_Beta1.ipynb). The other functions were copied
from the notebooks that each had their own copy; behaviour unchanged. The M3 settings (max_len,
subpix_step, margin_frac, min_half) used to be notebook globals; they are now arguments whose
defaults are the values every notebook used (60, 0.5, 0.5, 6.0 pixels).
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import curve_fit


def _walk_m2(mask, x0, y0, dx_, dy_, step=1, max_steps=1000):
    """Production M2 walk (edge-stop). Returns (distance, endpoint)."""
    nx, ny = mask.shape
    xf, yf, s = float(x0), float(y0), 0
    while s < max_steps:
        xn, yn = xf + dx_ * step, yf + dy_ * step
        xi, yi = int(round(xn)), int(round(yn))
        if xi < 0 or yi < 0 or xi >= nx or yi >= ny:
            break
        if not mask[xi, yi]:
            break
        xf, yf, s = xn, yn, s + 1
    return np.hypot(xf - x0, yf - y0), (xf, yf)


def gaussian(t, A, mu, sigma, B):
    return A * np.exp(-(t - mu) ** 2 / (2 * sigma ** 2)) + B


def bell(t, A, mu, sigma, B):
    return A * np.exp(-(t - mu) ** 2 / (2 * sigma ** 2)) + B


def _m3_profile_and_fit(J2, mask, xm, ym, dx_, dy_,
                        max_len=60, subpix_step=0.5, margin_frac=0.5, min_half=6.0):
    """Returns (xs, vals, popt, fwhm, mu) for the Gaussian-FWHM measurement."""
    if dx_ == 0 and dy_ == 0:
        return None, None, None, np.nan, np.nan
    nx, ny = mask.shape

    def edge(sx, sy):
        d = 0.0
        while d < max_len:
            dn = d + subpix_step
            xi, yi = int(round(xm + sx * dn)) % nx, int(round(ym + sy * dn)) % ny
            if not mask[xi, yi]:
                break
            d = dn
        return d
    half_f = min(max_len, max(min_half, edge(dx_, dy_) * (1 + margin_frac)))
    half_b = min(max_len, max(min_half, edge(-dx_, -dy_) * (1 + margin_frac)))
    xs = np.arange(-half_b, half_f + subpix_step, subpix_step)
    vals = ndi.map_coordinates(J2, np.vstack([xm + xs * dx_, ym + xs * dy_]),
                               order=1, mode="grid-wrap")
    if len(vals) < 4:
        return xs, vals, None, np.nan, np.nan
    B0, A0 = float(vals.min()), float(vals.max() - vals.min())
    if A0 <= 0:
        return xs, vals, None, np.nan, np.nan
    mu0 = float(xs[np.argmax(vals)])
    above = np.where(vals >= B0 + 0.5 * A0)[0]
    sigma0 = (max(1.0, (xs[above[-1]] - xs[above[0]]) / 2.3548)
              if above.size >= 2 else max(2.0, (half_f + half_b) / 6.0))
    span = float(xs.max() - xs.min())
    sigma0 = min(sigma0, 0.99 * span)
    bounds = ([0.0, xs.min() - 1, 1e-3, -np.inf], [np.inf, xs.max() + 1, span, np.inf])
    try:
        popt, _ = curve_fit(gaussian, xs, vals, p0=[A0, mu0, sigma0, B0],
                            bounds=bounds, maxfev=10000)
        return xs, vals, popt, 2 * np.sqrt(2 * np.log(2)) * abs(popt[2]), popt[1]
    except (RuntimeError, ValueError):
        return xs, vals, None, np.nan, np.nan


def _m3_fwhm(J2, mask, xm, ym, dx_, dy_,
             max_len=60, subpix_step=0.5, margin_frac=0.5, min_half=6.0):
    if dx_ == 0 and dy_ == 0:
        return np.nan
    nx, ny = mask.shape

    def edge(sx, sy):
        d = 0.0
        while d < max_len:
            dn = d + subpix_step
            xi, yi = int(round(xm + sx * dn)) % nx, int(round(ym + sy * dn)) % ny
            if not mask[xi, yi]:
                break
            d = dn
        return d
    half_f = min(max_len, max(min_half, edge(dx_, dy_) * (1 + margin_frac)))
    half_b = min(max_len, max(min_half, edge(-dx_, -dy_) * (1 + margin_frac)))
    xs = np.arange(-half_b, half_f + subpix_step, subpix_step)
    vals = ndi.map_coordinates(J2, np.vstack([xm + xs * dx_, ym + xs * dy_]),
                               order=1, mode="grid-wrap")
    if len(vals) < 4:
        return np.nan
    B0, A0 = float(vals.min()), float(vals.max() - vals.min())
    if A0 <= 0:
        return np.nan
    mu0 = float(xs[np.argmax(vals)])
    above = np.where(vals >= B0 + 0.5 * A0)[0]
    sigma0 = (max(1.0, (xs[above[-1]] - xs[above[0]]) / 2.3548)
              if above.size >= 2 else max(2.0, (half_f + half_b) / 6.0))
    span = float(xs.max() - xs.min())
    sigma0 = min(sigma0, 0.99 * span)
    bounds = ([0.0, xs.min() - 1, 1e-3, -np.inf], [np.inf, xs.max() + 1, span, np.inf])
    try:
        popt, _ = curve_fit(gaussian, xs, vals, p0=[A0, mu0, sigma0, B0], bounds=bounds, maxfev=10000)
        return 2 * np.sqrt(2 * np.log(2)) * abs(popt[2])
    except (RuntimeError, ValueError):
        return np.nan


def _walk_to_mask_edge(mask, x0, y0, dx, dy, step=0.5, max_dist=60):   # the notebooks' subpix_step, max_len
    nx, ny = mask.shape
    dist = 0.0
    while dist < max_dist:
        d_next = dist + step
        xi = int(round(x0 + dx * d_next)) % nx
        yi = int(round(y0 + dy * d_next)) % ny
        if not mask[xi, yi]:
            break
        dist = d_next
    return dist


def thickness_m1(region, step=1.0, pad=50):
    xs, ys = np.where(region)
    x0 = max(0, xs.min() - pad); x1 = min(region.shape[0], xs.max() + pad + 1)
    y0 = max(0, ys.min() - pad); y1 = min(region.shape[1], ys.max() + pad + 1)
    edt = ndi.distance_transform_edt(region[x0:x1, y0:y1])
    return 2.0 * edt.max() * step


def caliper_length(region, step=1.0, connectivity=2):
    from skimage import measure      # imported here so the module works without scikit-image
    lab = measure.label(region.astype(int), connectivity=connectivity)
    return float(measure.regionprops(lab)[0].feret_diameter_max) * step


def crop_with_border(region, return_offset=False):
    """Cut the sheet out of the big frame, leaving a 2-pixel empty border.

    The border is not cosmetic. The thickness is measured as "distance to the
    nearest pixel OUTSIDE the sheet", so if we cropped tightly there would be no
    outside to measure to, and the thickness would come out ~20x too big.

    With return_offset=True, returns (cropped sheet, offset): add the offset to any pixel
    coordinate from the crop to get back to the position in the full frame (for drawing).
    """
    xs, ys = np.where(region)
    small = region[xs.min():xs.max() + 1, ys.min():ys.max() + 1]
    cropped = np.pad(small, 2, constant_values=False)
    if return_offset:
        return cropped, np.array([xs.min() - 2, ys.min() - 2])
    return cropped
