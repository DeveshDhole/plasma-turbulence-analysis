"""Radial (omnidirectional) spectra of 2D fields on a periodic grid, x along axis 0."""
import numpy as np

from .fields import current_density, parallel_component
from .run import as_run


def compute_radial_spectrum(fld, dx, dy, dtype=float):
    """Radial spectrum of a 2D field: a recarray with kk (k_perp, bins of width dk = 2 pi / (nx dx)) and
    sp (power per unit k). Each Fourier mode is shared linearly between the two bins either side of |k|."""
    nx, ny = fld.shape

    # Fourier wavenumbers
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    K_flat = np.sqrt(kx[:, None]**2 + ky[None, :]**2).ravel()

    # Power spectrum density
    P_flat = (np.abs(np.fft.ifft2(fld))**2).ravel()

    # Radial bins
    dk = kx[1] - kx[0]
    k_bins = np.arange(0, np.max(K_flat) + dk, dk)
    nk = len(k_bins)

    # Each mode goes to the bins below (weight 1 - w) and above (weight w) its |k|
    idx_float = K_flat / dk
    i1 = np.floor(idx_float).astype(int)
    w = idx_float - i1
    inside = (i1 >= 0) & (i1 + 1 < nk)
    P_k = (np.bincount(i1[inside], weights=P_flat[inside] * (1 - w[inside]), minlength=nk)
           + np.bincount(i1[inside] + 1, weights=P_flat[inside] * w[inside], minlength=nk)).astype(dtype)

    # Pack into recarray
    sps = np.recarray((nk,), dtype=[(t, dtype) for t in ("kk", "sp")])
    sps["kk"] = k_bins
    sps["sp"] = P_k / dk
    return sps


def summed_spectrum(components, dx, dy):
    """Radial spectrum of a vector (or of some of its components): (k, sum of the components' spectra)."""
    spectra = [compute_radial_spectrum(component, dx, dy) for component in components]
    return spectra[0].kk, sum(spectrum.sp for spectrum in spectra)


def compute_spectra_at_time(run, time, dx=None, dy=None):
    """(k, B spectrum, J spectrum, density spectrum) at one output time of a run (a Run or a folder).
    dx and dy default to the run's grid spacing."""
    run = as_run(run)
    dx = run.dx if dx is None else dx
    dy = run.dy if dy is None else dy
    Bx, By, Bz, density = run.load_many(["Bx", "By", "Bz", "Dn1"], time)
    kk, spectrum_B = summed_spectrum((Bx, By, Bz), dx, dy)
    _, spectrum_J = summed_spectrum(current_density(Bx, By, Bz, dx, dy), dx, dy)
    return kk, spectrum_B, spectrum_J, compute_radial_spectrum(density, dx, dy).sp


def parallel_perpendicular_spectra(vector, field, dx, dy):
    """(k, spectrum of the vector, of its component along the field, and the rest): the perpendicular spectrum is
    the total minus the parallel one, e.g. for J and B."""
    kk, total = summed_spectrum(vector, dx, dy)
    parallel = compute_radial_spectrum(parallel_component(vector, field), dx, dy).sp
    return kk, total, parallel, total - parallel
