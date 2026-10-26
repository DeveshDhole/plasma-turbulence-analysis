"""Iterative filtering (fifpy) of the magnetic field, and a Gaussian low-pass filter in Fourier space.

fifpy is imported inside the iterative-filtering functions, so gaussian_fft_filter_2d and
k_to_mask_length work on machines without fifpy.

Functions copied verbatim from the Paper 1 pipeline notebook
(CSDetectionMaxTurbulenceFullPipelineWithKCutoff_Beta1.ipynb); behaviour unchanged.
iterative_filtering1DWithKCutOff is the Paper 1 version: the mask length comes from k.
"""
import numpy as np


def k_to_mask_length(k, dx=0.0625, angular=True, prefactor=1.0, min_len=3):
    """
    FIF mask length (in grid points) for a desired cutoff wavenumber k.
    Scales finer than ~this length get pushed into the discarded fine IMF.

    k         : cutoff wavenumber, in the SAME convention/units as your
                compute_radial_spectrum output.
    dx        : grid spacing you passed to compute_radial_spectrum (0.0625 here).
    angular   : True  -> k is angular       (rad/length,   wavelength = 2*pi/k)
                False -> k is ordinary freq  (cycles/length, wavelength = 1/k)
    prefactor : order-1 calibration. 1.0 = geometric estimate; tune it if you
                want the spectral rolloff to land exactly on k (see note below).
    min_len   : floor so a tiny mask stays valid.
    """
    wavelength = (2.0 * np.pi / k) if angular else (1.0 / k)   # in the length units of dx
    L = prefactor * wavelength / dx
    return max(min_len, int(round(L)))


def iterative_filtering1D(field, index, Xi=1.6):
    import fifpy                     # imported here so the rest of the module works without it
    fif = fifpy.MIF(NIMFs=4,alpha=30,Xi=Xi)
    fif.run(field)
    print(fif.IMC.shape)
    filtered_signal = np.sum(fif.IMC[index:],axis=0)
    return filtered_signal


def iterative_filtering1DWithKCutOff(field, Xi=1.6, k=10):
    import fifpy                     # imported here so the rest of the module works without it
    fif = fifpy.MIF(NIMFs=1, alpha=30, Xi=Xi)
    mask_length = k_to_mask_length(k)
    fif.run(field, M=[mask_length])          # one mask length per IMF (sequence, not scalar)
    print(fif.IMC.shape)              # sanity-check the component layout
    return np.sum(fif.IMC[1:], axis=0)


def iterative_filtering3D(fieldX, fieldY, fieldZ, index, Xi=1.6):
    import fifpy                     # imported here so the rest of the module works without it
    fif = fifpy.MIF(NIMFs=4,alpha=30, Xi=Xi)
    fif.run(fieldX)
    mif_x = fif.IMC
    logm = np.array([fif.data['stats_list']['logMx'],fif.data['stats_list']['logMy']]).min(axis=0)
    fif.run(fieldY, M=logm)
    mif_y = fif.IMC
    fif.run(fieldZ, M=logm)
    mif_z = fif.IMC
    filtered_x = np.sum(mif_x[index:], axis=0)
    filtered_y = np.sum(mif_y[index:], axis=0)
    filtered_z = np.sum(mif_z[index:], axis=0)
    return filtered_x, filtered_y, filtered_z


def gaussian_fft_filter_2d(J_squared, k0):
    """
    Apply a Gaussian low-pass filter in Fourier space to a 2D field.

    Parameters
    ----------
    J_squared : 2D numpy array
        Input real-space data
    k0 : float
        Filter width in k-space (grid units). Smaller = stronger smoothing.

    Returns
    -------
    J_filtered : 2D numpy array
        Smoothed real-space data
    """

    # FFT
    F = np.fft.fft2(J_squared)
    F_shifted = np.fft.fftshift(F)

    # k-grid
    nx, ny = J_squared.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
    ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2

    # Gaussian filter in k-space
    gaussian = np.exp(-K2 / (2 * k0**2))

    # Apply filter
    F_filt = F_shifted * gaussian

    # Inverse FFT
    F_filt = np.fft.ifftshift(F_filt)
    J_filtered = np.fft.ifft2(F_filt)

    # Return real part
    return np.real(J_filtered)
