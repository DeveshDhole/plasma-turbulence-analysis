"""Spectral derivatives and spectra on the periodic simulation grid.

Functions copied verbatim from the Paper 1 pipeline notebook
(CSDetectionMaxTurbulenceFullPipelineWithKCutoff_Beta1.ipynb); behaviour unchanged.
"""
import numpy as np


def partial_x_fft(f, dx):
    nx, ny = f.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    f_fft = np.fft.fft(f, axis=0)
    df_fft = 1j * kx[:, None] * f_fft
    return np.real(np.fft.ifft(df_fft, axis=0))


def partial_y_fft(f, dy):
    nx, ny = f.shape
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    f_fft = np.fft.fft(f, axis=1)
    df_fft = 1j * ky[None, :] * f_fft
    return np.real(np.fft.ifft(df_fft, axis=1))


def partial_x_2d(f,dx):
    """Compute ∂f/∂x using 2nd-order central differences with periodic BCs."""
    Nx, Ny = f.shape
    df_dx = np.zeros_like(f)
    
    # Central difference in x-direction (axis=0)
    df_dx[1:Nx-1, :] = (f[2:Nx, :] - f[0:Nx-2, :]) / (2 * dx)
    
    # Periodic boundaries (x-direction)
    df_dx[0, :] = (f[1, :] - f[-1, :]) / (2 * dx)
    df_dx[-1, :] = (f[0, :] - f[-2, :]) / (2 * dx)
    
    return df_dx


def partial_y_2d(f,dy):
    """Compute ∂f/∂y using 2nd-order central differences with periodic BCs."""
    Nx, Ny = f.shape
    df_dy = np.zeros_like(f)
    
    # Central difference in y-direction (axis=1)
    df_dy[:, 1:Ny-1] = (f[:, 2:Ny] - f[:, 0:Ny-2]) / (2 * dy)
    
    # Periodic boundaries (y-direction)
    df_dy[:, 0] = (f[:, 1] - f[:, -1]) / (2 * dy)
    df_dy[:, -1] = (f[:, 0] - f[:, -2]) / (2 * dy)
    
    return df_dy


def compute_radial_spectrum(fld, dx, dy, dtype=float):
    nx, ny = fld.shape

    # Fourier wavenumbers
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_perp = np.sqrt(KX**2 + KY**2)

    # Power spectrum density
    ff = np.abs(np.fft.ifft2(fld))**2   # use fft2, not ifft2
    P_flat = ff.ravel()
    K_flat = K_perp.ravel()

    # Radial bins
    dk = kx[1] - kx[0]
    k_bins = np.arange(0, np.max(K_flat) + dk, dk)
    nk = len(k_bins)
    P_k = np.zeros(nk, dtype=dtype)

    # Bin indices (vectorized)
    idx_float = K_flat / dk
    i1 = np.floor(idx_float).astype(int)
    i2 = i1 + 1
    w = idx_float - i1   # weight to upper bin

    # Accumulate into bins
    for i in range(len(K_flat)):
        if i1[i] >= 0 and i2[i] < nk:
            P_k[i1[i]] += P_flat[i] * (1 - w[i])
            P_k[i2[i]] += P_flat[i] * w[i]

    # Pack into recarray
    tags = ['kk', 'sp']
    sps = np.recarray((nk,), dtype=[(t, dtype) for t in tags])
    sps['kk'] = k_bins
    sps['sp'] = P_k / dk

    return sps


def compute_spectra_at_time(base_path, t, dx, dy):
    """Radial spectra of B, J and the density at output time t of the run in base_path.
    Returns (kk, spectrum_B, spectrum_J, spectrum_D)."""
    from .loading import load_fields
    data_Bx, data_By, data_Bz, data_Dn = load_fields(["Bx", "By", "Bz", "Dn1"], t, base_path)

    # Currents
    data_Jx =  partial_y_fft(data_Bz, dx)
    data_Jy = -partial_x_fft(data_Bz, dx)
    data_Jz =  partial_x_fft(data_By, dx) - partial_y_fft(data_Bx, dx)

    # Spectra
    spectrum_Bx = compute_radial_spectrum(data_Bx, dx, dy)
    spectrum_By = compute_radial_spectrum(data_By, dx, dy)
    spectrum_Bz = compute_radial_spectrum(data_Bz, dx, dy)

    spectrum_Jx = compute_radial_spectrum(data_Jx, dx, dy)
    spectrum_Jy = compute_radial_spectrum(data_Jy, dx, dy)
    spectrum_Jz = compute_radial_spectrum(data_Jz, dx, dy)

    spectrum_B = spectrum_Bx.sp + spectrum_By.sp + spectrum_Bz.sp
    spectrum_J = spectrum_Jx.sp + spectrum_Jy.sp + spectrum_Jz.sp

    spectrum_D = compute_radial_spectrum(data_Dn, dx, dy).sp

    kk = spectrum_Bx.kk  # assume same grid

    return kk, spectrum_B, spectrum_J, spectrum_D
