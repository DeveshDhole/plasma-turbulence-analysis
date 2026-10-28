"""Derivatives and derived fields on the periodic simulation grid (2D arrays with x along axis 0).

partial_x_fft and partial_y_fft were copied verbatim from the Paper 1 pipeline notebook
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


def current_density(Bx, By, Bz, dx, dy):
    """J = curl B of a 2D field with x along axis 0: (Jx, Jy, Jz) = (dBz/dy, -dBz/dx, dBy/dx - dBx/dy)."""
    return partial_y_fft(Bz, dy), -partial_x_fft(Bz, dx), partial_x_fft(By, dx) - partial_y_fft(Bx, dy)


def parallel_component(vector, field):
    """(v . b) / |b| at every point: the signed component of a vector along a field (each a tuple of 3 arrays),
    e.g. J_par = parallel_component((Jx, Jy, Jz), (Bx, By, Bz))."""
    (vx, vy, vz), (bx, by, bz) = vector, field
    return (vx * bx + vy * by + vz * bz) / np.sqrt(bx**2 + by**2 + bz**2)


def parallel_perpendicular_squared(vector, field):
    """(|v_par|^2, |v_perp|^2) of a vector with respect to a field: (v . b)^2 / |b|^2, and |v|^2 minus that."""
    (vx, vy, vz), (bx, by, bz) = vector, field
    total = vx**2 + vy**2 + vz**2
    parallel = ((vx * bx + vy * by + vz * bz)**2) / (bx**2 + by**2 + bz**2)
    return parallel, total - parallel


def resistive_dissipation_wavenumber(mean_J_squared, eta):
    """k_diss = <J^2>^(1/4) / eta^(1/2): the wavenumber where resistive dissipation, eta <J^2>, balances the
    cascade (the Kolmogorov scale (eta^3 / epsilon)^(1/4) with epsilon = eta <J^2>)."""
    return mean_J_squared**0.25 / np.sqrt(eta)
