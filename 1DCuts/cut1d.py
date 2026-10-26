"""
1D cut through a 2D periodic simulation at an (ir)rational angle.

The line wraps around the periodic box many times without closing on itself,
so it samples the domain far more uniformly than an axis-aligned cut
(much like a spacecraft trajectory through a turbulent plasma).

Arrays are assumed to be indexed [ix, iy]. If yours are [iy, ix], transpose
them first (arr.T), or pass swap_axes=True.
"""
import numpy as np
from scipy.ndimage import map_coordinates

GOLDEN = (1 + np.sqrt(5)) / 2


def irrational_cut(fields, Lx, Ly, theta=None, slope=1 / GOLDEN,
                   x0=0.0, y0=0.0, length=None, n_wraps=5,
                   ds=None, order=1, vectors=None, swap_axes=False):
    """
    Parameters
    ----------
    fields  : dict name -> 2D array (Nx, Ny) of the quantities to sample
              (e.g. {'bx': bx, 'by': by, 'bz': bz, 'n': density}).
    Lx, Ly  : physical box size (same units as x0, y0, length).
    theta   : cut angle in radians from the x axis. If None, uses
              theta = arctan(slope), with slope defaulting to 1/golden ratio.
    x0, y0  : starting point of the cut.
    length  : total path length. Default: n_wraps * Lx (roughly n_wraps passes).
    ds      : sampling step along the cut. Default: min(dx, dy).
    order   : interpolation order (1 = bilinear, 3 = cubic spline).
    vectors : optional dict name -> (compx, compy, compz) key names in `fields`.
              Adds components rotated into the cut frame:
              <name>_par  (along the cut), <name>_perp (in-plane, perpendicular),
              and <name>_z is just the z component.
              e.g. vectors={'B': ('bx', 'by', 'bz')}

    Returns
    -------
    s   : distance along the cut
    x,y : wrapped positions of the samples
    out : dict name -> 1D sampled array
    """
    if swap_axes:
        fields = {k: v.T for k, v in fields.items()}

    shape = next(iter(fields.values())).shape
    Nx, Ny = shape
    dx, dy = Lx / Nx, Ly / Ny

    if theta is None:
        theta = np.arctan(slope)
    c, sn = np.cos(theta), np.sin(theta)

    if ds is None:
        ds = min(dx, dy)
    if length is None:
        length = n_wraps * Lx
    s = np.arange(0.0, length, ds)

    x = (x0 + s * c) % Lx
    y = (y0 + s * sn) % Ly

    # physical -> fractional index (grid points at i*dx; shift by 0.5 if your
    # data is cell-centred and you've defined x0 relative to the cell edge)
    coords = np.vstack([x / dx, y / dy])

    out = {}
    for name, arr in fields.items():
        out[name] = map_coordinates(arr, coords, order=order, mode='grid-wrap')

    if vectors:
        for name, (cx, cy, cz) in vectors.items():
            vx, vy = out[cx], out[cy]
            out[f'{name}_par'] = vx * c + vy * sn
            out[f'{name}_perp'] = -vx * sn + vy * c
            out[f'{name}_z'] = out[cz]

    return s, x, y, out


if __name__ == '__main__':
    # quick demo with synthetic data
    import matplotlib.pyplot as plt

    Lx, Ly, N = 128.0, 64.0, 512
    X, Y = np.meshgrid(np.linspace(0, Lx, N, endpoint=False),
                       np.linspace(0, Ly, N // 2, endpoint=False), indexing='ij')
    bx = np.sin(2 * np.pi * X / Lx * 3) * np.cos(2 * np.pi * Y / Ly * 2)
    by = np.cos(2 * np.pi * X / Lx * 5 + 2 * np.pi * Y / Ly)
    bz = np.sin(2 * np.pi * Y / Ly * 4)

    s, x, y, out = irrational_cut({'bx': bx, 'by': by, 'bz': bz}, Lx, Ly,
                                  vectors={'B': ('bx', 'by', 'bz')})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].pcolormesh(X, Y, by, shading='auto')
    ax[0].plot(x, y, 'k.', ms=0.5)
    ax[1].plot(s, out['B_par'], label='B_par')
    ax[1].plot(s, out['B_perp'], label='B_perp')
    ax[1].plot(s, out['B_z'], label='Bz')
    ax[1].legend()
    plt.show()