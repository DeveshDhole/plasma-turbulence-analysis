"""
Implementation of the current-sheet identification & characterization algorithm
following Zhdankin et al. (2013) as adapted by Chatraee Azizabadi et al. (2021).

Includes improved plotting that handles periodic boundaries gracefully by shifting
regions so they are displayed continuously rather than split across domain edges.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from scipy.ndimage import maximum_filter, map_coordinates
import matplotlib.pyplot as plt

@dataclass
class Sheet:
    id: int
    peak_ij: Tuple[int, int]
    peak_xy: Tuple[float, float]
    J_peak: float
    J_peak_abs: float
    points: np.ndarray
    centroid_xy: Tuple[float, float]
    length: float
    thickness: float
    aspect_ratio: float
    theta_deg: float

# ------------------------- Utility helpers ------------------------- #

def _wrap_index(i: int, n: int) -> int:
    return (i + n) % n

def _neighbors(i: int, j: int, nx: int, ny: int, connectivity: int,
               periodic: bool) -> List[Tuple[int,int]]:
    neigh = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            if connectivity == 4 and abs(di) + abs(dj) != 1:
                continue
            ii, jj = i + di, j + dj
            if periodic:
                ii = _wrap_index(ii, nx)
                jj = _wrap_index(jj, ny)
                neigh.append((ii, jj))
            else:
                if 0 <= ii < nx and 0 <= jj < ny:
                    neigh.append((ii, jj))
    return neigh

# ------------------------- Core algorithm ------------------------- #

def detect_current_sheets(J: np.ndarray, dx: float, dy: float,
                          n: int = 9, a: float = 3.0, fmin: float = 0.4,
                          connectivity: int = 8, periodic: bool = True,
                          min_points: int = 5,
                          ) -> List[Sheet]:
    assert n % 2 == 1, "n must be odd"
    nx, ny = J.shape
    Jabs = np.abs(J)
    J_rms = np.sqrt(np.mean(J**2))

    size = (n, n)
    local_max = (Jabs == maximum_filter(Jabs, size=size, mode='wrap' if periodic else 'nearest'))
    peak_mask = local_max & (Jabs >= a * J_rms)
    peak_indices = np.argwhere(peak_mask)

    visited = np.zeros_like(Jabs, dtype=bool)
    sheets: List[Sheet] = []
    sid = 0
    for (i0, j0) in peak_indices:
        if visited[i0, j0]:
            continue
        J_peak = J[i0, j0]
        J_peak_abs = abs(J_peak)
        J_min = fmin * J_peak_abs

        stack = [(i0, j0)]
        region = []
        while stack:
            i, j = stack.pop()
            if visited[i, j]:
                continue
            if Jabs[i, j] < J_min:
                continue
            visited[i, j] = True
            region.append((i, j))
            for ii, jj in _neighbors(i, j, nx, ny, connectivity, periodic):
                if not visited[ii, jj] and Jabs[ii, jj] >= J_min:
                    stack.append((ii, jj))

        if len(region) < min_points:
            continue

        points = np.array(region, dtype=int)
        x = points[:, 1] * dx
        y = points[:, 0] * dy
        coords = np.stack([x, y], axis=1)
        centroid = coords.mean(axis=0)
        U, S, Vt = np.linalg.svd(coords - centroid, full_matrices=False)
        v_long, v_short = Vt[0], Vt[1]
        proj_long = (coords - centroid) @ v_long
        proj_short = (coords - centroid) @ v_short
        length = proj_long.max() - proj_long.min()
        width_bbox = proj_short.max() - proj_short.min()

        x0, y0 = j0 * dx, i0 * dy
        L = max(3 * width_bbox, max(dx, dy))
        s = np.linspace(-L, L, 1001)
        xs = x0 + s * v_short[0]
        ys = y0 + s * v_short[1]
        jj_float = xs / dx
        ii_float = ys / dy
        if periodic:
            ii_float = np.mod(ii_float, nx - 1)
            jj_float = np.mod(jj_float, ny - 1)
        prof = map_coordinates(Jabs, [ii_float, jj_float], order=1, mode='wrap' if periodic else 'nearest')
        half = 0.5 * J_peak_abs
        center_idx = np.argmin(np.abs(s))
        left = prof[:center_idx+1][::-1]
        s_left = s[:center_idx+1][::-1]
        i_cross_left = None
        for k in range(len(left)-1):
            if left[k] >= half and left[k+1] < half:
                frac = (left[k] - half) / (left[k] - left[k+1] + 1e-12)
                i_cross_left = s_left[k] + frac * (s_left[k+1] - s_left[k])
                break
        right = prof[center_idx:]
        s_right = s[center_idx:]
        i_cross_right = None
        for k in range(len(right)-1):
            if right[k] >= half and right[k+1] < half:
                frac = (right[k] - half) / (right[k] - right[k+1] + 1e-12)
                i_cross_right = s_right[k] + frac * (s_right[k+1] - s_right[k])
                break
        if i_cross_left is None or i_cross_right is None:
            thickness = width_bbox
        else:
            thickness = (i_cross_right - i_cross_left)

        theta = np.degrees(np.arctan2(v_long[1], v_long[0]))
        aspect = np.inf if thickness <= 0 else (length / thickness)

        sheet = Sheet(
            id=sid,
            peak_ij=(int(i0), int(j0)),
            peak_xy=(x0, y0),
            J_peak=J[i0, j0],
            J_peak_abs=J_peak_abs,
            points=points,
            centroid_xy=(centroid[0], centroid[1]),
            length=float(length),
            thickness=float(thickness),
            aspect_ratio=float(aspect),
            theta_deg=float(theta),
        )
        sheets.append(sheet)
        sid += 1

    return sheets

# ------------------------- Visualization ------------------------- #

def draw_overlays(J: np.ndarray, sheets: List[Sheet], dx: float = 1.0, dy: float = 1.0,
                  vmin: Optional[float] = None, vmax: Optional[float] = None,
                  cmap: str = 'seismic', figsize=(9, 8),
                  periodic: bool = True):
    nx, ny = J.shape
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(J, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=r"$J_\parallel$")

    for s in sheets:
        pts = s.points.copy()
        # Shift region so centroid is inside domain and avoid wrap artifacts
        ci, cj = s.peak_ij
        shift_i = 0
        shift_j = 0
        if periodic:
            if ci < nx * 0.25:
                shift_i = nx
            elif ci > nx * 0.75:
                shift_i = -nx
            if cj < ny * 0.25:
                shift_j = ny
            elif cj > ny * 0.75:
                shift_j = -ny
        pts[:,0] = pts[:,0] + shift_i
        pts[:,1] = pts[:,1] + shift_j

        ax.plot(pts[:,1], pts[:,0], ',', alpha=0.5)
        ax.plot(s.peak_ij[1] + shift_j, s.peak_ij[0] + shift_i, 'ko', ms=3)

        cx, cy = s.centroid_xy
        v = np.array([np.cos(np.radians(s.theta_deg)), np.sin(np.radians(s.theta_deg))])
        L2 = 0.5 * s.length
        x1, y1 = (cx - L2 * v[0]) / dx + shift_j, (cy - L2 * v[1]) / dy + shift_i
        x2, y2 = (cx + L2 * v[0]) / dx + shift_j, (cy + L2 * v[1]) / dy + shift_i
        ax.plot([x1, x2], [y1, y2], 'k-', lw=1)

    ax.set_xlabel(r"$x$ [cells]")
    ax.set_ylabel(r"$y$ [cells]")
    ax.set_title(r"Detected current sheets and peaks (wrapped cleanly)")
    fig.tight_layout()
    return fig, ax

# ------------------------- Summary ------------------------- #
def summarize_sheets(sheets: List[Sheet]):
    import pandas as pd
    rows = [asdict(s) for s in sheets]
    for r in rows:
        r.pop('points', None)
    df = pd.DataFrame(rows)
    cols = ['id','J_peak','J_peak_abs','length','thickness','aspect_ratio',
            'theta_deg','peak_xy','centroid_xy','peak_ij']
    df = df[cols]
    return df

