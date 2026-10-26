"""Thickness-method helpers (production versions from Paper 1).

Functions copied verbatim from the Paper 1 pipeline notebook
(CSDetectionMaxTurbulenceFullPipelineWithKCutoff_Beta1.ipynb); behaviour unchanged.
"""
import numpy as np


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
