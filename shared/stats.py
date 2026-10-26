"""Statistics helpers.

Functions copied verbatim from the Paper 1 pipeline notebook
(CSDetectionMaxTurbulenceFullPipelineWithKCutoff_Beta1.ipynb) and, for the lognormal-mode
helpers, from PooledStatisticsProtonBeta.py; behaviour unchanged.
"""
import numpy as np
from scipy.stats import lognorm


def bootstrap_lognorm_median(x_di, n_boot, rng):
    """Vectorised bootstrap SE + 95% CI of the lognormal median (= exp(mean(log x)))."""
    x = np.asarray(x_di, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 5:
        return np.nan, np.nan, np.nan
    logx = np.log(x); n = x.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_med = np.exp(logx[idx].mean(axis=1))
    return (float(boot_med.std(ddof=1)),
            float(np.percentile(boot_med, 2.5)),
            float(np.percentile(boot_med, 97.5)))


def lognorm_mode_with_error(x):
    """
    Mode of a lognormal (floc=0) plus its fitting uncertainty,
    from the analytic maximum-likelihood standard errors.

    For floc=0:  mu_hat = mean(ln x),  sigma_hat = std(ln x)
        SE(mu)    = sigma / sqrt(N)
        SE(sigma) = sigma / sqrt(2N)
    Mode = exp(mu - sigma^2). Error propagated in quadrature.
    """
    x = np.asarray(x)
    x = x[x > 0]
    N = x.size
    logx = np.log(x)
    mu_hat    = logx.mean()
    sigma_hat = logx.std(ddof=1)

    mu_err    = sigma_hat / np.sqrt(N)
    sigma_err = sigma_hat / np.sqrt(2*N)

    mode = np.exp(mu_hat - sigma_hat**2)
    # partial derivatives of mode = exp(mu - sigma^2)
    dmode_dmu    = mode
    dmode_dsigma = -2.0 * sigma_hat * mode
    mode_err = np.sqrt((dmode_dmu * mu_err)**2 + (dmode_dsigma * sigma_err)**2)

    return mode, mode_err


def lognorm_mode(shape, scale):
    """Mode of a lognormal distribution with loc=0.
    For lognorm.fit(..., floc=0): scale = exp(mu), shape = sigma.
    Mode = exp(mu - sigma^2) = scale * exp(-shape**2)."""
    return scale * np.exp(-shape**2)


def fit_and_plot_pdf(ax, x_di, title, color=None, bins=30, ylabel=False, xmax=None):
    """Per-method PDF panel: histogram + lognormal fit + median. Returns a dict of statistics.
    color (histogram and fit) and xmax (shared x-limit) are optional; without them the panel is
    exactly the original production version."""
    x = np.asarray(x_di, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    ax.set_xlabel(r"$d_i$")
    if ylabel:
        ax.set_ylabel("Probability density")
    if x.size < 5:
        ax.set_title(f"{title}  (N={x.size})")
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        if xmax is not None:
            ax.set_xlim(0, xmax)
        return {"n": int(x.size), "median_lognorm": np.nan, "sigma_lognorm": np.nan,
                "mean": np.nan, "median_emp": np.nan}
    shape, loc, scale = lognorm.fit(x, floc=0)
    xs_fit = np.linspace(0, xmax if xmax is not None else x.max(), 400)
    colour = {} if color is None else {"color": color}
    ax.hist(x, bins=bins, density=True, alpha=0.4, label="Histogram", **colour)
    ax.plot(xs_fit, lognorm.pdf(xs_fit, shape, loc, scale), label="Log-normal fit", **colour)
    ax.axvline(scale, color="k", ls="--", lw=1.5, label=f"Median = {scale:.3f}")
    ax.set_title(f"{title}  (N={x.size})")
    ax.legend()
    if xmax is not None:
        ax.set_xlim(0, xmax)
    return {"n": int(x.size), "median_lognorm": float(scale),
            "sigma_lognorm": float(shape), "mean": float(x.mean()),
            "median_emp": float(np.median(x))}


def avg_beta(region_mask, P_scalar, B2):
    """Region-averaged plasma beta = 2<P>/<B^2> (ratio of averages, not mean of ratios)."""
    return float(2.0 * P_scalar[region_mask].mean() / B2[region_mask].mean())
