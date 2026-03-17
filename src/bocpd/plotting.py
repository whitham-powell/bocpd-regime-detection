"""Plotting utilities for BOCPD results.

All panel functions follow the axes-level convention: they draw on a
provided ``ax`` and never create figures or call ``plt.show()``.

This module is deliberately not re-exported from ``bocpd.__init__`` so
that ``import bocpd`` never triggers a matplotlib import.
"""

from types import SimpleNamespace

import numpy as np

try:
    import matplotlib.colors as mcolors
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for bocpd.plotting — install it with "
        "`pip install matplotlib`"
    ) from _exc

COLORS = SimpleNamespace(
    cp="#7F77DD",  # purple  — change point markers
    ci="#AFA9EC",  # light purple — credible interval bands
    erl="#534AB7",  # dark purple — ERL line
    event="#C04040",  # muted red — known event markers
    price="#2C2C2A",  # near black — price line
    pred="#1D9E75",  # teal — predictive envelope
    ret="#888886",  # mid gray — log return scatter
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def build_rl_matrix(posteriors, clip_lo=1e-6, clip_hi=1.0):
    """Build (max_run_length, T) matrix from variable-length posterior list.

    Parameters
    ----------
    posteriors : list[np.ndarray]
        Per-timestep run-length posterior vectors (variable length).
    clip_lo, clip_hi : float
        Clipping bounds applied element-wise to the assembled matrix.

    Returns
    -------
    np.ndarray
        Shape ``(max_run_length, T)`` with values clipped to
        ``[clip_lo, clip_hi]``.
    """
    T = len(posteriors)
    max_rl = max(len(p) for p in posteriors)
    rl_matrix = np.zeros((max_rl, T))
    for t, p in enumerate(posteriors):
        rl_matrix[: len(p), t] = p
    return np.clip(rl_matrix, clip_lo, clip_hi)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def mark_events(
    ax,
    events,
    dates_index,
    *,
    color=COLORS.event,
    alpha=0.5,
    lw=1.0,
    label_first=True,
    show_labels=True,
):
    """Draw dashed vertical lines for a dict of ``{name: date_string}`` events.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    events : dict[str, str]
        Mapping of event name to date string (parseable by ``pd.Timestamp``).
    dates_index : pandas.DatetimeIndex
        Used to check whether each event falls within the visible range.
    color, alpha, lw : aesthetic overrides.
    label_first : bool
        If *True*, the first visible line gets ``label="Known events"``.
    show_labels : bool
        If *True*, place the event name as rotated text near the top of the axes.
    """
    import pandas as pd

    first = True
    for name, ds in events.items():
        dt = pd.Timestamp(ds)
        if dates_index[0] <= dt <= dates_index[-1]:
            ax.axvline(
                dt,
                color=color,
                lw=lw,
                alpha=alpha,
                ls="--",
                label="Known events" if (first and label_first) else None,
            )
            if show_labels:
                ax.text(
                    dt,
                    0.95,
                    name,
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    fontsize=7,
                    va="top",
                    ha="right",
                    color=color,
                )
            first = False


def format_xaxis(ax, interval=3):
    """Apply ``DateFormatter('%Y-%m')``, ``MonthLocator``, and rotated labels."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def draw_change_points(
    ax,
    boundaries,
    dates_index,
    *,
    color=COLORS.cp,
    alpha_line=0.8,
    draw_ci=True,
    ci_alpha=0.08,
):
    """Draw vertical lines at change points with optional CI bands.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    boundaries : list[dict]
        Dicts from ``extract_change_points_with_bounds()``, each containing
        ``"index"``, ``"lower"``, and ``"upper"`` keys.
    dates_index : pandas.DatetimeIndex
    color : str
    alpha_line : float
    draw_ci : bool
        Whether to draw shaded credible-interval bands.
    ci_alpha : float
        Alpha for the CI ``axvspan``.
    """
    for cp in boundaries:
        dt = dates_index[cp["index"]]
        ax.axvline(dt, color=color, lw=1.8, alpha=alpha_line)
        if draw_ci:
            ax.axvspan(
                dates_index[cp["lower"]],
                dates_index[cp["upper"]],
                alpha=ci_alpha,
                color=color,
            )


# ---------------------------------------------------------------------------
# Panel functions
# ---------------------------------------------------------------------------


def plot_run_length_heatmap(
    ax,
    posteriors,
    dates_index,
    *,
    y_max=500,
    cmap="gray_r",
    vmin=1e-5,
    vmax=1.0,
    clip_lo=1e-6,
    rasterized=True,
    colorbar=True,
    colorbar_kw=None,
):
    """Pcolormesh heatmap of the run-length posterior with LogNorm.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    posteriors : list[np.ndarray]
    dates_index : pandas.DatetimeIndex
    y_max : int
        Upper limit for the y-axis (run length).
    cmap : str
    vmin, vmax : float
        Bounds for the ``LogNorm`` colour mapping.
    clip_lo : float
        Lower clip for ``build_rl_matrix``.
    rasterized : bool
    colorbar : bool
        Whether to add a colorbar.
    colorbar_kw : dict or None
        Extra keyword arguments forwarded to ``fig.colorbar``.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The pcolormesh artist (useful for external colorbar calls).
    """
    import pandas as pd

    rl_matrix = build_rl_matrix(posteriors, clip_lo=clip_lo)
    max_rl = rl_matrix.shape[0]

    date_edges = np.append(dates_index, dates_index[-1] + pd.Timedelta(days=1))
    im = ax.pcolormesh(
        date_edges,
        np.arange(max_rl + 1),
        rl_matrix,
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        shading="flat",
        rasterized=rasterized,
    )
    ax.set_ylim(0, min(y_max, max_rl))
    ax.set_ylabel("Run length (days)")

    if colorbar:
        kw = {"label": r"$P(r_t \mid x_{1:t})$", "shrink": 0.8, "pad": 0.01}
        if colorbar_kw:
            kw.update(colorbar_kw)
        ax.get_figure().colorbar(im, ax=ax, **kw)

    return im


def plot_erl(
    ax, erl, dates_index, *, color=COLORS.erl, lw=1.0, label="E[r_t]", grid=True
):
    """Plot expected run length time series."""
    ax.plot(dates_index, erl, color=color, lw=lw, label=label)
    ax.set_ylabel("Expected run length")
    if grid:
        ax.grid(True, alpha=0.25)


def plot_price(
    ax, prices, dates_index, *, color=COLORS.price, lw=0.8, label="Close", grid=True
):
    """Plot a price time series."""
    ax.plot(dates_index, prices, color=color, lw=lw, label=label)
    ax.set_ylabel("Price ($)")
    if grid:
        ax.grid(True, alpha=0.25)


def plot_predictive_envelope(
    ax,
    dates_index,
    observations,
    pred_mean,
    pred_var,
    *,
    n_std=1.0,
    color_obs=COLORS.ret,
    color_pred=COLORS.pred,
    grid=True,
):
    """Plot observations as scatter with predictive mean +/- n_std band.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    dates_index : pandas.DatetimeIndex
    observations : array-like
        Raw observation values (e.g. log returns).
    pred_mean : array-like
    pred_var : array-like
        Predictive variance (square-rooted internally).
    n_std : float
        Number of standard deviations for the band.
    color_obs, color_pred : str
    grid : bool
    """
    pred_std = np.sqrt(pred_var)
    ax.plot(
        dates_index,
        observations,
        color=color_obs,
        lw=0.3,
        alpha=0.6,
        label="Log return",
    )
    ax.plot(dates_index, pred_mean, color=color_pred, lw=1, label="Predictive mean")
    ax.fill_between(
        dates_index,
        pred_mean - n_std * pred_std,
        pred_mean + n_std * pred_std,
        color=color_pred,
        alpha=0.15,
        label=f"\u00b1{n_std:.0f} std",
    )
    ax.set_ylabel("Log return")
    if grid:
        ax.grid(True, alpha=0.25)
