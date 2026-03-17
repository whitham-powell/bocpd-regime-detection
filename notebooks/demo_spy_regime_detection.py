# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SPY Regime Detection with BOCPD
#
# Bayesian Online Change Point Detection applied to S&P 500 ETF (SPY)
# daily log returns. We use a univariate Normal-Inverse-Gamma model
# and constant hazard function.
#
# This notebook showcases the **predictive envelope** — the mixture
# predictive mean and standard deviation computed from the run-length
# posterior — which is available for univariate models (NIG, PoissonGamma)
# but not for the multivariate NIW used in the experiments notebook.
#
# Reference: Adams & MacKay (2007), *Bayesian Online Changepoint Detection*.

# %%
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from finfeatures.sources import YFinanceSource

from bocpd import (
    BOCPD,
    ConstantHazard,
    UnivariateNormalNIG,
    extract_change_points_with_bounds,
)

# Colour palette — consistent with experiments notebook
C_CP = "#7F77DD"  # purple  — change point markers
C_CI = "#AFA9EC"  # light purple — credible interval bands
C_ERL = "#534AB7"  # dark purple — ERL line
C_EVENT = "#444441"  # dark gray — known event markers
C_PRICE = "#2C2C2A"  # near black — price line
C_PRED = "#1D9E75"  # teal — predictive envelope
C_RETURN = "#888886"  # mid gray — log return scatter

KNOWN_EVENTS = {
    "COVID crash": "2020-02-19",
    "COVID bottom": "2020-03-23",
    "2022 drawdown": "2022-01-03",
    "2022 bottom": "2022-10-13",
    "2023 rally": "2023-01-03",
}


def mark_events(ax, dates_index, alpha=0.3, label_first=True):
    """Draw dotted vertical lines for known market events."""
    first = True
    for _name, ds in KNOWN_EVENTS.items():
        dt = pd.Timestamp(ds)
        if dates_index[0] <= dt <= dates_index[-1]:
            ax.axvline(
                dt,
                color=C_EVENT,
                lw=0.8,
                alpha=alpha,
                ls=":",
                label="Known events" if (first and label_first) else None,
            )
            first = False


def format_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def draw_change_points(ax, cps, dates_index, color=C_CP, alpha_line=0.8, draw_ci=True):
    """Draw change point verticals and optional credible interval bands."""
    for cp in cps:
        dt = dates_index[cp["index"]]
        ax.axvline(dt, color=color, lw=1.8, alpha=alpha_line)
        if draw_ci:
            ax.axvspan(
                dates_index[cp["lower"]],
                dates_index[cp["upper"]],
                alpha=0.08,
                color=color,
            )


print("Imports OK")

# %% [markdown]
# ## Fetch SPY data

# %%
source = YFinanceSource()
df = source.fetch("SPY", start="2018-01-01", end="2024-12-31")
close = df["close"].values
dates = df.index

# Compute log returns
log_returns = np.diff(np.log(close))
dates_returns = dates[1:]

print(f"SPY observations: {len(log_returns)} trading days")
print(f"Date range: {dates_returns[0].date()} to {dates_returns[-1].date()}")

# %% [markdown]
# ## Run BOCPD

# %%
detector = BOCPD(
    model_factory=lambda: UnivariateNormalNIG(
        mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=0.0001
    ),
    hazard_fn=ConstantHazard(lam=100),
)
result = detector.run(log_returns)

boundaries = extract_change_points_with_bounds(result, credible_mass=0.90, min_gap=20)

print(f"Detected {len(boundaries)} change points:")
for b in boundaries:
    idx = b["index"]
    lo = dates_returns[b["lower"]]
    hi = dates_returns[b["upper"]]
    ci_width = (hi - lo).days
    print(
        f"  {dates_returns[idx].strftime('%Y-%m-%d')}  "
        f"90% CI [{lo.strftime('%Y-%m-%d')} -- {hi.strftime('%Y-%m-%d')}]  "
        f"({ci_width}d wide)  severity={b['severity']:.2f}"
    )

# %% [markdown]
# ## Overview: price, predictive envelope, heatmap, and ERL
#
# Four-panel figure combining the key outputs. The predictive envelope
# (panel 2) is the unique feature of univariate models — it shows the
# one-step-ahead mixture predictive mean $\pm$ one standard deviation,
# weighted by the run-length posterior. The envelope widens after change
# points as the fresh prior contributes high uncertainty, then tightens
# as within-regime data accumulates.

# %%
T = len(log_returns)
posteriors = result["run_length_posterior"]
max_rl = max(len(p) for p in posteriors)
rl_matrix = np.zeros((max_rl, T))
for t, p in enumerate(posteriors):
    rl_matrix[: len(p), t] = p
rl_matrix = np.clip(rl_matrix, 1e-6, 1.0)

pred_mean = result["predictive_mean"]
pred_std = np.sqrt(result["predictive_var"])
erl = result["expected_run_length"]

fig = plt.figure(figsize=(14, 14))
gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1.0, 1.2, 2, 1], hspace=0.07)

ax_price = fig.add_subplot(gs[0])
ax_pred = fig.add_subplot(gs[1], sharex=ax_price)
ax_rl = fig.add_subplot(gs[2], sharex=ax_price)
ax_erl = fig.add_subplot(gs[3], sharex=ax_price)

# -- Panel 1: price --
ax_price.plot(dates, close, color=C_PRICE, lw=0.8, label="SPY close")
draw_change_points(ax_price, boundaries, dates_returns)
mark_events(ax_price, dates_returns)
ax_price.set_ylabel("Price ($)")
ax_price.set_title(
    "SPY regime detection — UnivariateNormalNIG  (λ=100, κ₀=0.1)",
    fontsize=11,
    fontweight="bold",
)
ax_price.legend(fontsize=8, loc="upper left")
ax_price.grid(True, alpha=0.25)
plt.setp(ax_price.get_xticklabels(), visible=False)

# -- Panel 2: predictive envelope --
ax_pred.plot(
    dates_returns, log_returns, color=C_RETURN, lw=0.3, alpha=0.6, label="Log return"
)
ax_pred.plot(dates_returns, pred_mean, color=C_PRED, lw=1, label="Predictive mean")
ax_pred.fill_between(
    dates_returns,
    pred_mean - pred_std,
    pred_mean + pred_std,
    color=C_PRED,
    alpha=0.15,
    label="±1 std",
)
draw_change_points(ax_pred, boundaries, dates_returns, draw_ci=False, alpha_line=0.5)
mark_events(ax_pred, dates_returns, label_first=False)
ax_pred.set_ylabel("Log return")
ax_pred.legend(fontsize=8, loc="upper right")
ax_pred.grid(True, alpha=0.25)
plt.setp(ax_pred.get_xticklabels(), visible=False)

# -- Panel 3: run-length posterior heatmap --
date_edges = np.append(dates_returns, dates_returns[-1] + pd.Timedelta(days=1))
im = ax_rl.pcolormesh(
    date_edges,
    np.arange(max_rl + 1),
    rl_matrix,
    cmap="gray_r",
    norm=mcolors.LogNorm(vmin=1e-5, vmax=1.0),
    shading="flat",
    rasterized=True,
)
ax_rl.set_ylim(0, min(500, max_rl))
draw_change_points(ax_rl, boundaries, dates_returns, draw_ci=False, alpha_line=0.5)
mark_events(ax_rl, dates_returns, label_first=False)
ax_rl.set_ylabel("Run length (days)")
fig.colorbar(im, ax=ax_rl, label=r"$P(r_t \mid x_{1:t})$", shrink=0.8, pad=0.01)
plt.setp(ax_rl.get_xticklabels(), visible=False)

# -- Panel 4: expected run length with credible bands --
ax_erl.plot(dates_returns, erl, color=C_ERL, lw=1, label="E[r_t]")
draw_change_points(ax_erl, boundaries, dates_returns)
mark_events(ax_erl, dates_returns, label_first=False)
ax_erl.set_ylabel("Expected run length")
ax_erl.set_xlabel("Date")
ax_erl.legend(fontsize=8)
ax_erl.grid(True, alpha=0.25)
format_xaxis(ax_erl)

fig.tight_layout()
plt.show()

# %% [markdown]
# ### Reading this figure
#
# **Price** (top): detected change points align with major market turning
# points — COVID crash, recovery, 2022 drawdown, and subsequent rally.
# The credible interval bands (shaded purple) show uncertainty about
# the exact transition date.
#
# **Predictive envelope** (second): the mixture predictive distribution
# adapts in real time. After a change point, the envelope widens as the
# fresh NIG prior (with infinite Student-t variance at α₀=1) enters the
# mixture. As within-regime observations accumulate, alpha grows, the
# Student-t degrees of freedom increase, and the envelope tightens.
# The envelope is narrow during calm periods and blows out during volatile
# regimes — this is the model correctly tracking regime-specific variance.
#
# **Run-length posterior** (third): the primary output. Each column is
# the full posterior over run lengths at that time step. Before a change
# point, most mass sits on long run lengths — visible as a bright
# diagonal band growing from the bottom-left. At a change point the
# posterior collapses: mass drains from long run lengths and concentrates
# near zero, appearing as a vertical bright stripe.
#
# **Expected run length** (bottom): a scalar summary of the posterior.
# Sharp drops correspond to detected change points. The 90% credible
# interval bands are derived from aggregating retrospective
# change-time distributions across nearby time steps.

# %%
