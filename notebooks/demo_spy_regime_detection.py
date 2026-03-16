# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
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
# Reference: Adams & MacKay (2007), *Bayesian Online Changepoint Detection*.

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from finfeatures.sources import YFinanceSource

from bocpd import (
    BOCPD,
    ConstantHazard,
    UnivariateNormalNIG,
    extract_change_points_with_bounds,
)

sns.set_theme(style="whitegrid")

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
print(f"Date range: {dates_returns[0]} to {dates_returns[-1]}")

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
    print(
        f"  t={idx:4d} ({dates_returns[idx].strftime('%Y-%m-%d')})  "
        f"90% CI: [{b['lower']}, {b['upper']}]  "
        f"severity={b['severity']:.2f}"
    )

# %% [markdown]
# ## Run-length posterior heatmap

# %%
posteriors = result["run_length_posterior"]
T = len(posteriors)
max_rl = max(len(p) for p in posteriors)
rl_matrix = np.zeros((max_rl, T))
for t, p in enumerate(posteriors):
    rl_matrix[: len(p), t] = p

fig, ax = plt.subplots(figsize=(14, 5))
ax.imshow(
    rl_matrix[:200, :],
    aspect="auto",
    origin="lower",
    cmap="hot",
    interpolation="none",
)
ax.set_xlabel("Time (trading days)")
ax.set_ylabel("Run length")
ax.set_title("Run-length posterior P(r_t | x_{1:t})")
fig.tight_layout()
fig.savefig("figures/run_length_posterior.png", dpi=150)
plt.show()

# %% [markdown]
# ## Expected run length with change points

# %%
erl = result["expected_run_length"]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(dates_returns, erl, color="steelblue", linewidth=0.8, label="E[r_t]")

for b in boundaries:
    idx = b["index"]
    ax.axvline(dates_returns[idx], color="red", alpha=0.7, linewidth=1.2)
    ax.axvspan(
        dates_returns[b["lower"]],
        dates_returns[b["upper"]],
        alpha=0.15,
        color="red",
    )

ax.set_xlabel("Date")
ax.set_ylabel("Expected run length")
ax.set_title("Expected run length with detected change points (90% CI bands)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/erl_change_points.png", dpi=150)
plt.show()

# %% [markdown]
# ## SPY price with regime boundaries

# %%
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates, close, color="black", linewidth=0.7, label="SPY close")

for b in boundaries:
    idx = b["index"]
    ax.axvline(
        dates_returns[idx],
        color="red",
        alpha=0.6,
        linewidth=1.2,
        linestyle="--",
    )

ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.set_title("SPY daily close with BOCPD regime boundaries")
ax.legend()
fig.tight_layout()
fig.savefig("figures/spy_regimes.png", dpi=150)
plt.show()
