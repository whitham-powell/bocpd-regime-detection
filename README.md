# BOCPD — Bayesian Online Change Point Detection

A modular Python implementation of Adams & MacKay (2007) for detecting regime changes in time series data.

## Installation

### As a dependency in another project (uv)

```bash
uv add bocpd @ git+https://github.com/whitham-powell/bocpd-regime-detection.git
```

### Local development

```bash
git clone https://github.com/whitham-powell/bocpd-regime-detection.git
cd bocpd-regime-detection
uv sync          # installs core + dev dependencies
```

### Google Colab

```python
!pip install git+https://github.com/whitham-powell/bocpd-regime-detection.git
```

## Quick start

```python
import numpy as np
from bocpd import BOCPD, ConstantHazard, UnivariateNormalNIG, extract_change_points_with_bounds

# Synthetic data with a mean shift at t=100
data = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(3, 1, 100)])

detector = BOCPD(
    model_factory=lambda: UnivariateNormalNIG(mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0),
    hazard_fn=ConstantHazard(lam=100),
)
result = detector.run(data)

boundaries = extract_change_points_with_bounds(result, credible_mass=0.90)
for b in boundaries:
    print(f"Change point at t={b['index']}, 90% CI: [{b['lower']}, {b['upper']}]")
```

## Notebooks

Example notebooks live in `examples/` (as Jupytext `.py` files). Rendered versions are in [`examples/rendered/`](examples/rendered/):

- [SPY regime detection](examples/rendered/demo_spy_regime_detection.md) — univariate NIG model with predictive envelope
- [Sensitivity analysis](examples/rendered/bocpd_experiments.md) — systematic sweep of λ, hazard shape, and prior strength
- [Adams & MacKay Fig. 3 replication](examples/rendered/replicate_adams_mackay_fig3.md) — reproducing the original paper's well-log example

## Reference

Adams, R. P., & MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection. arXiv:0710.3742.
