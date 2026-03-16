# BOCPD — Bayesian Online Change Point Detection

A modular Python implementation of Adams & MacKay (2007) for detecting regime changes in time series data.

## Installation

```bash
# Core package
pip install -e .

# With all development dependencies
pip install -e ".[dev]"
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

## Demo notebook

See `notebooks/demo_spy_regime_detection.py` for a full example applying BOCPD to SPY daily returns. Convert to a Jupyter notebook with:

```bash
make notebooks
```

## Reference

Adams, R. P., & MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection. arXiv:0710.3742.
