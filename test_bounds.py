"""Test confidence bounds extraction on synthetic data."""

import sys
import numpy as np

sys.path.insert(0, "/home/claude/bocpd")
from src import (
    UnivariateNormalNIG,
    MultivariateNormalNIW,
    ConstantHazard,
    BOCPD,
    extract_change_points_with_bounds,
)


def test_bounds(name, data, model_factory, hazard_fn, true_cps):
    detector = BOCPD(model_factory=model_factory, hazard_fn=hazard_fn)
    result = detector.run(data)
    bounds = extract_change_points_with_bounds(result, credible_mass=0.90)

    print(f"\n{name}")
    print(f"  True CPs: {true_cps}")
    for b in bounds:
        print(f"  Detected: t={b['index']:4d}  "
              f"90% CI: [{b['lower']:4d}, {b['upper']:4d}]  "
              f"width={b['upper'] - b['lower']:3d}  "
              f"severity={b['severity']:.2f}")
        # Check if true CP falls within bounds
        for tcp in true_cps:
            if b['lower'] <= tcp <= b['upper']:
                print(f"    ✓ true CP {tcp} is within bounds")


# Sharp mean shift — should have tight bounds
np.random.seed(42)
test_bounds(
    "Sharp mean shift (0 → 3) at t=100",
    np.concatenate([np.random.normal(0, 1, 100), np.random.normal(3, 1, 100)]),
    lambda: UnivariateNormalNIG(mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0),
    ConstantHazard(lam=100),
    [100],
)

# Subtle mean shift — should have wider bounds
np.random.seed(42)
test_bounds(
    "Subtle mean shift (0 → 0.5) at t=100",
    np.concatenate([np.random.normal(0, 1, 100), np.random.normal(0.5, 1, 100)]),
    lambda: UnivariateNormalNIG(mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0),
    ConstantHazard(lam=100),
    [100],
)

# Multivariate
np.random.seed(42)
dim = 3
test_bounds(
    "Multivariate (D=3) mean shift at t=100",
    np.vstack([
        np.random.multivariate_normal([0, 0, 0], np.eye(dim), 100),
        np.random.multivariate_normal([2, -1, 1], np.eye(dim), 100),
    ]),
    lambda: MultivariateNormalNIW(dim=dim, kappa0=0.1, nu0=float(dim), Psi0=np.eye(dim)),
    ConstantHazard(lam=100),
    [100],
)

# Multiple change points
np.random.seed(42)
test_bounds(
    "Multiple CPs at t=80, 160",
    np.concatenate([
        np.random.normal(0, 1, 80),
        np.random.normal(3, 0.5, 80),
        np.random.normal(-1, 2, 80),
    ]),
    lambda: UnivariateNormalNIG(mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0),
    ConstantHazard(lam=80),
    [80, 160],
)

# Variance shift — typically harder, wider bounds expected
np.random.seed(42)
dim = 2
test_bounds(
    "Variance shift (D=2) at t=150",
    np.vstack([
        np.random.multivariate_normal([0, 0], 0.5 * np.eye(dim), 150),
        np.random.multivariate_normal([0, 0], 3.0 * np.eye(dim), 150),
    ]),
    lambda: MultivariateNormalNIW(dim=dim, kappa0=0.1, nu0=float(dim)+1, Psi0=np.eye(dim)),
    ConstantHazard(lam=150),
    [150],
)
