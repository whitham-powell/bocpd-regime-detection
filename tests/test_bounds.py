"""Test confidence bounds extraction on synthetic data."""

import numpy as np

from bocpd import (
    BOCPD,
    ConstantHazard,
    MultivariateNormalNIW,
    UnivariateNormalNIG,
    extract_change_points_with_bounds,
)


def _run_with_bounds(data, model_factory, hazard_fn):
    """Run BOCPD and extract change points with bounds."""
    detector = BOCPD(model_factory=model_factory, hazard_fn=hazard_fn)
    result = detector.run(data)
    return extract_change_points_with_bounds(result, credible_mass=0.90)


def test_sharp_mean_shift():
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.normal(0, 1, 100),
            np.random.normal(3, 1, 100),
        ]
    )
    bounds = _run_with_bounds(
        data,
        lambda: UnivariateNormalNIG(mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0),
        ConstantHazard(lam=100),
    )
    assert len(bounds) > 0, "No change point detected"
    b = bounds[0]
    assert b["lower"] <= 100 <= b["upper"], (
        f"True CP 100 not in [{b['lower']}, {b['upper']}]"
    )
    assert b["severity"] > 0


def test_subtle_mean_shift():
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.normal(0, 1, 100),
            np.random.normal(0.5, 1, 100),
        ]
    )
    bounds = _run_with_bounds(
        data,
        lambda: UnivariateNormalNIG(mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0),
        ConstantHazard(lam=100),
    )
    # Subtle shift may or may not be detected; if detected, check bounds
    if len(bounds) > 0:
        b = bounds[0]
        assert b["severity"] >= 0


def test_multivariate_bounds():
    np.random.seed(42)
    dim = 3
    data = np.vstack(
        [
            np.random.multivariate_normal([0, 0, 0], np.eye(dim), 100),
            np.random.multivariate_normal([2, -1, 1], np.eye(dim), 100),
        ]
    )
    bounds = _run_with_bounds(
        data,
        lambda: MultivariateNormalNIW(
            dim=dim, kappa0=0.1, nu0=float(dim), Psi0=np.eye(dim)
        ),
        ConstantHazard(lam=100),
    )
    assert len(bounds) > 0, "No change point detected"
    b = bounds[0]
    assert b["lower"] <= 100 <= b["upper"], (
        f"True CP 100 not in [{b['lower']}, {b['upper']}]"
    )
    assert b["severity"] > 0


def test_multiple_cps_bounds():
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.normal(0, 1, 80),
            np.random.normal(3, 0.5, 80),
            np.random.normal(-1, 2, 80),
        ]
    )
    bounds = _run_with_bounds(
        data,
        lambda: UnivariateNormalNIG(mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0),
        ConstantHazard(lam=80),
    )
    assert len(bounds) >= 2, f"Expected >=2 CPs, got {len(bounds)}"
    for b in bounds:
        assert b["severity"] > 0


def test_variance_shift_bounds():
    np.random.seed(42)
    dim = 2
    data = np.vstack(
        [
            np.random.multivariate_normal([0, 0], 0.5 * np.eye(dim), 150),
            np.random.multivariate_normal([0, 0], 3.0 * np.eye(dim), 150),
        ]
    )
    bounds = _run_with_bounds(
        data,
        lambda: MultivariateNormalNIW(
            dim=dim, kappa0=0.1, nu0=float(dim) + 1, Psi0=np.eye(dim)
        ),
        ConstantHazard(lam=150),
    )
    assert len(bounds) > 0, "No change point detected for variance shift"
    b = bounds[0]
    assert b["severity"] > 0
