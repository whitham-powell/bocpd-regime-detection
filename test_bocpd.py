"""
Smoke tests for BOCPD implementation.
Tests on synthetic data where we know ground truth change points.
"""

import sys
import numpy as np

sys.path.insert(0, "/home/claude/bocpd")
from src import (
    UnivariateNormalNIG,
    MultivariateNormalNIW,
    PoissonGamma,
    ConstantHazard,
    BOCPD,
    extract_change_points,
)


def run_and_report(name, data, model_factory, hazard_fn, true_cps):
    """Run BOCPD and report results with multiple extraction methods."""
    detector = BOCPD(model_factory=model_factory, hazard_fn=hazard_fn)
    result = detector.run(data)

    print(f"\n{name}")
    print(f"  True change points: {true_cps}")
    print(f"  E[r_t] range: [{result['expected_run_length'].min():.1f}, "
          f"{result['expected_run_length'].max():.1f}]")
    print(f"  MAP r_t range: [{result['map_run_length'].min()}, "
          f"{result['map_run_length'].max()}]")

    for method in ["expected_run_length", "map_run_length", "posterior_mass"]:
        cps = extract_change_points(result, method=method)
        print(f"  [{method}] detected: {cps}")

    return result


def test_univariate_synthetic():
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(3, 1, 100),
    ])
    run_and_report(
        "Univariate: mean shift at t=100",
        data,
        model_factory=lambda: UnivariateNormalNIG(
            mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0
        ),
        hazard_fn=ConstantHazard(lam=100),
        true_cps=[100],
    )


def test_multivariate_synthetic():
    np.random.seed(42)
    dim = 3
    data = np.vstack([
        np.random.multivariate_normal([0, 0, 0], np.eye(dim), 100),
        np.random.multivariate_normal([2, -1, 1], np.eye(dim), 100),
    ])
    run_and_report(
        "Multivariate (D=3): mean shift at t=100",
        data,
        model_factory=lambda: MultivariateNormalNIW(
            dim=dim, kappa0=0.1, nu0=float(dim), Psi0=np.eye(dim)
        ),
        hazard_fn=ConstantHazard(lam=100),
        true_cps=[100],
    )


def test_multivariate_variance_shift():
    np.random.seed(42)
    dim = 2
    data = np.vstack([
        np.random.multivariate_normal([0, 0], 0.5 * np.eye(dim), 150),
        np.random.multivariate_normal([0, 0], 3.0 * np.eye(dim), 150),
    ])
    run_and_report(
        "Variance shift (D=2): covariance change at t=150",
        data,
        model_factory=lambda: MultivariateNormalNIW(
            dim=dim, kappa0=0.1, nu0=float(dim) + 1, Psi0=np.eye(dim)
        ),
        hazard_fn=ConstantHazard(lam=150),
        true_cps=[150],
    )


def test_multiple_change_points():
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(0, 1, 80),
        np.random.normal(3, 0.5, 80),
        np.random.normal(-1, 2, 80),
    ])
    run_and_report(
        "Multiple CPs: shifts at t=80, 160",
        data,
        model_factory=lambda: UnivariateNormalNIG(
            mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0
        ),
        hazard_fn=ConstantHazard(lam=80),
        true_cps=[80, 160],
    )


def test_poisson_synthetic():
    np.random.seed(42)
    data = np.concatenate([
        np.random.poisson(3, 100),
        np.random.poisson(10, 100),
    ]).astype(float)
    run_and_report(
        "Poisson: rate change 3->10 at t=100",
        data,
        model_factory=lambda: PoissonGamma(alpha0=1.0, beta0=0.25),
        hazard_fn=ConstantHazard(lam=100),
        true_cps=[100],
    )


def test_no_change():
    np.random.seed(42)
    data = np.random.normal(0, 1, 200)
    run_and_report(
        "No change (stationary N(0,1))",
        data,
        model_factory=lambda: UnivariateNormalNIG(
            mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0
        ),
        hazard_fn=ConstantHazard(lam=200),
        true_cps=[],
    )


def test_log_predictive_sanity():
    print("\nLog-predictive sanity checks:")
    model = UnivariateNormalNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    lp_near = model.log_predictive(np.array(0.0))
    lp_far = model.log_predictive(np.array(10.0))
    assert lp_near > lp_far
    print(f"  Univariate: near={lp_near:.4f}, far={lp_far:.4f} ok")

    model_mv = MultivariateNormalNIW(dim=3, kappa0=1.0, nu0=4.0, Psi0=np.eye(3))
    lp_near_mv = model_mv.log_predictive(np.zeros(3))
    lp_far_mv = model_mv.log_predictive(np.ones(3) * 10)
    assert lp_near_mv > lp_far_mv
    print(f"  Multivariate: near={lp_near_mv:.4f}, far={lp_far_mv:.4f} ok")

    model_p = PoissonGamma(alpha0=4.0, beta0=1.0)
    lp_near_p = model_p.log_predictive(np.array(4.0))
    lp_far_p = model_p.log_predictive(np.array(50.0))
    assert lp_near_p > lp_far_p
    print(f"  Poisson: near={lp_near_p:.4f}, far={lp_far_p:.4f} ok")


if __name__ == "__main__":
    print("=" * 60)
    print("BOCPD Smoke Tests")
    print("=" * 60)

    test_log_predictive_sanity()
    test_univariate_synthetic()
    test_multivariate_synthetic()
    test_multivariate_variance_shift()
    test_multiple_change_points()
    test_poisson_synthetic()
    test_no_change()

    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)
