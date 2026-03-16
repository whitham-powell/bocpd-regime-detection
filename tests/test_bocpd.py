"""
Smoke tests for BOCPD implementation.
Tests on synthetic data where we know ground truth change points.
"""

import numpy as np

from bocpd import (
    BOCPD,
    ConstantHazard,
    MultivariateNormalNIW,
    PoissonGamma,
    UnivariateNormalNIG,
    extract_change_points,
)

CP_TOLERANCE = 20


def _run_bocpd(data, model_factory, hazard_fn):
    """Run BOCPD and return result dict."""
    detector = BOCPD(model_factory=model_factory, hazard_fn=hazard_fn)
    return detector.run(data)


def _assert_cps_detected(result, true_cps, tolerance=CP_TOLERANCE):
    """Assert each true CP is detected within tolerance by at least one method."""
    for method in ["expected_run_length", "map_run_length", "posterior_mass"]:
        detected = extract_change_points(result, method=method)
        for tcp in true_cps:
            assert any(abs(int(d) - tcp) <= tolerance for d in detected), (
                f"[{method}] true CP {tcp} not detected "
                f"within +/-{tolerance}; got {detected}"
            )


def test_log_predictive_sanity():
    model = UnivariateNormalNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    lp_near = model.log_predictive(np.array(0.0))
    lp_far = model.log_predictive(np.array(10.0))
    assert lp_near > lp_far

    model_mv = MultivariateNormalNIW(dim=3, kappa0=1.0, nu0=4.0, Psi0=np.eye(3))
    lp_near_mv = model_mv.log_predictive(np.zeros(3))
    lp_far_mv = model_mv.log_predictive(np.ones(3) * 10)
    assert lp_near_mv > lp_far_mv

    model_p = PoissonGamma(alpha0=4.0, beta0=1.0)
    lp_near_p = model_p.log_predictive(np.array(4.0))
    lp_far_p = model_p.log_predictive(np.array(50.0))
    assert lp_near_p > lp_far_p


def test_univariate_synthetic():
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.normal(0, 1, 100),
            np.random.normal(3, 1, 100),
        ]
    )
    result = _run_bocpd(
        data,
        model_factory=lambda: UnivariateNormalNIG(
            mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0
        ),
        hazard_fn=ConstantHazard(lam=100),
    )
    _assert_cps_detected(result, [100])


def test_multivariate_synthetic():
    np.random.seed(42)
    dim = 3
    data = np.vstack(
        [
            np.random.multivariate_normal([0, 0, 0], np.eye(dim), 100),
            np.random.multivariate_normal([2, -1, 1], np.eye(dim), 100),
        ]
    )
    result = _run_bocpd(
        data,
        model_factory=lambda: MultivariateNormalNIW(
            dim=dim, kappa0=0.1, nu0=float(dim), Psi0=np.eye(dim)
        ),
        hazard_fn=ConstantHazard(lam=100),
    )
    _assert_cps_detected(result, [100])


def test_multivariate_variance_shift():
    np.random.seed(42)
    dim = 2
    data = np.vstack(
        [
            np.random.multivariate_normal([0, 0], 0.5 * np.eye(dim), 150),
            np.random.multivariate_normal([0, 0], 3.0 * np.eye(dim), 150),
        ]
    )
    result = _run_bocpd(
        data,
        model_factory=lambda: MultivariateNormalNIW(
            dim=dim, kappa0=0.1, nu0=float(dim) + 1, Psi0=np.eye(dim)
        ),
        hazard_fn=ConstantHazard(lam=150),
    )
    _assert_cps_detected(result, [150])


def test_multiple_change_points():
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.normal(0, 1, 80),
            np.random.normal(3, 0.5, 80),
            np.random.normal(-1, 2, 80),
        ]
    )
    result = _run_bocpd(
        data,
        model_factory=lambda: UnivariateNormalNIG(
            mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0
        ),
        hazard_fn=ConstantHazard(lam=80),
    )
    _assert_cps_detected(result, [80, 160])


def test_poisson_synthetic():
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.poisson(3, 100),
            np.random.poisson(10, 100),
        ]
    ).astype(float)
    result = _run_bocpd(
        data,
        model_factory=lambda: PoissonGamma(alpha0=1.0, beta0=0.25),
        hazard_fn=ConstantHazard(lam=100),
    )
    _assert_cps_detected(result, [100])


def test_no_change():
    np.random.seed(42)
    data = np.random.normal(0, 1, 200)
    result = _run_bocpd(
        data,
        model_factory=lambda: UnivariateNormalNIG(
            mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0
        ),
        hazard_fn=ConstantHazard(lam=200),
    )
    detected = extract_change_points(result, method="expected_run_length")
    assert len(detected) == 0, f"Expected no CPs, got {detected}"
