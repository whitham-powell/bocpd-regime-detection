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


# =============================================================================
# Tests for predictive_mean_var on each observation model
# =============================================================================


def test_predictive_mean_var_nig():
    """NIG predictive_mean_var: finite after updates, inf variance with alpha0=1."""
    # Default alpha0=1.0 gives df=2, so variance = scale * 2 / (2-2) = inf
    model = UnivariateNormalNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    m, v = model.predictive_mean_var()
    assert np.isfinite(m), "Mean should be finite even before updates"
    assert v == np.inf, "Variance should be inf when alpha=1 (df=2)"

    # After a few updates, alpha grows and variance becomes finite
    for x in [1.0, 2.0, 3.0]:
        model.update(np.array(x))
    m, v = model.predictive_mean_var()
    assert np.isfinite(m), "Mean should be finite after updates"
    assert np.isfinite(v) and v > 0, (
        "Variance should be finite and positive after updates"
    )


def test_predictive_mean_var_poisson():
    """PoissonGamma predictive_mean_var: always finite with default priors."""
    model = PoissonGamma(alpha0=1.0, beta0=1.0)
    m, v = model.predictive_mean_var()
    assert np.isfinite(m) and m > 0, "Mean should be finite and positive"
    assert np.isfinite(v) and v > 0, "Variance should be finite and positive"

    # After updates
    for x in [3, 5, 2]:
        model.update(np.array(x))
    m, v = model.predictive_mean_var()
    assert np.isfinite(m) and m > 0
    assert np.isfinite(v) and v > 0


def test_predictive_mean_var_niw():
    """NIW predictive_mean_var: returns arrays with correct shapes."""
    dim = 3
    model = MultivariateNormalNIW(dim=dim, kappa0=1.0, nu0=float(dim), Psi0=np.eye(dim))
    m, v = model.predictive_mean_var()
    assert isinstance(m, np.ndarray) and m.shape == (dim,)
    assert isinstance(v, np.ndarray) and v.shape == (dim, dim)

    # With nu0=dim, df=1, so variance should be inf
    assert np.all(np.isinf(v)), "Variance should be inf when df <= 2"

    # With higher nu0, variance becomes finite after updates
    model2 = MultivariateNormalNIW(
        dim=dim, kappa0=1.0, nu0=float(dim) + 3, Psi0=np.eye(dim)
    )
    for _ in range(5):
        model2.update(np.random.randn(dim))
    m2, v2 = model2.predictive_mean_var()
    assert np.all(np.isfinite(m2))
    assert np.all(np.isfinite(v2))


# =============================================================================
# Tests for BOCPD predictive output fields
# =============================================================================


def test_predictive_output_univariate():
    """BOCPD with NIG produces finite predictive_mean and predictive_var."""
    np.random.seed(42)
    data = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(3, 1, 100)])
    result = _run_bocpd(
        data,
        model_factory=lambda: UnivariateNormalNIG(
            mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0
        ),
        hazard_fn=ConstantHazard(lam=100),
    )
    pm = result["predictive_mean"]
    pv = result["predictive_var"]

    assert pm.shape == (200,)
    assert pv.shape == (200,)

    # At most 1 NaN (t=0 when only the fresh prior exists with inf variance)
    assert np.sum(np.isnan(pm)) <= 1, (
        f"Too many NaN in predictive_mean: {np.sum(np.isnan(pm))}"
    )
    assert np.sum(np.isnan(pv)) <= 1, (
        f"Too many NaN in predictive_var: {np.sum(np.isnan(pv))}"
    )

    # No inf values
    assert not np.any(np.isinf(pm)), "predictive_mean contains inf"
    assert not np.any(np.isinf(pv)), "predictive_var contains inf"

    # Variance is non-negative where finite
    finite_v = pv[np.isfinite(pv)]
    assert np.all(finite_v >= 0), "predictive_var has negative values"

    # Predictive mean is roughly within data range
    finite_m = pm[np.isfinite(pm)]
    assert np.min(finite_m) > -10 and np.max(finite_m) < 15, (
        f"predictive_mean out of expected range: [{np.min(finite_m)}, {np.max(finite_m)}]"
    )


def test_predictive_output_poisson():
    """BOCPD with PoissonGamma produces finite predictive output from the start."""
    np.random.seed(42)
    data = np.concatenate(
        [np.random.poisson(3, 100), np.random.poisson(10, 100)]
    ).astype(float)
    result = _run_bocpd(
        data,
        model_factory=lambda: PoissonGamma(alpha0=1.0, beta0=0.25),
        hazard_fn=ConstantHazard(lam=100),
    )
    pm = result["predictive_mean"]
    pv = result["predictive_var"]

    # PoissonGamma has finite variance from the start (NegBin with alpha>0, beta>0)
    assert np.all(np.isfinite(pm)), "predictive_mean has non-finite values"
    assert np.all(np.isfinite(pv)), "predictive_var has non-finite values"
    assert np.all(pv >= 0), "predictive_var has negative values"


def test_predictive_output_multivariate():
    """BOCPD with NIW produces all-NaN predictive output (scalar storage can't hold it)."""
    np.random.seed(42)
    dim = 3
    data = np.vstack(
        [
            np.random.multivariate_normal([0, 0, 0], np.eye(dim), 50),
            np.random.multivariate_normal([2, -1, 1], np.eye(dim), 50),
        ]
    )
    result = _run_bocpd(
        data,
        model_factory=lambda: MultivariateNormalNIW(
            dim=dim, kappa0=0.1, nu0=float(dim), Psi0=np.eye(dim)
        ),
        hazard_fn=ConstantHazard(lam=50),
    )
    pm = result["predictive_mean"]
    pv = result["predictive_var"]

    # Multivariate models are skipped — all NaN is expected
    assert np.all(np.isnan(pm)), "Expected all NaN for multivariate predictive_mean"
    assert np.all(np.isnan(pv)), "Expected all NaN for multivariate predictive_var"


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
