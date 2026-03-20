"""
Tests for new observation models: log_predictive sanity, synthetic change point
detection, predictive_mean_var correctness, and analytical verification.
"""

import numpy as np
from scipy.stats import t as scipy_t

from bocpd import (
    BOCPD,
    BernoulliBeta,
    ConstantHazard,
    ExponentialGamma,
    GeometricBeta,
    MultinomialDirichlet,
    MultivariateNormalKnownCov,
    MultivariateNormalKnownMean,
    MultivariateStudentTNIW,
    NormalKnownMean,
    NormalKnownVariance,
    StudentTNIG,
    UnivariateNormalNIG,
    extract_change_points,
)

CP_TOLERANCE = 25


def _assert_cps_detected(result, true_cps, tolerance=CP_TOLERANCE):
    """Assert each true CP is detected within tolerance by at least one method."""
    for method in ["expected_run_length", "map_run_length", "posterior_mass"]:
        detected = extract_change_points(result, method=method)
        for tcp in true_cps:
            assert any(abs(int(d) - tcp) <= tolerance for d in detected), (
                f"[{method}] true CP {tcp} not detected "
                f"within +/-{tolerance}; got {detected}"
            )


# =============================================================================
# log_predictive sanity: near-prior observation scores higher than far
# =============================================================================


def test_log_predictive_bernoulli_beta():
    model = BernoulliBeta(alpha0=5.0, beta0=1.0)  # prior favors 1
    lp_near = model.log_predictive(np.array(1.0))
    lp_far = model.log_predictive(np.array(0.0))
    assert lp_near > lp_far


def test_log_predictive_exponential_gamma():
    model = ExponentialGamma(alpha0=5.0, beta0=1.0)  # E[rate]=5, E[x]=0.2
    lp_near = model.log_predictive(np.array(0.2))
    lp_far = model.log_predictive(np.array(10.0))
    assert lp_near > lp_far


def test_log_predictive_normal_known_variance():
    model = NormalKnownVariance(mu0=0.0, sigma0_sq=1.0, sigma2=1.0)
    lp_near = model.log_predictive(np.array(0.0))
    lp_far = model.log_predictive(np.array(10.0))
    assert lp_near > lp_far


def test_log_predictive_normal_known_mean():
    model = NormalKnownMean(mu_known=0.0, alpha0=3.0, beta0=1.0)
    lp_near = model.log_predictive(np.array(0.5))
    lp_far = model.log_predictive(np.array(20.0))
    assert lp_near > lp_far


def test_log_predictive_geometric_beta():
    model = GeometricBeta(alpha0=5.0, beta0=1.0)  # high p -> few failures
    lp_near = model.log_predictive(np.array(0))
    lp_far = model.log_predictive(np.array(50))
    assert lp_near > lp_far


def test_log_predictive_multinomial_dirichlet():
    model = MultinomialDirichlet(alpha0=[10.0, 1.0, 1.0])
    lp_near = model.log_predictive(np.array([1, 0, 0]))
    lp_far = model.log_predictive(np.array([0, 0, 1]))
    assert lp_near > lp_far


def test_log_predictive_mv_normal_known_cov():
    model = MultivariateNormalKnownCov(dim=3)
    lp_near = model.log_predictive(np.zeros(3))
    lp_far = model.log_predictive(np.ones(3) * 10)
    assert lp_near > lp_far


def test_log_predictive_mv_normal_known_mean():
    model = MultivariateNormalKnownMean(dim=3, nu0=5.0)
    lp_near = model.log_predictive(np.array([0.5, -0.5, 0.3]))
    lp_far = model.log_predictive(np.ones(3) * 20)
    assert lp_near > lp_far


# =============================================================================
# Analytical verification: BernoulliBeta and NormalKnownVariance
# =============================================================================


def test_bernoulli_beta_analytical():
    """Verify log_predictive matches closed-form Beta-Bernoulli."""
    model = BernoulliBeta(alpha0=3.0, beta0=2.0)
    # Predictive P(x=1) = alpha / (alpha + beta) = 3/5
    lp1 = model.log_predictive(np.array(1.0))
    np.testing.assert_allclose(lp1, np.log(3.0 / 5.0), atol=1e-12)
    lp0 = model.log_predictive(np.array(0.0))
    np.testing.assert_allclose(lp0, np.log(2.0 / 5.0), atol=1e-12)

    # After observing x=1, alpha becomes 4
    model.update(np.array(1.0))
    lp1 = model.log_predictive(np.array(1.0))
    np.testing.assert_allclose(lp1, np.log(4.0 / 6.0), atol=1e-12)


def test_normal_known_variance_analytical():
    """Verify log_predictive matches closed-form Normal predictive."""
    # Prior N(0, 1) on mu, observation variance sigma2=1
    # Predictive: N(0, 1 + 1) = N(0, 2)
    model = NormalKnownVariance(mu0=0.0, sigma0_sq=1.0, sigma2=1.0)
    x = 1.5
    expected = -0.5 * np.log(2 * np.pi * 2.0) - x**2 / (2 * 2.0)
    actual = model.log_predictive(np.array(x))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


# =============================================================================
# Synthetic change point detection
# =============================================================================


def test_bernoulli_beta_synthetic():
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.binomial(1, 0.2, 100),
            np.random.binomial(1, 0.8, 100),
        ]
    ).astype(float)
    det = BOCPD(
        model_factory=lambda: BernoulliBeta(alpha0=1.0, beta0=1.0),
        hazard_fn=ConstantHazard(lam=100),
    )
    result = det.run(data)
    _assert_cps_detected(result, [100])


def test_exponential_gamma_synthetic():
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.exponential(1.0, 100),
            np.random.exponential(0.2, 100),
        ]
    )
    det = BOCPD(
        model_factory=lambda: ExponentialGamma(alpha0=1.0, beta0=1.0),
        hazard_fn=ConstantHazard(lam=100),
    )
    result = det.run(data)
    _assert_cps_detected(result, [100])


def test_normal_known_variance_synthetic():
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.normal(0, 1, 100),
            np.random.normal(3, 1, 100),
        ]
    )
    det = BOCPD(
        model_factory=lambda: NormalKnownVariance(mu0=0.0, sigma0_sq=10.0, sigma2=1.0),
        hazard_fn=ConstantHazard(lam=100),
    )
    result = det.run(data)
    _assert_cps_detected(result, [100])


def test_normal_known_mean_synthetic():
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.normal(0, 1, 100),
            np.random.normal(0, 3, 100),
        ]
    )
    det = BOCPD(
        model_factory=lambda: NormalKnownMean(mu_known=0.0, alpha0=1.0, beta0=1.0),
        hazard_fn=ConstantHazard(lam=100),
    )
    result = det.run(data)
    _assert_cps_detected(result, [100])


def test_geometric_beta_synthetic():
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.geometric(0.3, 100) - 1,  # -1 for 0-indexed failures
            np.random.geometric(0.8, 100) - 1,
        ]
    ).astype(float)
    det = BOCPD(
        model_factory=lambda: GeometricBeta(alpha0=1.0, beta0=1.0),
        hazard_fn=ConstantHazard(lam=100),
    )
    result = det.run(data)
    _assert_cps_detected(result, [100])


def test_multinomial_dirichlet_synthetic():
    np.random.seed(42)
    K = 3
    probs1 = [0.7, 0.2, 0.1]
    probs2 = [0.1, 0.2, 0.7]
    data = np.vstack(
        [
            np.eye(K)[np.random.choice(K, size=100, p=probs1)],
            np.eye(K)[np.random.choice(K, size=100, p=probs2)],
        ]
    )
    det = BOCPD(
        model_factory=lambda: MultinomialDirichlet(alpha0=np.ones(K)),
        hazard_fn=ConstantHazard(lam=100),
    )
    result = det.run(data)
    _assert_cps_detected(result, [100])


def test_mv_normal_known_cov_synthetic():
    np.random.seed(42)
    dim = 3
    data = np.vstack(
        [
            np.random.multivariate_normal([0, 0, 0], np.eye(dim), 100),
            np.random.multivariate_normal([3, -2, 1], np.eye(dim), 100),
        ]
    )
    det = BOCPD(
        model_factory=lambda: MultivariateNormalKnownCov(
            dim=dim, Sigma0=10.0 * np.eye(dim), Sigma=np.eye(dim)
        ),
        hazard_fn=ConstantHazard(lam=100),
    )
    result = det.run(data)
    _assert_cps_detected(result, [100])


def test_mv_normal_known_mean_synthetic():
    np.random.seed(42)
    dim = 2
    data = np.vstack(
        [
            np.random.multivariate_normal([0, 0], 0.5 * np.eye(dim), 100),
            np.random.multivariate_normal([0, 0], 3.0 * np.eye(dim), 100),
        ]
    )
    det = BOCPD(
        model_factory=lambda: MultivariateNormalKnownMean(
            dim=dim, nu0=float(dim) + 1, Psi0=np.eye(dim)
        ),
        hazard_fn=ConstantHazard(lam=100),
    )
    result = det.run(data)
    _assert_cps_detected(result, [100])


# =============================================================================
# predictive_mean_var correctness
# =============================================================================


def test_predictive_mean_var_bernoulli_beta():
    model = BernoulliBeta(alpha0=1.0, beta0=1.0)
    m, v = model.predictive_mean_var()
    assert np.isfinite(m) and 0.0 <= m <= 1.0
    assert np.isfinite(v) and v > 0

    for x in [1.0, 0.0, 1.0]:
        model.update(np.array(x))
    m, v = model.predictive_mean_var()
    assert np.isfinite(m) and 0.0 <= m <= 1.0
    assert np.isfinite(v) and v > 0


def test_predictive_mean_var_exponential_gamma():
    model = ExponentialGamma(alpha0=1.0, beta0=1.0)
    m, v = model.predictive_mean_var()
    assert m == np.inf  # alpha=1, need alpha>1

    for x in [0.5, 1.0, 0.3]:
        model.update(np.array(x))
    m, v = model.predictive_mean_var()
    assert np.isfinite(m) and m > 0
    assert np.isfinite(v) and v > 0


def test_predictive_mean_var_normal_known_variance():
    model = NormalKnownVariance(mu0=0.0, sigma0_sq=1.0, sigma2=1.0)
    m, v = model.predictive_mean_var()
    assert np.isfinite(m)
    assert np.isfinite(v) and v > 0
    # Predictive var = sigma2 + 1/tau = 1 + 1 = 2
    np.testing.assert_allclose(v, 2.0, atol=1e-12)

    model.update(np.array(2.0))
    m, v = model.predictive_mean_var()
    assert np.isfinite(m) and m > 0  # shifted toward 2
    assert v < 2.0  # tighter after one observation


def test_predictive_mean_var_normal_known_mean():
    model = NormalKnownMean(mu_known=0.0, alpha0=1.0, beta0=1.0)
    m, v = model.predictive_mean_var()
    assert m == 0.0
    assert v == np.inf  # alpha=1, df=2 -> inf var

    for x in [1.0, -0.5, 0.3]:
        model.update(np.array(x))
    m, v = model.predictive_mean_var()
    assert m == 0.0
    assert np.isfinite(v) and v > 0


def test_predictive_mean_var_geometric_beta():
    model = GeometricBeta(alpha0=1.0, beta0=1.0)
    m, v = model.predictive_mean_var()
    assert m == np.inf  # alpha=1, need alpha>1

    for x in [2, 0, 1, 3, 0]:
        model.update(np.array(x))
    m, v = model.predictive_mean_var()
    assert np.isfinite(m) and m >= 0
    assert np.isfinite(v) and v > 0


def test_predictive_mean_var_multinomial_dirichlet():
    model = MultinomialDirichlet(alpha0=[1.0, 1.0, 1.0])
    m, v = model.predictive_mean_var()
    assert m.shape == (3,)
    assert v.shape == (3, 3)
    np.testing.assert_allclose(m, [1 / 3, 1 / 3, 1 / 3], atol=1e-12)
    assert np.all(np.isfinite(v))


def test_predictive_mean_var_mv_normal_known_cov():
    dim = 3
    model = MultivariateNormalKnownCov(dim=dim)
    m, v = model.predictive_mean_var()
    assert m.shape == (dim,)
    assert v.shape == (dim, dim)
    assert np.all(np.isfinite(m))
    assert np.all(np.isfinite(v))
    # Predictive cov = Sigma + inv(Lambda) = I + I = 2I
    np.testing.assert_allclose(v, 2.0 * np.eye(dim), atol=1e-12)


def test_predictive_mean_var_mv_normal_known_mean():
    dim = 3
    model = MultivariateNormalKnownMean(dim=dim, nu0=float(dim) + 1)
    m, v = model.predictive_mean_var()
    assert m.shape == (dim,)
    assert v.shape == (dim, dim)
    # nu0=4, D=3, df=2 -> inf variance
    assert np.all(np.isinf(v))

    model2 = MultivariateNormalKnownMean(dim=dim, nu0=float(dim) + 3)
    for _ in range(5):
        model2.update(np.random.randn(dim))
    m2, v2 = model2.predictive_mean_var()
    assert np.all(np.isfinite(m2))
    assert np.all(np.isfinite(v2))


# =============================================================================
# StudentTNIG tests
# =============================================================================


def test_log_predictive_student_t_nig():
    """Near-prior observation scores higher than far."""
    model = StudentTNIG(nu=4.0, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    lp_near = model.log_predictive(np.array(0.0))
    lp_far = model.log_predictive(np.array(10.0))
    assert lp_near > lp_far


def test_student_t_nig_analytical():
    """Verify log_predictive matches scipy.stats.t.logpdf at prior."""
    nu = 5.0
    mu0 = 2.0
    kappa0 = 3.0
    alpha0 = 4.0
    beta0 = 2.0
    model = StudentTNIG(nu=nu, mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0)

    scale = beta0 * (kappa0 + 1.0) / (alpha0 * kappa0)
    for x in [-3.0, 0.0, 2.0, 5.0, 20.0]:
        expected = scipy_t.logpdf(x, df=nu, loc=mu0, scale=np.sqrt(scale))
        actual = model.log_predictive(np.array(x))
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_student_t_nig_analytical_after_updates():
    """Verify log_predictive matches scipy after several updates."""
    nu = 4.0
    model = StudentTNIG(nu=nu, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)

    np.random.seed(123)
    for _ in range(10):
        model.update(np.array(np.random.normal(1.0, 0.5)))

    scale = model.beta * (model.kappa + 1.0) / (model.alpha * model.kappa)
    for x in [-2.0, 0.0, 1.0, 5.0]:
        expected = scipy_t.logpdf(x, df=nu, loc=model.mu, scale=np.sqrt(scale))
        actual = model.log_predictive(np.array(x))
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_student_t_nig_outlier_downweighting():
    """Outlier moves StudentTNIG location less than UnivariateNormalNIG."""
    np.random.seed(42)
    inliers = np.random.normal(0, 1, 20)

    # Build up both models on the same inliers
    nig = UnivariateNormalNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    t_nig = StudentTNIG(nu=4.0, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)

    for x in inliers:
        nig.update(np.array(x))
        t_nig.update(np.array(x))

    mu_nig_before = nig.mu
    mu_t_before = t_nig.mu

    # Hit both with a massive outlier
    outlier = np.array(100.0)
    nig.update(outlier)
    t_nig.update(outlier)

    shift_nig = abs(nig.mu - mu_nig_before)
    shift_t = abs(t_nig.mu - mu_t_before)

    # StudentTNIG should move LESS
    assert shift_t < shift_nig, (
        f"StudentTNIG shifted {shift_t:.4f} vs NIG {shift_nig:.4f} — "
        "expected StudentTNIG to be more robust"
    )


def test_student_t_nig_persistent_heavy_tails():
    """After many observations, StudentTNIG keeps heavy tails while NIG thins."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 200)

    nig = UnivariateNormalNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    t_nig = StudentTNIG(nu=4.0, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)

    for x in data:
        nig.update(np.array(x))
        t_nig.update(np.array(x))

    _, var_nig = nig.predictive_mean_var()
    _, var_t = t_nig.predictive_mean_var()

    # NIG df = 2*alpha = 2*(1 + 200*0.5) = 202, almost Gaussian
    # StudentTNIG df = 4, var = scale * 4/2 = 2*scale — much heavier
    # The Student-t variance with df=4 should be meaningfully larger
    assert var_t > var_nig, (
        f"StudentTNIG var={var_t:.4f} should exceed NIG var={var_nig:.4f} "
        "due to persistent heavy tails"
    )


def test_student_t_nig_synthetic_cp():
    """Detect change point in Student-t data with mean shift."""
    np.random.seed(42)
    # Student-t(nu=4) data with mean shift at t=100
    from scipy.stats import t as scipy_t_dist

    data = np.concatenate(
        [
            scipy_t_dist.rvs(df=4, loc=0, scale=1, size=100),
            scipy_t_dist.rvs(df=4, loc=4, scale=1, size=100),
        ]
    )
    det = BOCPD(
        model_factory=lambda: StudentTNIG(
            nu=4.0, mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0
        ),
        hazard_fn=ConstantHazard(lam=100),
    )
    result = det.run(data)
    _assert_cps_detected(result, [100])


def test_predictive_mean_var_student_t_nig():
    model = StudentTNIG(nu=4.0, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    m, v = model.predictive_mean_var()
    assert m == 0.0
    assert np.isfinite(v) and v > 0  # nu=4 > 2 so var is finite

    # nu <= 2 -> inf variance
    model2 = StudentTNIG(nu=2.0, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    _, v2 = model2.predictive_mean_var()
    assert v2 == np.inf


# =============================================================================
# MultivariateStudentTNIW tests
# =============================================================================


def test_log_predictive_mv_student_t_niw():
    """Near-prior observation scores higher than far."""
    model = MultivariateStudentTNIW(dim=3, nu=4.0)
    lp_near = model.log_predictive(np.zeros(3))
    lp_far = model.log_predictive(np.ones(3) * 10)
    assert lp_near > lp_far


def test_mv_student_t_niw_synthetic_cp():
    """Detect change point in multivariate Student-t data."""
    np.random.seed(42)
    dim = 2

    # Generate MV Student-t by scale mixture: x = mu + z/sqrt(w), w ~ Gamma(nu/2, nu/2)
    def mv_student_t(mu, cov, nu, size):
        D = len(mu)
        w = np.random.gamma(nu / 2.0, 2.0 / nu, size=size)
        z = np.random.multivariate_normal(np.zeros(D), cov, size=size)
        return mu + z / np.sqrt(w)[:, None]

    data = np.vstack(
        [
            mv_student_t(np.zeros(dim), np.eye(dim), 4.0, 100),
            mv_student_t(np.array([5.0, -4.0]), np.eye(dim), 4.0, 100),
        ]
    )
    det = BOCPD(
        model_factory=lambda: MultivariateStudentTNIW(
            dim=dim, nu=4.0, kappa0=0.1, nu0=float(dim) + 1, Psi0=np.eye(dim)
        ),
        hazard_fn=ConstantHazard(lam=100),
    )
    result = det.run(data)
    # Heavy-tailed data makes posterior_mass noisy; check ERL and MAP
    for method in ["expected_run_length", "map_run_length"]:
        detected = extract_change_points(result, method=method)
        assert any(abs(int(d) - 100) <= CP_TOLERANCE for d in detected), (
            f"[{method}] true CP 100 not detected within +/-{CP_TOLERANCE}; "
            f"got {detected}"
        )


def test_predictive_mean_var_mv_student_t_niw():
    dim = 3
    model = MultivariateStudentTNIW(dim=dim, nu=4.0)
    m, v = model.predictive_mean_var()
    assert m.shape == (dim,)
    assert v.shape == (dim, dim)
    assert np.all(np.isfinite(m))
    assert np.all(np.isfinite(v))  # nu=4 > 2

    # nu <= 2 -> inf
    model2 = MultivariateStudentTNIW(dim=dim, nu=2.0)
    _, v2 = model2.predictive_mean_var()
    assert np.all(np.isinf(v2))
