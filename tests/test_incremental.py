"""
Tests for incremental BOCPD: step(), warm_up(), save_state(), load_state().

Verifies:
1. step()-by-step produces identical results to run()
2. warm_up() produces identical results to run()
3. Checkpoint round-trip: save_state() + load_state() + step() matches
   running the full series
4. All model types (NIG, NIW sequential, NIW vectorized, PoissonGamma)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bocpd import (
    BOCPD,
    ConstantHazard,
    MultivariateNormalNIW,
    PoissonGamma,
    UnivariateNormalNIG,
)

# =============================================================================
# Fixtures: synthetic data + model configs
# =============================================================================


def _nig_config():
    return {
        "model_factory": lambda: UnivariateNormalNIG(
            mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0
        ),
        "hazard_fn": ConstantHazard(lam=100),
        "r_max": None,
    }


def _niw_sequential_config():
    dim = 3
    return {
        "model_factory": lambda: MultivariateNormalNIW(
            dim=dim, kappa0=0.1, nu0=float(dim) + 1, Psi0=np.eye(dim)
        ),
        "hazard_fn": ConstantHazard(lam=80),
        "r_max": None,  # no r_max -> sequential path
    }


def _niw_vectorized_config():
    dim = 3
    return {
        "model_factory": lambda: MultivariateNormalNIW(
            dim=dim, kappa0=0.1, nu0=float(dim) + 1, Psi0=np.eye(dim)
        ),
        "hazard_fn": ConstantHazard(lam=80),
        "r_max": 100,  # r_max + NIW -> vectorized path
    }


def _poisson_config():
    return {
        "model_factory": lambda: PoissonGamma(alpha0=1.0, beta0=0.25),
        "hazard_fn": ConstantHazard(lam=100),
        "r_max": None,
    }


def _univariate_data():
    np.random.seed(42)
    return np.concatenate(
        [
            np.random.normal(0, 1, 100),
            np.random.normal(3, 1, 100),
        ]
    )


def _multivariate_data():
    np.random.seed(42)
    dim = 3
    return np.vstack(
        [
            np.random.multivariate_normal([0, 0, 0], np.eye(dim), 80),
            np.random.multivariate_normal([3, -2, 1], 0.5 * np.eye(dim), 70),
        ]
    )


def _poisson_data():
    np.random.seed(42)
    return np.concatenate(
        [
            np.random.poisson(3, 100),
            np.random.poisson(10, 100),
        ]
    ).astype(float)


# =============================================================================
# Test: step-by-step equivalence with run()
# =============================================================================


@pytest.mark.parametrize(
    "config_fn, data_fn",
    [
        (_nig_config, _univariate_data),
        (_niw_sequential_config, _multivariate_data),
        (_niw_vectorized_config, _multivariate_data),
        (_poisson_config, _poisson_data),
    ],
    ids=["NIG", "NIW-sequential", "NIW-vectorized", "PoissonGamma"],
)
def test_step_matches_run(config_fn, data_fn):
    """Calling step() in a loop must produce identical results to run()."""
    config = config_fn()
    data = data_fn()
    T = len(data)

    # run() path
    det_run = BOCPD(**config)
    result_run = det_run.run(data)

    # step() path
    det_step = BOCPD(**config)
    step_cp = np.zeros(T)
    step_map = np.zeros(T, dtype=int)
    step_erl = np.zeros(T)
    step_pm = np.full(T, np.nan)
    step_pv = np.full(T, np.nan)
    step_posteriors = []

    for t in range(T):
        summary = det_step.step(data[t])
        step_cp[t] = summary["change_point_prob"]
        step_map[t] = summary["map_run_length"]
        step_erl[t] = summary["expected_run_length"]
        step_pm[t] = summary["predictive_mean"]
        step_pv[t] = summary["predictive_var"]
        step_posteriors.append(det_step._joint.copy())

    np.testing.assert_allclose(
        step_cp,
        result_run["change_point_prob"],
        atol=1e-12,
        err_msg="change_point_prob mismatch",
    )
    np.testing.assert_array_equal(
        step_map, result_run["map_run_length"], err_msg="map_run_length mismatch"
    )
    np.testing.assert_allclose(
        step_erl,
        result_run["expected_run_length"],
        atol=1e-12,
        err_msg="expected_run_length mismatch",
    )

    # Predictive: compare only finite values
    run_pm = result_run["predictive_mean"]
    run_pv = result_run["predictive_var"]
    finite_mask = np.isfinite(run_pm)
    if np.any(finite_mask):
        np.testing.assert_allclose(
            step_pm[finite_mask],
            run_pm[finite_mask],
            atol=1e-12,
            err_msg="predictive_mean mismatch",
        )
        np.testing.assert_allclose(
            step_pv[finite_mask],
            run_pv[finite_mask],
            atol=1e-12,
            err_msg="predictive_var mismatch",
        )

    # Posteriors
    for t in range(T):
        np.testing.assert_allclose(
            step_posteriors[t],
            result_run["run_length_posterior"][t],
            atol=1e-12,
            err_msg=f"run_length_posterior mismatch at t={t}",
        )


# =============================================================================
# Test: warm_up() equivalence with run()
# =============================================================================


@pytest.mark.parametrize(
    "config_fn, data_fn",
    [
        (_nig_config, _univariate_data),
        (_niw_vectorized_config, _multivariate_data),
        (_poisson_config, _poisson_data),
    ],
    ids=["NIG", "NIW-vectorized", "PoissonGamma"],
)
def test_warm_up_matches_run(config_fn, data_fn):
    """warm_up() must produce identical results to run()."""
    config = config_fn()
    data = data_fn()

    det_run = BOCPD(**config)
    result_run = det_run.run(data)

    det_warm = BOCPD(**config)
    result_warm = det_warm.warm_up(data)

    np.testing.assert_allclose(
        result_warm["change_point_prob"],
        result_run["change_point_prob"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result_warm["expected_run_length"],
        result_run["expected_run_length"],
        atol=1e-12,
    )


# =============================================================================
# Test: checkpoint round-trip
# =============================================================================


@pytest.mark.parametrize(
    "config_fn, data_fn",
    [
        (_nig_config, _univariate_data),
        (_niw_sequential_config, _multivariate_data),
        (_niw_vectorized_config, _multivariate_data),
        (_poisson_config, _poisson_data),
    ],
    ids=["NIG", "NIW-sequential", "NIW-vectorized", "PoissonGamma"],
)
def test_checkpoint_roundtrip(config_fn, data_fn):
    """save_state + load_state + step must match running the full series."""
    config = config_fn()
    data = data_fn()
    T = len(data)
    split = T // 2

    # Full run for ground truth
    det_full = BOCPD(**config)
    det_full.run(data)

    # Warm up on first half, save, load, continue on second half
    det_first = BOCPD(**config)
    det_first.warm_up(data[:split])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmppath = f.name
    det_first.save_state(tmppath)

    det_loaded = BOCPD.load_state(tmppath)
    Path(tmppath).unlink()

    # Continue stepping through the second half
    for t in range(split, T):
        det_loaded.step(data[t])

    # Compare the final posterior
    np.testing.assert_allclose(
        det_loaded._joint,
        det_full._joint,
        atol=1e-10,
        err_msg="Final joint posterior mismatch after checkpoint round-trip",
    )
    assert det_loaded._t == det_full._t


@pytest.mark.parametrize(
    "config_fn, data_fn",
    [
        (_nig_config, _univariate_data),
        (_niw_vectorized_config, _multivariate_data),
        (_poisson_config, _poisson_data),
    ],
    ids=["NIG", "NIW-vectorized", "PoissonGamma"],
)
def test_checkpoint_step_by_step_equivalence(config_fn, data_fn):
    """After checkpoint, step-by-step results must match full run exactly."""
    config = config_fn()
    data = data_fn()
    T = len(data)
    split = T // 2

    # Full run
    det_full = BOCPD(**config)
    result_full = det_full.run(data)

    # Split run with checkpoint
    det_a = BOCPD(**config)
    det_a.warm_up(data[:split])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmppath = f.name
    det_a.save_state(tmppath)
    det_b = BOCPD.load_state(tmppath)
    Path(tmppath).unlink()

    for t in range(split, T):
        summary_b = det_b.step(data[t])
        np.testing.assert_allclose(
            summary_b["change_point_prob"],
            result_full["change_point_prob"][t],
            atol=1e-10,
            err_msg=f"change_point_prob mismatch at t={t}",
        )
        np.testing.assert_allclose(
            summary_b["expected_run_length"],
            result_full["expected_run_length"][t],
            atol=1e-10,
            err_msg=f"expected_run_length mismatch at t={t}",
        )


# =============================================================================
# Test: save_state raises before initialization
# =============================================================================


def test_save_state_before_init_raises():
    det = BOCPD(**_nig_config())
    with pytest.raises(RuntimeError, match="No state to save"):
        det.save_state("/tmp/should_not_exist.json")


# =============================================================================
# Test: state file is valid JSON with expected structure
# =============================================================================


def test_state_file_is_valid_json():
    config = _nig_config()
    data = _univariate_data()
    det = BOCPD(**config)
    det.warm_up(data[:50])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmppath = f.name
    det.save_state(tmppath)

    state = json.loads(Path(tmppath).read_text())
    Path(tmppath).unlink()

    assert "t" in state
    assert "joint" in state
    assert "hazard" in state
    assert state["t"] == 50
    assert state["hazard"]["type"] == "ConstantHazard"


# =============================================================================
# Test: existing tests still pass (regression guard)
# =============================================================================


def test_run_still_works():
    """Ensure run() still produces correct results after refactor."""
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.normal(0, 1, 100),
            np.random.normal(3, 1, 100),
        ]
    )
    det = BOCPD(
        model_factory=lambda: UnivariateNormalNIG(
            mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0
        ),
        hazard_fn=ConstantHazard(lam=100),
    )
    result = det.run(data)
    assert result["change_point_prob"].shape == (200,)
    assert len(result["run_length_posterior"]) == 200
    # Should detect the change point near t=100
    from bocpd import extract_change_points

    cps = extract_change_points(result, method="expected_run_length")
    assert any(abs(int(cp) - 100) <= 20 for cp in cps)
