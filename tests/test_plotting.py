"""Tests for bocpd.plotting utilities."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from bocpd.plotting import (
    COLORS,
    build_rl_matrix,
    draw_change_points,
    format_xaxis,
    mark_events,
    plot_erl,
    plot_predictive_envelope,
    plot_price,
    plot_run_length_heatmap,
)

# ---------------------------------------------------------------------------
# build_rl_matrix
# ---------------------------------------------------------------------------


def _make_posteriors(T=50, max_rl=30):
    """Create synthetic posteriors that grow in length like real BOCPD output."""
    posteriors = []
    for t in range(T):
        length = min(t + 2, max_rl)
        p = np.random.dirichlet(np.ones(length))
        posteriors.append(p)
    return posteriors


def test_build_rl_matrix_shape():
    posteriors = _make_posteriors(T=50, max_rl=30)
    mat = build_rl_matrix(posteriors)
    assert mat.shape == (30, 50)
    assert mat.min() >= 1e-6
    assert mat.max() <= 1.0


def test_build_rl_matrix_custom_clip():
    posteriors = _make_posteriors(T=20, max_rl=10)
    mat = build_rl_matrix(posteriors, clip_lo=1e-4, clip_hi=0.5)
    assert mat.min() >= 1e-4
    assert mat.max() <= 0.5


# ---------------------------------------------------------------------------
# Panel functions — smoke tests (no exceptions)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """Create minimal synthetic data for panel function tests."""
    T = 100
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    prices = np.cumsum(np.random.randn(T)) + 100
    observations = np.random.randn(T) * 0.01
    pred_mean = np.zeros(T)
    pred_var = np.ones(T) * 0.001
    erl = np.random.rand(T) * 100
    posteriors = _make_posteriors(T=T, max_rl=50)
    boundaries = [
        {"index": 25, "lower": 22, "upper": 28, "severity": 0.8},
        {"index": 60, "lower": 57, "upper": 63, "severity": 0.6},
    ]
    events = {"Event A": "2020-02-15", "Event B": "2020-04-01"}
    return {
        "T": T,
        "dates": dates,
        "prices": prices,
        "observations": observations,
        "pred_mean": pred_mean,
        "pred_var": pred_var,
        "erl": erl,
        "posteriors": posteriors,
        "boundaries": boundaries,
        "events": events,
    }


def test_plot_price(synthetic_data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_price(ax, synthetic_data["prices"], synthetic_data["dates"])
    plt.close(fig)


def test_plot_erl(synthetic_data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_erl(ax, synthetic_data["erl"], synthetic_data["dates"])
    plt.close(fig)


def test_plot_predictive_envelope(synthetic_data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_predictive_envelope(
        ax,
        synthetic_data["dates"],
        synthetic_data["observations"],
        synthetic_data["pred_mean"],
        synthetic_data["pred_var"],
    )
    plt.close(fig)


def test_plot_run_length_heatmap(synthetic_data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = plot_run_length_heatmap(
        ax, synthetic_data["posteriors"], synthetic_data["dates"]
    )
    assert im is not None
    plt.close(fig)


def test_plot_run_length_heatmap_no_colorbar(synthetic_data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_run_length_heatmap(
        ax, synthetic_data["posteriors"], synthetic_data["dates"], colorbar=False
    )
    plt.close(fig)


def test_draw_change_points(synthetic_data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_change_points(ax, synthetic_data["boundaries"], synthetic_data["dates"])
    plt.close(fig)


def test_draw_change_points_no_ci(synthetic_data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_change_points(
        ax, synthetic_data["boundaries"], synthetic_data["dates"], draw_ci=False
    )
    plt.close(fig)


def test_mark_events(synthetic_data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    mark_events(ax, synthetic_data["events"], synthetic_data["dates"])
    plt.close(fig)


def test_format_xaxis(synthetic_data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(synthetic_data["dates"], synthetic_data["erl"])
    format_xaxis(ax)
    format_xaxis(ax, interval=6)
    plt.close(fig)


def test_colors_namespace():
    assert COLORS.cp == "#7F77DD"
    assert COLORS.erl == "#534AB7"
    assert COLORS.pred == "#1D9E75"
