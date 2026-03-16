"""
Bayesian Online Change Point Detection (BOCPD).

Implementation of Adams & MacKay (2007), generalized to accept any
observation model and hazard function.

The algorithm maintains a posterior distribution over run lengths
(time since last change point) at each time step. It is:
    - Online: processes one observation at a time
    - Exact: no approximation (for conjugate observation models)
    - Modular: observation model and hazard function are pluggable

References
----------
Adams, R. P., & MacKay, D. J. C. (2007).
    Bayesian Online Changepoint Detection. arXiv:0710.3742.
"""

from __future__ import annotations

import numpy as np

from .hazard import ConstantHazard


class BOCPD:
    """Bayesian Online Change Point Detection.

    Parameters
    ----------
    model_factory : callable
        A callable that returns a fresh ObservationModel instance
        initialized from the prior. Called once per hypothesized run
        length. Example: lambda: MultivariateNormalNIW(dim=5)

    hazard_fn : callable, optional
        Hazard function H(r) -> probability of change point at run
        length r. Default: ConstantHazard(lam=200).

    r_max : int or None, optional
        Maximum run length to track. If None, the run length vector
        grows with time (exact). If set, truncate and renormalize
        to cap memory/compute. Recommended for long time series.

    Attributes
    ----------
    run_length_posterior : list of np.ndarray
        run_length_posterior[t] is the posterior P(r_t | x_{1:t}).
        Stored for all time steps for later analysis/visualization.

    change_point_prob : np.ndarray
        change_point_prob[t] = P(r_t = 0 | x_{1:t}), the posterior
        probability of a change point at time t.
    """

    def __init__(
        self,
        model_factory: callable,
        hazard_fn: callable | None = None,
        r_max: int | None = None,
    ):
        self.model_factory = model_factory
        self.hazard_fn = hazard_fn or ConstantHazard(lam=200)
        self.r_max = r_max

    def run(self, data: np.ndarray) -> dict:
        """Run BOCPD on a sequence of observations.

        Parameters
        ----------
        data : np.ndarray
            Observations. Shape (T,) for univariate, (T, D) for
            multivariate.

        Returns
        -------
        dict with keys:
            'run_length_posterior' : list of np.ndarray
                Full posterior over run lengths at each time step.
            'change_point_prob' : np.ndarray, shape (T,)
                P(r_t = 0 | x_{1:t}) at each time step.
            'map_run_length' : np.ndarray, shape (T,)
                Most probable run length at each time step.
            'expected_run_length' : np.ndarray, shape (T,)
                E[r_t | x_{1:t}] at each time step.
            'predictive_mean' : np.ndarray, shape (T,)
                One-step-ahead predictive mean E[x_t | x_{1:t-1}],
                averaged over run lengths.
            'predictive_var' : np.ndarray, shape (T,)
                One-step-ahead predictive variance Var[x_t | x_{1:t-1}],
                averaged over run lengths (includes mixture variance).
        """
        T = len(data)

        # --- Storage ---
        run_length_posterior = []
        change_point_prob = np.zeros(T)
        map_run_length = np.zeros(T, dtype=int)
        expected_run_length = np.zeros(T)
        predictive_mean = np.full(T, np.nan)
        predictive_var = np.full(T, np.nan)

        # --- Initialization ---
        # At t=0, before seeing any data, we have a single run of length 0
        # with probability 1. This is the "message" passed forward.
        # joint[r] ∝ P(r_{t-1} = r, x_{1:t-1})
        joint = np.array([1.0])  # P(r_0 = 0) = 1

        # One observation model per hypothesized run length.
        # models[r] has been updated with the data assigned to run length r.
        models = [self.model_factory()]

        # --- Main loop ---
        for t in range(T):
            x = data[t]
            n_run_lengths = len(joint)

            # Step 0: One-step-ahead predictive (before observing x_t)
            # Mixture over run lengths: E[x_t] = sum_r P(r) * mu_r
            # Var[x_t] = sum_r P(r) * (var_r + mu_r^2) - E[x_t]^2
            # Only computed for univariate models (scalar mean/var).
            #
            # Some run lengths may have non-finite predictive moments
            # (e.g. NIG with alpha <= 1 gives a Student-t with df <= 2,
            # whose variance is infinite). These are excluded from the
            # mixture and the weights renormalized over the remaining
            # run lengths. At t=0, when only the fresh prior exists and
            # it has infinite variance, no finite run lengths are
            # available, so the predictive is left as NaN for that step.
            m0, _ = models[0].predictive_mean_var()
            is_scalar = np.ndim(m0) == 0 and not np.isnan(m0)

            if is_scalar:
                means = np.zeros(n_run_lengths)
                varis = np.zeros(n_run_lengths)
                for r in range(n_run_lengths):
                    means[r], varis[r] = models[r].predictive_mean_var()

                # Exclude run lengths with infinite variance (e.g. NIG
                # with alpha <= 1 where the Student-t has no finite
                # variance) and renormalize the weights.
                finite = np.isfinite(varis) & np.isfinite(means)
                w = np.where(finite, joint, 0.0)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w /= w_sum
                    m_fin = np.where(finite, means, 0.0)
                    v_fin = np.where(finite, varis, 0.0)
                    predictive_mean[t] = np.sum(w * m_fin)
                    predictive_var[t] = (
                        np.sum(w * (v_fin + m_fin**2)) - predictive_mean[t] ** 2
                    )

            # Step 1: Evaluate predictive probability under each run length
            log_pred = np.zeros(n_run_lengths)
            for r in range(n_run_lengths):
                log_pred[r] = models[r].log_predictive(x)

            # Step 2: Compute growth probabilities
            # P(r_t = r+1, x_{1:t}) = P(r_{t-1}=r, x_{1:t-1}) * pred * (1-H)
            run_lengths = np.arange(n_run_lengths)
            H = self.hazard_fn(run_lengths)

            growth = joint * np.exp(log_pred) * (1.0 - H)

            # Step 3: Compute change point probability (run length resets to 0)
            # P(r_t = 0, x_{1:t}) = sum_r P(r_{t-1}=r, x_{1:t-1}) * pred * H
            changepoint = np.sum(joint * np.exp(log_pred) * H)

            # Step 4: Assemble new joint distribution
            # new_joint[0] = changepoint mass
            # new_joint[1:] = growth mass
            new_joint = np.empty(n_run_lengths + 1)
            new_joint[0] = changepoint
            new_joint[1:] = growth

            # Step 5: Normalize to get posterior
            evidence = np.sum(new_joint)
            if evidence > 0:
                new_joint /= evidence
            else:
                # Numerical underflow — reset to uniform
                new_joint = np.ones_like(new_joint) / len(new_joint)

            # Step 6: Optional truncation
            if self.r_max is not None and len(new_joint) > self.r_max:
                new_joint = new_joint[: self.r_max]
                total = np.sum(new_joint)
                if total > 0:
                    new_joint /= total

            # Step 7: Store results
            run_length_posterior.append(new_joint.copy())
            change_point_prob[t] = new_joint[0]
            map_run_length[t] = np.argmax(new_joint)
            expected_run_length[t] = np.sum(np.arange(len(new_joint)) * new_joint)

            # Step 8: Update observation models
            # - Each existing model incorporates x (for its run length hypothesis)
            # - A new model is prepended (for the r=0 hypothesis at next step)
            new_models = [self.model_factory()]  # fresh model for new run
            for r in range(min(len(models), len(new_joint) - 1)):
                models[r].update(x)
                new_models.append(models[r])

            models = new_models
            joint = new_joint

        return {
            "run_length_posterior": run_length_posterior,
            "change_point_prob": change_point_prob,
            "map_run_length": map_run_length,
            "expected_run_length": expected_run_length,
            "predictive_mean": predictive_mean,
            "predictive_var": predictive_var,
        }


# =============================================================================
# Post-processing: extract change points from BOCPD output
# =============================================================================


def extract_change_points(
    result: dict,
    method: str = "expected_run_length",
    threshold: float | None = None,
    min_gap: int = 20,
) -> np.ndarray:
    """Extract discrete change point indices from BOCPD output.

    With constant hazard, P(r_t=0) is always 1/lambda (the hazard
    factors out of the posterior ratio). Change points are instead
    detected by looking for drops in the expected or MAP run length.

    Parameters
    ----------
    result : dict
        Output from BOCPD.run().
    method : str
        How to detect change points:
        - 'expected_run_length': flag when E[r_t] drops significantly
        - 'map_run_length': flag when MAP r_t drops to near 0
        - 'posterior_mass': flag when P(r_t < k) exceeds threshold
          (mass concentrated on short run lengths)
    threshold : float or None
        Method-dependent threshold. If None, uses sensible defaults:
        - 'expected_run_length': drop of 50% from recent maximum
        - 'map_run_length': MAP drops to below 10
        - 'posterior_mass': P(r_t < 20) > 0.5
    min_gap : int
        Minimum number of time steps between successive change points.

    Returns
    -------
    np.ndarray
        Indices of detected change points.
    """
    if method == "expected_run_length":
        return _extract_from_expected_run_length(
            result["expected_run_length"],
            threshold=threshold,
            min_gap=min_gap,
        )
    elif method == "map_run_length":
        return _extract_from_map_run_length(
            result["map_run_length"],
            threshold=threshold,
            min_gap=min_gap,
        )
    elif method == "posterior_mass":
        return _extract_from_posterior_mass(
            result["run_length_posterior"],
            threshold=threshold,
            min_gap=min_gap,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def _extract_from_expected_run_length(
    erl: np.ndarray,
    threshold: float | None = None,
    min_gap: int = 20,
) -> np.ndarray:
    """Detect change points as sharp drops in expected run length.

    A change point is flagged when E[r_t] drops by more than `threshold`
    fraction from its recent running maximum.
    """
    if threshold is None:
        threshold = 0.5  # 50% drop from recent max

    running_max = np.maximum.accumulate(erl)

    # Flag where ERL drops significantly relative to recent max
    # Avoid division by zero for early time steps
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_drop = 1.0 - erl / np.maximum(running_max, 1.0)

    candidates = np.where(relative_drop > threshold)[0]
    return _merge_nearby(candidates, erl, min_gap, pick="max_drop", score=relative_drop)


def _extract_from_map_run_length(
    map_rl: np.ndarray,
    threshold: float | None = None,
    min_gap: int = 20,
) -> np.ndarray:
    """Detect change points where MAP run length drops to near zero."""
    if threshold is None:
        threshold = 10  # MAP run length < 10 steps

    candidates = np.where(map_rl < threshold)[0]

    # Among candidates, prefer the point where MAP is smallest
    return _merge_nearby(candidates, map_rl, min_gap, pick="min_value", score=map_rl)


def _extract_from_posterior_mass(
    posteriors: list,
    threshold: float | None = None,
    min_gap: int = 20,
    short_run_max: int = 20,
) -> np.ndarray:
    """Detect change points where posterior mass concentrates on short runs."""
    if threshold is None:
        threshold = 0.5

    T = len(posteriors)
    short_mass = np.zeros(T)
    for t in range(T):
        post = posteriors[t]
        k = min(short_run_max, len(post))
        short_mass[t] = np.sum(post[:k])

    candidates = np.where(short_mass > threshold)[0]
    return _merge_nearby(
        candidates, short_mass, min_gap, pick="max_value", score=short_mass
    )


def _merge_nearby(
    candidates: np.ndarray,
    data: np.ndarray,
    min_gap: int,
    pick: str,
    score: np.ndarray,
) -> np.ndarray:
    """Merge nearby candidate change points into clusters."""
    if len(candidates) == 0:
        return np.array([], dtype=int)

    change_points = []
    cluster_start = 0

    for i in range(1, len(candidates)):
        if candidates[i] - candidates[i - 1] > min_gap:
            cluster = candidates[cluster_start:i]
            best = _pick_from_cluster(cluster, score, pick)
            change_points.append(best)
            cluster_start = i

    # Last cluster
    cluster = candidates[cluster_start:]
    best = _pick_from_cluster(cluster, score, pick)
    change_points.append(best)

    return np.array(change_points, dtype=int)


def _pick_from_cluster(cluster, score, pick):
    """Pick the best candidate from a cluster."""
    if pick == "max_drop" or pick == "max_value":
        return cluster[np.argmax(score[cluster])]
    elif pick == "min_value":
        return cluster[np.argmin(score[cluster])]
    else:
        return cluster[0]


# =============================================================================
# Confidence bounds for detected change points
# =============================================================================


def extract_change_points_with_bounds(
    result: dict,
    method: str = "expected_run_length",
    threshold: float | None = None,
    min_gap: int = 20,
    credible_mass: float = 0.90,
    aggregation_window: int = 10,
    min_width: int = 3,
) -> list[dict]:
    """Extract change points with credible intervals from BOCPD output.

    For each detected change point, computes a credible interval
    representing uncertainty about exactly when the change occurred.

    The interval is derived by aggregating retrospective change-time
    distributions from multiple time steps around the detection point.
    At each time t near a change, the run-length posterior P(r_t = r)
    implies a distribution over when the change occurred (at time t - r).
    Averaging these across a window of time steps gives a more robust
    estimate than using a single posterior snapshot.

    Parameters
    ----------
    result : dict
        Output from BOCPD.run().
    method : str
        Detection method (passed to extract_change_points).
    threshold : float or None
        Detection threshold (passed to extract_change_points).
    min_gap : int
        Minimum gap between detections.
    credible_mass : float
        Target probability mass for the credible interval (e.g. 0.90).
    aggregation_window : int
        Number of time steps after detection to aggregate posteriors from.
        Larger values give more stable bounds but add latency.
    min_width : int
        Minimum half-width of the credible interval (in time steps).
        Ensures even confident detections show a visible band.
        Set to 0 to get the raw posterior-derived bounds.

    Returns
    -------
    list of dict, each with:
        'index'    : int   — detected change point index (best estimate)
        'lower'    : int   — lower bound of credible interval
        'upper'    : int   — upper bound of credible interval
        'severity' : float — magnitude of the change (peak ERL drop fraction)
    """
    # Step 1: get point estimates
    change_points = extract_change_points(
        result, method=method, threshold=threshold, min_gap=min_gap
    )

    if len(change_points) == 0:
        return []

    posteriors = result["run_length_posterior"]
    erl = result["expected_run_length"]
    T = len(erl)

    boundaries = []

    for cp in change_points:
        # Step 2: aggregate retrospective change-time distributions
        # from multiple time steps around the detection point.
        #
        # At each time t, the posterior P(r_t = r) implies:
        #   "change occurred at time (t - r)" with probability P(r_t = r)
        #
        # We build a histogram over implied change times by summing
        # these distributions from t = cp to t = cp + aggregation_window.

        change_time_hist = np.zeros(T)
        t_start = max(0, cp - aggregation_window)
        t_end = min(T, cp + aggregation_window + 1)

        for t in range(t_start, t_end):
            post = posteriors[t]
            for r in range(len(post)):
                implied_time = t - r
                if 0 <= implied_time < T:
                    change_time_hist[implied_time] += post[r]

        total_mass = np.sum(change_time_hist)
        if total_mass < 1e-12:
            boundaries.append(
                {
                    "index": int(cp),
                    "lower": int(max(0, cp - min_width)),
                    "upper": int(min(T - 1, cp + min_width)),
                    "severity": 0.0,
                }
            )
            continue

        # Normalize to a proper distribution
        change_time_hist /= total_mass

        # Step 3: find the highest-density credible interval
        # (shortest interval containing credible_mass probability)
        nonzero_idx = np.where(change_time_hist > 1e-15)[0]
        if len(nonzero_idx) == 0:
            boundaries.append(
                {
                    "index": int(cp),
                    "lower": int(max(0, cp - min_width)),
                    "upper": int(min(T - 1, cp + min_width)),
                    "severity": 0.0,
                }
            )
            continue

        best_width = T
        best_lower = int(nonzero_idx[0])
        best_upper = int(nonzero_idx[-1])

        cumsum = np.cumsum(change_time_hist)

        for i in nonzero_idx:
            # Find smallest j >= i such that mass in [i, j] >= credible_mass
            # Mass in [i, j] = cumsum[j] - cumsum[i-1]
            mass_before_i = cumsum[i - 1] if i > 0 else 0.0
            needed_at_j = mass_before_i + credible_mass

            # If not enough mass from i onward, skip this starting point
            if needed_at_j > cumsum[-1] + 1e-12:
                continue

            j_candidates = np.where(cumsum[i:] >= needed_at_j)[0]
            if len(j_candidates) == 0:
                continue

            j = i + j_candidates[0]
            width = j - i
            if width < best_width:
                best_width = width
                best_lower = int(i)
                best_upper = int(j)

        # Step 4: apply minimum width
        center = (best_lower + best_upper) // 2
        half = max(min_width, (best_upper - best_lower) // 2)
        lower = max(0, min(best_lower, center - half))
        upper = min(T - 1, max(best_upper, center + half))

        # Step 5: compute severity as the ERL drop magnitude
        lookback = min(cp, min_gap * 2)
        if lookback > 0:
            pre_peak = np.max(erl[max(0, cp - lookback) : cp + 1])
            post_trough = np.min(erl[cp : min(T, cp + lookback + 1)])
            severity = 1.0 - post_trough / max(pre_peak, 1.0)
        else:
            severity = 0.0

        boundaries.append(
            {
                "index": int(cp),
                "lower": int(lower),
                "upper": int(upper),
                "severity": float(severity),
            }
        )

    return boundaries
