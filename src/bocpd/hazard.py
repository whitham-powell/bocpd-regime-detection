"""
Hazard functions for Bayesian Online Change Point Detection.

The hazard function H(r) gives the probability of a change point occurring
at run length r. It encodes prior beliefs about regime duration.

BOCPD expects a callable: hazard_fn(r) -> float in [0, 1].
"""

import numpy as np


class ConstantHazard:
    """Constant hazard function: H(r) = 1/lambda for all r.

    Implies a geometric distribution on run lengths with
    expected run length = lambda.

    This is the standard choice from Adams & MacKay (2007).
    Memoryless: the probability of a change does not depend on
    how long the current regime has lasted.

    Parameters
    ----------
    lam : float
        Expected run length (in time steps). Larger values mean
        fewer expected change points.
    """

    def __init__(self, lam: float = 200.0):
        if lam <= 0:
            raise ValueError(f"lambda must be positive, got {lam}")
        self.lam = lam

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Evaluate hazard at run length(s) r.

        Parameters
        ----------
        r : int or np.ndarray
            Run length(s). Not actually used (constant hazard),
            but accepted for interface consistency.

        Returns
        -------
        float or np.ndarray
            Hazard value(s), always 1/lambda.
        """
        return np.ones_like(r, dtype=float) / self.lam

    def __repr__(self) -> str:
        return f"ConstantHazard(lam={self.lam})"


class IncreasingHazard:
    """Hazard that increases with run length (wear-out / fatigue).

    H(r) = 1 - S(r+1)/S(r) where S is the survival function
    of a Weibull distribution with shape > 1.

    Regimes become more likely to end the longer they last.

    Parameters
    ----------
    scale : float
        Characteristic run length (Weibull scale parameter).
    shape : float
        Controls how quickly hazard increases. Must be > 1.
        shape = 2 gives linearly increasing hazard.
    """

    def __init__(self, scale: float = 200.0, shape: float = 2.0):
        if shape <= 1.0:
            raise ValueError(f"shape must be > 1 for increasing hazard, got {shape}")
        self.scale = scale
        self.shape = shape

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        # Weibull hazard: (shape/scale) * (r/scale)^(shape-1)
        # Clipped to [0, 1] for valid probability
        h = (self.shape / self.scale) * (r / self.scale) ** (self.shape - 1)
        return np.clip(h, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"IncreasingHazard(scale={self.scale}, shape={self.shape})"


class DecreasingHazard:
    """Hazard that decreases with run length (sticky regimes).

    H(r) = max(h_min, a / (r + b))

    Regimes become less likely to end the longer they last — the
    "sticky" intuition. Useful when regimes tend to persist once
    established.

    Parameters
    ----------
    a : float
        Numerator scale. Controls initial hazard.
    b : float
        Offset. Prevents division by zero at r=0.
    h_min : float
        Floor on the hazard (so it never reaches zero).
    """

    def __init__(self, a: float = 10.0, b: float = 5.0, h_min: float = 0.001):
        self.a = a
        self.b = b
        self.h_min = h_min

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        h = self.a / (r + self.b)
        return np.clip(h, self.h_min, 1.0)

    def __repr__(self) -> str:
        return f"DecreasingHazard(a={self.a}, b={self.b}, h_min={self.h_min})"
