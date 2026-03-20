"""
Observation models for Bayesian Online Change Point Detection.

Architecture:
    ObservationModel                 <- BOCPD talks to this
    ├── ExponentialFamilyModel       <- generic conjugate machinery
    │   ├── UnivariateNormalNIG
    │   ├── MultivariateNormalNIW
    │   └── (any conjugate model)
    └── (any non-conjugate model)    <- just implement the 3 methods

BOCPD only requires the ObservationModel interface. The ExponentialFamilyModel
base class provides a generic implementation of log_predictive via the
normalizing-constant ratio trick, so subclasses only need to define:
    - sufficient_statistic(x)
    - log_base_measure(x)
    - log_normalizer(**params)
    - _get_params() / _updated_params(stat)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from scipy.special import gammaln, multigammaln

# =============================================================================
# Base Interface — this is all BOCPD sees
# =============================================================================


class ObservationModel(ABC):
    """Abstract interface for BOCPD observation models.

    Any model that can produce a predictive log-probability and incorporate
    new observations can be used with BOCPD. No distributional assumptions
    are imposed at this level.
    """

    @abstractmethod
    def log_predictive(self, x: np.ndarray) -> float:
        """Log probability of x given all data observed in this run so far.

        Parameters
        ----------
        x : np.ndarray
            New observation. Shape depends on the model (scalar, vector, etc.).

        Returns
        -------
        float
            Log predictive probability.
        """

    @abstractmethod
    def update(self, x: np.ndarray) -> None:
        """Incorporate observation x into the model's sufficient statistics.

        Parameters
        ----------
        x : np.ndarray
            New observation.
        """

    def predictive_mean_var(self) -> tuple:
        """Return (mean, variance) of the predictive distribution.

        For univariate models, returns (float, float).
        For multivariate models, returns (ndarray, ndarray) where
        mean is shape (D,) and variance is the (D, D) covariance matrix.

        Used to construct mixture predictive envelopes. Models that do
        not support this should return (nan, nan).

        Implementations may return inf for the variance (or, less
        commonly, the mean) when the predictive distribution's moments
        do not exist. For example, the Student-t predictive of NIG has
        infinite variance when df <= 2 (i.e. alpha <= 1). This is not
        a bug — it reflects the prior's genuine uncertainty before
        enough data has been observed. The BOCPD loop handles this by
        excluding such run lengths from the mixture computation.
        """
        return (np.nan, np.nan)

    def copy(self) -> ObservationModel:
        """Return an independent copy of this model (including current state).

        The default implementation uses deepcopy. Subclasses may override
        for efficiency.
        """
        return deepcopy(self)


# =============================================================================
# Exponential Family Base — generic conjugate machinery
# =============================================================================


class ExponentialFamilyModel(ObservationModel, ABC):
    """Base class for conjugate exponential family observation models.

    For any exponential family likelihood with a conjugate prior, the
    predictive probability of a new observation is:

        p(x_{n+1} | x_{1:n}) = Z(params_updated) / Z(params_current) * h(x)

    where Z is the normalizing constant of the conjugate prior, params_updated
    are the prior hyperparameters after hypothetically incorporating x_{n+1},
    and h(x) is the base measure.

    Subclasses implement four methods that define the specific distribution.
    Everything else — log_predictive, update — is handled generically.
    """

    @abstractmethod
    def sufficient_statistic(self, x: np.ndarray) -> dict:
        """Compute sufficient statistic T(x).

        Returns a dict whose keys match the prior hyperparameter names.
        The values are the additive contributions of one observation.
        """

    @abstractmethod
    def log_base_measure(self, x: np.ndarray) -> float:
        """Compute log h(x), the base measure term."""

    @abstractmethod
    def log_normalizer(self, **params) -> float:
        """Log normalizing constant of the conjugate prior.

        Parameters are the prior hyperparameters (distribution-specific).
        """

    @abstractmethod
    def _get_params(self) -> dict:
        """Return current prior hyperparameters as a dict."""

    @abstractmethod
    def _updated_params(self, stat: dict) -> dict:
        """Return hyperparameters that would result from incorporating one
        observation with sufficient statistic `stat`, WITHOUT mutating state."""

    # --- Generic implementations (same for all exponential family models) ---

    def log_predictive(self, x: np.ndarray) -> float:
        """Predictive probability via normalizing constant ratio.

        This is the core trick: we never need to name or evaluate the
        predictive distribution (Student-t, negative binomial, etc.)
        explicitly. It falls out of the conjugate structure.
        """
        current_params = self._get_params()
        stat = self.sufficient_statistic(x)
        new_params = self._updated_params(stat)

        log_Z_new = self.log_normalizer(**new_params)
        log_Z_cur = self.log_normalizer(**current_params)

        return log_Z_new - log_Z_cur + self.log_base_measure(x)

    def update(self, x: np.ndarray) -> None:
        """Incorporate observation by updating prior hyperparameters."""
        stat = self.sufficient_statistic(x)
        updated = self._updated_params(stat)
        self._set_params(updated)

    @abstractmethod
    def _set_params(self, params: dict) -> None:
        """Set hyperparameters from a dict (inverse of _get_params)."""


# =============================================================================
# Univariate Normal with Normal-Inverse-Gamma Prior
# =============================================================================


class UnivariateNormalNIG(ExponentialFamilyModel):
    """Univariate Gaussian with unknown mean and variance.

    Conjugate prior: Normal-Inverse-Gamma(mu0, kappa0, alpha0, beta0)

    Prior hyperparameters:
        mu0    : prior mean
        kappa0 : pseudo-observations for the mean (controls mean certainty)
        alpha0 : shape parameter for inverse-gamma on variance
        beta0  : scale parameter for inverse-gamma on variance

    Predictive distribution: Student-t with
        df    = 2 * alpha
        loc   = mu
        scale = beta * (kappa + 1) / (alpha * kappa)

    Note: The predictive df grows with sample size (df = 2*alpha, alpha
    increases by 0.5 per observation), so tails lighten toward Gaussian
    within each regime. For data with persistent heavy tails (e.g., stock
    returns), use ``StudentTNIG`` instead, which fixes the tail index.
    """

    def __init__(
        self,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ):
        self.mu = mu0
        self.kappa = kappa0
        self.alpha = alpha0
        self.beta = beta0

        # Store initial values for reference
        self._mu0 = mu0
        self._kappa0 = kappa0
        self._alpha0 = alpha0
        self._beta0 = beta0

    def sufficient_statistic(self, x: np.ndarray) -> dict:
        x = float(x)
        return {"x": x, "n": 1}

    def log_base_measure(self, x: np.ndarray) -> float:
        return -0.5 * np.log(2 * np.pi)

    def log_normalizer(self, *, mu, kappa, alpha, beta) -> float:
        """Log normalizing constant of the NIG distribution.

        Z(mu, kappa, alpha, beta) =
            (beta^alpha / Gamma(alpha)) * (2*pi / kappa)^{1/2}

        So log Z = alpha * log(beta) - gammaln(alpha) + 0.5 * log(2*pi/kappa)

        Note: we need the INVERSE of the normalizing constant for the prior
        density, but since we take a ratio, the convention just needs to be
        consistent. We use log Z = gammaln(alpha) - alpha*log(beta) + 0.5*log(kappa)
        which corresponds to the numerator terms that survive in the ratio.
        """
        return gammaln(alpha) - alpha * np.log(beta) - 0.5 * np.log(kappa)

    def _get_params(self) -> dict:
        return {
            "mu": self.mu,
            "kappa": self.kappa,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def _updated_params(self, stat: dict) -> dict:
        x, n = stat["x"], stat["n"]
        kappa_new = self.kappa + n
        mu_new = (self.kappa * self.mu + n * x) / kappa_new
        alpha_new = self.alpha + n / 2.0
        beta_new = self.beta + 0.5 * n * (x - self.mu) ** 2 * self.kappa / kappa_new
        return {
            "mu": mu_new,
            "kappa": kappa_new,
            "alpha": alpha_new,
            "beta": beta_new,
        }

    def _set_params(self, params: dict) -> None:
        self.mu = params["mu"]
        self.kappa = params["kappa"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]

    def predictive_mean_var(self) -> tuple[float, float]:
        """Predictive Student-t mean and variance.

        The predictive distribution is Student-t with:
            df    = 2 * alpha
            loc   = mu
            scale = beta * (kappa + 1) / (alpha * kappa)

        Variance = scale * df / (df - 2)  when df > 2.
        """
        df = 2.0 * self.alpha
        scale = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa)
        mean = self.mu
        var = scale * df / (df - 2.0) if df > 2.0 else np.inf
        return (mean, var)

    def to_dict(self) -> dict:
        return {
            "type": "UnivariateNormalNIG",
            "prior": {
                "mu0": self._mu0,
                "kappa0": self._kappa0,
                "alpha0": self._alpha0,
                "beta0": self._beta0,
            },
            "state": {
                "mu": self.mu,
                "kappa": self.kappa,
                "alpha": self.alpha,
                "beta": self.beta,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> UnivariateNormalNIG:
        prior = d["prior"]
        obj = cls(
            mu0=prior["mu0"],
            kappa0=prior["kappa0"],
            alpha0=prior["alpha0"],
            beta0=prior["beta0"],
        )
        if "state" in d:
            s = d["state"]
            obj.mu = s["mu"]
            obj.kappa = s["kappa"]
            obj.alpha = s["alpha"]
            obj.beta = s["beta"]
        return obj


# =============================================================================
# Multivariate Normal with Normal-Inverse-Wishart Prior
# =============================================================================


class MultivariateNormalNIW(ExponentialFamilyModel):
    """Multivariate Gaussian with unknown mean and covariance.

    Conjugate prior: Normal-Inverse-Wishart(mu0, kappa0, nu0, Psi0)

    Prior hyperparameters:
        mu0    : prior mean vector (D,)
        kappa0 : pseudo-observations for the mean
        nu0    : prior degrees of freedom (must be > D - 1)
        Psi0   : prior scale matrix (D, D), positive definite

    Predictive distribution: Multivariate Student-t with
        df    = nu - D + 1
        loc   = mu
        shape = Psi * (kappa + 1) / (kappa * (nu - D + 1))

    Note: The predictive df grows with sample size (df = nu_n - D + 1,
    nu_n increases by 1 per observation), so tails lighten toward Gaussian
    within each regime. For data with persistent heavy tails, use
    ``MultivariateStudentTNIW`` instead, which fixes the tail index.

    Implementation uses Welford-style online sufficient statistics
    (n, x_bar, S) internally, and derives NIW posterior parameters from these
    plus the prior.
    """

    def __init__(
        self,
        dim: int,
        mu0: np.ndarray = None,
        kappa0: float = 1.0,
        nu0: float | None = None,
        Psi0: np.ndarray = None,
    ):
        self.dim = dim

        # Prior hyperparameters
        self.mu0 = mu0 if mu0 is not None else np.zeros(dim)
        self.kappa0 = kappa0
        self.nu0 = nu0 if nu0 is not None else float(dim)  # minimum valid
        self.Psi0 = Psi0 if Psi0 is not None else np.eye(dim)

        if self.nu0 <= self.dim - 1:
            raise ValueError(f"nu0 must be > dim - 1 = {self.dim - 1}, got {self.nu0}")

        # Sufficient statistics for current run (Welford-style)
        self.n = 0
        self.x_bar = np.zeros(dim)
        self.S = np.zeros((dim, dim))

        # slogdet cache: avoids recomputing slogdet(Psi_n) when it was
        # already computed as the hypothetical slogdet(Psi_n') last step.
        self._cached_logdet = None  # (sign, logdet) or None
        self._staged_logdet = None  # hypothetical, promoted on update()

    def sufficient_statistic(self, x: np.ndarray) -> dict:
        return {"x": np.asarray(x, dtype=float)}

    def log_base_measure(self, x: np.ndarray) -> float:
        return -0.5 * self.dim * np.log(2 * np.pi)

    def log_normalizer(self, *, kappa, nu, Psi) -> float:
        """Log normalizing constant of the NIW distribution.

        The relevant terms that survive in the predictive ratio are:

        log Z(kappa, nu, Psi) = multigammaln(nu/2, D)
                                - (nu/2) * log|Psi|
                                - (D/2) * log(kappa)
        """
        D = self.dim
        sign, logdet_Psi = np.linalg.slogdet(Psi)
        if sign <= 0:
            return -np.inf

        return (
            multigammaln(nu / 2.0, D)
            - (nu / 2.0) * logdet_Psi
            - (D / 2.0) * np.log(kappa)
        )

    def _log_normalizer_with_logdet(self, kappa, nu, sign, logdet_Psi):
        """Log normalizer when slogdet is already computed."""
        if sign <= 0:
            return -np.inf
        D = self.dim
        return (
            multigammaln(nu / 2.0, D)
            - (nu / 2.0) * logdet_Psi
            - (D / 2.0) * np.log(kappa)
        )

    def _posterior_params(self, n, x_bar, S):
        """Derive NIW posterior parameters from sufficient statistics."""
        kappa_n = self.kappa0 + n
        nu_n = self.nu0 + n

        if n == 0:
            Psi_n = self.Psi0.copy()
        else:
            diff = x_bar - self.mu0
            Psi_n = self.Psi0 + S + (self.kappa0 * n / kappa_n) * np.outer(diff, diff)

        return {"kappa": kappa_n, "nu": nu_n, "Psi": Psi_n}

    def _get_params(self) -> dict:
        return self._posterior_params(self.n, self.x_bar, self.S)

    def _updated_params(self, stat: dict) -> dict:
        """Compute what posterior params would be after incorporating x."""
        x = stat["x"]

        # Hypothetical Welford update (don't mutate self)
        n_new = self.n + 1
        dx = x - self.x_bar
        x_bar_new = self.x_bar + dx / n_new
        S_new = self.S + (self.n / n_new) * np.outer(dx, dx)

        return self._posterior_params(n_new, x_bar_new, S_new)

    def log_predictive(self, x: np.ndarray) -> float:
        """Predictive probability with slogdet caching.

        Overrides the generic ExponentialFamilyModel.log_predictive to
        reuse slogdet(Psi_n) from the previous timestep's hypothetical
        computation, halving the number of slogdet calls.
        """
        current_params = self._get_params()
        stat = self.sufficient_statistic(x)
        new_params = self._updated_params(stat)

        # Current log normalizer: use cache if available
        if self._cached_logdet is not None:
            sign_cur, logdet_cur = self._cached_logdet
            log_Z_cur = self._log_normalizer_with_logdet(
                current_params["kappa"],
                current_params["nu"],
                sign_cur,
                logdet_cur,
            )
        else:
            log_Z_cur = self.log_normalizer(**current_params)

        # Hypothetical log normalizer: always fresh, stash for reuse
        sign_new, logdet_new = np.linalg.slogdet(new_params["Psi"])
        self._staged_logdet = (sign_new, logdet_new)
        log_Z_new = self._log_normalizer_with_logdet(
            new_params["kappa"],
            new_params["nu"],
            sign_new,
            logdet_new,
        )

        return log_Z_new - log_Z_cur + self.log_base_measure(x)

    def _set_params(self, params: dict) -> None:
        """For NIW, we don't set params directly — we update sufficient stats.

        This is called by the generic ExponentialFamilyModel.update().
        We override update() instead.
        """
        raise NotImplementedError(
            "NIW uses Welford sufficient statistics; use update() directly."
        )

    def update(self, x: np.ndarray) -> None:
        """Welford online update of sufficient statistics."""
        x = np.asarray(x, dtype=float)
        self.n += 1
        dx = x - self.x_bar
        self.x_bar = self.x_bar + dx / self.n
        self.S = self.S + ((self.n - 1) / self.n) * np.outer(dx, dx)
        # Promote staged hypothetical logdet to current cache
        if self._staged_logdet is not None:
            self._cached_logdet = self._staged_logdet
            self._staged_logdet = None

    def copy(self) -> MultivariateNormalNIW:
        """Efficient copy — avoid deepcopy of numpy arrays."""
        new = MultivariateNormalNIW.__new__(MultivariateNormalNIW)
        new.dim = self.dim
        new.mu0 = self.mu0  # shared ref is fine (immutable during run)
        new.kappa0 = self.kappa0
        new.nu0 = self.nu0
        new.Psi0 = self.Psi0  # shared ref is fine
        new.n = self.n
        new.x_bar = self.x_bar.copy()
        new.S = self.S.copy()
        new._cached_logdet = self._cached_logdet
        new._staged_logdet = self._staged_logdet
        return new

    def predictive_mean_var(self) -> tuple[np.ndarray, np.ndarray]:
        """Predictive multivariate Student-t mean and covariance.

        The predictive is multivariate Student-t with:
            df    = nu - D + 1
            loc   = mu_n  (posterior mean)
            shape = Psi_n * (kappa_n + 1) / (kappa_n * df)

        Covariance = shape * df / (df - 2)  when df > 2.
        """
        params = self._get_params()
        D = self.dim
        df = params["nu"] - D + 1.0
        mu_n = (self.kappa0 * self.mu0 + self.n * self.x_bar) / params["kappa"]
        shape = params["Psi"] * (params["kappa"] + 1.0) / (params["kappa"] * df)
        cov = shape * df / (df - 2.0) if df > 2.0 else np.full_like(shape, np.inf)
        return (mu_n, cov)

    def to_dict(self) -> dict:
        return {
            "type": "MultivariateNormalNIW",
            "prior": {
                "dim": self.dim,
                "mu0": self.mu0.tolist(),
                "kappa0": self.kappa0,
                "nu0": self.nu0,
                "Psi0": self.Psi0.tolist(),
            },
            "state": {
                "n": self.n,
                "x_bar": self.x_bar.tolist(),
                "S": self.S.tolist(),
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> MultivariateNormalNIW:
        prior = d["prior"]
        obj = cls(
            dim=prior["dim"],
            mu0=np.array(prior["mu0"]),
            kappa0=prior["kappa0"],
            nu0=prior["nu0"],
            Psi0=np.array(prior["Psi0"]),
        )
        if "state" in d:
            s = d["state"]
            obj.n = s["n"]
            obj.x_bar = np.array(s["x_bar"])
            obj.S = np.array(s["S"])
        return obj


# =============================================================================
# Poisson-Gamma (example conjugate model for count data)
# =============================================================================


class PoissonGamma(ExponentialFamilyModel):
    """Poisson likelihood with Gamma conjugate prior on the rate.

    Prior: Gamma(alpha0, beta0) on rate parameter lambda.
    Predictive: Negative Binomial.

    Prior hyperparameters:
        alpha0 : shape (pseudo-counts of events)
        beta0  : rate (pseudo-observations)
    """

    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0):
        self.alpha = alpha0
        self.beta = beta0
        self._alpha0 = alpha0
        self._beta0 = beta0

    def sufficient_statistic(self, x: np.ndarray) -> dict:
        x = int(x)
        return {"sum_x": x, "n": 1}

    def log_base_measure(self, x: np.ndarray) -> float:
        x = int(x)
        return -gammaln(x + 1)  # -log(x!)

    def log_normalizer(self, *, alpha, beta) -> float:
        return gammaln(alpha) - alpha * np.log(beta)

    def _get_params(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    def _updated_params(self, stat: dict) -> dict:
        return {
            "alpha": self.alpha + stat["sum_x"],
            "beta": self.beta + stat["n"],
        }

    def _set_params(self, params: dict) -> None:
        self.alpha = params["alpha"]
        self.beta = params["beta"]

    def predictive_mean_var(self) -> tuple[float, float]:
        """Predictive Negative Binomial mean and variance.

        The predictive is NegBin(r=alpha, p=beta/(beta+1)):
            mean = alpha / beta
            var  = alpha * (beta + 1) / beta^2
        """
        mean = self.alpha / self.beta
        var = self.alpha * (self.beta + 1.0) / (self.beta**2)
        return (mean, var)

    def to_dict(self) -> dict:
        return {
            "type": "PoissonGamma",
            "prior": {"alpha0": self._alpha0, "beta0": self._beta0},
            "state": {"alpha": self.alpha, "beta": self.beta},
        }

    @classmethod
    def from_dict(cls, d: dict) -> PoissonGamma:
        prior = d["prior"]
        obj = cls(alpha0=prior["alpha0"], beta0=prior["beta0"])
        if "state" in d:
            s = d["state"]
            obj.alpha = s["alpha"]
            obj.beta = s["beta"]
        return obj


# =============================================================================
# Batched NIW sufficient statistics (internal, not exported)
# =============================================================================


class _NIWBatch:
    """Pre-allocated batched NIW sufficient statistics for vectorized BOCPD.

    Stores (n, x_bar, S) arrays for up to `capacity` run lengths, with an
    active count `R`. All posterior and predictive computations are vectorized
    over the active run lengths — no Python loop required.
    """

    def __init__(self, probe: MultivariateNormalNIW, capacity: int):
        D = probe.dim
        self.D = D
        self.capacity = capacity

        # Prior hyperparameters (shared, immutable during run)
        self.mu0 = probe.mu0
        self.kappa0 = probe.kappa0
        self.nu0 = probe.nu0
        self.Psi0 = probe.Psi0

        # Precompute slogdet of prior scale matrix and log base measure
        sign0, logdet0 = np.linalg.slogdet(self.Psi0)
        self._logdet_Psi0 = logdet0
        self._sign_Psi0 = sign0
        self._log_base = -0.5 * D * np.log(2 * np.pi)

        # Pre-allocated sufficient statistics
        self.n = np.zeros(capacity)  # (capacity,)
        self.x_bar = np.zeros((capacity, D))  # (capacity, D)
        self.S = np.zeros((capacity, D, D))  # (capacity, D, D)

        # Active run-length count
        self.R = 0

        # slogdet cache: shape (R,) logdets for the current Psi_n
        self._cached_logdet = None  # (R,) or None
        self._staged_logdet = None  # (R,) staged from hypothetical

    def prepend_fresh(self):
        """Shift active region right, insert a fresh run at position 0."""
        if self.capacity <= self.R:
            # At capacity — drop the oldest run length
            pass
        else:
            self.R += 1

        # Shift existing entries right by 1
        R = self.R
        self.n[1:R] = self.n[: R - 1]
        self.x_bar[1:R] = self.x_bar[: R - 1]
        self.S[1:R] = self.S[: R - 1]

        # Fresh run at position 0
        self.n[0] = 0
        self.x_bar[0] = 0.0
        self.S[0] = 0.0

        # Update cache: shift and insert prior logdet at position 0
        if self._cached_logdet is not None:
            new_cache = np.empty(R)
            new_cache[0] = self._logdet_Psi0
            new_cache[1:R] = self._cached_logdet[: R - 1]
            self._cached_logdet = new_cache
        else:
            self._cached_logdet = None

    def log_predictive_all(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log predictive for all active run lengths.

        Parameters
        ----------
        x : np.ndarray, shape (D,)

        Returns
        -------
        np.ndarray, shape (R,)
        """
        R, D = self.R, self.D
        n = self.n[:R]  # (R,)
        x_bar = self.x_bar[:R]  # (R, D)
        S = self.S[:R]  # (R, D, D)

        # Current posterior parameters (batched)
        kappa_n = self.kappa0 + n  # (R,)
        nu_n = self.nu0 + n  # (R,)

        # Psi_n: when n=0, scale=0 and S=0 so Psi_n = Psi0 automatically
        diff_cur = x_bar - self.mu0  # (R, D)
        scale_cur = np.where(n > 0, self.kappa0 * n / kappa_n, 0.0)  # (R,)
        Psi_n = (
            self.Psi0
            + S
            + scale_cur[:, None, None] * (diff_cur[:, :, None] * diff_cur[:, None, :])
        )  # (R, D, D)

        # Hypothetical Welford update (batched, no mutation)
        n_new = n + 1  # (R,)
        dx = x - x_bar  # (R, D)
        # x_bar_new = x_bar + dx / n_new[:, None]  # not needed for Psi
        S_new = S + (n / n_new)[:, None, None] * (
            dx[:, :, None] * dx[:, None, :]
        )  # (R, D, D)

        # Hypothetical posterior parameters
        kappa_n_new = self.kappa0 + n_new  # (R,)
        nu_n_new = self.nu0 + n_new  # (R,)
        x_bar_new = x_bar + dx / n_new[:, None]  # (R, D)
        diff_new = x_bar_new - self.mu0  # (R, D)
        scale_new = self.kappa0 * n_new / kappa_n_new  # (R,) — n_new >= 1 always
        Psi_n_new = (
            self.Psi0
            + S_new
            + scale_new[:, None, None] * (diff_new[:, :, None] * diff_new[:, None, :])
        )  # (R, D, D)

        # slogdet: use cache for current, compute fresh for hypothetical
        if self._cached_logdet is not None and len(self._cached_logdet) >= R:
            logdet_cur = self._cached_logdet[:R]
        else:
            signs_cur, logdet_cur = np.linalg.slogdet(Psi_n)
            # Guard: where sign <= 0, logdet doesn't matter (handled below)
            logdet_cur = np.where(signs_cur > 0, logdet_cur, 0.0)

        signs_new, logdet_new = np.linalg.slogdet(Psi_n_new)  # 1 batched call
        self._staged_logdet = np.where(signs_new > 0, logdet_new, 0.0)

        # Log normalizer ratio
        log_Z_cur = (
            multigammaln(nu_n / 2.0, D)
            - (nu_n / 2.0) * logdet_cur
            - (D / 2.0) * np.log(kappa_n)
        )
        log_Z_new = (
            multigammaln(nu_n_new / 2.0, D)
            - (nu_n_new / 2.0) * logdet_new
            - (D / 2.0) * np.log(kappa_n_new)
        )

        result = log_Z_new - log_Z_cur + self._log_base

        # Guard: where hypothetical sign <= 0, result is -inf
        result = np.where(signs_new > 0, result, -np.inf)

        return result

    def update_all(self, x: np.ndarray):
        """Vectorized Welford update on all active run lengths."""
        R = self.R
        n = self.n[:R]
        x_bar = self.x_bar[:R]
        S = self.S[:R]

        n_new = n + 1
        dx = x - x_bar
        x_bar_new = x_bar + dx / n_new[:, None]
        S_new = S + (n / n_new)[:, None, None] * (dx[:, :, None] * dx[:, None, :])

        self.n[:R] = n_new
        self.x_bar[:R] = x_bar_new
        self.S[:R] = S_new

        # Promote staged logdet to current cache
        if self._staged_logdet is not None:
            self._cached_logdet = self._staged_logdet
            self._staged_logdet = None

    def truncate(self, n: int):
        """Reduce active run lengths to at most n."""
        self.R = min(self.R, n)
        if self._cached_logdet is not None:
            self._cached_logdet = self._cached_logdet[: self.R]

    def to_dict(self) -> dict:
        R = self.R
        return {
            "type": "_NIWBatch",
            "prior": {
                "dim": self.D,
                "mu0": self.mu0.tolist(),
                "kappa0": self.kappa0,
                "nu0": self.nu0,
                "Psi0": self.Psi0.tolist(),
            },
            "state": {
                "R": R,
                "capacity": self.capacity,
                "n": self.n[:R].tolist(),
                "x_bar": self.x_bar[:R].tolist(),
                "S": self.S[:R].tolist(),
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> _NIWBatch:
        prior = d["prior"]
        probe = MultivariateNormalNIW(
            dim=prior["dim"],
            mu0=np.array(prior["mu0"]),
            kappa0=prior["kappa0"],
            nu0=prior["nu0"],
            Psi0=np.array(prior["Psi0"]),
        )
        s = d["state"]
        obj = cls(probe, capacity=s["capacity"])
        R = s["R"]
        obj.R = R
        obj.n[:R] = np.array(s["n"])
        obj.x_bar[:R] = np.array(s["x_bar"])
        obj.S[:R] = np.array(s["S"])
        obj._cached_logdet = None
        obj._staged_logdet = None
        return obj


# =============================================================================
# Bernoulli with Beta Prior
# =============================================================================


class BernoulliBeta(ExponentialFamilyModel):
    """Bernoulli likelihood with Beta conjugate prior on p.

    Useful for detecting changes in binary event probability
    (e.g., defect rates, click-through rates, alarm states).

    Prior hyperparameters:
        alpha0 : pseudo-count of successes
        beta0  : pseudo-count of failures

    Predictive distribution: Beta-Bernoulli.
    """

    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0):
        self.alpha = alpha0
        self.beta = beta0
        self._alpha0 = alpha0
        self._beta0 = beta0

    def sufficient_statistic(self, x: np.ndarray) -> dict:
        return {"sum_x": float(x), "n": 1}

    def log_base_measure(self, x: np.ndarray) -> float:
        return 0.0

    def log_normalizer(self, *, alpha, beta) -> float:
        return gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)

    def _get_params(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    def _updated_params(self, stat: dict) -> dict:
        return {
            "alpha": self.alpha + stat["sum_x"],
            "beta": self.beta + stat["n"] - stat["sum_x"],
        }

    def _set_params(self, params: dict) -> None:
        self.alpha = params["alpha"]
        self.beta = params["beta"]

    def predictive_mean_var(self) -> tuple[float, float]:
        """Beta-Bernoulli predictive mean and variance."""
        s = self.alpha + self.beta
        mean = self.alpha / s
        var = self.alpha * self.beta / (s**2 * (s + 1.0))
        return (mean, var)

    def to_dict(self) -> dict:
        return {
            "type": "BernoulliBeta",
            "prior": {"alpha0": self._alpha0, "beta0": self._beta0},
            "state": {"alpha": self.alpha, "beta": self.beta},
        }

    @classmethod
    def from_dict(cls, d: dict) -> BernoulliBeta:
        prior = d["prior"]
        obj = cls(alpha0=prior["alpha0"], beta0=prior["beta0"])
        if "state" in d:
            s = d["state"]
            obj.alpha = s["alpha"]
            obj.beta = s["beta"]
        return obj


# =============================================================================
# Exponential with Gamma Prior
# =============================================================================


class ExponentialGamma(ExponentialFamilyModel):
    """Exponential likelihood with Gamma conjugate prior on the rate.

    Useful for detecting changes in event rates or durations
    (e.g., time between failures, inter-arrival times).

    Prior hyperparameters:
        alpha0 : shape (pseudo-count of events)
        beta0  : rate (pseudo-total of durations)

    Predictive distribution: Lomax (Pareto Type II).
    """

    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0):
        self.alpha = alpha0
        self.beta = beta0
        self._alpha0 = alpha0
        self._beta0 = beta0

    def sufficient_statistic(self, x: np.ndarray) -> dict:
        return {"sum_x": float(x), "n": 1}

    def log_base_measure(self, x: np.ndarray) -> float:
        return 0.0

    def log_normalizer(self, *, alpha, beta) -> float:
        return gammaln(alpha) - alpha * np.log(beta)

    def _get_params(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    def _updated_params(self, stat: dict) -> dict:
        return {
            "alpha": self.alpha + stat["n"],
            "beta": self.beta + stat["sum_x"],
        }

    def _set_params(self, params: dict) -> None:
        self.alpha = params["alpha"]
        self.beta = params["beta"]

    def predictive_mean_var(self) -> tuple[float, float]:
        """Lomax (Pareto Type II) predictive mean and variance."""
        mean = self.beta / (self.alpha - 1.0) if self.alpha > 1.0 else np.inf
        var = (
            self.beta**2 * self.alpha / ((self.alpha - 1.0) ** 2 * (self.alpha - 2.0))
            if self.alpha > 2.0
            else np.inf
        )
        return (mean, var)

    def to_dict(self) -> dict:
        return {
            "type": "ExponentialGamma",
            "prior": {"alpha0": self._alpha0, "beta0": self._beta0},
            "state": {"alpha": self.alpha, "beta": self.beta},
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExponentialGamma:
        prior = d["prior"]
        obj = cls(alpha0=prior["alpha0"], beta0=prior["beta0"])
        if "state" in d:
            s = d["state"]
            obj.alpha = s["alpha"]
            obj.beta = s["beta"]
        return obj


# =============================================================================
# Normal with Known Variance (unknown mean)
# =============================================================================


class NormalKnownVariance(ExponentialFamilyModel):
    """Normal likelihood with known variance and Normal prior on the mean.

    Detects changes in the mean while assuming constant variance.
    Faster and sharper than NIG when variance is truly known.

    Parameters:
        mu0       : prior mean
        sigma0_sq : prior variance on the mean
        sigma2    : known observation variance

    Working parameters: tau (precision = 1/posterior_var), mu (posterior mean).
    Predictive distribution: Normal.
    """

    def __init__(
        self,
        mu0: float = 0.0,
        sigma0_sq: float = 1.0,
        sigma2: float = 1.0,
    ):
        self._mu0 = mu0
        self._sigma0_sq = sigma0_sq
        self._sigma2 = sigma2

        self.tau = 1.0 / sigma0_sq
        self.mu = mu0

    def sufficient_statistic(self, x: np.ndarray) -> dict:
        return {"x": float(x), "n": 1}

    def log_base_measure(self, x: np.ndarray) -> float:
        x = float(x)
        return -0.5 * np.log(2 * np.pi * self._sigma2) - x**2 / (2 * self._sigma2)

    def log_normalizer(self, *, tau, mu) -> float:
        return -0.5 * np.log(tau) + 0.5 * tau * mu**2

    def _get_params(self) -> dict:
        return {"tau": self.tau, "mu": self.mu}

    def _updated_params(self, stat: dict) -> dict:
        tau_new = self.tau + stat["n"] / self._sigma2
        mu_new = (self.tau * self.mu + stat["x"] / self._sigma2) / tau_new
        return {"tau": tau_new, "mu": mu_new}

    def _set_params(self, params: dict) -> None:
        self.tau = params["tau"]
        self.mu = params["mu"]

    def predictive_mean_var(self) -> tuple[float, float]:
        """Normal predictive mean and variance."""
        return (self.mu, self._sigma2 + 1.0 / self.tau)

    def to_dict(self) -> dict:
        return {
            "type": "NormalKnownVariance",
            "prior": {
                "mu0": self._mu0,
                "sigma0_sq": self._sigma0_sq,
                "sigma2": self._sigma2,
            },
            "state": {"tau": self.tau, "mu": self.mu},
        }

    @classmethod
    def from_dict(cls, d: dict) -> NormalKnownVariance:
        prior = d["prior"]
        obj = cls(
            mu0=prior["mu0"],
            sigma0_sq=prior["sigma0_sq"],
            sigma2=prior["sigma2"],
        )
        if "state" in d:
            s = d["state"]
            obj.tau = s["tau"]
            obj.mu = s["mu"]
        return obj


# =============================================================================
# Normal with Known Mean (unknown variance)
# =============================================================================


class NormalKnownMean(ExponentialFamilyModel):
    """Normal likelihood with known mean and InverseGamma prior on variance.

    Detects changes in variance (volatility) while assuming constant mean.

    Parameters:
        mu_known : known observation mean
        alpha0   : InverseGamma shape
        beta0    : InverseGamma scale

    Predictive distribution: Student-t.
    """

    def __init__(
        self,
        mu_known: float = 0.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ):
        self._mu_known = mu_known
        self._alpha0 = alpha0
        self._beta0 = beta0

        self.alpha = alpha0
        self.beta = beta0

    def sufficient_statistic(self, x: np.ndarray) -> dict:
        x = float(x)
        return {"sq_dev": (x - self._mu_known) ** 2, "n": 1}

    def log_base_measure(self, x: np.ndarray) -> float:
        return -0.5 * np.log(2 * np.pi)

    def log_normalizer(self, *, alpha, beta) -> float:
        return gammaln(alpha) - alpha * np.log(beta)

    def _get_params(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    def _updated_params(self, stat: dict) -> dict:
        return {
            "alpha": self.alpha + 0.5 * stat["n"],
            "beta": self.beta + 0.5 * stat["sq_dev"],
        }

    def _set_params(self, params: dict) -> None:
        self.alpha = params["alpha"]
        self.beta = params["beta"]

    def predictive_mean_var(self) -> tuple[float, float]:
        """Student-t predictive mean and variance."""
        df = 2.0 * self.alpha
        scale = self.beta / self.alpha
        var = scale * df / (df - 2.0) if df > 2.0 else np.inf
        return (self._mu_known, var)

    def to_dict(self) -> dict:
        return {
            "type": "NormalKnownMean",
            "prior": {
                "mu_known": self._mu_known,
                "alpha0": self._alpha0,
                "beta0": self._beta0,
            },
            "state": {"alpha": self.alpha, "beta": self.beta},
        }

    @classmethod
    def from_dict(cls, d: dict) -> NormalKnownMean:
        prior = d["prior"]
        obj = cls(
            mu_known=prior["mu_known"],
            alpha0=prior["alpha0"],
            beta0=prior["beta0"],
        )
        if "state" in d:
            s = d["state"]
            obj.alpha = s["alpha"]
            obj.beta = s["beta"]
        return obj


# =============================================================================
# Geometric with Beta Prior
# =============================================================================


class GeometricBeta(ExponentialFamilyModel):
    """Geometric likelihood with Beta conjugate prior on p.

    Models the number of failures before the first success.
    Detects changes in success probability.

    Prior hyperparameters:
        alpha0 : pseudo-count of successes
        beta0  : pseudo-count of failures

    Predictive distribution: Beta-Geometric.
    """

    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0):
        self.alpha = alpha0
        self.beta = beta0
        self._alpha0 = alpha0
        self._beta0 = beta0

    def sufficient_statistic(self, x: np.ndarray) -> dict:
        return {"sum_x": int(x), "n": 1}

    def log_base_measure(self, x: np.ndarray) -> float:
        return 0.0

    def log_normalizer(self, *, alpha, beta) -> float:
        return gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)

    def _get_params(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    def _updated_params(self, stat: dict) -> dict:
        return {
            "alpha": self.alpha + stat["n"],
            "beta": self.beta + stat["sum_x"],
        }

    def _set_params(self, params: dict) -> None:
        self.alpha = params["alpha"]
        self.beta = params["beta"]

    def predictive_mean_var(self) -> tuple[float, float]:
        """Beta-Geometric predictive mean and variance."""
        mean = self.beta / (self.alpha - 1.0) if self.alpha > 1.0 else np.inf
        var = (
            self.alpha
            * self.beta
            * (self.alpha + self.beta - 1.0)
            / ((self.alpha - 1.0) ** 2 * (self.alpha - 2.0))
            if self.alpha > 2.0
            else np.inf
        )
        return (mean, var)

    def to_dict(self) -> dict:
        return {
            "type": "GeometricBeta",
            "prior": {"alpha0": self._alpha0, "beta0": self._beta0},
            "state": {"alpha": self.alpha, "beta": self.beta},
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeometricBeta:
        prior = d["prior"]
        obj = cls(alpha0=prior["alpha0"], beta0=prior["beta0"])
        if "state" in d:
            s = d["state"]
            obj.alpha = s["alpha"]
            obj.beta = s["beta"]
        return obj


# =============================================================================
# Multinomial with Dirichlet Prior (Categorical)
# =============================================================================


class MultinomialDirichlet(ExponentialFamilyModel):
    """Categorical/Multinomial likelihood with Dirichlet conjugate prior.

    Inherently multivariate (K categories). For binary data (K=2),
    BernoulliBeta is a simpler alternative.

    Prior hyperparameters:
        alpha0 : concentration parameters, shape (K,)

    Observations are one-hot vectors (single categorical draw) or
    count vectors (multiple draws from the same distribution).

    Predictive distribution: Dirichlet-Multinomial (Polya).
    """

    def __init__(self, alpha0: np.ndarray | list[float]):
        self._alpha0 = np.asarray(alpha0, dtype=float)
        self.alpha = self._alpha0.copy()
        self.K = len(self._alpha0)

    def sufficient_statistic(self, x: np.ndarray) -> dict:
        return {"counts": np.asarray(x, dtype=float)}

    def log_base_measure(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        n = np.sum(x)
        return float(gammaln(n + 1) - np.sum(gammaln(x + 1)))

    def log_normalizer(self, *, alpha) -> float:
        return float(np.sum(gammaln(alpha)) - gammaln(np.sum(alpha)))

    def _get_params(self) -> dict:
        return {"alpha": self.alpha}

    def _updated_params(self, stat: dict) -> dict:
        return {"alpha": self.alpha + stat["counts"]}

    def _set_params(self, params: dict) -> None:
        self.alpha = params["alpha"]

    def predictive_mean_var(self) -> tuple[np.ndarray, np.ndarray]:
        """Dirichlet-Multinomial predictive mean and covariance."""
        alpha_sum = np.sum(self.alpha)
        p = self.alpha / alpha_sum
        cov = (np.diag(p) - np.outer(p, p)) / (alpha_sum + 1.0)
        return (p, cov)

    def to_dict(self) -> dict:
        return {
            "type": "MultinomialDirichlet",
            "prior": {"alpha0": self._alpha0.tolist()},
            "state": {"alpha": self.alpha.tolist()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> MultinomialDirichlet:
        prior = d["prior"]
        obj = cls(alpha0=np.array(prior["alpha0"]))
        if "state" in d:
            obj.alpha = np.array(d["state"]["alpha"])
        return obj


# =============================================================================
# Multivariate Normal with Known Covariance (unknown mean)
# =============================================================================


class MultivariateNormalKnownCov(ExponentialFamilyModel):
    """Multivariate Normal with known covariance, Normal prior on mean.

    Detects changes in the mean vector while assuming constant covariance.
    Predictive distribution is Normal (not Student-t), giving sharper
    detection when the covariance is truly known.

    Parameters:
        dim    : dimensionality
        mu0    : prior mean vector (D,)
        Sigma0 : prior covariance on the mean (D, D)
        Sigma  : known observation covariance (D, D)

    Working parameters: Lambda (precision matrix), mu (posterior mean).
    """

    def __init__(
        self,
        dim: int,
        mu0: np.ndarray = None,
        Sigma0: np.ndarray = None,
        Sigma: np.ndarray = None,
    ):
        self.dim = dim
        self._mu0 = mu0 if mu0 is not None else np.zeros(dim)
        self._Sigma0 = Sigma0 if Sigma0 is not None else np.eye(dim)
        self._Sigma = Sigma if Sigma is not None else np.eye(dim)

        # Precompute observation precision and its log determinant
        self._Sigma_inv = np.linalg.inv(self._Sigma)
        _, self._log_det_Sigma = np.linalg.slogdet(self._Sigma)

        # Working parameters: posterior precision and mean
        self.Lambda = np.linalg.inv(self._Sigma0)
        self.mu = self._mu0.copy()

    def sufficient_statistic(self, x: np.ndarray) -> dict:
        return {"x": np.asarray(x, dtype=float), "n": 1}

    def log_base_measure(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        D = self.dim
        return (
            -0.5 * D * np.log(2 * np.pi)
            - 0.5 * self._log_det_Sigma
            - 0.5 * float(x @ self._Sigma_inv @ x)
        )

    def log_normalizer(self, *, Lambda, mu) -> float:
        sign, logdet = np.linalg.slogdet(Lambda)
        if sign <= 0:
            return -np.inf
        return -0.5 * logdet + 0.5 * float(mu @ Lambda @ mu)

    def _get_params(self) -> dict:
        return {"Lambda": self.Lambda, "mu": self.mu}

    def _updated_params(self, stat: dict) -> dict:
        Lambda_new = self.Lambda + stat["n"] * self._Sigma_inv
        mu_new = np.linalg.solve(
            Lambda_new, self.Lambda @ self.mu + self._Sigma_inv @ stat["x"]
        )
        return {"Lambda": Lambda_new, "mu": mu_new}

    def _set_params(self, params: dict) -> None:
        self.Lambda = params["Lambda"]
        self.mu = params["mu"]

    def predictive_mean_var(self) -> tuple[np.ndarray, np.ndarray]:
        """Normal predictive mean and covariance."""
        cov = self._Sigma + np.linalg.inv(self.Lambda)
        return (self.mu.copy(), cov)

    def to_dict(self) -> dict:
        return {
            "type": "MultivariateNormalKnownCov",
            "prior": {
                "dim": self.dim,
                "mu0": self._mu0.tolist(),
                "Sigma0": self._Sigma0.tolist(),
                "Sigma": self._Sigma.tolist(),
            },
            "state": {
                "Lambda": self.Lambda.tolist(),
                "mu": self.mu.tolist(),
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> MultivariateNormalKnownCov:
        prior = d["prior"]
        obj = cls(
            dim=prior["dim"],
            mu0=np.array(prior["mu0"]),
            Sigma0=np.array(prior["Sigma0"]),
            Sigma=np.array(prior["Sigma"]),
        )
        if "state" in d:
            s = d["state"]
            obj.Lambda = np.array(s["Lambda"])
            obj.mu = np.array(s["mu"])
        return obj


# =============================================================================
# Multivariate Normal with Known Mean (unknown covariance)
# =============================================================================


class MultivariateNormalKnownMean(ExponentialFamilyModel):
    """Multivariate Normal with known mean, InverseWishart prior on covariance.

    Detects changes in the covariance structure while assuming constant mean.
    Useful for volatility regime detection in multivariate time series.

    Parameters:
        dim      : dimensionality
        mu_known : known mean vector (D,)
        nu0      : InverseWishart degrees of freedom (must be > D - 1)
        Psi0     : InverseWishart scale matrix (D, D)

    Predictive distribution: Multivariate Student-t.
    """

    def __init__(
        self,
        dim: int,
        mu_known: np.ndarray = None,
        nu0: float | None = None,
        Psi0: np.ndarray = None,
    ):
        self.dim = dim
        self._mu_known = mu_known if mu_known is not None else np.zeros(dim)
        self._nu0 = nu0 if nu0 is not None else float(dim) + 1.0
        self._Psi0 = Psi0 if Psi0 is not None else np.eye(dim)

        if self._nu0 <= self.dim - 1:
            raise ValueError(f"nu0 must be > dim - 1 = {self.dim - 1}, got {self._nu0}")

        self.nu = self._nu0
        self.Psi = self._Psi0.copy()

    def sufficient_statistic(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float)
        d = x - self._mu_known
        return {"scatter": np.outer(d, d), "n": 1}

    def log_base_measure(self, x: np.ndarray) -> float:
        return -0.5 * self.dim * np.log(2 * np.pi)

    def log_normalizer(self, *, nu, Psi) -> float:
        D = self.dim
        sign, logdet = np.linalg.slogdet(Psi)
        if sign <= 0:
            return -np.inf
        return multigammaln(nu / 2.0, D) - (nu / 2.0) * logdet

    def _get_params(self) -> dict:
        return {"nu": self.nu, "Psi": self.Psi}

    def _updated_params(self, stat: dict) -> dict:
        return {
            "nu": self.nu + stat["n"],
            "Psi": self.Psi + stat["scatter"],
        }

    def _set_params(self, params: dict) -> None:
        self.nu = params["nu"]
        self.Psi = params["Psi"].copy()

    def predictive_mean_var(self) -> tuple[np.ndarray, np.ndarray]:
        """Multivariate Student-t predictive mean and covariance."""
        D = self.dim
        df = self.nu - D + 1.0
        shape = self.Psi / df
        cov = shape * df / (df - 2.0) if df > 2.0 else np.full((D, D), np.inf)
        return (self._mu_known.copy(), cov)

    def to_dict(self) -> dict:
        return {
            "type": "MultivariateNormalKnownMean",
            "prior": {
                "dim": self.dim,
                "mu_known": self._mu_known.tolist(),
                "nu0": self._nu0,
                "Psi0": self._Psi0.tolist(),
            },
            "state": {
                "nu": self.nu,
                "Psi": self.Psi.tolist(),
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> MultivariateNormalKnownMean:
        prior = d["prior"]
        obj = cls(
            dim=prior["dim"],
            mu_known=np.array(prior["mu_known"]),
            nu0=prior["nu0"],
            Psi0=np.array(prior["Psi0"]),
        )
        if "state" in d:
            s = d["state"]
            obj.nu = s["nu"]
            obj.Psi = np.array(s["Psi"])
        return obj


# =============================================================================
# Student-t with NIG Prior (univariate, persistent heavy tails)
# =============================================================================


class StudentTFixedDf(ObservationModel):
    """Univariate Student-t with fixed (known) degrees of freedom.

    Models data with persistent heavy tails by writing the Student-t(nu)
    distribution as a scale mixture of normals:

        t(x | nu, mu, sigma^2) = int N(x | mu, sigma^2/w) Gamma(w | nu/2, nu/2) dw

    Unlike ``UnivariateNormalNIG`` whose predictive df grows with sample size
    (tails lighten toward Gaussian), this model fixes df = nu forever.
    Outliers are automatically downweighted via the posterior expected
    mixing weight E[w | x].

    For unknown df, see ``StudentTGridDf`` which learns nu from data.

    Parameters
    ----------
    nu : float
        Fixed Student-t degrees of freedom (controls tail heaviness).
        Lower values = heavier tails. Must be > 0.
    mu0 : float
        NIG prior mean.
    kappa0 : float
        NIG pseudo-observations for the mean.
    alpha0 : float
        NIG shape for inverse-gamma on variance.
    beta0 : float
        NIG scale for inverse-gamma on variance.
    """

    def __init__(
        self,
        nu: float = 4.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ):
        if nu <= 0:
            raise ValueError(f"nu must be > 0, got {nu}")
        self.nu = nu

        self.mu = mu0
        self.kappa = kappa0
        self.alpha = alpha0
        self.beta = beta0

        self._mu0 = mu0
        self._kappa0 = kappa0
        self._alpha0 = alpha0
        self._beta0 = beta0

    def log_predictive(self, x: np.ndarray) -> float:
        """Exact Student-t predictive log-density with fixed df = nu."""
        x = float(x)
        nu = self.nu
        scale = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa)
        z2 = (x - self.mu) ** 2 / (nu * scale)

        return (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi * scale)
            - ((nu + 1.0) / 2.0) * np.log(1.0 + z2)
        )

    def update(self, x: np.ndarray) -> None:
        """Weighted NIG update — outliers are automatically downweighted."""
        x = float(x)
        nu = self.nu

        # Current variance estimate
        sigma2_hat = self.beta / self.alpha
        z2 = (x - self.mu) ** 2 / sigma2_hat

        # Posterior expected mixing weight (downweights outliers)
        w_hat = (nu + 1.0) / (nu + z2)

        # Weighted NIG update
        kappa_new = self.kappa + w_hat
        mu_new = (self.kappa * self.mu + w_hat * x) / kappa_new
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + 0.5 * w_hat * (x - self.mu) ** 2 * self.kappa / kappa_new

        self.mu = mu_new
        self.kappa = kappa_new
        self.alpha = alpha_new
        self.beta = beta_new

    def predictive_mean_var(self) -> tuple[float, float]:
        """Student-t predictive mean and variance with fixed df = nu."""
        nu = self.nu
        scale = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa)
        mean = self.mu
        var = scale * nu / (nu - 2.0) if nu > 2.0 else np.inf
        return (mean, var)

    def to_dict(self) -> dict:
        return {
            "type": "StudentTFixedDf",
            "prior": {
                "nu": self.nu,
                "mu0": self._mu0,
                "kappa0": self._kappa0,
                "alpha0": self._alpha0,
                "beta0": self._beta0,
            },
            "state": {
                "mu": self.mu,
                "kappa": self.kappa,
                "alpha": self.alpha,
                "beta": self.beta,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> StudentTFixedDf:
        prior = d["prior"]
        obj = cls(
            nu=prior["nu"],
            mu0=prior["mu0"],
            kappa0=prior["kappa0"],
            alpha0=prior["alpha0"],
            beta0=prior["beta0"],
        )
        if "state" in d:
            s = d["state"]
            obj.mu = s["mu"]
            obj.kappa = s["kappa"]
            obj.alpha = s["alpha"]
            obj.beta = s["beta"]
        return obj


# =============================================================================
# Multivariate Student-t with Fixed Degrees of Freedom
# =============================================================================


class MultivariateStudentTFixedDf(ObservationModel):
    """Multivariate Student-t with fixed (known) degrees of freedom.

    The multivariate analogue of ``StudentTFixedDf``. Fixes the predictive
    degrees of freedom at ``nu`` regardless of sample size, unlike
    ``MultivariateNormalNIW`` whose df grows with data.

    Outliers (by Mahalanobis distance) are automatically downweighted
    via the posterior expected mixing weight.

    For unknown df, see ``MultivariateStudentTGridDf``.

    Parameters
    ----------
    dim : int
        Dimensionality.
    nu : float
        Fixed Student-t degrees of freedom (tail heaviness). Must be > 0.
    mu0 : np.ndarray, optional
        NIW prior mean vector (D,). Defaults to zeros.
    kappa0 : float
        NIW pseudo-observations for the mean.
    nu0 : float, optional
        NIW degrees of freedom for inverse-Wishart. Must be > D - 1.
        Defaults to D + 1.
    Psi0 : np.ndarray, optional
        NIW scale matrix (D, D). Defaults to identity.
    """

    def __init__(
        self,
        dim: int,
        nu: float = 4.0,
        mu0: np.ndarray = None,
        kappa0: float = 1.0,
        nu0: float | None = None,
        Psi0: np.ndarray = None,
    ):
        if nu <= 0:
            raise ValueError(f"nu must be > 0, got {nu}")
        self.dim = dim
        self.nu = nu

        self.mu0 = mu0 if mu0 is not None else np.zeros(dim)
        self.kappa0 = kappa0
        self.nu0 = nu0 if nu0 is not None else float(dim) + 1.0
        self.Psi0 = Psi0 if Psi0 is not None else np.eye(dim)

        if self.nu0 <= self.dim - 1:
            raise ValueError(f"nu0 must be > dim - 1 = {self.dim - 1}, got {self.nu0}")

        # Welford sufficient statistics
        self.n = 0.0
        self.x_bar = np.zeros(dim)
        self.S = np.zeros((dim, dim))

    def _posterior_params(self):
        """Derive NIW posterior parameters from sufficient statistics."""
        kappa_n = self.kappa0 + self.n
        nu_n = self.nu0 + self.n

        if self.n == 0:
            Psi_n = self.Psi0.copy()
        else:
            diff = self.x_bar - self.mu0
            Psi_n = (
                self.Psi0
                + self.S
                + (self.kappa0 * self.n / kappa_n) * np.outer(diff, diff)
            )

        mu_n = (self.kappa0 * self.mu0 + self.n * self.x_bar) / kappa_n
        return kappa_n, nu_n, Psi_n, mu_n

    def log_predictive(self, x: np.ndarray) -> float:
        """Exact multivariate Student-t predictive with fixed df = nu."""
        x = np.asarray(x, dtype=float)
        D = self.dim
        nu = self.nu

        kappa_n, _nu_n, Psi_n, mu_n = self._posterior_params()

        Sigma_pred = Psi_n * (kappa_n + 1.0) / (kappa_n * nu)
        delta = x - mu_n

        # Use solve instead of inv for numerical stability
        solved = np.linalg.solve(Sigma_pred, delta)
        maha = float(delta @ solved)

        sign, logdet = np.linalg.slogdet(Sigma_pred)
        if sign <= 0:
            return -np.inf

        return (
            gammaln((nu + D) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * D * np.log(nu * np.pi)
            - 0.5 * logdet
            - ((nu + D) / 2.0) * np.log(1.0 + maha / nu)
        )

    def update(self, x: np.ndarray) -> None:
        """Weighted Welford update — multivariate outliers downweighted."""
        x = np.asarray(x, dtype=float)
        D = self.dim
        nu = self.nu

        _kappa_n, nu_n, Psi_n, mu_n = self._posterior_params()

        # Posterior expected covariance
        denom = nu_n - D - 1.0
        Sigma_hat = Psi_n / denom if denom > 0 else Psi_n

        # Mahalanobis distance
        delta = x - mu_n
        solved = np.linalg.solve(Sigma_hat, delta)
        z2 = float(delta @ solved)

        # Posterior expected mixing weight
        w_hat = (nu + D) / (nu + z2)

        # Weighted Welford update
        n_new = self.n + w_hat
        dx = x - self.x_bar
        x_bar_new = self.x_bar + w_hat * dx / n_new
        S_new = self.S + w_hat * (self.n / n_new) * np.outer(dx, dx)

        self.n = n_new
        self.x_bar = x_bar_new
        self.S = S_new

    def predictive_mean_var(self) -> tuple[np.ndarray, np.ndarray]:
        """Multivariate Student-t predictive mean and covariance."""
        D = self.dim
        nu = self.nu

        kappa_n, _nu_n, Psi_n, mu_n = self._posterior_params()

        Sigma_pred = Psi_n * (kappa_n + 1.0) / (kappa_n * nu)
        cov = Sigma_pred * nu / (nu - 2.0) if nu > 2.0 else np.full((D, D), np.inf)
        return (mu_n, cov)

    def copy(self) -> MultivariateStudentTFixedDf:
        """Efficient copy — avoid deepcopy of numpy arrays."""
        new = MultivariateStudentTFixedDf.__new__(MultivariateStudentTFixedDf)
        new.dim = self.dim
        new.nu = self.nu
        new.mu0 = self.mu0
        new.kappa0 = self.kappa0
        new.nu0 = self.nu0
        new.Psi0 = self.Psi0
        new.n = self.n
        new.x_bar = self.x_bar.copy()
        new.S = self.S.copy()
        return new

    def to_dict(self) -> dict:
        return {
            "type": "MultivariateStudentTFixedDf",
            "prior": {
                "dim": self.dim,
                "nu": self.nu,
                "mu0": self.mu0.tolist(),
                "kappa0": self.kappa0,
                "nu0": self.nu0,
                "Psi0": self.Psi0.tolist(),
            },
            "state": {
                "n": self.n,
                "x_bar": self.x_bar.tolist(),
                "S": self.S.tolist(),
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> MultivariateStudentTFixedDf:
        prior = d["prior"]
        obj = cls(
            dim=prior["dim"],
            nu=prior["nu"],
            mu0=np.array(prior["mu0"]),
            kappa0=prior["kappa0"],
            nu0=prior["nu0"],
            Psi0=np.array(prior["Psi0"]),
        )
        if "state" in d:
            s = d["state"]
            obj.n = s["n"]
            obj.x_bar = np.array(s["x_bar"])
            obj.S = np.array(s["S"])
        return obj


# =============================================================================
# Student-t with Adaptive (Unknown) Degrees of Freedom — Base Class
# =============================================================================


class StudentTAdaptiveDf(ObservationModel, ABC):
    """Abstract base for Student-t models that learn df from data.

    Subclasses implement different estimation strategies for the
    degrees of freedom parameter nu:

    - ``StudentTGridDf``: discrete Bayesian model averaging over a
      grid of candidate nu values (exact within grid resolution)
    - ``StudentTOnlineEmDf``: online EM point estimate of nu
      (lightweight, approximate) — not yet implemented

    All subclasses share:
    - NIG prior on (mu, sigma^2) within each nu hypothesis
    - An ``estimated_nu`` property returning the current best estimate
    """

    @property
    @abstractmethod
    def estimated_nu(self) -> float:
        """Current best estimate of the degrees of freedom."""


class MultivariateStudentTAdaptiveDf(ObservationModel, ABC):
    """Abstract base for multivariate Student-t models that learn df.

    Subclasses implement different estimation strategies:

    - ``MultivariateStudentTGridDf``: discrete Bayesian model averaging
    - ``MultivariateStudentTOnlineEmDf``: online EM — not yet implemented
    """

    @property
    @abstractmethod
    def estimated_nu(self) -> float:
        """Current best estimate of the degrees of freedom."""


# =============================================================================
# Student-t with Grid-Based Df Estimation (Univariate)
# =============================================================================


class StudentTGridDf(StudentTAdaptiveDf):
    """Univariate Student-t with unknown df learned via grid model averaging.

    Maintains a discrete set of candidate nu values, each backed by its
    own ``StudentTFixedDf`` model. Posterior weights over the grid are
    updated via Bayes rule at each observation, giving a full posterior
    over nu.

    Parameters
    ----------
    nu_grid : list of float, optional
        Candidate df values. Default: ``[2.0, 3.0, 5.0, 10.0, 30.0]``,
        covering heavy-tailed (nu=2) through near-Gaussian (nu=30).
        Users may supply any list of positive floats to match their
        domain (e.g., a denser grid around suspected values).
    mu0, kappa0, alpha0, beta0 : float
        NIG prior hyperparameters (shared across all grid points).
    """

    def __init__(
        self,
        nu_grid: list[float] | None = None,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ):
        if nu_grid is None:
            nu_grid = [2.0, 3.0, 5.0, 10.0, 30.0]
        self._nu_grid = list(nu_grid)
        self.K = len(self._nu_grid)
        self._mu0 = mu0
        self._kappa0 = kappa0
        self._alpha0 = alpha0
        self._beta0 = beta0

        self._models = [
            StudentTFixedDf(nu=nu, mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0)
            for nu in self._nu_grid
        ]
        self._log_weights = np.full(self.K, -np.log(self.K))  # uniform

        # Cache for log_predictive values (consumed by update)
        self._cached_log_preds: np.ndarray | None = None

    @property
    def estimated_nu(self) -> float:
        """Posterior-weighted expected nu."""
        weights = np.exp(self._log_weights)
        return float(np.dot(weights, self._nu_grid))

    def log_predictive(self, x: np.ndarray) -> float:
        log_preds = np.array([m.log_predictive(x) for m in self._models])
        self._cached_log_preds = log_preds
        # logsumexp: log(sum(w_k * pred_k))
        return float(_logsumexp(self._log_weights + log_preds))

    def update(self, x: np.ndarray) -> None:
        # Update grid weights via Bayes rule
        if self._cached_log_preds is not None:
            log_preds = self._cached_log_preds
            self._cached_log_preds = None
        else:
            log_preds = np.array([m.log_predictive(x) for m in self._models])

        self._log_weights = self._log_weights + log_preds
        self._log_weights -= _logsumexp(self._log_weights)

        # Update each grid model
        for m in self._models:
            m.update(x)

    def predictive_mean_var(self) -> tuple[float, float]:
        weights = np.exp(self._log_weights)
        means = np.zeros(self.K)
        varis = np.zeros(self.K)
        for k in range(self.K):
            means[k], varis[k] = self._models[k].predictive_mean_var()

        finite = np.isfinite(varis) & np.isfinite(means)
        w = np.where(finite, weights, 0.0)
        w_sum = np.sum(w)
        if w_sum <= 0:
            return (np.nan, np.nan)
        w /= w_sum
        m_fin = np.where(finite, means, 0.0)
        v_fin = np.where(finite, varis, 0.0)
        mean = float(np.sum(w * m_fin))
        var = float(np.sum(w * (v_fin + m_fin**2)) - mean**2)
        return (mean, var)

    def copy(self) -> StudentTGridDf:
        new = StudentTGridDf.__new__(StudentTGridDf)
        new._nu_grid = self._nu_grid
        new.K = self.K
        new._mu0 = self._mu0
        new._kappa0 = self._kappa0
        new._alpha0 = self._alpha0
        new._beta0 = self._beta0
        new._models = [deepcopy(m) for m in self._models]
        new._log_weights = self._log_weights.copy()
        new._cached_log_preds = None
        return new

    def to_dict(self) -> dict:
        return {
            "type": "StudentTGridDf",
            "prior": {
                "nu_grid": self._nu_grid,
                "mu0": self._mu0,
                "kappa0": self._kappa0,
                "alpha0": self._alpha0,
                "beta0": self._beta0,
            },
            "state": {
                "log_weights": self._log_weights.tolist(),
                "models": [m.to_dict() for m in self._models],
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> StudentTGridDf:
        prior = d["prior"]
        obj = cls(
            nu_grid=prior["nu_grid"],
            mu0=prior["mu0"],
            kappa0=prior["kappa0"],
            alpha0=prior["alpha0"],
            beta0=prior["beta0"],
        )
        if "state" in d:
            s = d["state"]
            obj._log_weights = np.array(s["log_weights"])
            obj._models = [StudentTFixedDf.from_dict(md) for md in s["models"]]
        return obj


# =============================================================================
# Multivariate Student-t with Grid-Based Df Estimation
# =============================================================================


class MultivariateStudentTGridDf(MultivariateStudentTAdaptiveDf):
    """Multivariate Student-t with unknown df learned via grid model averaging.

    Maintains a discrete set of candidate nu values, each backed by its
    own ``MultivariateStudentTFixedDf`` model.

    Parameters
    ----------
    dim : int
        Dimensionality.
    nu_grid : list of float, optional
        Candidate df values. Default: ``[2.0, 3.0, 5.0, 10.0, 30.0]``.
        See ``StudentTGridDf`` for guidance on choosing the grid.
    mu0, kappa0, nu0, Psi0 :
        NIW prior hyperparameters (shared across all grid points).
    """

    def __init__(
        self,
        dim: int,
        nu_grid: list[float] | None = None,
        mu0: np.ndarray = None,
        kappa0: float = 1.0,
        nu0: float | None = None,
        Psi0: np.ndarray = None,
    ):
        if nu_grid is None:
            nu_grid = [2.0, 3.0, 5.0, 10.0, 30.0]
        self._nu_grid = list(nu_grid)
        self.K = len(self._nu_grid)
        self.dim = dim
        self._mu0 = mu0
        self._kappa0 = kappa0
        self._nu0 = nu0
        self._Psi0 = Psi0

        self._models = [
            MultivariateStudentTFixedDf(
                dim=dim, nu=nu, mu0=mu0, kappa0=kappa0, nu0=nu0, Psi0=Psi0
            )
            for nu in self._nu_grid
        ]
        self._log_weights = np.full(self.K, -np.log(self.K))
        self._cached_log_preds: np.ndarray | None = None

    @property
    def estimated_nu(self) -> float:
        weights = np.exp(self._log_weights)
        return float(np.dot(weights, self._nu_grid))

    def log_predictive(self, x: np.ndarray) -> float:
        log_preds = np.array([m.log_predictive(x) for m in self._models])
        self._cached_log_preds = log_preds
        return float(_logsumexp(self._log_weights + log_preds))

    def update(self, x: np.ndarray) -> None:
        if self._cached_log_preds is not None:
            log_preds = self._cached_log_preds
            self._cached_log_preds = None
        else:
            log_preds = np.array([m.log_predictive(x) for m in self._models])

        self._log_weights = self._log_weights + log_preds
        self._log_weights -= _logsumexp(self._log_weights)

        for m in self._models:
            m.update(x)

    def predictive_mean_var(self) -> tuple[np.ndarray, np.ndarray]:
        weights = np.exp(self._log_weights)
        D = self.dim

        # Mixture mean
        mean = np.zeros(D)
        for k in range(self.K):
            mk, _ = self._models[k].predictive_mean_var()
            if np.all(np.isfinite(mk)):
                mean += weights[k] * mk

        # Mixture covariance (law of total variance)
        cov = np.zeros((D, D))
        for k in range(self.K):
            mk, vk = self._models[k].predictive_mean_var()
            if np.all(np.isfinite(mk)) and np.all(np.isfinite(vk)):
                cov += weights[k] * (vk + np.outer(mk, mk))
        cov -= np.outer(mean, mean)

        return (mean, cov)

    def copy(self) -> MultivariateStudentTGridDf:
        new = MultivariateStudentTGridDf.__new__(MultivariateStudentTGridDf)
        new._nu_grid = self._nu_grid
        new.K = self.K
        new.dim = self.dim
        new._mu0 = self._mu0
        new._kappa0 = self._kappa0
        new._nu0 = self._nu0
        new._Psi0 = self._Psi0
        new._models = [m.copy() for m in self._models]
        new._log_weights = self._log_weights.copy()
        new._cached_log_preds = None
        return new

    def to_dict(self) -> dict:
        return {
            "type": "MultivariateStudentTGridDf",
            "prior": {
                "dim": self.dim,
                "nu_grid": self._nu_grid,
                "mu0": self._mu0.tolist() if self._mu0 is not None else None,
                "kappa0": self._kappa0,
                "nu0": self._nu0,
                "Psi0": self._Psi0.tolist() if self._Psi0 is not None else None,
            },
            "state": {
                "log_weights": self._log_weights.tolist(),
                "models": [m.to_dict() for m in self._models],
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> MultivariateStudentTGridDf:
        prior = d["prior"]
        obj = cls(
            dim=prior["dim"],
            nu_grid=prior["nu_grid"],
            mu0=np.array(prior["mu0"]) if prior["mu0"] is not None else None,
            kappa0=prior["kappa0"],
            nu0=prior["nu0"],
            Psi0=np.array(prior["Psi0"]) if prior["Psi0"] is not None else None,
        )
        if "state" in d:
            s = d["state"]
            obj._log_weights = np.array(s["log_weights"])
            obj._models = [
                MultivariateStudentTFixedDf.from_dict(md) for md in s["models"]
            ]
        return obj


# =============================================================================
# Student-t with Online EM Df Estimation — TODO
# =============================================================================


class StudentTOnlineEmDf(StudentTAdaptiveDf):
    """Univariate Student-t with nu estimated via online EM.

    Maintains a single point estimate of nu, updated each step via
    the EM fixed-point iteration for the Student-t distribution.
    Lighter than ``StudentTGridDf`` but gives a point estimate, not
    a full posterior over nu.

    .. note:: Not yet implemented.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "StudentTOnlineEmDf is planned but not yet implemented. "
            "Use StudentTGridDf for adaptive df estimation."
        )

    @property
    def estimated_nu(self) -> float:
        raise NotImplementedError

    def log_predictive(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def update(self, x: np.ndarray) -> None:
        raise NotImplementedError


class MultivariateStudentTOnlineEmDf(MultivariateStudentTAdaptiveDf):
    """Multivariate Student-t with nu estimated via online EM.

    .. note:: Not yet implemented.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MultivariateStudentTOnlineEmDf is planned but not yet implemented. "
            "Use MultivariateStudentTGridDf for adaptive df estimation."
        )

    @property
    def estimated_nu(self) -> float:
        raise NotImplementedError

    def log_predictive(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def update(self, x: np.ndarray) -> None:
        raise NotImplementedError


# =============================================================================
# Numerics
# =============================================================================


def _logsumexp(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    a_max = np.max(a)
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


# =============================================================================
# Model Registry
# =============================================================================

_MODEL_REGISTRY = {
    "UnivariateNormalNIG": UnivariateNormalNIG,
    "MultivariateNormalNIW": MultivariateNormalNIW,
    "PoissonGamma": PoissonGamma,
    "BernoulliBeta": BernoulliBeta,
    "ExponentialGamma": ExponentialGamma,
    "NormalKnownVariance": NormalKnownVariance,
    "NormalKnownMean": NormalKnownMean,
    "GeometricBeta": GeometricBeta,
    "MultinomialDirichlet": MultinomialDirichlet,
    "MultivariateNormalKnownCov": MultivariateNormalKnownCov,
    "MultivariateNormalKnownMean": MultivariateNormalKnownMean,
    "StudentTFixedDf": StudentTFixedDf,
    "MultivariateStudentTFixedDf": MultivariateStudentTFixedDf,
    "StudentTGridDf": StudentTGridDf,
    "MultivariateStudentTGridDf": MultivariateStudentTGridDf,
}


def model_from_dict(d: dict):
    """Reconstruct an observation model from its dict representation."""
    cls = _MODEL_REGISTRY[d["type"]]
    return cls.from_dict(d)
