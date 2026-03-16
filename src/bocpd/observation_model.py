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
            # Psi should be positive definite; if not, return -inf
            return -np.inf

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
        return new


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
