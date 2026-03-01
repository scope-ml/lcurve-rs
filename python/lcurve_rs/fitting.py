"""
Parameter estimation for lcurve models.

Provides a generic :class:`Fitter` that wraps lcurve_rs.Model and drives
nested-sampling (UltraNest, dynesty) or MCMC (emcee) backends.  Backend
libraries are imported lazily and are optional dependencies.

Example
-------
>>> from lcurve_rs.fitting import Fitter, Prior
>>> import lcurve_rs
>>> model = lcurve_rs.Model("model.dat")
>>> fitter = Fitter(model, data="lightcurve.dat")
>>> result = fitter.run_emcee(nwalkers=32, nsteps=5000, burn=1000)
>>> print(result.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

@dataclass
class Prior:
    """Prior specification for a single parameter.

    Attributes
    ----------
    name : str
        Parameter name.
    low, high : float
        Hard bounds.
    transform : callable or None
        Maps unit cube [0, 1] -> parameter value (nested sampling).
    log_prob : callable or None
        Returns log-probability for a given value (MCMC).
    """

    name: str
    low: float
    high: float
    transform: Optional[Callable[[float], float]] = None
    log_prob: Optional[Callable[[float], float]] = None

    @staticmethod
    def uniform(name: str, low: float, high: float) -> "Prior":
        """Uniform prior on [low, high]."""
        span = high - low

        def _transform(u: float) -> float:
            return low + u * span

        def _log_prob(x: float) -> float:
            if low <= x <= high:
                return -math.log(span)
            return -math.inf

        return Prior(name=name, low=low, high=high,
                     transform=_transform, log_prob=_log_prob)

    @staticmethod
    def gaussian(name: str, mean: float, sigma: float,
                 low: float, high: float) -> "Prior":
        """Truncated Gaussian prior clipped to [low, high]."""
        from scipy.stats import truncnorm
        a = (low - mean) / sigma
        b = (high - mean) / sigma
        dist = truncnorm(a, b, loc=mean, scale=sigma)

        def _transform(u: float) -> float:
            return float(dist.ppf(u))

        def _log_prob(x: float) -> float:
            if low <= x <= high:
                return float(dist.logpdf(x))
            return -math.inf

        return Prior(name=name, low=low, high=high,
                     transform=_transform, log_prob=_log_prob)


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Uniform container for results from any sampling backend.

    Attributes
    ----------
    backend : str
        ``"ultranest"``, ``"dynesty"``, ``"emcee"``, or ``"eryn"``.
    param_names : list[str]
        Ordered parameter names.
    samples : np.ndarray
        Shape ``(n_samples, n_params)``.
    log_likelihood : np.ndarray
        Shape ``(n_samples,)``.
    weights : np.ndarray or None
        Sample weights (nested sampling only).
    log_evidence : float or None
        log(Z) (nested sampling only).
    log_evidence_err : float or None
        Uncertainty on log(Z).
    raw_result : Any
        The backend's native result object.
    """

    backend: str
    param_names: list
    samples: np.ndarray
    log_likelihood: np.ndarray
    weights: Optional[np.ndarray] = None
    log_evidence: Optional[float] = None
    log_evidence_err: Optional[float] = None
    raw_result: Any = field(default=None, repr=False)

    @property
    def median(self) -> dict:
        """Median value for each parameter."""
        if self.weights is not None:
            # weighted median via sorted cumulative weights
            out = {}
            for i, name in enumerate(self.param_names):
                col = self.samples[:, i]
                idx = np.argsort(col)
                cw = np.cumsum(self.weights[idx])
                cw /= cw[-1]
                out[name] = float(col[idx][np.searchsorted(cw, 0.5)])
            return out
        return {name: float(np.median(self.samples[:, i]))
                for i, name in enumerate(self.param_names)}

    @property
    def best_fit(self) -> dict:
        """Parameter values at the maximum likelihood sample."""
        idx = int(np.argmax(self.log_likelihood))
        return {name: float(self.samples[idx, i])
                for i, name in enumerate(self.param_names)}

    def summary(self, ci: float = 68.27) -> str:
        """Return a text summary with credible intervals.

        Parameters
        ----------
        ci : float
            Credible interval width in percent (default 68.27 ~ 1-sigma).
        """
        lo_q = (100.0 - ci) / 2.0
        hi_q = 100.0 - lo_q
        lines = []
        if self.log_evidence is not None:
            lines.append(
                f"log(Z) = {self.log_evidence:.2f}"
                f" +/- {self.log_evidence_err:.2f}"
            )
        lines.append(f"Backend: {self.backend}  |  Samples: {len(self.samples)}")
        lines.append("")
        lines.append(f"{'Parameter':>20s}  {'Median':>12s}  {'Lower':>12s}  {'Upper':>12s}")
        lines.append("-" * 62)
        for i, name in enumerate(self.param_names):
            col = self.samples[:, i]
            if self.weights is not None:
                # weighted percentiles
                idx = np.argsort(col)
                cw = np.cumsum(self.weights[idx])
                cw /= cw[-1]
                med = float(col[idx][np.searchsorted(cw, 0.5)])
                lo = float(col[idx][np.searchsorted(cw, lo_q / 100.0)])
                hi = float(col[idx][np.searchsorted(cw, hi_q / 100.0)])
            else:
                med = float(np.median(col))
                lo = float(np.percentile(col, lo_q))
                hi = float(np.percentile(col, hi_q))
            lines.append(f"{name:>20s}  {med:12.6g}  {lo:12.6g}  {hi:12.6g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Eryn prior adapter
# ---------------------------------------------------------------------------

class _ErynPriorAdapter:
    """Bridge an lcurve :class:`Prior` to the eryn distribution interface.

    Eryn's :class:`ProbDistContainer` expects distributions with
    ``logpdf(x)`` and ``rvs(size)`` methods that accept array inputs.
    It also sets ``use_cupy`` and ``return_gpu`` attributes on each
    distribution during ``__init__``.
    """

    def __init__(self, prior: Prior):
        self._prior = prior
        self.use_cupy = False
        self.return_gpu = False

    def logpdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        flat = x.ravel()
        out = np.array([self._prior.log_prob(float(v)) for v in flat])
        return out.reshape(x.shape)

    def rvs(self, size=1):
        if isinstance(size, int):
            size = (size,)
        u = np.random.uniform(0.0, 1.0, size)
        flat = u.ravel()
        out = np.array([self._prior.transform(float(v)) for v in flat])
        return out.reshape(size)


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

class Fitter:
    """Drive parameter estimation for an lcurve model.

    Parameters
    ----------
    model : lcurve_rs.Model
        The model to fit (a copy is made internally).
    data : str
        Path to an lcurve data file.
    scale : bool
        Whether to autoscale the model to the data (default True).
    priors : dict[str, Prior] or None
        Override priors keyed by parameter name.  Parameters not listed
        get a default uniform prior from the model's ``value +/- range``.
    params : list[str] or None
        Restrict fitting to these parameter names.  Default: all
        parameters with ``vary=True`` in the model file.
    """

    def __init__(
        self,
        model,
        data: str,
        scale: bool = True,
        priors: Optional[dict] = None,
        params: Optional[list] = None,
    ):
        self._model = model.copy()
        self._data = data
        self._scale = scale

        # Discover free parameters
        if params is not None:
            free_info = []
            for name in params:
                info = self._model.get_pparam(name)
                free_info.append({"name": name, **info})
        else:
            free_info = self._model.get_free_params()

        if not free_info:
            raise ValueError("No free parameters found. Set vary=True in your model file or pass params=.")

        self.param_names: list[str] = [p["name"] for p in free_info]
        self.ndim: int = len(self.param_names)

        # Build priors
        user_priors = priors or {}
        self.priors: list[Prior] = []
        for p in free_info:
            name = p["name"]
            if name in user_priors:
                self.priors.append(user_priors[name])
            else:
                lo = p["value"] - p["range"]
                hi = p["value"] + p["range"]
                self.priors.append(Prior.uniform(name, lo, hi))

    # -- Core functions used by all backends --

    def log_likelihood(self, theta) -> float:
        """Compute log-likelihood for parameter vector *theta*.

        Sets parameters on the internal model copy, computes the light
        curve against the data file, and returns ``-0.5 * chisq``.
        Unphysical parameter combinations that raise exceptions return
        ``-inf``.
        """
        try:
            self._model.set_params(self.param_names, list(theta))
            result = self._model.light_curve(data=self._data, scale=self._scale)
            ll = -0.5 * result.chisq
            if not math.isfinite(ll):
                return -math.inf
            return ll
        except Exception:
            return -math.inf

    def prior_transform(self, cube) -> np.ndarray:
        """Map unit cube [0,1]^d -> parameter space (nested sampling)."""
        out = np.empty(self.ndim)
        for i, prior in enumerate(self.priors):
            out[i] = prior.transform(cube[i])
        return out

    def log_prior(self, theta) -> float:
        """Sum of prior log-probabilities (MCMC)."""
        lp = 0.0
        for i, prior in enumerate(self.priors):
            val = prior.log_prob(theta[i])
            if not math.isfinite(val):
                return -math.inf
            lp += val
        return lp

    def log_probability(self, theta) -> float:
        """log_prior + log_likelihood (emcee)."""
        lp = self.log_prior(theta)
        if not math.isfinite(lp):
            return -math.inf
        return lp + self.log_likelihood(theta)

    # -- Batch (vectorised) core functions --

    def log_likelihood_batch(self, theta_batch: np.ndarray) -> np.ndarray:
        """Vectorised log-likelihood for an (N, ndim) array of parameter sets.

        Uses ``chisq_batch`` on the Rust side so the data file is read
        once and evaluations run in parallel via Rayon.
        """
        theta_batch = np.ascontiguousarray(theta_batch, dtype=np.float64)
        chisq = self._model.chisq_batch(
            self._data, self.param_names, theta_batch, self._scale,
        )
        ll = -0.5 * np.asarray(chisq, dtype=np.float64)
        ll[~np.isfinite(ll)] = -np.inf
        return ll

    def log_probability_batch(self, theta_batch: np.ndarray) -> np.ndarray:
        """Vectorised log_prior + log_likelihood for (N, ndim) array.

        Priors are evaluated per-row in Python (cheap).  Only rows that
        pass the prior are forwarded to the Rust batch evaluator.
        """
        theta_batch = np.atleast_2d(theta_batch)
        n = theta_batch.shape[0]
        lp = np.empty(n)
        for i in range(n):
            lp[i] = self.log_prior(theta_batch[i])

        result = np.full(n, -np.inf)
        finite_mask = np.isfinite(lp)
        if finite_mask.any():
            ll = self.log_likelihood_batch(theta_batch[finite_mask])
            result[finite_mask] = lp[finite_mask] + ll
        return result

    # -- Backend runners --

    def run_ultranest(self, vectorize: bool = True, **kwargs) -> FitResult:
        """Run UltraNest nested sampling.

        Parameters
        ----------
        vectorize : bool
            Use batch evaluation of live points (default True).

        All other keyword arguments are forwarded to
        ``ultranest.ReactiveNestedSampler.run()``.

        Returns
        -------
        FitResult
        """
        import ultranest

        if vectorize:
            log_fn = self.log_likelihood_batch
        else:
            log_fn = self.log_likelihood

        sampler = ultranest.ReactiveNestedSampler(
            self.param_names,
            log_fn,
            self.prior_transform,
            vectorized=vectorize,
        )
        raw = sampler.run(**kwargs)
        samples = np.array(raw["weighted_samples"]["points"])
        weights = np.array(raw["weighted_samples"]["weights"])
        logl = np.array(raw["weighted_samples"]["logl"])
        return FitResult(
            backend="ultranest",
            param_names=list(self.param_names),
            samples=samples,
            log_likelihood=logl,
            weights=weights,
            log_evidence=raw["logz"],
            log_evidence_err=raw["logzerr"],
            raw_result=raw,
        )

    def run_dynesty(self, nlive: int = 500, **kwargs) -> FitResult:
        """Run dynesty nested sampling.

        Parameters
        ----------
        nlive : int
            Number of live points (default 500).

        All other keyword arguments are forwarded to
        ``dynesty.NestedSampler.run_nested()``.

        Returns
        -------
        FitResult
        """
        import dynesty

        sampler = dynesty.NestedSampler(
            self.log_likelihood,
            self.prior_transform,
            self.ndim,
            nlive=nlive,
        )
        sampler.run_nested(**kwargs)
        raw = sampler.results

        from dynesty.utils import resample_equal
        weights = np.exp(raw.logwt - raw.logz[-1])
        weights /= weights.sum()

        return FitResult(
            backend="dynesty",
            param_names=list(self.param_names),
            samples=raw.samples,
            log_likelihood=raw.logl,
            weights=weights,
            log_evidence=float(raw.logz[-1]),
            log_evidence_err=float(raw.logzerr[-1]),
            raw_result=raw,
        )

    def run_eryn(
        self,
        nwalkers: int = 0,
        nsteps: int = 5000,
        burn: int = 1000,
        ntemps: int = 1,
        vectorize: bool = True,
        thin_by: int = 1,
        **kwargs,
    ) -> FitResult:
        """Run eryn parallel-tempered MCMC.

        Parameters
        ----------
        nwalkers : int
            Number of walkers (default: ``max(32, 2*ndim + 2)``).
        nsteps : int
            Total MCMC steps after burn-in (default 5000).
        burn : int
            Burn-in steps to discard (default 1000).
        ntemps : int
            Number of temperatures for parallel tempering (default 1).
        vectorize : bool
            Use batch evaluation across walkers (default True).
        thin_by : int
            Thinning factor for stored chain (default 1).

        All other keyword arguments are forwarded to
        ``eryn.ensemble.EnsembleSampler``.

        Returns
        -------
        FitResult
        """
        from eryn.ensemble import EnsembleSampler
        from eryn.prior import ProbDistContainer

        if nwalkers <= 0:
            nwalkers = max(32, 2 * self.ndim + 2)

        # Build eryn prior container from lcurve Prior objects
        eryn_priors = ProbDistContainer(
            {i: _ErynPriorAdapter(p) for i, p in enumerate(self.priors)}
        )

        # Likelihood: eryn handles priors internally, so pass only likelihood
        if vectorize:
            log_fn = self.log_likelihood_batch
        else:
            log_fn = self.log_likelihood

        # Tempering
        tempering_kwargs = kwargs.pop("tempering_kwargs", {})
        if ntemps > 1:
            tempering_kwargs.setdefault("ntemps", ntemps)

        # Initial coordinates: shape (ntemps, nwalkers, 1, ndim)
        actual_ntemps = tempering_kwargs.get("ntemps", ntemps) if ntemps > 1 else 1
        coords = eryn_priors.rvs(size=(actual_ntemps, nwalkers, 1))

        sampler = EnsembleSampler(
            nwalkers,
            self.ndim,
            log_fn,
            eryn_priors,
            tempering_kwargs=tempering_kwargs,
            vectorize=vectorize,
            **kwargs,
        )
        sampler.run_mcmc(coords, nsteps, burn=burn, progress=True, thin_by=thin_by)

        # Extract cold-chain results (temp_index=0)
        chain = sampler.get_chain(temp_index=0)["model_0"]
        # shape: (nsteps, nwalkers, 1, ndim) -> flatten
        chain = chain.reshape(-1, self.ndim)

        logl = sampler.get_log_like(temp_index=0)
        # shape: (nsteps, nwalkers) -> flatten
        logl = logl.reshape(-1)

        return FitResult(
            backend="eryn",
            param_names=list(self.param_names),
            samples=chain,
            log_likelihood=logl,
            raw_result=sampler,
        )

    def run_emcee(
        self,
        nwalkers: int = 0,
        nsteps: int = 5000,
        burn: int = 1000,
        vectorize: bool = True,
        **kwargs,
    ) -> FitResult:
        """Run emcee MCMC.

        Parameters
        ----------
        nwalkers : int
            Number of walkers (default: ``max(32, 2*ndim + 2)``).
        nsteps : int
            Total MCMC steps (default 5000).
        burn : int
            Burn-in steps to discard (default 1000).
        vectorize : bool
            Use batch evaluation across walkers (default True).
            This reads the data file once per step instead of once
            per walker and parallelises across walkers via Rayon.

        All other keyword arguments are forwarded to
        ``emcee.EnsembleSampler.run_mcmc()``.

        Returns
        -------
        FitResult
        """
        import emcee

        if nwalkers <= 0:
            nwalkers = max(32, 2 * self.ndim + 2)

        # Initialise walkers as a small ball around current values
        p0_center = np.array([
            prior.transform(0.5) for prior in self.priors
        ])
        # small perturbation within 1% of prior range
        spreads = np.array([
            0.01 * (prior.high - prior.low) for prior in self.priors
        ])
        rng = np.random.default_rng()
        p0 = p0_center + spreads * rng.standard_normal((nwalkers, self.ndim))
        # clip to bounds
        for i, prior in enumerate(self.priors):
            p0[:, i] = np.clip(p0[:, i], prior.low, prior.high)

        if vectorize:
            log_fn = self.log_probability_batch
        else:
            log_fn = self.log_probability

        sampler = emcee.EnsembleSampler(
            nwalkers, self.ndim, log_fn,
            vectorize=vectorize,
        )
        sampler.run_mcmc(p0, nsteps, **kwargs)

        chain = sampler.get_chain(discard=burn, flat=True)
        logl = sampler.get_log_prob(discard=burn, flat=True)

        return FitResult(
            backend="emcee",
            param_names=list(self.param_names),
            samples=chain,
            log_likelihood=logl,
            raw_result=sampler,
        )
