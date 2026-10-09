"""Shared prior learning and per-problem parameter fitting for regularized frameworks."""

from collections.abc import Callable
from dataclasses import asdict
from math import inf

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from discovery_execution.frameworks.base.base_framework import BaseFramework
from discovery_execution.frameworks.models import BaseParameters, Observation, PriorFit
from discovery_execution.frameworks.utils.constants import (
    PRIOR_FUNCTION_TOLERANCE,
    PRIOR_GRADIENT_TOLERANCE,
    PRIOR_LOG_SHAPE_BOUNDS,
    PRIOR_MAX_ITERATIONS,
    PRIOR_MAX_LINE_SEARCH_STEPS,
    PRIOR_NLL_TOLERANCE,
    PRIOR_NUM_INITIALIZATIONS,
    PRIOR_OPTIMIZER,
)


class BaseRegularized[Parameters: BaseParameters, Prior: BaseParameters](BaseFramework[Parameters]):
    """Implement _fit_prior, _fit_parameters, and _success_probability; use _optimize_prior for repeated optimization. fit preserves the last successful state, and get_parameters includes the prior and convergence diagnostics."""

    def __init__(self) -> None:
        """Keep the shared prior alongside the inherited per-problem parameters."""
        super().__init__()
        self.prior: Prior | None = None
        self.prior_fits: list[PriorFit] = []

    def fit(self, observations: list[Observation]) -> None:
        """Learn one prior across all input problems, then update each problem using its own counts."""
        self._validate_observations(observations)
        previous_prior, previous_fits = self.prior, self.prior_fits
        try:
            self.prior = self._fit_prior(observations)
            super().fit(observations)
        except BaseException:
            # BaseFramework preserves the previous parameters; restore their prior and diagnostics too.
            self.prior, self.prior_fits = previous_prior, previous_fits
            raise

    def get_parameters(self, dataset: str) -> dict[str, object]:
        """Include the shared prior and convergence diagnostics alongside this dataset's parameters."""
        if self.prior is None:
            raise RuntimeError("fit the framework before reading its prior")
        fitted = super().get_parameters(dataset)
        fitted["prior"] = asdict(self.prior)
        fitted["optimization"] = {
            "starts": [asdict(fit) for fit in self.prior_fits],
            "best_nll": min(fit.nll for fit in self.prior_fits if fit.nll is not None),
            "best_nll_tolerance": PRIOR_NLL_TOLERANCE,
            "starts_reaching_best": sum(fit.reached_best for fit in self.prior_fits),
        }
        return fitted

    def _optimize_prior(
        self, objective: Callable[[NDArray[np.float64]], float | list[float | NDArray[np.float64]]],
        initializations: list[list[float]], use_gradient: bool,
    ) -> NDArray[np.float64]:
        """Fit from ten positive starts and report NLL agreement; objective returns a gradient too when use_gradient is true."""
        if len(initializations) != PRIOR_NUM_INITIALIZATIONS or any(initializations.count(initial) > 1 for initial in initializations):
            raise ValueError(f"prior fitting requires {PRIOR_NUM_INITIALIZATIONS} distinct initializations")
        fits = []
        for initial in initializations:
            fits.append(minimize(
                objective, np.log(initial), method=PRIOR_OPTIMIZER, jac=use_gradient,
                bounds=[PRIOR_LOG_SHAPE_BOUNDS] * len(initial),
                options={
                    "maxiter": PRIOR_MAX_ITERATIONS, "maxls": PRIOR_MAX_LINE_SEARCH_STEPS,
                    "ftol": PRIOR_FUNCTION_TOLERANCE, "gtol": PRIOR_GRADIENT_TOLERANCE,
                },
            ))

        # Reject an unsuccessful best fit instead of silently accepting a worse converged fit.
        best = min(fits, key=lambda fit: fit.fun if np.isfinite(fit.fun) else inf)
        if not best.success or not np.isfinite(best.fun) or not np.all(np.isfinite(best.x)):
            raise RuntimeError(f"{self.name} prior optimization failed: {best.message}")

        diagnostics = []
        for initial, fit in zip(initializations, fits):
            nll = float(fit.fun) if np.isfinite(fit.fun) else None
            gap = nll - float(best.fun) if nll is not None else None
            success = bool(fit.success and nll is not None and np.all(np.isfinite(fit.x)))
            diagnostics.append(PriorFit(
                initialization=initial, success=success, nll=nll, nll_gap=gap,
                reached_best=success and gap is not None and abs(gap) <= PRIOR_NLL_TOLERANCE,
                message=str(fit.message),
            ))
        self.prior_fits = diagnostics
        return np.exp(best.x)

    def _fit_prior(self, observations: list[Observation]) -> Prior:
        """Subclasses estimate shared prior parameters from all selected problems."""
        raise NotImplementedError
