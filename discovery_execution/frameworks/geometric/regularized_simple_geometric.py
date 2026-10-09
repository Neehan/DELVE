"""Empirical-Bayes SG with a shared Beta prior and posterior predictive allocation probabilities."""

from functools import partial

import numpy as np

from discovery_execution.frameworks.base.base_regularized import BaseRegularized
from discovery_execution.frameworks.models import BetaParameters, Observation, RSGParameters
from discovery_execution.frameworks.utils.constants import RSG, RSG_INITIALIZATIONS
from discovery_execution.frameworks.utils.probabilities import beta_binomial_nll


class RegularizedSimpleGeometric(BaseRegularized[RSGParameters, BetaParameters]):
    """Use inherited fit/predict to learn a Beta prior from short runs and integrate geometric success over each problem's posterior."""

    name = RSG

    def _fit_prior(self, observations: list[Observation]) -> BetaParameters:
        """Maximize the Beta-binomial marginal likelihood across all selected problems."""
        successes = np.array([obs.short_successes for obs in observations], dtype=float)
        attempts = np.array([obs.short_attempts for obs in observations], dtype=float)
        initializations = []
        for mean, concentration in RSG_INITIALIZATIONS:
            initializations.append([mean * concentration, (1 - mean) * concentration])
        objective = partial(beta_binomial_nll, successes=successes, attempts=attempts)
        a, b = self._optimize_prior(objective, initializations, use_gradient=True)
        return BetaParameters(a=float(a), b=float(b))

    def _fit_parameters(self, observation: Observation) -> RSGParameters:
        """Keep this problem's observed counts and the learned prior as explicit prediction inputs."""
        if self.prior is None:
            raise ValueError("R-SG requires a fitted prior")
        return RSGParameters(
            successes=observation.short_successes,
            trials=observation.short_attempts,
            a=self.prior.a,
            b=self.prior.b,
        )

    def _success_probability(self, params: RSGParameters, n: int, k: int) -> float:
        """Integrate 1-(1-q)^(n*k) over the Beta posterior, using its exact failure moment."""
        # Update the prior with this problem's observed successes and failures.
        posterior_a = params.a + params.successes
        posterior_b = params.b + (params.trials - params.successes)
        failure_probability = 1.0
        for attempt in range(n * k):
            # The product is E[(1-q)^(n*k)]; using the posterior mean of q would lose uncertainty.
            failure_probability *= (posterior_b + attempt) / (posterior_a + posterior_b + attempt)
        return 1 - failure_probability
