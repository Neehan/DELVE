"""Simple geometric predictions using a finite-sample estimator."""

from math import comb

from discovery_execution.frameworks.base.base_framework import BaseFramework
from discovery_execution.frameworks.models import Observation, SGParameters
from discovery_execution.frameworks.utils.constants import SG


class SimpleGeometric(BaseFramework[SGParameters]):
    """Use fit/predict with the unbiased pass@a estimator: 1 - C(T-S, a)/C(T, a), where T is short trials, S is successes, and a=n*k is total attempts."""

    name = SG

    def _fit_parameters(self, observation: Observation) -> SGParameters:
        """Keep the measured successes and trials needed by the finite-sample estimator."""
        return SGParameters(
            successes=observation.short_successes,
            trials=observation.short_attempts,
        )

    def _success_probability(self, params: SGParameters, n: int, k: int) -> float:
        """Estimate pass@(n*k) by counting subsets of observed trials with at least one success."""
        # SG treats n trajectories of k blocks as n*k independent success opportunities.
        attempts = n * k
        if params.trials < attempts:
            raise ValueError("SG requires at least n*k short observations per problem")

        failed_trials = params.trials - params.successes
        # Their ratio is the fraction of attempt-sized subsets containing only failures.
        unsuccessful_combinations = comb(failed_trials, attempts)
        total_combinations = comb(params.trials, attempts)
        return 1 - unsuccessful_combinations / total_combinations
