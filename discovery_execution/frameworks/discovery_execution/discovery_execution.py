"""Discovery-execution estimates from short successes and oracle completion times."""

from discovery_execution.frameworks.base.base_framework import BaseFramework
from discovery_execution.frameworks.models import DEParameters, Observation
from discovery_execution.frameworks.utils.constants import DE


class DiscoveryExecution(BaseFramework[DEParameters]):
    """Fit per-problem discovery and execution parameters and predict their convolution."""

    name = DE

    def _fit_parameters(self, observation: Observation) -> DEParameters:
        """Estimate constrained parameters, pooling first-block evidence at the boundary."""
        execution_curve = []
        for successes in observation.oracle_successes:
            execution_curve.append(successes / observation.oracle_attempts)

        short_rate = observation.short_successes / observation.short_attempts
        epsilon_first = execution_curve[0]
        adjusted_curve = execution_curve

        # problem never observed to be solved in the first block
        is_unidentified = (
            observation.short_successes == 0 and observation.oracle_successes[0] == 0
        )
        if is_unidentified:
            alpha = 0.0

        # normal case: execution exceeds short rate
        elif short_rate <= epsilon_first:
            alpha = short_rate / epsilon_first

        # edge case: short rate exceeds execution. Could happen since execution has 3 seeds only
        else:
            # Pool first-block evidence when unconstrained discovery would exceed one.
            alpha = 1.0
            pooled_successes = (
                observation.short_successes + observation.oracle_successes[0]
            )
            pooled_attempts = observation.short_attempts + observation.oracle_attempts
            first_execution = pooled_successes / pooled_attempts

            # Preserve the measured tail proportions after adjusting the first block.
            adjusted_curve = []
            for probability in execution_curve:
                # Among runs unsolved at 1x, the fraction solved by this checkpoint.
                later_success_fraction = (probability - epsilon_first) / (
                    1 - epsilon_first
                )
                # Apply that fraction to the runs remaining after the updated first-block success.
                adjusted = (
                    first_execution + (1 - first_execution) * later_success_fraction
                )
                adjusted_curve.append(adjusted)

        return DEParameters(
            alpha=alpha,
            epsilon=adjusted_curve,
            at_boundary=short_rate > epsilon_first,
            is_unidentified=is_unidentified,
        )

    def _success_probability(self, params: DEParameters, n: int, k: int) -> float:
        """Combine geometric discovery with execution, then calculate any-of-n success."""
        success = 0.0
        for step in range(k):
            # First discovery after exactly `step` blocks without discovery.
            discovery_probability = params.alpha * (1 - params.alpha) ** step
            # Execution has k-step blocks, including the discovery block; epsilon uses zero-based indices.
            execution_probability = params.epsilon[k - step - 1]
            # Sum success probabilities over mutually exclusive discovery times for one trajectory.
            success += discovery_probability * execution_probability

        # At least one success across n independent trajectories: one minus all n failing.
        return 1 - (1 - success) ** n
