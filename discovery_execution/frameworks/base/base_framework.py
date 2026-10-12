"""Shared parameter ownership, fitting, and prediction for allocation frameworks."""

from dataclasses import asdict
from math import isfinite

from discovery_execution.frameworks.models import BaseParameters, BasePrediction, Observation
from discovery_execution.frameworks.utils.constants import NUM_ARMS, PROBABILITY_TOLERANCE, TOTAL_BUDGET


class BaseFramework[Parameters: BaseParameters]:
    """fit stores params by dataset/problem; predict evaluates the fixed N and K values.

    Subclasses declare name and implement _fit_parameters and _success_probability; joint fits override fit and reuse _validate_observations. get_parameters prepares one dataset's fitted parameters for saving.
    """

    name: str

    def __init__(self) -> None:
        """Keep one parameter dictionary as this framework's fitted state."""
        self.params: dict[Observation, Parameters] = {}

    def fit(self, observations: list[Observation]) -> None:
        """Validate input data and replace fitted parameters only after the complete fit succeeds."""
        self._validate_observations(observations)
        params = {}
        for obs in observations:
            params[obs] = self._fit_parameters(obs)

        # replace old dict with new in one atomic swap
        self.params = params

    def get_parameters(self, dataset: str) -> dict[str, object]:
        """Prepare one dataset's fitted parameters for saving."""
        parameters = {}
        for obs, params in self.params.items():
            if obs.dataset == dataset:
                parameters[obs.problem_id] = asdict(params)
        return {"parameters": parameters}

    def predict(self) -> list[BasePrediction]:
        """Predict each problem for N=1,2,4 and measured depths within the configured total budget."""
        if not self.params:
            raise RuntimeError("fit the framework before predicting")

        predictions = []
        for n in NUM_ARMS:
            for k in range(1, TOTAL_BUDGET // n + 1):
                for observation, params in self.params.items():
                    probability = self._success_probability(params, n, k)
                    if not self._is_valid_probability(probability):
                        raise ArithmeticError("framework produced an invalid probability")

                    # Clamp rounding error only after rejecting invalid probabilities.
                    prediction = BasePrediction(
                        dataset=observation.dataset,
                        problem_id=observation.problem_id,
                        n=n,
                        k=k,
                        success_probability=min(1, max(0, probability)),
                    )
                    predictions.append(prediction)
        return predictions

    def _is_valid_probability(self, probability: float) -> bool:
        """Check that a probability is finite and within the allowed rounding tolerance."""
        return isfinite(probability) and -PROBABILITY_TOLERANCE <= probability <= 1 + PROBABILITY_TOLERANCE

    def _validate_observations(self, observations: list[Observation]) -> None:
        """Require a nonempty input with one observation per dataset/problem."""
        if not observations or len(set(observations)) != len(observations):
            raise ValueError("fitting requires a nonempty list of unique problems")

    def _fit_parameters(self, observation: Observation) -> Parameters:
        """Subclasses estimate one problem's parameters from its observations."""
        raise NotImplementedError

    def _success_probability(self, params: Parameters, n: int, k: int) -> float:
        """Subclasses calculate at-least-one success for n trajectories of k blocks."""
        raise NotImplementedError
