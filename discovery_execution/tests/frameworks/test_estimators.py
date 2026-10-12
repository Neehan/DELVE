"""Check SG/DE mathematics through the fit/predict interface."""

from dataclasses import replace
from math import comb

import pytest
from discovery_execution.frameworks.discovery_execution.discovery_execution import DiscoveryExecution
from discovery_execution.frameworks.geometric.simple_geometric import SimpleGeometric
from discovery_execution.frameworks.geometric.regularized_simple_geometric import RegularizedSimpleGeometric

from discovery_execution.tests.frameworks.constants import EXPECTED_ALLOCATIONS, NUMERICAL_TOLERANCE, ROWS


@pytest.mark.parametrize("attempts, successes", [
    [0, [0, 0, 0, 0, 0]],
    [4, [1, 2]],
    [4, [1, 2, 2, 3, 5]],
    [4, [1, 2, 1, 3, 3]],
])
def test_observation_rejects_invalid_oracle_curve(attempts: int, successes: list[int]) -> None:
    """Oracle curves need positive attempts, every checkpoint, bounded counts, and cumulative successes."""
    with pytest.raises(ValueError, match="oracle"):
        replace(ROWS[0], oracle_attempts=attempts, oracle_successes=successes)


def test_sg_preserves_unbiased_short_bank_prediction() -> None:
    """Predictions use combinations of observed counts, not powers of a fitted success rate."""
    framework = SimpleGeometric()
    framework.fit(ROWS)
    for prediction in framework.predict():
        observation = next(row for row in ROWS if (row.dataset, row.problem_id) == (prediction.dataset, prediction.problem_id))
        parameters = framework.params[observation]
        attempts = prediction.n * prediction.k
        expected = 1 - comb(parameters.trials - parameters.successes, attempts) / comb(parameters.trials, attempts)
        assert prediction.success_probability == expected
    framework.fit([replace(ROWS[0], short_successes=1, short_attempts=3)])
    with pytest.raises(ValueError, match="short observations"):
        framework.predict()


def test_de_joint_boundary_and_unidentified_discovery() -> None:
    """Joint fitting adjusts first-block execution at the constraint and exposes unidentified discovery."""
    framework = DiscoveryExecution()
    framework.fit(ROWS)
    first = framework.params[ROWS[0]]
    assert first.alpha == 1 and first.at_boundary
    assert first.epsilon[0] == pytest.approx(13 / 28)
    assert first.epsilon[1] == pytest.approx(9 / 14)
    assert framework.params[ROWS[2]].is_unidentified
    for observation in ROWS[1:]:
        params = framework.params[observation]
        assert not params.at_boundary
        assert params.epsilon == [count / observation.oracle_attempts for count in observation.oracle_successes]
    predictions = {(row.problem_id, row.n, row.k): row.success_probability for row in framework.predict()}
    assert predictions["p1", 2, 2] == pytest.approx(1 - (5 / 14) ** 2)
    assert predictions["p3", 2, 2] == 0
    equal_rates = replace(ROWS[0], short_successes=6)
    framework.fit([equal_rates])
    assert framework.params[equal_rates].alpha == 1
    assert not framework.params[equal_rates].at_boundary


def test_de_saturated_execution_reduces_to_geometric() -> None:
    """When oracle execution always finishes immediately, DE equals the geometric model's formula."""
    framework = DiscoveryExecution()
    framework.fit([replace(ROWS[0], oracle_successes=[4, 4, 4, 4, 4])])
    short_rate = ROWS[0].short_successes / ROWS[0].short_attempts
    for prediction in framework.predict():
        assert prediction.success_probability == pytest.approx(1 - (1 - short_rate) ** (prediction.n * prediction.k))


@pytest.mark.parametrize("framework_type", [SimpleGeometric, DiscoveryExecution, RegularizedSimpleGeometric])
def test_fixed_predictions_and_failed_refit(
    framework_type: type[SimpleGeometric] | type[DiscoveryExecution] | type[RegularizedSimpleGeometric],
) -> None:
    """All methods share fixed predictions and retain the last successful fit when input validation fails."""
    framework = framework_type()
    with pytest.raises(RuntimeError, match="fit"):
        framework.predict()
    framework.fit(ROWS)
    predictions = framework.predict()
    for observation in ROWS:
        allocations = [[row.n, row.k] for row in predictions if row.problem_id == observation.problem_id]
        assert allocations == EXPECTED_ALLOCATIONS
        curve = [row.success_probability for row in predictions if row.problem_id == observation.problem_id and row.n == 2]
        assert all(0 <= value <= 1 for value in curve)
        assert all(later >= earlier - NUMERICAL_TOLERANCE for earlier, later in zip(curve, curve[1:]))
    duplicate = replace(ROWS[0], short_successes=0)
    assert framework.params[duplicate] == framework.params[ROWS[0]]
    with pytest.raises(ValueError, match="unique"):
        framework.fit([ROWS[0], duplicate])
    assert framework.predict() == predictions
    framework.fit([ROWS[0], replace(ROWS[0], dataset="imoproofbench")])
    assert len(framework.params) == 2
