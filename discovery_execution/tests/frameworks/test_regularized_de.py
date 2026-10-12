"""Check joint R-DE inference against independent integration, sparse support, and saved-parameter reconstruction."""

import json
from dataclasses import asdict, replace
from math import log
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from scipy.special import gammaln, roots_jacobi

from discovery_execution.frameworks.discovery_execution.regularized_discovery_execution import RegularizedDiscoveryExecution
from discovery_execution.frameworks.models import BetaParameters, DirichletParameters, RDEParameters, RDEPrior
from discovery_execution.frameworks.utils.constants import NUM_ARMS, PRIOR_LOG_SHAPE_BOUNDS, RDE_INITIALIZATIONS, TOTAL_BUDGET
from discovery_execution.tests.frameworks.constants import NUMERICAL_TOLERANCE, QUADRATURE_ORDER, ROWS


def beta_quadrature(a: float, b: float) -> list[NDArray[np.float64]]:
    """Integrate polynomials against a Beta law using independent Gauss-Jacobi quadrature."""
    points, weights = roots_jacobi(QUADRATURE_ORDER, b - 1, a - 1)
    return [(points + 1) / 2, weights / weights.sum()]


@pytest.mark.parametrize("successes", [0, 6, 24])
def test_joint_likelihood_and_all_allocations_match_quadrature(successes: int) -> None:
    """Integrate the coupled likelihood directly, including extreme short counts and excluded categories."""
    prior = RDEPrior(BetaParameters(0.7, 2.3), DirichletParameters([1.2, 0.8, *([0] * (TOTAL_BUDGET - 2)), 1.5]))
    counts = np.array([[2, 1, *([0] * (TOTAL_BUDGET - 2)), 1]], dtype=float)
    params = RDEParameters(prior, successes, 24, [int(value) for value in counts[0]])
    discovery, discovery_weights = beta_quadrature(prior.discovery.a, prior.discovery.b)
    first, first_weights = beta_quadrature(3.2, 4.3)
    later, later_weights = beta_quadrature(1.8, 2.5)
    discovery, first, later = discovery[:, None, None], first[None, :, None], later[None, None, :]
    weights = discovery_weights[:, None, None] * first_weights[None, :, None] * later_weights[None, None, :]
    # Oracle counts update the execution prior; short-run outcomes still couple discovery and first execution.
    short_rate = discovery * first
    likelihood = short_rate ** successes * (1 - short_rate) ** (params.trials - successes)
    evidence = float(np.sum(weights * likelihood))
    active = counts[0] > 0
    concentrations = np.array(prior.execution.concentrations)[active]
    oracle_evidence = gammaln(concentrations.sum()) - gammaln(concentrations.sum() + counts.sum())
    oracle_evidence += np.sum(gammaln(concentrations + counts[0, active]) - gammaln(concentrations))
    log_shapes = np.log([prior.discovery.a, prior.discovery.b, *concentrations])
    framework = RegularizedDiscoveryExecution()
    nll = framework._negative_log_likelihood(log_shapes, np.array([successes], dtype=float), np.array([24.0]), counts)
    assert nll == pytest.approx(-oracle_evidence - log(evidence), abs=NUMERICAL_TOLERANCE)

    for n in NUM_ARMS:
        for k in range(1, TOTAL_BUDGET // n + 1):
            success = np.zeros_like(weights)
            for delay in range(k):
                execution = first if k - delay == 1 else first + (1 - first) * later
                success += discovery * (1 - discovery) ** delay * execution
            expected = float(np.sum(weights * likelihood * (1 - (1 - success) ** n)) / evidence)
            assert framework._success_probability(params, n, k) == pytest.approx(expected, abs=NUMERICAL_TOLERANCE)


def test_fit_uses_ten_starts_and_saved_parameters_reproduce_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check active support, all ten initializations, and reconstruction without needing the fitted instance."""
    # Total first-completion counts for ROWS include unsolved and exclude two unobserved blocks.
    total_oracle_counts = np.array([3, 2, 0, 1, 0, 6])
    active = total_oracle_counts > 0
    shapes = np.array([0.7, 2.3, *([1.2] * int(active.sum()))])
    optimizer = Mock(return_value=OptimizeResult(success=True, fun=1.0, x=np.log(shapes), message="converged"))
    monkeypatch.setattr("discovery_execution.frameworks.base.base_regularized.minimize", optimizer)
    framework = RegularizedDiscoveryExecution()
    with pytest.raises(ValueError, match="requires a fitted prior"):
        framework._fit_parameters(ROWS[0])
    framework.fit(ROWS)
    assert optimizer.call_count == len(RDE_INITIALIZATIONS)
    for call, (mean, discovery_concentration, execution_concentration) in zip(optimizer.call_args_list, RDE_INITIALIZATIONS):
        expected = [mean * discovery_concentration, (1 - mean) * discovery_concentration,
                    *(execution_concentration * total_oracle_counts[active] / total_oracle_counts.sum())]
        np.testing.assert_allclose(call.args[1], np.log(expected))
        assert call.kwargs["jac"] is False
    assert framework.prior is not None
    assert np.array(framework.prior.execution.concentrations) == pytest.approx(np.where(active, 1.2, 0))
    saved = json.loads(json.dumps(asdict(framework.params[ROWS[0]]), allow_nan=False))
    prior = saved.pop("prior")
    restored = RDEParameters(
        prior=RDEPrior(BetaParameters(**prior["discovery"]), DirichletParameters(prior["execution"]["concentrations"])),
        successes=saved["successes"], trials=saved["trials"], oracle_counts=saved["oracle_counts"],
    )
    independent = RegularizedDiscoveryExecution()
    for prediction in framework.predict():
        if prediction.problem_id == ROWS[0].problem_id:
            assert independent._success_probability(restored, prediction.n, prediction.k) == pytest.approx(prediction.success_probability)


@pytest.mark.parametrize("oracle_successes", [[0,] * TOTAL_BUDGET, [4,] * TOTAL_BUDGET, [0,] + [4,] * (TOTAL_BUDGET - 1)])
def test_unsupported_oracle_data_preserves_previous_fit(oracle_successes: list[int]) -> None:
    """Reject missing first-block support or missing later/unsolved support before replacing a successful fit."""
    framework = RegularizedDiscoveryExecution()
    framework.fit(ROWS)
    prior, parameters, diagnostics = framework.prior, framework.params, framework.prior_fits
    with pytest.raises(ValueError, match="first-block"):
        framework.fit([replace(row, oracle_successes=oracle_successes) for row in ROWS])
    assert framework.prior is prior and framework.params is parameters and framework.prior_fits is diagnostics


@pytest.mark.parametrize("later_category", [1, TOTAL_BUDGET])
def test_single_supported_later_category_has_exact_moments(later_category: int) -> None:
    """A conditional execution category can be deterministic, including all later outcomes being unsolved."""
    concentrations = [0.0] * (TOTAL_BUDGET + 1)
    concentrations[0], concentrations[later_category] = 1.2, 0.8
    counts = [0] * (TOTAL_BUDGET + 1)
    counts[0] = counts[later_category] = 2
    params = RDEParameters(RDEPrior(BetaParameters(0.7, 2.3), DirichletParameters(concentrations)), 6, 24, counts)
    discovery, discovery_weights = beta_quadrature(0.7, 2.3)
    first, first_weights = beta_quadrature(3.2, 2.8)
    discovery, first = discovery[:, None], first[None, :]
    weights = discovery_weights[:, None] * first_weights[None, :]
    rate = discovery * first
    likelihood = rate ** params.successes * (1 - rate) ** (params.trials - params.successes)
    framework = RegularizedDiscoveryExecution()
    for n in NUM_ARMS:
        for k in range(1, TOTAL_BUDGET // n + 1):
            success = np.zeros_like(weights)
            for delay in range(k):
                execution = 1 if k - delay > later_category else first
                success += discovery * (1 - discovery) ** delay * execution
            expected = float(np.sum(weights * likelihood * (1 - (1 - success) ** n)) / np.sum(weights * likelihood))
            assert framework._success_probability(params, n, k) == pytest.approx(expected, abs=NUMERICAL_TOLERANCE)


@pytest.mark.parametrize("successes", [0, 24])
def test_near_certain_predictions_remain_valid_at_optimizer_bounds(successes: int) -> None:
    """Mixed extreme shapes previously produced probabilities above one through log-Beta cancellation."""
    lower, upper = np.exp(PRIOR_LOG_SHAPE_BOUNDS)
    prior = RDEPrior(
        BetaParameters(float(upper), float(lower)),
        DirichletParameters([float(upper), *([float(lower)] * TOTAL_BUDGET)]),
    )
    params = RDEParameters(prior, successes, 24, [1, 1, *([0] * (TOTAL_BUDGET - 2)), 2])
    framework = RegularizedDiscoveryExecution()
    for n in NUM_ARMS:
        for k in range(1, TOTAL_BUDGET // n + 1):
            probability = framework._success_probability(params, n, k)
            assert framework._is_valid_probability(probability)
    assert framework._success_probability(params, 8, 1) == pytest.approx(1, abs=NUMERICAL_TOLERANCE, rel=0)
