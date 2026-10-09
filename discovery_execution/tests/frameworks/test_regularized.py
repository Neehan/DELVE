"""Check R-SG's marginal likelihood, posterior predictions, and failed-fit state preservation."""

import json
from dataclasses import asdict, replace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import OptimizeResult, check_grad
from scipy.stats import beta, betabinom, binom

from discovery_execution.frameworks.geometric.regularized_simple_geometric import RegularizedSimpleGeometric
from discovery_execution.frameworks.models import RSGParameters, Observation
from discovery_execution.frameworks.utils.constants import (
    PRIOR_MAX_LINE_SEARCH_STEPS,
    PRIOR_NLL_TOLERANCE,
    PRIOR_NUM_INITIALIZATIONS,
    RSG_INITIALIZATIONS,
)
from discovery_execution.frameworks.utils.probabilities import beta_binomial_nll
from discovery_execution.tests.frameworks.constants import GRADIENT_TOLERANCE, NUMERICAL_TOLERANCE, ROWS


def test_marginal_likelihood_and_gradient_match_independent_references() -> None:
    """Check the fitting objective against SciPy's distribution and its gradient against finite differences."""
    successes = np.array([row.short_successes for row in ROWS], dtype=float)
    attempts = np.array([row.short_attempts for row in ROWS], dtype=float)
    shapes = np.array([0.4, 1.7])
    value, _ = beta_binomial_nll(np.log(shapes), successes, attempts)
    log_combinations = binom.logpmf(successes, attempts, 0.5) + attempts * np.log(2)
    expected = -np.sum(betabinom.logpmf(successes, attempts, *shapes) - log_combinations)
    assert value == pytest.approx(expected)
    error = check_grad(
        lambda shapes: beta_binomial_nll(shapes, successes, attempts)[0],
        lambda shapes: beta_binomial_nll(shapes, successes, attempts)[1],
        np.log(shapes),
    )
    assert error < GRADIENT_TOLERANCE


@pytest.mark.parametrize("shapes", [[1, 1], [0.4, 1.7], [12, 3]])
def test_posterior_prediction_integrates_uncertainty(shapes: list[float]) -> None:
    """Compare exact predictions with numerical integration, including equal-budget allocation invariance."""
    framework = RegularizedSimpleGeometric()
    params = RSGParameters(successes=6, trials=24, a=shapes[0], b=shapes[1])
    posterior_a, posterior_b = params.a + 6, params.b + 18
    expected, _ = quad(lambda q: (1 - (1 - q) ** 8) * beta.pdf(q, posterior_a, posterior_b), 0, 1)
    for n, k in [[8, 1], [4, 2], [2, 4]]:
        actual = framework._success_probability(params, n, k)
        assert actual == pytest.approx(expected, abs=NUMERICAL_TOLERANCE)
        posterior_mean = posterior_a / (posterior_a + posterior_b)
        assert actual < 1 - (1 - posterior_mean) ** (n * k)


def test_stored_parameters_keep_prior_and_counts_and_ignore_oracle_measurements() -> None:
    """Store all four prediction inputs; changing oracle outcomes cannot alter the R-SG fit."""
    framework = RegularizedSimpleGeometric()
    with pytest.raises(ValueError, match="requires a fitted prior"):
        framework._fit_parameters(ROWS[0])
    framework.fit(ROWS)
    prior = framework.prior
    params = framework.params.copy()
    assert prior is not None
    for obs in ROWS:
        assert params[obs] == RSGParameters(successes=obs.short_successes, trials=obs.short_attempts, a=prior.a, b=prior.b)
    framework.fit([replace(row, oracle_successes=[0,] * len(row.oracle_successes)) for row in ROWS])
    assert framework.prior == prior and framework.params == params


@pytest.mark.parametrize("all_solved", [False, True])
def test_prior_fit_accepts_all_failure_or_all_success_data(all_solved: bool) -> None:
    """Extreme input counts produce finite predictions through the bounded positive prior fit."""
    framework = RegularizedSimpleGeometric()
    framework.fit([replace(row, short_successes=row.short_attempts if all_solved else 0) for row in ROWS])
    assert framework.prior is not None
    assert all(0 <= row.success_probability <= 1 for row in framework.predict())


def test_failed_posterior_update_preserves_prior_and_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newly learned prior must not replace the previous fit if a posterior update fails."""
    framework = RegularizedSimpleGeometric()
    framework.fit(ROWS)
    prior, params, diagnostics = framework.prior, framework.params, framework.prior_fits

    def fail_parameters(observation: Observation) -> RSGParameters:
        """Inject a failure after prior fitting and before new parameters are published."""
        raise ArithmeticError("injected posterior failure")

    monkeypatch.setattr(framework, "_fit_parameters", fail_parameters)
    with pytest.raises(ArithmeticError, match="injected"):
        framework.fit([replace(row, short_successes=row.short_attempts) for row in ROWS])
    assert framework.prior is prior and framework.params is params
    assert framework.prior_fits is diagnostics


@pytest.mark.parametrize("failure", ["all", "best", "nonfinite"])
def test_optimizer_failure_does_not_replace_previous_fit(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """Reject failed best results and wholly nonfinite fits without replacing the previous state."""
    framework = RegularizedSimpleGeometric()
    framework.fit(ROWS)
    prior, params, diagnostics = framework.prior, framework.params, framework.prior_fits
    fits = [OptimizeResult(
        success=failure == "nonfinite" or (failure == "best" and index > 0),
        fun=float("nan") if failure == "nonfinite" else float(index),
        x=np.zeros(2), message="injected optimizer failure",
    ) for index in range(PRIOR_NUM_INITIALIZATIONS)]
    monkeypatch.setattr("discovery_execution.frameworks.base.base_regularized.minimize", Mock(side_effect=fits))
    with pytest.raises(RuntimeError, match="injected optimizer failure"):
        framework.fit(ROWS)
    assert framework.prior is prior and framework.params is params
    assert framework.prior_fits is diagnostics


def test_ten_starts_record_agreement_without_requiring_eight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain a valid best fit with only two agreeing starts; record unsuccessful and nonfinite starts honestly."""
    fits = [OptimizeResult(success=True, fun=float(index + 1), x=np.log([2, 3]), message="converged")
            for index in range(PRIOR_NUM_INITIALIZATIONS)]
    fits[1].fun = 1 + PRIOR_NLL_TOLERANCE / 2
    fits[2].fun, fits[2].success = 1 + PRIOR_NLL_TOLERANCE / 4, False
    fits[3].fun, fits[3].success = float("nan"), False
    optimizer = Mock(side_effect=fits)
    monkeypatch.setattr("discovery_execution.frameworks.base.base_regularized.minimize", optimizer)
    framework = RegularizedSimpleGeometric()
    framework.fit(ROWS)

    assert optimizer.call_count == PRIOR_NUM_INITIALIZATIONS
    expected_initializations = [[mean * concentration, (1 - mean) * concentration] for mean, concentration in RSG_INITIALIZATIONS]
    assert all(expected_initializations.count(initial) == 1 for initial in expected_initializations)
    assert [fit.initialization for fit in framework.prior_fits] == expected_initializations
    assert sum(fit.reached_best for fit in framework.prior_fits) == 2
    assert not framework.prior_fits[2].success and not framework.prior_fits[2].reached_best
    assert framework.prior_fits[3].nll is None and framework.prior_fits[3].nll_gap is None
    assert framework.prior is not None and asdict(framework.prior) == pytest.approx({"a": 2, "b": 3})
    json.dumps(framework.get_parameters(ROWS[0].dataset), allow_nan=False)
    for call, initial in zip(optimizer.call_args_list, expected_initializations):
        np.testing.assert_allclose(call.args[1], np.log(initial))
        assert call.kwargs["jac"] is True
        assert call.kwargs["options"]["maxls"] == PRIOR_MAX_LINE_SEARCH_STEPS
