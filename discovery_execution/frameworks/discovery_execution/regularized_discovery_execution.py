"""Empirical-Bayes DE with joint Beta/Dirichlet fitting and exact posterior allocation predictions."""

from functools import partial
from itertools import product
from math import comb

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, gammaln, logsumexp

from discovery_execution.frameworks.base.base_regularized import BaseRegularized
from discovery_execution.frameworks.models import (
    BetaParameters,
    DirichletParameters,
    Observation,
    RDEConditionalPosterior,
    RDEParameters,
    RDEPosterior,
    RDEPrior,
)
from discovery_execution.frameworks.utils.constants import RDE, RDE_INITIALIZATIONS, TOTAL_BUDGET
from discovery_execution.frameworks.utils.probabilities import (
    expected_beta_product,
    expected_dirichlet_product,
)


class RegularizedDiscoveryExecution(BaseRegularized[RDEParameters, RDEPrior]):
    """Use inherited fit/predict to learn Beta/Dirichlet priors and evaluate posterior DE success.

    In the formulas, u is short successes, m is short trials, c_j is oracle first-completion counts, and D=(u,m,c). h counts failed short attempts where discovery succeeded. alpha is discovery probability; pi_j is execution-completion probability in block j, with the final category denoting noncompletion. The priors are alpha~Beta(a,b) and pi~Dirichlet(d).

    Code arguments n and k are the paper's arm count N and depth K. Formulas use one-based pi_j; arrays store pi_1 at index zero. Posterior helpers construct distributions and Pr(h|D); prediction helpers evaluate the inner expectation and then average over h.
    """

    name = RDE

    def _fit_prior(self, observations: list[Observation]) -> RDEPrior:
        """Fit (a,b,d) = argmin [-sum_i log E_prior[L_i(alpha,pi)]] across problems via _negative_log_likelihood."""
        successes = np.array([obs.short_successes for obs in observations], dtype=float)
        trials = np.array([obs.short_attempts for obs in observations], dtype=float)
        oracle_counts = np.array(
            [self._get_oracle_counts(obs) for obs in observations], dtype=float
        )
        total_oracle_counts = oracle_counts.sum(axis=0)
        has_oracle_observations = total_oracle_counts > 0
        if not has_oracle_observations[0] or has_oracle_observations.sum() < 2:
            raise ValueError(
                "R-DE requires some first-block and some later or unsolved oracle outcomes"
            )

        execution_proportions = (
            total_oracle_counts[has_oracle_observations] / total_oracle_counts.sum()
        )
        initializations = []
        for (
            mean,
            discovery_concentration,
            execution_concentration,
        ) in RDE_INITIALIZATIONS:
            initializations.append(
                [
                    mean * discovery_concentration,
                    (1 - mean) * discovery_concentration,
                    *(
                        float(value)
                        for value in execution_concentration * execution_proportions
                    ),
                ]
            )
        objective = partial(
            self._negative_log_likelihood,
            successes=successes,
            trials=trials,
            oracle_counts=oracle_counts,
        )
        parameters = self._optimize_prior(
            objective, initializations, use_gradient=False
        )
        return self._build_prior(parameters, has_oracle_observations)

    def _negative_log_likelihood(
        self,
        log_parameters: NDArray[np.float64],
        successes: NDArray[np.float64],
        trials: NDArray[np.float64],
        oracle_counts: NDArray[np.float64],
    ) -> float:
        """Return -sum_i log E_prior[(alpha*pi_1)^u_i * (1-alpha*pi_1)^(m_i-u_i) * product_j pi_j^c_ij].

        The expectation uses alpha~Beta(a,b) and pi~Dirichlet(d), with shapes obtained from exp(log_parameters). Binomial and multinomial coefficients independent of the shapes are omitted.
        """
        has_oracle_observations = oracle_counts.sum(axis=0) > 0
        prior = self._build_prior(np.exp(log_parameters), has_oracle_observations)
        posterior = self._calculate_posterior(prior, successes, trials, oracle_counts)
        return -float(posterior.log_evidence.sum())

    def _build_prior(
        self,
        parameters: NDArray[np.float64],
        has_oracle_observations: NDArray[np.bool_],
    ) -> RDEPrior:
        """Read positive optimizer values as a, b, then concentrations for observed outcomes; excluded outcomes stay zero."""
        a, b, *execution_concentrations = parameters
        concentrations = np.zeros(TOTAL_BUDGET + 1)
        concentrations[has_oracle_observations] = execution_concentrations
        return RDEPrior(
            discovery=BetaParameters(a=float(a), b=float(b)),
            execution=DirichletParameters([float(value) for value in concentrations]),
        )

    def _fit_parameters(self, observation: Observation) -> RDEParameters:
        """Keep the prior and both types of counts so saved predictions can be reconstructed."""
        if self.prior is None:
            raise ValueError("R-DE requires a fitted prior")
        return RDEParameters(
            prior=self.prior,
            successes=observation.short_successes,
            trials=observation.short_attempts,
            oracle_counts=self._get_oracle_counts(observation),
        )

    def _get_oracle_counts(self, observation: Observation) -> list[int]:
        """Return c_j=C_j-C_(j-1) for cumulative oracle counts C_0=0; append oracle_attempts-C_B for noncompletion."""
        cumulative = [0, *observation.oracle_successes, observation.oracle_attempts]
        return [later - earlier for earlier, later in zip(cumulative, cumulative[1:])]

    def _calculate_posterior(
        self,
        prior: RDEPrior,
        successes: NDArray[np.float64],
        trials: NDArray[np.float64],
        oracle_counts: NDArray[np.float64],
    ) -> RDEPosterior:
        """Return p(alpha,pi|h,D), Pr(h|D), and each problem's marginal log likelihood up to data-only constants.

        If ell_h is the unnormalized log weight from _failure_count_log_weights, Pr(h|D)=exp(ell_h-logsumexp_h(ell_h)). The marginal log likelihood adds the oracle log evidence to logsumexp_h(ell_h).
        """
        concentrations = np.array(prior.execution.concentrations)
        # Oracle observations give pi|c ~ Dirichlet(d+c), before the short-attempt update.
        oracle_concentrations = concentrations + oracle_counts
        conditional = self._conditional_posterior(
            prior.discovery, successes, trials, oracle_concentrations
        )
        log_case_weights = self._failure_count_log_weights(
            prior.discovery, successes, trials, oracle_concentrations, conditional
        )
        short_log_evidence = np.asarray(
            logsumexp(log_case_weights, axis=1), dtype=float
        )
        return RDEPosterior(
            log_evidence=self._oracle_log_evidence(
                concentrations, oracle_counts, oracle_concentrations
            )
            + short_log_evidence,
            case_probabilities=np.exp(log_case_weights - short_log_evidence[:, None]),
            discovery_a=conditional.discovery_a,
            discovery_b=conditional.discovery_b,
            first_execution_a=conditional.first_execution_a,
            first_execution_b=conditional.first_execution_b,
            later_execution_concentrations=conditional.later_execution_concentrations,
        )

    def _conditional_posterior(
        self,
        discovery_prior: BetaParameters,
        successes: NDArray[np.float64],
        trials: NDArray[np.float64],
        oracle_concentrations: NDArray[np.float64],
    ) -> RDEConditionalPosterior:
        """Return alpha|h,D ~ Beta(a+u+h,b+m-u-h) and pi_1|h,D ~ Beta(d_1+c_1+u,sum_(j>1)(d_j+c_j)+h).

        The relative shares pi_j/(1-pi_1), j>1, retain Dirichlet shapes d_j+c_j independently of the two Beta variables. Rows are problems; Beta arrays broadcast across columns h=0,...,max(m-u).
        """
        short_successes = successes[:, None]
        short_failures = (trials - successes)[:, None]
        execution_failures = np.arange(int(short_failures.max()) + 1)[None, :]
        # Columns with h>m-u are padding for shorter rows; their final probability is zero.
        discovery_failures = np.maximum(short_failures - execution_failures, 0)
        later_concentrations = oracle_concentrations[:, 1:]
        return RDEConditionalPosterior(
            discovery_a=discovery_prior.a + short_successes + execution_failures,
            discovery_b=discovery_prior.b + discovery_failures,
            first_execution_a=oracle_concentrations[:, :1] + short_successes,
            first_execution_b=later_concentrations.sum(axis=1, keepdims=True)
            + execution_failures,
            later_execution_concentrations=later_concentrations,
        )

    def _failure_count_log_weights(
        self,
        discovery_prior: BetaParameters,
        successes: NDArray[np.float64],
        trials: NDArray[np.float64],
        oracle_concentrations: NDArray[np.float64],
        conditional: RDEConditionalPosterior,
    ) -> NDArray[np.float64]:
        """Return log w_h for w_h=binom(m-u,h) * E[alpha^(u+h)*(1-alpha)^(m-u-h)] * E[pi_1^u*(1-pi_1)^h].

        The first expectation uses Beta(a,b); the second uses Beta(d_1+c_1,sum_(j>1)(d_j+c_j)). Both precede the short-attempt update. Beta-function ratios evaluate these expectations; invalid h>m-u receives log weight -inf.
        """
        short_failures = (trials - successes)[:, None]
        execution_failures = np.arange(conditional.discovery_a.shape[1])[None, :]
        discovery_failures = np.maximum(short_failures - execution_failures, 0)
        # log binom(m-u,h): select which observed failures occurred after successful discovery.
        log_combinations = np.where(
            execution_failures <= short_failures,
            gammaln(short_failures + 1)
            - gammaln(execution_failures + 1)
            - gammaln(discovery_failures + 1),
            -np.inf,
        )
        log_weights = log_combinations + betaln(
            conditional.discovery_a, conditional.discovery_b
        )
        log_weights -= betaln(discovery_prior.a, discovery_prior.b)
        execution_log_likelihood = betaln(
            conditional.first_execution_a, conditional.first_execution_b
        ) - betaln(
            oracle_concentrations[:, :1],
            oracle_concentrations[:, 1:].sum(axis=1, keepdims=True),
        )
        return log_weights + execution_log_likelihood

    def _oracle_log_evidence(
        self,
        concentrations: NDArray[np.float64],
        oracle_counts: NDArray[np.float64],
        oracle_concentrations: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return log E_(pi~Dirichlet(d))[product_j pi_j^c_j], omitting the data-only multinomial coefficient.

        This is gammaln(sum d)-gammaln(sum(d+c)) + sum_j[gammaln(d_j+c_j)-gammaln(d_j)], over supported categories. The result has one value per problem.
        """
        supported_categories = concentrations > 0
        oracle_trials = oracle_counts.sum(axis=1)
        total_concentration = concentrations.sum()
        log_evidence = gammaln(total_concentration) - gammaln(
            total_concentration + oracle_trials
        )
        log_evidence += (
            gammaln(oracle_concentrations[:, supported_categories])
            - gammaln(concentrations[supported_categories])
        ).sum(axis=1)
        return log_evidence

    def _success_probability(self, params: RDEParameters, n: int, k: int) -> float:
        """Return sum_h Pr(h|D) * E[1-(1-s_k(alpha,pi))^n | h,D], the posterior probability any arm solves."""
        posterior = self._calculate_posterior(
            prior=params.prior,
            successes=np.array([params.successes], dtype=float),
            trials=np.array([params.trials], dtype=float),
            oracle_counts=np.array([params.oracle_counts], dtype=float),
        )
        success_by_case = self._conditional_success_probability(posterior, n, k)
        return float(np.sum(posterior.case_probabilities * success_by_case))

    def _conditional_success_probability(
        self, posterior: RDEPosterior, n: int, k: int
    ) -> NDArray[np.float64]:
        """Return E[1-(1-s_k)^n | h,D] = sum_(r=1)^n (-1)^(r+1)*binom(n,r)*E[s_k^r | h,D] for every h.

        Here s_k=sum_(f=0)^(k-1) sum_(j=1)^(k-f) alpha*(1-alpha)^f*pi_j is one arm's DE success probability. The returned array has the same problem/case shape as posterior.case_probabilities.
        """
        success_by_case = np.zeros_like(posterior.case_probabilities)
        for power in range(1, n + 1):
            # Each binomial term contributes (-1)^(r+1)*binom(n,r)*E[s_k^r | h,D].
            expected_power = self._expected_success_power(posterior, k, power)
            coefficient = (-1) ** (power + 1) * comb(n, power)
            success_by_case += coefficient * expected_power
        return success_by_case

    def _expected_success_power(
        self, posterior: RDEPosterior, k: int, power: int
    ) -> NDArray[np.float64]:
        """Return E[s_k^r | h,D], r=power, by expanding the r-fold product of the successful-path sum.

        Each ordered selection of r paths contributes E[alpha^r*(1-alpha)^sum_i(f_i) | h,D] * E[product_i pi_(j_i+1) | h,D]. f_i counts discovery failures and j_i is the zero-based execution block; conditional independence permits the product of expectations.
        """
        success_paths = self._get_success_paths(k)

        # Ordered selections retain each term's multiplicity in (sum_path probability(path))^r.
        expected_power = np.zeros_like(posterior.case_probabilities)
        for paths in product(success_paths, repeat=power):
            discovery_failures = sum(failures for failures, _ in paths)
            execution_blocks = [block for _, block in paths]
            expected_discovery = expected_beta_product(
                posterior.discovery_a,
                posterior.discovery_b,
                power,
                discovery_failures,
            )
            expected_execution = self._expected_execution_product(
                posterior, execution_blocks
            )
            expected_power += expected_discovery * expected_execution
        return expected_power

    def _get_success_paths(self, k: int) -> list[list[int]]:
        """Return [f,j] with f+j+1<=k, where f is discovery failures and j is the zero-based execution block.

        Each pair contributes alpha*(1-alpha)^f*pi_(j+1) to s_k; f=0,j=0 is first-block discovery and execution success.
        """
        return [
            [discovery_failures, execution_block]
            for discovery_failures in range(k)
            for execution_block in range(k - discovery_failures)
        ]

    def _expected_execution_product(
        self, posterior: RDEPosterior, blocks: list[int]
    ) -> NDArray[np.float64]:
        """Return E[product_i pi_(blocks[i]+1) | h,D] as a product of independent Beta and Dirichlet averages.

        If r=len(blocks) and r_1 counts block zero, the Beta factor is E[pi_1^r_1*(1-pi_1)^(r-r_1) | h,D]. The Dirichlet factor is E[product_(j>1)(pi_j/(1-pi_1))^r_j | h,D], where r_j counts occurrences of execution category j.
        """
        first_count = blocks.count(0)
        later_count = len(blocks) - first_count
        expected_first_block = expected_beta_product(
            posterior.first_execution_a,
            posterior.first_execution_b,
            first_count,
            later_count,
        )
        later_counts = np.asarray(
            np.bincount(
                blocks, minlength=posterior.later_execution_concentrations.shape[1] + 1
            )[1:],
            dtype=np.int64,
        )
        expected_later_shares = expected_dirichlet_product(
            posterior.later_execution_concentrations, later_counts
        )
        return expected_first_block * expected_later_shares[:, None]
