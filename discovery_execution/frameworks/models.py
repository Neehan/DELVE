"""Input selections, observations, fitted parameters, and predictions for SG and DE."""

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from discovery_execution.frameworks.utils.constants import FRAMEWORK_NAMES, TOTAL_BUDGET
from discovery_execution.harness.constants import COMPETITIONS, SUPPORTED_MODELS


@dataclass(frozen=True)
class BaseParameters:
    """Base for fitted parameter dataclasses; subclasses declare their framework's fields."""


@dataclass(frozen=True)
class PriorFit:
    """Record one optimizer start; initialization holds positive parameters before log transformation, and nonfinite NLLs are recorded as None."""

    initialization: list[float]
    success: bool
    nll: float | None
    nll_gap: float | None
    reached_best: bool
    message: str


@dataclass(frozen=True)
class SGParameters(BaseParameters):
    """Store the short success and trial counts used by SG."""

    successes: int
    trials: int


@dataclass(frozen=True)
class BetaParameters(BaseParameters):
    """Store the positive Beta shapes a and b of a learned prior."""

    a: float
    b: float

    def __post_init__(self) -> None:
        """Require finite, positive Beta shapes before fitting or predicting."""
        if not isfinite(self.a) or not isfinite(self.b) or self.a <= 0 or self.b <= 0:
            raise ValueError("Beta shapes must be finite and positive")


@dataclass(frozen=True)
class RSGParameters(SGParameters):
    """Keep observed successes and trials together with prior shapes a and b for prediction and reporting."""

    a: float
    b: float


@dataclass(frozen=True)
class DirichletParameters(BaseParameters):
    """Store execution concentrations for blocks 1..8 and unsolved; zero marks an excluded category."""

    concentrations: list[float]

    def __post_init__(self) -> None:
        """Require finite nonnegative concentrations and at least two supported execution outcomes."""
        if len(self.concentrations) != TOTAL_BUDGET + 1:
            raise ValueError("execution requires one concentration per block and one for unsolved")
        if any(not isfinite(value) or value < 0 for value in self.concentrations):
            raise ValueError("Dirichlet concentrations must be finite and nonnegative")
        if sum(value > 0 for value in self.concentrations) < 2:
            raise ValueError("R-DE requires some first-block and some later or unsolved oracle outcomes")


@dataclass(frozen=True)
class RDEPrior(BaseParameters):
    """Hold the independent Beta discovery prior and Dirichlet execution prior."""

    discovery: BetaParameters
    execution: DirichletParameters

    def __post_init__(self) -> None:
        """Require first-block support so short successes can inform both discovery and execution."""
        if self.execution.concentrations[0] <= 0:
            raise ValueError("R-DE requires some first-block oracle completions")

@dataclass(frozen=True)
class RDEParameters(BaseParameters):
    """Retain the prior and short counts; oracle_counts contains first completions by block, then unsolved."""

    prior: RDEPrior
    successes: int
    trials: int
    oracle_counts: list[int]


@dataclass(frozen=True)
class RDEConditionalPosterior:
    """Store shapes for alpha|h,D, pi_1|h,D, and the relative later-execution shares pi_j/(1-pi_1).

    h counts short failures after discovery succeeded. Beta arrays broadcast to (problems, failure-count cases); later_execution_concentrations has shape (problems, later execution categories including noncompletion).
    """

    discovery_a: NDArray[np.float64]
    discovery_b: NDArray[np.float64]
    first_execution_a: NDArray[np.float64]
    first_execution_b: NDArray[np.float64]
    later_execution_concentrations: NDArray[np.float64]


@dataclass(frozen=True)
class RDEPosterior(RDEConditionalPosterior):
    """Add case_probabilities[i,h]=Pr(h|D_i) and log_evidence[i]=log E_prior[L_i] to the conditional shapes.

    log_evidence omits data-only binomial and multinomial coefficients. Cases beyond a problem's observed short-failure count have probability zero.
    """

    log_evidence: NDArray[np.float64]
    case_probabilities: NDArray[np.float64]


@dataclass(frozen=True)
class DEParameters(BaseParameters):
    """Store discovery probability alpha, cumulative execution probabilities epsilon, and fit diagnostics."""

    alpha: float
    epsilon: list[float]
    at_boundary: bool
    is_unidentified: bool


@dataclass(frozen=True)
class BasePrediction:
    """Hold shared prediction fields; subclasses may add framework-specific outputs."""

    dataset: str
    problem_id: str
    n: int
    k: int
    success_probability: float


@dataclass(frozen=True)
class Observation:
    """Hold problem measurements; equality and hashing use only dataset and problem ID."""

    dataset: str
    problem_id: str
    short_attempts: int = field(compare=False)
    short_successes: int = field(compare=False)
    oracle_attempts: int = field(compare=False)
    # Cumulative successes by checkpoint, from 1x through TOTAL_BUDGET.
    oracle_successes: list[int] = field(compare=False)

    def __post_init__(self) -> None:
        """Require valid counts for both types of input data."""
        if not self.dataset or not self.problem_id:
            raise ValueError("observations require dataset and problem identity")

        counts = [self.short_attempts, self.short_successes, self.oracle_attempts, *self.oracle_successes]
        for count in counts:
            if type(count) is not int or count < 0:
                raise ValueError("observation counts must be nonnegative integers")

        if self.short_attempts < 1 or self.short_successes > self.short_attempts:
            raise ValueError("invalid short-success counts")

        if self.oracle_attempts < 1 or len(self.oracle_successes) != TOTAL_BUDGET:
            raise ValueError("oracle observations require attempts and one success count per checkpoint")
        if self.oracle_successes[-1] > self.oracle_attempts:
            raise ValueError("oracle successes cannot exceed attempts")
        for earlier, later in zip(self.oracle_successes, self.oracle_successes[1:]):
            if later < earlier:
                raise ValueError("oracle successes must be cumulative")


class InputSelection(BaseModel):
    """Select audited competition/year partitions from one model."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str
    competitions: list[str]
    years: list[int]
    frameworks: list[str]

    @model_validator(mode="after")
    def validate_selection(self) -> "InputSelection":
        """Reject unsupported selections, duplicate partitions, and unsafe output paths."""
        if self.model not in SUPPORTED_MODELS:
            raise ValueError("invalid model")
        if not self.competitions or len(set(self.competitions)) != len(self.competitions):
            raise ValueError("competitions must be nonempty and unique")
        if any(competition not in COMPETITIONS for competition in self.competitions):
            raise ValueError("unsupported competition")
        if not self.years or len(set(self.years)) != len(self.years) or min(self.years) < 1:
            raise ValueError("years must be unique positive integers")
        if not self.frameworks or len(set(self.frameworks)) != len(self.frameworks):
            raise ValueError("frameworks must be nonempty and unique")
        if any(name not in FRAMEWORK_NAMES for name in self.frameworks):
            raise ValueError("unsupported framework")
        return self
