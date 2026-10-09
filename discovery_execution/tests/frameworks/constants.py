"""Small complete audit banks for SG/DE regression and input-selection tests."""

from discovery_execution.frameworks.models import Observation

MODEL = "qwen3-4b-instruct-2507"
DATASET = "aime/2024"
SHORT_SEEDS = list(range(1, 25))
ORACLE_SEEDS = list(range(1, 5))
EXTRA_SEED = 101
EXPECTED_ALLOCATIONS = [
    [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
    [2, 1], [2, 2], [2, 3], [2, 4],
    [4, 1], [4, 2],
]
ROWS = [
    Observation(DATASET, "p1", short_attempts=24, short_successes=12, oracle_attempts=4, oracle_successes=[1, 2, 2, 3, 3, 3, 3, 3]),
    Observation(DATASET, "p2", short_attempts=24, short_successes=3, oracle_attempts=4, oracle_successes=[2, 2, 2, 2, 2, 2, 2, 2]),
    Observation(DATASET, "p3", short_attempts=24, short_successes=0, oracle_attempts=4, oracle_successes=[0, 1, 1, 1, 1, 2, 2, 2]),
]
NUMERICAL_TOLERANCE = 1e-9
GRADIENT_TOLERANCE = 1e-5
QUADRATURE_ORDER = 32
