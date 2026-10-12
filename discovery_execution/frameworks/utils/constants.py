"""Estimator settings retained from DE-Framework and DELVE's fitting arms."""

from discovery_execution.harness.constants import DEFAULT_BLOCKS

SG = "sg"
DE = "de"
RSG = "r-sg"
RDE = "r-de"
FRAMEWORKS_DIRECTORY = "frameworks"
FIT_FILENAME = "fit.json"
PREDICTIONS_FILENAME = "predictions.jsonl"
JSON_INDENT = 2
TOTAL_BUDGET = DEFAULT_BLOCKS
NUM_ARMS = [1, 2, 4]
MIN_SHORT_SEEDS = 16
MIN_ORACLE_SEEDS = 3
MIN_AUDIT_SCORE = 1
PROBABILITY_TOLERANCE = 1e-9
# (Mean, concentration): distinct discovery starts; plus two centered starts.
RSG_INITIALIZATIONS = [
    [0.6, 0.5],
    [0.6, 2],
    [0.6, 10],
    [0.6, 50],
    [0.2, 1],
    [0.8, 1],
    [0.35, 5],
    [0.9, 5],
    [0.5, 1],
    [0.5, 50],
]
# (Discovery mean, discovery concentration, execution concentration).
RDE_INITIALIZATIONS = [
    [0.6, 0.5, 0.5],
    [0.6, 2, 2],
    [0.6, 10, 0.5],
    [0.6, 0.5, 10],
    [0.6, 10, 10],
    [0.6, 50, 50],
    [0.2, 1, 1],
    [0.8, 1, 1],
    [0.35, 5, 5],
    [0.9, 5, 5],
]
PRIOR_NUM_INITIALIZATIONS = 10
PRIOR_LOG_SHAPE_BOUNDS = [-12, 12]
PRIOR_OPTIMIZER = "L-BFGS-B"
PRIOR_MAX_ITERATIONS = 1500
PRIOR_MAX_LINE_SEARCH_STEPS = 40
PRIOR_FUNCTION_TOLERANCE = 1e-12
PRIOR_GRADIENT_TOLERANCE = 1e-6
PRIOR_NLL_TOLERANCE = 1e-6

FRAMEWORK_NAMES = (SG, DE, RSG, RDE)
SHORT_ARM = "unaided-1x"
ORACLE_ARM = "oracle-at-0x-execution"
