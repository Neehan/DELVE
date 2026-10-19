"""Defaults for reproducible frozen-solver critic experiments."""

from pathlib import Path

DEFAULT_TEST_YEAR = 2026
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_SEED = 42
DEFAULT_HORIZON = 5
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_ACCUMULATION_STEPS = 8
DEFAULT_MAX_LENGTH = 49_152
DEFAULT_ORACLE_FRACTION = 1.0
DEFAULT_BRANCH_LIMIT = None
DEFAULT_ORACLE_WEIGHT = 1.0
DEFAULT_CORRECTION_WEIGHT = 0.01
DEFAULT_ATTENTION = "sdpa"
DEFAULT_REFERENCE_MODE = "none"
DEFAULT_VARIANT = "direct"
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_LOGGING_STEPS = 10
PROBABILITY_EPSILON = 1e-6
HASH_SCALE = 2**256
SPLITS = ("train", "validation", "test")
VARIANTS = ("direct", "auxiliary", "de")
CONDITIONS = ("continue", "continue-oracle")
ORDINARY = 0
ORACLE = 1
SUCCESS = 0
TRIALS = 1
CONDITION_COUNT = 2
DE_STATE_FIELDS = 2
PREPARED_DIRECTORY = Path("data/frozen_solver/prepared")
TOKENIZED_DIRECTORY = Path("data/frozen_solver/tokenized")
TRAINING_DIRECTORY = Path("output/frozen_solver")
MODEL_FILENAME = "critic.safetensors"
CONFIG_FILENAME = "critic.json"
MANIFEST_FILENAME = "manifest.json"
TRAINING_ARGS_FILENAME = "training_args.json"
PROMPT = (
    "Estimate final-answer correctness after each additional block of self-refinement. "
    "One block contains {block_tokens} generated tokens. The solver follows solve, critique, "
    "and revise instructions without correctness feedback. It may revise a correct answer incorrectly. "
    "The prefix ends after {fork} blocks; the total budget is {budget} blocks.\n\n"
    "Current phase: {phase}. The last response ended naturally: {response_finished}.\n\n"
    "Saved conversation:\n{conversation}"
)
REFERENCE_PROMPT = "\n\nVerified reference solution (critic only):\n{solution}"
