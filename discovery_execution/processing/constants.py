"""Set shared processing defaults, including analysis checkpoints independent of generation blocks."""

from pathlib import Path

PROCESSED_DIRECTORY = Path("data/outputs/processed")
SPLIT_FILENAME = "splits.parquet"
MANIFEST_FILENAME = "manifest.json"
SPLIT_SEED = 1234
TRAIN_PROBLEMS = 1000
TEST_YEAR = 2026
SPLITS = ("train", "validation", "test")
PARQUET_SHARD_BYTES = 256 * 1024 * 1024
PARQUET_COMPRESSION = "zstd"
HASH_CHUNK_BYTES = 1024 * 1024
TRAJECTORIES_SUBDIRECTORY = "trajectories"
CHECKPOINT_INTERVAL_TOKENS = 4096
# Legacy response text may retokenize slightly differently from the original generated IDs.
MAX_CHECKPOINT_TOKEN_DRIFT = 8
CHECKPOINT_RECONSTRUCTION = "request-anchored-text-offsets-v1"
