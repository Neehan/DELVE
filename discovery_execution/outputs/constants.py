"""Versioned publication schema and bounded Parquet I/O settings."""

from pathlib import Path

import pyarrow as pa

FORMAT_VERSION = b"1"
FORMAT_METADATA = {b"delve.outputs.version": FORMAT_VERSION}
PUBLISHED_DIRECTORY = Path("data/outputs/published")
EXPORT_SEEDS: tuple[int, ...] | None = None
COMPRESSION = "zstd"
ROW_GROUP_SIZE = 16
BATCH_SIZE = 16
SEED_PATTERN = "seed_*.parquet"
MESSAGE = pa.struct([("role", pa.string()), ("content", pa.large_string())])
BOUNDARY = pa.struct([("output_tokens", pa.int64()), ("message_count", pa.int64()),
                      ("last_message_chars", pa.int64())])
CONVERSATION = pa.struct([
    ("messages", pa.list_(pa.field("element", MESSAGE))), ("output_tokens", pa.int64()), ("phase", pa.string()),
    ("injected", pa.bool_()), ("response_finished", pa.bool_()),
    ("requests_json", pa.large_string()), ("checkpoints", pa.list_(pa.field("element", BOUNDARY))),
])
GRADE = pa.struct([("block", pa.int32()), ("score", pa.int8())])
SCHEMA = pa.schema([
    ("trajectory_id", pa.string()), ("competition", pa.string()), ("year", pa.int32()),
    ("model", pa.string()), ("arm", pa.string()), ("problem_id", pa.string()), ("seed", pa.int32()),
    ("source_seed", pa.int32()), ("branch_seed", pa.int32()), ("source_arm", pa.string()), ("continue_at", pa.int32()),
    ("metadata", pa.large_string()), ("conversation", CONVERSATION),
    ("audit", pa.large_string()), ("grades", pa.list_(pa.field("element", GRADE))),
], metadata=FORMAT_METADATA)
AUDIT_COLUMNS = tuple(field.name for field in SCHEMA if field.name != "conversation")
