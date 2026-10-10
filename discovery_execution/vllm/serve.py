"""Launch an installed vLLM server with fixed, model-specific collection settings."""

import os
import shlex
import sys
from argparse import ArgumentParser

from discovery_execution.harness.constants import DEFAULT_BATCH_SIZES, SUPPORTED_MODELS
from discovery_execution.vllm.constants import (
    DEFAULT_PORT, GPU_COUNT, GPU_MEMORY_UTILIZATION, HOST, KV_CACHE_DTYPE, MAX_MODEL_LEN,
    MAX_PORT, MODEL_ARGUMENTS, PARALLELISM,
)


def build_command(model: str, port: int, max_num_seqs: int | None) -> list[str]:
    """Build one server command without changing model weights or parsing reasoning."""
    if not 1 <= port <= MAX_PORT:
        raise ValueError(f"port must be between 1 and {MAX_PORT}")
    if max_num_seqs is None:
        max_num_seqs = DEFAULT_BATCH_SIZES[model][0]
    if max_num_seqs < 1:
        raise ValueError("max-num-seqs must be positive")
    return [
        "vllm", "serve", SUPPORTED_MODELS[model], "--served-model-name", model,
        "--host", HOST, "--port", str(port), "--max-model-len", str(MAX_MODEL_LEN),
        f"--{PARALLELISM[model]}-parallel-size", str(GPU_COUNT),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-num-seqs", str(max_num_seqs), "--kv-cache-dtype", KV_CACHE_DTYPE[model],
        "--generation-config", "vllm", *MODEL_ARGUMENTS[model],
    ]


def main() -> None:
    """Validate the selected alias and replace this process with the server."""
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-num-seqs", type=int,
                        help="Maximum active sequences (default: the model's k=8 batch)")
    args = parser.parse_args()
    try:
        command = build_command(args.model, args.port, args.max_num_seqs)
    except ValueError as error:
        parser.error(str(error))
    print(shlex.join(command), file=sys.stderr, flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
