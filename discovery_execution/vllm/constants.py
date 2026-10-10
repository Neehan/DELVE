"""Fixed serving settings for the four trajectory-collection models."""

import json
from pathlib import Path

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_PORT = 65535
MAX_MODEL_LEN = 131072
GPU_COUNT = 2
GPU_MEMORY_UTILIZATION = 0.92
KV_CACHE_DTYPE = {
    "qwen3-4b-instruct-2507": "fp8_e4m3",
    "qwen3-1.7b": "fp8_e4m3",
    "ministral-3-3b-instruct-2512": "auto",
    # Uncalibrated FP8 caused severe repetition in the AIME quality checks.
    "deepseek-r1-distill-qwen-1.5b": "auto",
}
TEMPLATES_DIRECTORY = Path(__file__).parent / "templates"
# Favor long trajectories on 2x L40S: tensor parallelism for the larger KV caches,
# and independent replicas for DeepSeek's small KV cache.
PARALLELISM = {
    "qwen3-4b-instruct-2507": "tensor",
    "qwen3-1.7b": "tensor",
    "ministral-3-3b-instruct-2512": "tensor",
    "deepseek-r1-distill-qwen-1.5b": "data",
}

# Use the same static YaRN extension for every Qwen3-1.7B arm and budget.
QWEN_YARN = {"rope_parameters": {"rope_type": "yarn", "factor": 4.0,
                                "original_max_position_embeddings": 32768, "rope_theta": 1000000}}
MODEL_ARGUMENTS = {
    "qwen3-4b-instruct-2507": [],
    "qwen3-1.7b": ["--hf-overrides", json.dumps(QWEN_YARN),
                   "--chat-template", str(TEMPLATES_DIRECTORY / "qwen3.jinja")],
    "ministral-3-3b-instruct-2512": ["--tokenizer-mode", "mistral", "--config-format", "mistral",
                                   "--load-format", "mistral", "--limit-mm-per-prompt", '{"image": 0}'],
    "deepseek-r1-distill-qwen-1.5b": ["--chat-template", str(TEMPLATES_DIRECTORY / "deepseek.jinja")],
}
