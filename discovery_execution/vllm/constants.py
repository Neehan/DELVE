"""Fixed serving settings for the four trajectory-collection models."""

import json
from pathlib import Path

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_PORT = 65535
MAX_MODEL_LEN = 131072
TENSOR_PARALLEL_SIZE = 2
GPU_MEMORY_UTILIZATION = 0.92
KV_CACHE_DTYPE = "fp8_e4m3"
TEMPLATES_DIRECTORY = Path(__file__).parent / "templates"

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
