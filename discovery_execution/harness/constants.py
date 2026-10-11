"""Shared protocol constants and repository paths."""

import re
from pathlib import Path

DATASETS_DIRECTORY = Path("data/datasets")
OUTPUT_DIRECTORY = Path("output")
LOGS_DIRECTORY = Path("logs")
PROMPTS_DIRECTORY = Path("discovery_execution/prompts")
ENV_FILE = Path(".env")
OUTPUT_TOKENS_PER_BLOCK = 8192
TOTAL_BLOCKS = 8
TOTAL_OUTPUT_TOKENS = OUTPUT_TOKENS_PER_BLOCK * TOTAL_BLOCKS
MIN_CONTEXT_TOKENS = 100_000
UNAIDED_EXPERIMENT = "unaided"
ORACLE_EXECUTION_EXPERIMENT = "oracle-execution"
SEED_MODULUS = 2**31
COMPETITIONS = ("aime", "hmmt", "brumo", "smt", "bmt", "cmimc")
# CLI and vLLM served names mapped to their Hugging Face model repositories.
SUPPORTED_MODELS = {
    "qwen3-4b-instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "ministral-3-3b-instruct-2512": "mistralai/Ministral-3-3B-Instruct-2512",
    "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}
# Require problem IDs to be single folder names without separators or traversal paths.
SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
# (k=8 batch, k=1 batch) tuned on 2x L40S. KV use grows with k, so other budgets scale the k=8
# batch by 8/k, capped at the k=1 batch. Standalone serving defaults to the k=8 batch;
# Slurm passes the selected harness batch explicitly to --max-num-seqs.
DEFAULT_BATCH_SIZES = {
    "qwen3-4b-instruct-2507": (24, 128),
    "qwen3-1.7b": (32, 128),
    "ministral-3-3b-instruct-2512": (16, 128),
    "deepseek-r1-distill-qwen-1.5b": (32, 256),
}
DEFAULT_INJECTION = 0
DEFAULT_SOLUTION_INDEX: int | None = None
UNAIDED_SOLUTION_INDEX = 0
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 20
VLLM_TIMEOUT_SECONDS = 600
GENERATION_RETRY_DELAYS = (1.0, 2.0, 4.0)
RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_PENALTY = 0.0
DEFAULT_MIN_P = 0.0
CHECKPOINT_FILENAME = "checkpoint_{k}x.json.zst"
CONVERSATION_FILENAME = "conversation.json.zst"
AUDIT_FILENAME = "audit.json"
META_FILENAME = "meta.json"
OUTPUT_UPLOAD_PATTERNS = [
    f"**/{META_FILENAME}", f"**/{CONVERSATION_FILENAME}",
    f"**/{CHECKPOINT_FILENAME.format(k='*')}", f"**/{AUDIT_FILENAME}", "**/audit.jsonl",
]
SOLVE_PHASE = "solve"
NEXT_PHASE = {"solve": "critique", "critique": "revise", "revise": "critique"}
LOCK_FILENAME = ".lock"
SUCCESS_EXIT_CODE = 0
FAILURE_EXIT_CODE = 1
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
SSE_PREFIX = "data: "
SSE_DONE = "[DONE]"
