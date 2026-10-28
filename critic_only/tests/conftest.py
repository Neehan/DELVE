"""Small local transformer, tokenizer, and exact DELVE output fixtures; no network or GPUs."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import zstandard
from datasets import config as datasets_config
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK
from discovery_execution.vllm.constants import TEMPLATES_DIRECTORY
from critic_only.tests.models import CollectionFixture
from critic_only.models.models import CriticConfig


@pytest.fixture(autouse=True)
def local_dataset_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep library-generated Arrow caches inside the test's disposable directory."""
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    """Use a deterministic local tokenizer with distinct padding and unknown tokens."""
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "problem": 2, "answer": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", padding_side="right",
                                   chat_template=(TEMPLATES_DIRECTORY / "qwen3.jinja").read_text())


@pytest.fixture
def critic_settings(tmp_path: Path) -> Callable[[str], CriticConfig]:
    """Save a real tiny Qwen3 model for extraction tests; MLP settings remain independent of model loading."""
    directory = tmp_path / "model"
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, head_dim=8, max_position_embeddings=256)
    Qwen3Model(config).save_pretrained(directory)

    def create(variant: str) -> CriticConfig:
        """Keep all settings except the prediction variant matched."""
        return CriticConfig.model_validate(dict(model=str(directory), revision="local", variant=variant, horizon=3,
                                                input_size=16, hidden_size=16, feature_signature="test-signature",
                                                oracle_weight=1))

    return create


def write_run(path: Path, metadata: dict[str, Any], histories: list[list[dict[str, str]]], outcomes: list[int]) -> None:
    """Save real checkpoint boundaries and later conversation text for processor integration tests."""
    path.mkdir(parents=True)
    (path / "meta.json").write_text(json.dumps(metadata))
    fork = metadata.get("continue_at", 0)
    oracle = metadata["experiment"] in ("oracle-execution", "continue-oracle")
    checkpoints, requests = [], []
    for block, messages in enumerate(histories, start=fork + 1):
        snapshot = {"output_tokens": block * OUTPUT_TOKENS_PER_BLOCK, "messages": messages}
        (path / f"checkpoint_{block}x.json.zst").write_bytes(zstandard.compress(json.dumps(snapshot).encode()))
        checkpoints.append({"output_tokens": snapshot["output_tokens"], "message_count": len(messages),
                            "last_message_chars": len(messages[-1]["content"])})
        requests.append({"output_tokens": snapshot["output_tokens"], "phase": "critique" if block == 2 else "revise",
                         "finish_reason": "stop" if block == 2 else "length", "injected": oracle})
    state = {"output_tokens": (fork + len(histories)) * OUTPUT_TOKENS_PER_BLOCK, "messages": histories[-1],
             "phase": "revise", "injected": oracle, "response_finished": False,
             "requests": requests, "checkpoints": checkpoints}
    (path / "conversation.json.zst").write_bytes(zstandard.compress(json.dumps(state).encode()))
    audit = {"complete": True, "output_tokens": len(histories) * OUTPUT_TOKENS_PER_BLOCK + fork * OUTPUT_TOKENS_PER_BLOCK,
             "checkpoints": {f"{block}x": {"correctness": {"score": score}}
                             for block, score in enumerate(outcomes, start=fork + 1)}}
    (path / "audit.json").write_text(json.dumps(audit))


@pytest.fixture
def prepared_inputs(tmp_path: Path) -> CollectionFixture:
    """Save root and forked runs with distinct observed reasoning and later correct-to-incorrect revisions."""
    root = tmp_path / "outputs"
    references = tmp_path / "datasets" / "aime"
    references.mkdir(parents=True)
    for year in (2025, 2026):
        problem = {"problem_id": f"AIME-{year}-I-1", "competition": "aime", "year": year,
                   "statement": "Find the number.", "final_answer": "1",
                   "solutions": [{"source": "reference", "solution": "The answer is one.", "sketch": "Use symmetry."}]}
        (references / f"{year}.jsonl").write_text(json.dumps(problem) + "\n")
        model_root = root / "aime" / str(year) / "tiny"
        common = {"competition": "aime", "year": year, "model": "tiny", "problem_id": problem["problem_id"],
                  "backend": {"temperature": 0.8, "top_p": 0.95, "top_k": 20}, "compute_multiplier_k": 3}
        messages = [{"role": "user", "content": "Find the number."}, {"role": "assistant", "content": "Thinking"}]
        source = model_root / "unaided-3x" / problem["problem_id"] / "seed_0"
        write_run(source, {**common, "experiment": "unaided", "seed": 0},
                  [messages, messages + [{"role": "assistant", "content": "Parent later reasoning"}],
                   messages + [{"role": "assistant", "content": "Parent later reasoning FINAL FUTURE"}]], [1, 1, 0])
        for condition in ("continue", "continue-oracle"):
            for seed in range(2):
                branch = model_root / f"{condition}-at-1x" / problem["problem_id"] / "seed_0" / f"branch_{seed}"
                metadata = {**common, "experiment": condition, "seed": seed, "source_seed": 0, "source_arm": "unaided-3x",
                            "continue_at": 1, "solution_index": 0}
                continued = messages + [{"role": "user", "content": "Use symmetry."}] if condition == "continue-oracle" else messages
                observed = continued + [{"role": "assistant", "content": f"Observed {condition} branch {seed}"}]
                final = continued + [{"role": "assistant", "content": f"Observed {condition} branch {seed} FINAL FUTURE"}]
                write_run(branch, metadata, [observed, final], [1, seed])
    extra = {**json.loads((references / "2025.jsonl").read_text()), "problem_id": "AIME-2025-I-2"}
    with (references / "2025.jsonl").open("a") as file:
        file.write(json.dumps(extra) + "\n")
    return CollectionFixture(outputs_directory=root, datasets_directory=references.parent, destination=tmp_path / "prepared",
                             solver_model="tiny", horizon=3)


@pytest.fixture
def replace_saved_text() -> Callable[[Path, str, str], None]:
    """Update test checkpoint text and its conversation copy without invalidating boundary offsets."""
    def replace(directory: Path, old: str, new: str) -> None:
        """Replace text in fixture messages; the edited instruction precedes the final assistant message."""
        for path in directory.glob("*.zst"):
            data = json.loads(zstandard.decompress(path.read_bytes()))
            for message in data["messages"]:
                message["content"] = message["content"].replace(old, new)
            path.write_bytes(zstandard.compress(json.dumps(data).encode()))
    return replace


@pytest.fixture
def token_rows() -> list[dict[str, Any]]:
    """Vary lengths and include missing oracle supervision and reference input in one row."""
    return [{"problem_key": "problem-A", "input_ids": [2, 3, 2], "length": 5,
             "privileged_input_ids": [2, 3, 2, 3, 3],
             "oracle": False, "remaining_blocks": 2, "labels": 0},
            {"problem_key": "problem-B", "input_ids": [2, 3], "length": 2,
             "privileged_input_ids": None,
             "oracle": False, "remaining_blocks": 1, "labels": 1}]


@pytest.fixture
def training_rows(token_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use fixed vectors with one missing reference to exercise paired loss weighting."""
    return [{"problem_key": row["problem_key"], "hidden_state": [float(index + 1)] * 16,
             "privileged_hidden_state": [0.5] * 16 if row["privileged_input_ids"] is not None else None,
             "oracle": row["oracle"], "remaining_blocks": row["remaining_blocks"], "labels": row["labels"]}
            for index, row in enumerate(token_rows)]
