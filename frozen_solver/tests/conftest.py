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
from frozen_solver.data.models import PreparationConfig
from frozen_solver.models.models import CriticConfig


@pytest.fixture(autouse=True)
def local_dataset_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep library-generated Arrow caches inside the test's disposable directory."""
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    """Use a deterministic local tokenizer with distinct padding and unknown tokens."""
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "problem": 2, "answer": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", padding_side="right")


@pytest.fixture
def critic_settings(tmp_path: Path) -> Callable[[str], CriticConfig]:
    """Save a real tiny Qwen3 backbone so PEFT construction and checkpointing use actual APIs."""
    directory = tmp_path / "backbone"
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, head_dim=8, max_position_embeddings=256)
    Qwen3Model(config).save_pretrained(directory)

    def create(variant: str) -> CriticConfig:
        """Keep all settings except the prediction variant matched."""
        return CriticConfig.model_validate(dict(backbone=str(directory), revision="local", variant=variant, horizon=3,
                                                reference_mode="none", lora_rank=2, lora_alpha=4,
                                                oracle_weight=1, correction_weight=0.01, attention="eager"))

    return create


@pytest.fixture
def prepared_inputs(tmp_path: Path) -> PreparationConfig:
    """Create one train-era and one test-era problem, with an observed success that later regresses."""
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
        source.mkdir(parents=True)
        (source / "meta.json").write_text(json.dumps({**common, "experiment": "unaided", "seed": 0}))
        snapshot = {"output_tokens": OUTPUT_TOKENS_PER_BLOCK, "messages": messages}
        (source / "checkpoint_1x.json.zst").write_bytes(zstandard.compress(json.dumps(snapshot).encode()))
        state = {**snapshot, "phase": "solve", "injected": False, "response_finished": False,
                 "requests": [{"output_tokens": OUTPUT_TOKENS_PER_BLOCK, "phase": "solve", "finish_reason": "length", "injected": False}],
                 "checkpoints": [{"output_tokens": OUTPUT_TOKENS_PER_BLOCK, "message_count": len(messages),
                                  "last_message_chars": len(messages[-1]["content"])}]}
        (source / "conversation.json.zst").write_bytes(zstandard.compress(json.dumps(state).encode()))
        for condition in ("continue", "continue-oracle"):
            for seed in range(2):
                branch = model_root / f"{condition}-at-1x" / problem["problem_id"] / "seed_0" / f"branch_{seed}"
                branch.mkdir(parents=True)
                metadata = {**common, "experiment": condition, "seed": seed, "source_seed": 0, "source_arm": "unaided-3x",
                            "continue_at": 1, "solution_index": 0}
                (branch / "meta.json").write_text(json.dumps(metadata))
                continued = messages + [{"role": "user", "content": "Use symmetry."}] if condition == "continue-oracle" else messages
                next_snapshot = {"output_tokens": 2 * OUTPUT_TOKENS_PER_BLOCK, "messages": continued}
                (branch / "checkpoint_2x.json.zst").write_bytes(zstandard.compress(json.dumps(next_snapshot).encode()))
                audit = {"complete": True, "output_tokens": 3 * OUTPUT_TOKENS_PER_BLOCK,
                         "checkpoints": {"2x": {"correctness": {"score": 1}}, "3x": {"correctness": {"score": seed}}}}
                (branch / "audit.json").write_text(json.dumps(audit))
    return PreparationConfig(outputs_directory=root, datasets_directory=references.parent, destination=tmp_path / "prepared",
                             solver_model="tiny", horizon=3, test_year=2026, validation_fraction=0.2,
                             split_seed=42, source_revision="fixture")


@pytest.fixture
def training_rows() -> list[dict[str, Any]]:
    """Vary lengths and include missing oracle supervision in one row."""
    return [{"input_ids": [2, 3, 2], "attention_mask": [1, 1, 1], "length": 3,
             "labels": [[[1, 2], [0, 2]], [[2, 2], [1, 2]]]},
            {"input_ids": [2, 3], "attention_mask": [1, 1], "length": 2,
             "labels": [[[0, 2], [1, 2]], [[0, 0], [0, 0]]]}]
