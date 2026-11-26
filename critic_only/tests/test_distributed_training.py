"""Compare native HF token-value training and evaluation on one process versus two CPU ranks."""

import json
import os
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from datasets import Dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer, PreTrainedTokenizerFast, TrainingArguments, set_seed

from critic_only.constants import MODEL_FILENAME
from critic_only.data.critic_batcher import CriticBatcher
from critic_only.models.loader import create_critic
from critic_only.models.models import CriticConfig
from critic_only.trainer.loader import get_critic_trainer


def run_training(directory: Path, batch_size: int, accumulation: int) -> None:
    """Train one update and save ordered endpoint predictions; arguments choose the same global batch on each launch."""
    set_seed(17)
    settings = CriticConfig.model_validate_json((directory / "settings.json").read_text())
    rows = json.loads((directory / "rows.json").read_text())
    output = directory / f"batch-{batch_size}"
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")
    args = TrainingArguments(output_dir=str(output), use_cpu=True, max_steps=1,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
        per_device_eval_batch_size=2, learning_rate=0.1, optim="sgd", max_grad_norm=0, weight_decay=0,
        save_strategy="no", logging_strategy="no", report_to="none", disable_tqdm=True,
        remove_unused_columns=False, label_names=["labels"], seed=17, ddp_find_unused_parameters=False)
    model = create_critic(settings, True)
    if settings.variant == "de-residual-critic":
        with torch.no_grad():
            cast(nn.Linear, model.get_submodule("residual_head")).bias.fill_(0.2)
    trainer = get_critic_trainer(settings.variant)(model=model, args=args,
        train_dataset=Dataset.from_list(rows[:4]), data_collator=CriticBatcher(tokenizer))
    trainer.train()
    prediction = trainer.predict(cast(Any, Dataset.from_list(rows)))
    trainer.save_model(str(output))
    if trainer.is_world_process_zero():
        (output / "predictions.json").write_text(json.dumps({"logits": np.asarray(prediction.predictions).tolist(),
                                                           "labels": np.asarray(prediction.label_ids).tolist()}))


@pytest.mark.parametrize("variant", ["ppo-oracle-critic", "de-critic", "de-residual-critic"])
@pytest.mark.parametrize("mode", ["head", "lora", "full"])
def test_two_rank_training_matches_single_batch(variant: str, mode: str, critic_settings: Callable[[str], CriticConfig],
    tokenizer: PreTrainedTokenizerFast, reference_token_rows: list[dict[str, Any]], tmp_path: Path) -> None:
    """Unequal token counts and oracle weights preserve updates; odd-sized evaluation keeps exact row order."""
    settings = critic_settings(variant).model_copy(update={"training_mode": mode,
        "trainable_layers": None if mode == "head" else 1, "oracle_weight": 0.4,
        "lora_rank": 2 if mode == "lora" else None, "lora_alpha": 4 if mode == "lora" else None})
    (tmp_path / "settings.json").write_text(settings.model_dump_json())
    rows = [{**reference_token_rows[index % 2], "privileged_input_ids": [2] + [3] * length,
             "privileged_solver_mask": [False] + [True] * length, "oracle": index % 3 == 0}
            for index, length in enumerate((1, 3, 2, 5, 4))]
    (tmp_path / "rows.json").write_text(json.dumps(rows))
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    commands = [[sys.executable, __file__, str(tmp_path), "4", "1"],
        [sys.executable, "-m", "torch.distributed.run", "--master-addr=127.0.0.1", f"--master-port={port}",
         "--nproc_per_node=2", __file__, str(tmp_path), "1", "2"]]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, env=environment)
        assert result.returncode == 0, result.stdout + result.stderr
    single, distributed = (load_file(str(tmp_path / f"batch-{batch}" / MODEL_FILENAME)) for batch in (4, 1))
    assert single.keys() == distributed.keys()
    for name in single:
        torch.testing.assert_close(distributed[name], single[name], atol=1e-6, rtol=1e-5, msg=name)
    expected, actual = (json.loads((tmp_path / f"batch-{batch}" / "predictions.json").read_text()) for batch in (4, 1))
    np.testing.assert_allclose(actual["logits"], expected["logits"], atol=1e-6, rtol=1e-5)
    assert actual["labels"] == expected["labels"] == [row["labels"] for row in rows]


if __name__ == "__main__":
    # On MPS hosts, Accelerate otherwise passes accelerator device IDs to CPU Gloo barriers.
    with patch("torch._C._get_accelerator", return_value=torch.device("cpu")):
        run_training(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
