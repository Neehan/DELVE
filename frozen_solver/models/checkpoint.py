"""Build pinned PEFT critics and save only trainable adapters and prediction heads."""

import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModel

from frozen_solver.constants import CONFIG_FILENAME, MODEL_FILENAME
from frozen_solver.models.base_critic import BaseCritic
from frozen_solver.models.de_critic import DECritic
from frozen_solver.models.direct_critic import DirectCritic
from frozen_solver.models.models import CriticConfig


def build_critic(settings: CriticConfig, dtype: torch.dtype) -> BaseCritic:
    """Create adapters on attention and MLP layers without loading an unused language-model head."""
    backbone = AutoModel.from_pretrained(settings.backbone, revision=settings.revision,
                                         dtype=dtype, attn_implementation=settings.attention)
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    adapter = LoraConfig(r=settings.lora_rank, lora_alpha=settings.lora_alpha, target_modules="all-linear")
    adapted = get_peft_model(backbone, adapter)
    critic_type = DECritic if settings.variant == "de" else DirectCritic
    return critic_type(adapted, backbone.config.hidden_size, settings)


def save_critic(model: BaseCritic, directory: Path) -> None:
    """Save trainable weights and architecture; HF Trainer separately owns optimizer/RNG state."""
    directory.mkdir(parents=True, exist_ok=True)
    weights = {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters()
               if parameter.requires_grad}
    save_file(weights, str(directory / MODEL_FILENAME))
    (directory / CONFIG_FILENAME).write_text(model.settings.model_dump_json(indent=2) + "\n")


def load_critic_weights(model: BaseCritic, directory: Path) -> None:
    """Reject architecture or adapter mismatches before restoring a checkpoint."""
    settings = CriticConfig.model_validate(json.loads((directory / CONFIG_FILENAME).read_text()))
    if settings != model.settings:
        raise ValueError("critic checkpoint configuration differs")
    weights = load_file(str(directory / MODEL_FILENAME))
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if parameters.keys() != weights.keys():
        raise ValueError("critic checkpoint trainable parameter names differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(weights[name])
