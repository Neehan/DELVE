"""Construct the selected critic and attach Qwen with LoRA adapters when requested."""

from typing import cast

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

from critic_only.constants import CPU_ATTENTION, DEFAULT_ATTENTION
from critic_only.models.base_critic import BaseCritic
from critic_only.models.de_critic import DECritic
from critic_only.models.models import CriticConfig
from critic_only.models.ppo_oracle_critic import PPOOracleCritic
from critic_only.models.ppo_critic import PPOCritic


def create_critic(settings: CriticConfig, cpu: bool) -> BaseCritic:
    """Build the selected prediction heads and optionally attach Qwen with LoRA.

    Frozen mode loads no Qwen weights. LoRA freezes pretrained Qwen weights and enables all-linear adapters and gradient checkpointing. Trainer handles device placement.

    Args:
        settings: Critic variant, dimensions, input-cache identity, and frozen/LoRA settings.
        cpu: Use FP32/SDPA for CPU LoRA or BF16/FlashAttention-2 for GPU LoRA. Frozen MLP construction is unaffected.

    Returns:
        BaseCritic: Fresh PPOCritic, PPOOracleCritic, or DECritic; call load_from_weights to restore trained weights.
    """
    classes = {"ppo-critic": PPOCritic, "ppo-oracle-critic": PPOOracleCritic, "de-critic": DECritic}
    critic = classes[settings.variant](settings)
    if settings.training_mode == "lora":
        model = AutoModel.from_pretrained(settings.model, revision=settings.revision,
                                          dtype=torch.float32 if cpu else torch.bfloat16,
                                          attn_implementation=CPU_ATTENTION if cpu else DEFAULT_ATTENTION)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        adapters = LoraConfig(r=cast(int, settings.lora_rank), lora_alpha=cast(int, settings.lora_alpha), target_modules="all-linear")
        critic.language_model = get_peft_model(model, adapters)
    return critic
