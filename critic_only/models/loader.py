"""Construct a critic using cached vectors, LoRA, or fully trainable final Qwen blocks."""

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
    """Build prediction heads and attach Qwen when training on tokens.

    Frozen mode loads no Qwen weights. LoRA adapts selected final layers. Full mode trains all parameters in selected final blocks and freezes the rest, including the output normalization. Trainer handles device placement and BF16 autocast; trainable full-block weights remain FP32.

    Args:
        settings: Critic variant, dimensions, input-cache identity, and training-mode settings.
        cpu: Use FP32/SDPA for CPU execution or BF16/FlashAttention-2 for GPU execution. Frozen MLP construction is unaffected.

    Returns:
        BaseCritic: Fresh PPOCritic, PPOOracleCritic, or DECritic; call load_from_weights to restore trained weights.
    """
    classes = {"ppo-critic": PPOCritic, "ppo-oracle-critic": PPOOracleCritic, "de-critic": DECritic}
    critic = classes[settings.variant](settings)
    if settings.training_mode != "frozen":
        model = AutoModel.from_pretrained(settings.model, revision=settings.revision,
                                          dtype=torch.float32 if cpu else torch.bfloat16,
                                          attn_implementation=CPU_ATTENTION if cpu else DEFAULT_ATTENTION)
        model.config.use_cache = False
        first_trainable_layer = max(0, model.config.num_hidden_layers - cast(int, settings.trainable_layers))
        if settings.training_mode == "lora":
            critic.language_model = get_peft_model(model, adapter_config(settings, first_trainable_layer))
        else:
            model.requires_grad_(False)
            # FP32 trainable weights retain small updates while autocast keeps matrix products in BF16.
            for layer_index in range(first_trainable_layer, model.config.num_hidden_layers):
                model.get_submodule(f"layers.{layer_index}").float().requires_grad_(True)
            critic.language_model = model
        # Non-reentrant checkpointing supports gradients in selected blocks without embedding-output gradients.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.disable_input_require_grads()
    return critic


def adapter_config(settings: CriticConfig, first_trainable_layer: int) -> LoraConfig:
    """Configure adapters while excluding blocks before the selected suffix.

    Args:
        settings: Validated LoRA rank and scaling settings.
        first_trainable_layer: Zero-based index of the first block receiving adapters.

    Returns:
        LoraConfig: Adapters on every linear projection within the selected blocks.
    """
    # PEFT's all-linear target supports module exclusions, but not layers_to_transform.
    excluded_modules = None
    if first_trainable_layer > 0:
        layer_indices = "|".join(str(index) for index in range(first_trainable_layer))
        excluded_modules = rf"(?:.*\.)?layers\.(?:{layer_indices})\..*"
    return LoraConfig(r=cast(int, settings.lora_rank), lora_alpha=cast(int, settings.lora_alpha),
                      target_modules="all-linear", exclude_modules=excluded_modules)
