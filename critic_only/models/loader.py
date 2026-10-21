"""Construct the configured critic variant for both training and evaluation."""

import torch

from critic_only.models.base_critic import BaseCritic
from critic_only.models.de_critic import DECritic
from critic_only.models.models import CriticConfig
from critic_only.models.ppo_oracle_critic import PPOOracleCritic
from critic_only.models.ppo_critic import PPOCritic


def create_critic_from_pretrained(settings: CriticConfig, dtype: torch.dtype) -> BaseCritic:
    """Load a pretrained model and fresh heads for the validated critic variant."""
    if settings.variant == "de-critic":
        return DECritic.create_from_pretrained(settings, dtype)
    if settings.variant == "ppo-oracle-critic":
        return PPOOracleCritic.create_from_pretrained(settings, dtype)
    if settings.variant == "ppo-critic":
        return PPOCritic.create_from_pretrained(settings, dtype)
    raise ValueError(f"unknown critic variant: {settings.variant}")
