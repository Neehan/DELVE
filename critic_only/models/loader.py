"""Construct the configured critic variant for both training and evaluation."""

import torch

from critic_only.models.base_critic import BaseCritic
from critic_only.models.de_critic import DECritic
from critic_only.models.models import CriticConfig
from critic_only.models.oracle_supervised import OracleSupervised
from critic_only.models.ordinary import Ordinary


def create_critic_from_pretrained(settings: CriticConfig, dtype: torch.dtype) -> BaseCritic:
    """Load a pretrained backbone and fresh heads for the validated critic variant."""
    if settings.variant == "de":
        return DECritic.create_from_pretrained(settings, dtype)
    if settings.variant == "oracle-supervised":
        return OracleSupervised.create_from_pretrained(settings, dtype)
    if settings.variant == "ordinary":
        return Ordinary.create_from_pretrained(settings, dtype)
    raise ValueError(f"unknown critic variant: {settings.variant}")
