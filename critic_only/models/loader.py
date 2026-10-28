"""Construct the configured critic variant for both training and evaluation."""

from critic_only.models.base_critic import BaseCritic
from critic_only.models.de_critic import DECritic
from critic_only.models.models import CriticConfig
from critic_only.models.ppo_oracle_critic import PPOOracleCritic
from critic_only.models.ppo_critic import PPOCritic


def create_critic(settings: CriticConfig) -> BaseCritic:
    """Construct a fresh MLP and the configured terminal-value heads."""
    if settings.variant == "de-critic":
        return DECritic(settings)
    if settings.variant == "ppo-oracle-critic":
        return PPOOracleCritic(settings)
    if settings.variant == "ppo-critic":
        return PPOCritic(settings)
    raise ValueError(f"unknown critic variant: {settings.variant}")
