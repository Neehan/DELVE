"""Direct value prediction with auxiliary supervision from oracle continuations."""

from critic_only.models.ppo_critic import PPOCritic


class PPOOracleCritic(PPOCritic):
    """Train the same direct head on ordinary histories and histories containing the oracle sketch."""
