"""Direct value prediction with auxiliary supervision from oracle continuations."""

from critic_only.models.ppo_critic import PPOCritic


class PPOOracleCritic(PPOCritic):
    """Train the same direct head on unaided histories and histories containing the oracle sketch and the solver's subsequent reasoning."""
