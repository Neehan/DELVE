"""Use the PPO critic architecture with additional training trajectories that contain oracle sketches."""

from critic_only.models.ppo_critic import PPOCritic


class PPOOracleCritic(PPOCritic):
    """Train the same direct head as PPOCritic on both unaided and oracle trajectories.

    All methods are inherited. This class tells the training CLI to retain source histories containing an injected sketch. Both variants receive critic-only reference solutions.
    """
