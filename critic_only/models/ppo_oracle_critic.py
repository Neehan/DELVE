"""Use the PPO critic architecture with additional training trajectories that contain oracle sketches."""

from critic_only.models.ppo_critic import PPOCritic


class PPOOracleCritic(PPOCritic):
    """Train the same direct head as PPOCritic on both unaided and oracle trajectories.

    All methods are inherited. This class tells the training CLI to retain histories containing a sketch and the solver's response. Critic-only reference solutions are a separate input choice used by both variants.
    """
