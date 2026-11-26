"""Select the trainer whose objective matches the saved critic variant."""

from critic_only.trainer.base_trainer import BaseTrainer
from critic_only.trainer.de_residual_trainer import DEResidualTrainer


def get_critic_trainer(variant: str) -> type[BaseTrainer]:
    """Return the trainer class for variant, shared by training and evaluation CLIs.

    Args:
        variant: Critic variant from CLI options or a checkpoint's configuration.

    Returns:
        type[BaseTrainer]: The residual trainer for de-residual-critic, otherwise the standard BCE trainer.
    """
    trainers = {"ppo-critic": BaseTrainer, "ppo-oracle-critic": BaseTrainer,
                "de-critic": BaseTrainer, "de-residual-critic": DEResidualTrainer}
    return trainers[variant]
