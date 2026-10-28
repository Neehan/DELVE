"""Model configuration shared by training and checkpoint restoration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from discovery_execution.harness.constants import MAX_BLOCKS


class CriticConfig(BaseModel):
    """Describe a feature cache and matched MLP capacity for one prediction variant."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str
    revision: str
    variant: Literal["ppo-critic", "ppo-oracle-critic", "de-critic"]
    horizon: int = Field(ge=2, le=MAX_BLOCKS)
    input_size: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    feature_signature: str
    oracle_weight: float = Field(ge=0)
