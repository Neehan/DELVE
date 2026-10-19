"""Model configuration shared by training and checkpoint restoration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from discovery_execution.harness.constants import MAX_BLOCKS


class CriticConfig(BaseModel):
    """Pin the backbone and adapter capacity independently of the prediction variant."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    backbone: str
    revision: str
    reference_mode: Literal["none", "solution"]
    variant: Literal["direct", "auxiliary", "de"]
    horizon: int = Field(ge=2, le=MAX_BLOCKS)
    lora_rank: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    oracle_weight: float = Field(ge=0)
    correction_weight: float = Field(ge=0)
    attention: Literal["sdpa", "flash_attention_2", "eager"]
