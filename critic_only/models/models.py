"""Define and validate the model settings saved with each critic checkpoint."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from discovery_execution.harness.constants import MAX_BLOCKS


class CriticConfig(BaseModel):
    """Store the architecture and input identity needed to train or restore a critic.

    Fields:
        model: Pretrained Qwen repository ID or local path.
        revision: Exact pretrained model revision.
        variant: ppo-critic, ppo-oracle-critic, or de-critic.
        horizon: Total solver budget in blocks, including the observed history.
        input_size: Number of values in one Qwen hidden vector.
        hidden_size: Width of both GELU MLP layers and input width of the prediction heads.
        data_signature: Hash of the prepared input cache, checked when resuming.
        training_mode: frozen reads cached vectors; lora runs Qwen with trainable adapters.
        lora_rank: Adapter rank in LoRA mode; None in frozen mode.
        lora_alpha: Adapter scaling numerator (scale = alpha/rank); unrelated to DE alpha. None in frozen mode.
        oracle_weight: Loss multiplier for solver histories containing sketches; unaided weight is one.

    validate_adapter_settings checks that adapter options match training_mode. The configuration is saved with each checkpoint.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str
    revision: str
    variant: Literal["ppo-critic", "ppo-oracle-critic", "de-critic"]
    horizon: int = Field(ge=2, le=MAX_BLOCKS)
    input_size: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    data_signature: str
    training_mode: Literal["frozen", "lora"]
    lora_rank: int | None = Field(gt=0)
    lora_alpha: int | None = Field(gt=0)
    oracle_weight: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_adapter_settings(self) -> Self:
        """Reject missing LoRA settings or adapter settings supplied for frozen mode.

        Returns:
            Self: Validated configuration; incompatible mode/adapter combinations raise ValueError.
        """
        if self.training_mode == "lora" and (self.lora_rank is None or self.lora_alpha is None):
            raise ValueError("LoRA requires rank and alpha")
        if self.training_mode == "frozen" and (self.lora_rank is not None or self.lora_alpha is not None):
            raise ValueError("frozen critics do not use LoRA settings")
        return self
