"""Define and validate the model settings saved with each critic checkpoint."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from discovery_execution.harness.constants import MAX_BLOCKS


class CriticConfig(BaseModel):
    """Store the architecture and input identity needed to train or restore a critic.

    Fields:
        model: Pretrained Qwen repository ID or local path.
        revision: Exact pretrained model revision.
        variant: ppo-critic, ppo-oracle-critic, de-critic, or de-residual-critic.
        horizon: Total solver budget in blocks, including the observed history.
        input_size: Number of values in one Qwen hidden vector.
        hidden_size: Width of both GELU MLP layers and input width of the prediction heads.
        data_signature: Hash of the prepared input cache, checked when resuming.
        training_mode: head runs frozen Qwen on tokens; lora/full adapt selected final blocks.
        lora_rank: Adapter rank in LoRA mode; None otherwise.
        lora_alpha: Adapter scaling numerator (scale = alpha/rank); unrelated to DE alpha. None outside LoRA mode.
        trainable_layers: Number of final blocks to train in LoRA/full modes, capped at model depth; None in head mode.
        oracle_weight: Loss multiplier for solver histories containing sketches; unaided weight is one.
        residual_penalty: Squared logit-correction penalty for de-residual-critic; None for other variants.

    validate_training_settings checks layer/adapter options and the variant-specific penalty. The configuration is saved with each checkpoint.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str
    revision: str
    variant: Literal["ppo-critic", "ppo-oracle-critic", "de-critic", "de-residual-critic"]
    horizon: int = Field(ge=2, le=MAX_BLOCKS)
    input_size: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    data_signature: str
    training_mode: Literal["head", "lora", "full"]
    lora_rank: int | None = Field(gt=0)
    lora_alpha: int | None = Field(gt=0)
    trainable_layers: int | None = Field(default=None, gt=0)
    oracle_weight: float = Field(ge=0)
    residual_penalty: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_training_settings(self) -> Self:
        """Reject layer, adapter or penalty settings incompatible with the selected critic.

        Returns:
            Self: Validated configuration; incompatible training or variant settings raise ValueError.
        """
        if self.training_mode == "lora" and (self.lora_rank is None or self.lora_alpha is None):
            raise ValueError("LoRA requires rank and alpha")
        if self.training_mode != "lora" and any(value is not None for value in
                                                 (self.lora_rank, self.lora_alpha)):
            raise ValueError("adapter settings require LoRA mode")
        if self.training_mode == "head" and self.trainable_layers is not None:
            raise ValueError("head critics do not use trainable layers")
        if self.training_mode in ("lora", "full") and self.trainable_layers is None:
            raise ValueError("LoRA and full training require trainable_layers")
        if (self.variant == "de-residual-critic") != (self.residual_penalty is not None):
            raise ValueError("residual_penalty is required only for de-residual-critic")
        return self
