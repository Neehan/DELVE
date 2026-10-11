"""Validated dataset, request, and JSON checkpoint models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from discovery_execution.harness.constants import (
    COMPETITIONS, SAFE_COMPONENT, SUPPORTED_MODELS, TOTAL_BLOCKS, TOTAL_OUTPUT_TOKENS,
)


class Solution(BaseModel):
    """A source-tagged solution and its corresponding strategy sketch."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    source: Literal["reference", "ai_generated"]
    solution: str
    sketch: str

    @field_validator("solution", "sketch")
    @classmethod
    def nonempty(cls, value: str) -> str:
        """Reject empty solution material."""
        if not value.strip():
            raise ValueError("solution and sketch must be nonempty")
        return value


class Problem(BaseModel):
    """The exact dataset row; references never enter unaided solver requests."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    problem_id: str
    competition: str
    year: int
    statement: str
    final_answer: str
    solutions: list[Solution]

    @model_validator(mode="after")
    def validate_problem(self) -> "Problem":
        """Validate identity, supported partition, and reference availability."""
        if not SAFE_COMPONENT.fullmatch(self.problem_id):
            raise ValueError("problem_id must be a safe nonempty path component")
        if self.competition not in COMPETITIONS or self.year < 1:
            raise ValueError("competition must be supported and year must be positive")
        if not self.statement.strip() or not self.final_answer.strip() or not self.solutions:
            raise ValueError("statement, final_answer, and solutions must be nonempty")
        return self


class DatasetConfig(BaseModel):
    """Identify one competition/year partition and its local or Hugging Face source."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    competition: str
    year: int
    datasets_directory: Path
    hf_repo: str | None

    @model_validator(mode="after")
    def validate_partition(self) -> "DatasetConfig":
        """Reject unsupported partitions before reading or downloading data."""
        if self.competition not in COMPETITIONS or self.year < 1:
            raise ValueError("competition must be supported and year must be positive")
        return self


class RunConfig(DatasetConfig):
    """Validated run selection with a chosen unaided budget or fixed eight-block oracle budget."""

    output_directory: Path
    model: str
    experiment: Literal["unaided", "oracle-execution"]
    compute_multiplier_k: int
    inject_sketch_at: int
    solution_index: int | None
    seeds: list[int]
    batch_size: int

    @model_validator(mode="after")
    def validate_selection(self) -> "RunConfig":
        """Reject invalid run combinations before reading or generating data."""
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"unsupported model: {self.model}; choose from {', '.join(SUPPORTED_MODELS)}")
        if not 1 <= self.compute_multiplier_k <= TOTAL_BLOCKS:
            raise ValueError("compute-multiplier-k must be 1..8")
        if self.experiment == "oracle-execution" and self.compute_multiplier_k != TOTAL_BLOCKS:
            raise ValueError("oracle-execution requires compute-multiplier-k=8")
        if not 0 <= self.inject_sketch_at < TOTAL_BLOCKS:
            raise ValueError("injection must be 0..7")
        if self.solution_index is not None and self.solution_index < 0:
            raise ValueError("solution index must be nonnegative")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        if not self.seeds or min(self.seeds) < 0 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique nonnegative integers")
        if self.experiment == "unaided" and (self.inject_sketch_at or self.solution_index):
            raise ValueError("sketch options apply only to oracle-execution")
        return self


class BackendConfig(BaseModel):
    """Serializable inference settings, excluding credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    base_url: str
    temperature: float
    top_p: float
    top_k: int
    timeout: float

    @model_validator(mode="after")
    def validate_sampling(self) -> "BackendConfig":
        """Reject invalid endpoints and nonsensical sampling settings."""
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password:
            raise ValueError("VLLM_BASE_URL must be an HTTP(S) URL without embedded credentials")
        if not url.path.endswith("/v1") or url.query or url.fragment:
            raise ValueError("VLLM_BASE_URL must end in /v1 with no query or fragment")
        if not 0 < self.temperature or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be positive; top_p must be in (0,1]")
        if self.top_k < 1 or self.timeout <= 0:
            raise ValueError("top_k and timeout must be positive")
        return self


@dataclass(frozen=True)
class GenerationRequest:
    """OpenAI messages and the remaining allowance for one server request."""

    model: str
    messages: list[dict[str, str]]
    max_tokens: int
    seed: int
    continue_response: bool


@dataclass(frozen=True)
class GenerationResponse:
    """A completed model response with validated server usage."""

    text: str
    finish_reason: str
    usage: dict[str, Any]


class Checkpoint(BaseModel):
    """Locate an exact block boundary in the accumulated conversation."""

    model_config = ConfigDict(extra="forbid", strict=True)
    output_tokens: int
    message_count: int
    last_message_chars: int


class RunState(BaseModel):
    """Directly persisted state; correctness is intentionally absent from control state."""

    model_config = ConfigDict(extra="forbid", strict=True)
    messages: list[dict[str, str]]
    output_tokens: int
    phase: str
    injected: bool
    response_finished: bool
    requests: list[dict[str, Any]]
    checkpoints: list[Checkpoint]

    @classmethod
    def empty(cls) -> "RunState":
        """Construct a fresh attempt without mutable shared defaults."""
        return cls(messages=[], output_tokens=0, phase="solve", injected=False,
                   response_finished=False,
                   requests=[], checkpoints=[])

    @model_validator(mode="after")
    def validate_state(self) -> "RunState":
        """Reject corrupted counters and invalid checkpoint references."""
        if not 0 <= self.output_tokens <= TOTAL_OUTPUT_TOKENS:
            raise ValueError("invalid checkpoint token count")
        for checkpoint in self.checkpoints:
            if not 0 < checkpoint.message_count <= len(self.messages):
                raise ValueError("invalid checkpoint message count")
            message = self.messages[checkpoint.message_count - 1]
            if not 0 <= checkpoint.last_message_chars <= len(message["content"]):
                raise ValueError("invalid checkpoint character boundary")
        return self
