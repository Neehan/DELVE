"""Execute one solve/critique/revise trajectory and persist its progress."""

import hashlib

from discovery_execution.harness.constants import (
    CONTINUATION_EXPERIMENTS, META_FILENAME, NEXT_PHASE, ORACLE_EXPERIMENTS,
    OUTPUT_TOKENS_PER_BLOCK, PROMPTS_DIRECTORY, UNAIDED_EXPERIMENT, UNAIDED_SOLUTION_INDEX, SEED_MODULUS, SOLVE_PHASE,
)
from discovery_execution.harness.models import Checkpoint, GenerationRequest, GenerationResponse, Problem, RunConfig, RunState
from discovery_execution.harness.openai_client import OpenAIClient
from discovery_execution.harness.storage import Storage, model_output_directory


class SelfRefine:
    """Own one trajectory's inputs, prompts, state, and storage; run resumes it through the full budget."""

    def __init__(self, config: RunConfig, problem: Problem, seed: int, client: OpenAIClient) -> None:
        """Derive this trajectory's sketch, timing, and storage location from shared settings."""
        self.config = config
        self.problem = problem
        self.seed = seed
        self.client = client
        oracle = config.experiment in ORACLE_EXPERIMENTS
        self.sketch: str | None = None
        injection = config.continue_at if config.continue_at is not None else config.inject_sketch_at
        self.injection_tokens = injection * OUTPUT_TOKENS_PER_BLOCK if oracle else None
        self.token_budget = config.compute_multiplier_k * OUTPUT_TOKENS_PER_BLOCK
        if config.experiment in CONTINUATION_EXPERIMENTS:
            self.arm = f"{config.experiment}-at-{config.continue_at}x"
        elif oracle:
            self.arm = f"oracle-at-{config.inject_sketch_at}x-execution"
        else:
            self.arm = f"{UNAIDED_EXPERIMENT}-{config.compute_multiplier_k}x"
        model_directory = model_output_directory(config.output_directory, config.competition, config.year, config.model)
        self.source_directory = (model_directory / config.source_arm / problem.problem_id / f"seed_{config.source_seed}"
                                 if config.source_arm is not None else None)
        directory = model_directory / self.arm / problem.problem_id
        directory = (directory / f"seed_{config.source_seed}" / f"branch_{seed}"
                     if config.experiment in CONTINUATION_EXPERIMENTS else directory / f"seed_{seed}")
        self.prompts = {
            name: (PROMPTS_DIRECTORY / f"{name}.md").read_text(encoding="utf-8").strip()
            for name in ("solve", "critique", "revise", "sketch")
        }
        self.state = RunState.empty()
        self.storage = Storage(directory)

    async def run(self) -> None:
        """Load progress, then prepare, generate, update, and save until the budget is exhausted."""
        with self.storage:
            self._load_trajectory()
            while self.state.output_tokens < self.token_budget:
                self._add_next_instruction()
                request = self._build_request()
                response = await self.client.generate(request)
                self._record_response(request, response)
                self.storage.save(self.state)

    def _load_trajectory(self) -> None:
        """Select the sketch, validate metadata, and restore saved progress or the source prefix."""
        oracle = self.config.experiment in ORACLE_EXPERIMENTS
        solution_index = self.config.solution_index
        if solution_index is None and (self.storage.directory / META_FILENAME).exists():
            solution_index = self.storage.read_metadata()["solution_index"]
        if oracle:
            sketch_seed = self.config.source_seed if self.config.source_seed is not None else self.seed
            solution_index = self.problem.select_solution_index(sketch_seed, solution_index)
            self.sketch = self.problem.solutions[solution_index].sketch
        elif solution_index is None:
            solution_index = UNAIDED_SOLUTION_INDEX
        excluded = {"seeds", "batch_size", "output_directory"}
        if self.config.continue_at is None:
            excluded.update({"source_arm", "source_seed", "continue_at"})
        metadata = {
            **self.config.model_dump(mode="json", exclude=excluded),
            "inject_sketch_at": self.injection_tokens // OUTPUT_TOKENS_PER_BLOCK if self.injection_tokens is not None else 0,
            "solution_index": solution_index,
            "problem_id": self.problem.problem_id, "seed": self.seed, "arm": self.arm,
            # Slurm may restart the same model on a different host or port.
            "backend": self.client.configuration.model_dump(exclude={"base_url"}),
        }
        self.state = self.storage.load(metadata)
        if self.source_directory is not None and not self.state.output_tokens:
            self.state = self.storage.load_source_prefix(self.source_directory, metadata)
            self.storage.save(self.state)

    def _add_next_instruction(self) -> None:
        """Start solving, inject a due sketch, or advance after a natural response ending."""
        intervention = self._get_sketch_prompt()
        if not self.state.messages:
            instruction = self.prompts["solve"].replace("{statement}", self.problem.statement)
            if intervention is not None:
                instruction += "\n\n" + intervention
            self._record_user_message(instruction, SOLVE_PHASE, intervention is not None)
        elif intervention is not None:
            self._record_user_message(intervention, SOLVE_PHASE, True)
        elif self.state.response_finished:
            phase = NEXT_PHASE[self.state.phase]
            self._record_user_message(self.prompts[phase], phase, False)

    def _get_sketch_prompt(self) -> str | None:
        """Render the optional sketch exactly once at its configured token boundary."""
        sketch = self.sketch
        if sketch is None or self.state.injected or self.state.output_tokens != self.injection_tokens:
            return None
        return self.prompts["sketch"].replace("{sketch}", sketch)

    def _record_user_message(self, instruction: str, phase: str, injection: bool) -> None:
        """Append a user instruction and record its phase and intervention status."""
        self.state.messages.append({"role": "user", "content": instruction})
        self.state.phase = phase
        self.state.injected = self.state.injected or injection
        self.state.response_finished = False

    def _build_request(self) -> GenerationRequest:
        """Derive a reproducible sampling seed and cap output at the next block boundary."""
        identity = f"{self.arm}:{self.seed}:{len(self.state.requests)}".encode("utf-8")
        if self.config.continue_at is not None:
            identity += f":{self.config.source_arm}:{self.config.source_seed}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(identity).digest()) % SEED_MODULUS
        boundary = (self.state.output_tokens // OUTPUT_TOKENS_PER_BLOCK + 1) * OUTPUT_TOKENS_PER_BLOCK
        allowance = min(boundary, self.token_budget) - self.state.output_tokens
        continuation = self.state.messages[-1]["role"] == "assistant"
        return GenerationRequest(self.config.model, [dict(item) for item in self.state.messages],
                                 allowance, seed, continuation)

    def _record_response(self, request: GenerationRequest, response: GenerationResponse) -> None:
        """Update the conversation, token usage, and checkpoint boundaries after a complete response."""
        if request.continue_response:
            self.state.messages[-1]["content"] += response.text
        else:
            self.state.messages.append({"role": "assistant", "content": response.text})
        self.state.output_tokens += response.usage["completion_tokens"]
        self.state.response_finished = response.finish_reason == "stop"
        self.state.requests.append({"seed": request.seed, "max_tokens": request.max_tokens,
                                    "usage": response.usage, "finish_reason": response.finish_reason,
                                    "phase": self.state.phase, "injected": self.state.injected,
                                    "output_tokens": self.state.output_tokens})
        if self.state.output_tokens % OUTPUT_TOKENS_PER_BLOCK == 0:
            self.state.checkpoints.append(Checkpoint(output_tokens=self.state.output_tokens,
                                                     message_count=len(self.state.messages),
                                                     last_message_chars=len(self.state.messages[-1]["content"])))
