"""Reconstruct finer checkpoints from request text while preserving exact saved boundaries."""

from typing import Any, Literal, cast

from transformers import PreTrainedTokenizerBase

from discovery_execution.audit.answer_grader import AnswerGrader
from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK
from discovery_execution.harness.models import Checkpoint, RunState
from discovery_execution.processing.constants import MAX_CHECKPOINT_TOKEN_DRIFT
from discovery_execution.processing.models import GradedCheckpoint
from discovery_execution.processing.storage import checkpoint_snapshot


class CheckpointReconstructor:
    """Build graded analysis checkpoints with reconstruct; never modify the source state.

    The tokenizer must match generation. Original boundaries keep their reviewed grades. Intermediate cuts use offsets within each original request, not a retokenized whole conversation. Unrecoverable token-count drift raises an error.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, interval_tokens: int) -> None:
        """Store tokenizer and analysis interval; reject unsupported intervals before reading data."""
        if not tokenizer.is_fast or interval_tokens <= 0 or OUTPUT_TOKENS_PER_BLOCK % interval_tokens:
            raise ValueError("analysis interval must divide the generation block; a fast tokenizer is required")
        self.tokenizer = tokenizer
        self.interval_tokens = interval_tokens

    def reconstruct(self, state: RunState, audit: dict[str, Any], grader: AnswerGrader) -> list[GradedCheckpoint]:
        """Return all graded boundaries, using state/audit for exact cuts and grader for new answers."""
        saved = self._saved_checkpoints(state, audit)
        known_answers: dict[str, Literal[0, 1]] = {grader._solver_answer(checkpoint_snapshot(state, item)["messages"]): item.correctness
                         for item in saved.values()}
        result = dict(saved)
        targets = set(range(self.interval_tokens, state.output_tokens + 1, self.interval_tokens))
        missing = targets - result.keys()
        if missing:
            for start, end, message_index, start_char, end_char in self._request_spans(state):
                selected = sorted(target for target in missing if start < target <= end)
                if not selected:
                    continue
                text = state.messages[message_index]["content"][start_char:end_char]
                # Natural request endings are exact even when repetitive text retokenizes poorly.
                offsets = self._token_offsets(text, end - start) if any(target != end for target in selected) else []
                for target in selected:
                    offset = target - start
                    if target != end and offset > len(offsets):
                        raise ValueError("retokenized response does not reach the requested cut")
                    char_end = end_char if target == end else start_char + offsets[offset - 1][1]
                    boundary = Checkpoint(output_tokens=target, message_count=message_index + 1, last_message_chars=char_end)
                    answer = grader._solver_answer(checkpoint_snapshot(state, boundary)["messages"])
                    if answer not in known_answers:
                        known_answers[answer] = 1 if grader.check(answer) else 0
                    result[target] = GradedCheckpoint(**boundary.model_dump(), correctness=known_answers[answer],
                        boundary_source="request" if target == end else "reconstructed")
        if not targets <= result.keys():
            raise ValueError("requests do not cover all analysis checkpoints")
        return [result[target] for target in sorted(targets)]

    def _saved_checkpoints(self, state: RunState, audit: dict[str, Any]) -> dict[int, GradedCheckpoint]:
        """Validate state/audit boundaries and return exact saved grades indexed by token count."""
        expected = list(range(OUTPUT_TOKENS_PER_BLOCK, state.output_tokens + 1, OUTPUT_TOKENS_PER_BLOCK))
        if [item.output_tokens for item in state.checkpoints] != expected:
            raise ValueError("missing or duplicate saved checkpoint boundary")
        request_ends = {request["output_tokens"] for request in state.requests}
        if not set(expected) <= request_ends:
            raise ValueError("saved checkpoint lacks its generation request")
        expected_keys = {f"{tokens // OUTPUT_TOKENS_PER_BLOCK}x" for tokens in expected}
        if set(audit["checkpoints"]) != expected_keys:
            raise ValueError("missing or unexpected audit checkpoints")
        return {item.output_tokens: GradedCheckpoint(**item.model_dump(), boundary_source="saved",
                correctness=audit["checkpoints"][f"{item.output_tokens // OUTPUT_TOKENS_PER_BLOCK}x"]["correctness"]["score"])
                for item in state.checkpoints}

    def _request_spans(self, state: RunState) -> list[tuple[int, int, int, int, int]]:
        """Return each request's token range, assistant-message index and character range from state."""
        assistants = [index for index, message in enumerate(state.messages) if message["role"] == "assistant"]
        boundaries = {item.output_tokens: item for item in state.checkpoints}
        spans = []
        assistant_index, start_tokens, start_char = -1, 0, 0
        previous_finish = "stop"
        for request in state.requests:
            if previous_finish == "stop":
                assistant_index += 1
                start_char = 0
            message_index = assistants[assistant_index]
            end_tokens = request["output_tokens"]
            if request["usage"]["completion_tokens"] != end_tokens - start_tokens:
                raise ValueError("request usage disagrees with cumulative generation count")
            finish = request["finish_reason"]
            if finish == "length":
                boundary = boundaries[end_tokens]
                if boundary.message_count != message_index + 1:
                    raise ValueError("request and checkpoint refer to different responses")
                end_char = boundary.last_message_chars
            elif finish == "stop":
                end_char = len(state.messages[message_index]["content"])
            else:
                raise ValueError(f"unsupported request finish reason: {finish}")
            if end_tokens <= start_tokens or end_char < start_char:
                raise ValueError("request token/character boundaries are not increasing")
            spans.append((start_tokens, end_tokens, message_index, start_char, end_char))
            start_tokens, start_char, previous_finish = end_tokens, end_char, finish
        if start_tokens != state.output_tokens:
            raise ValueError("requests do not reach the saved final budget")
        return spans

    def _token_offsets(self, text: str, generated_tokens: int) -> list[tuple[int, int]]:
        """Retokenize one response chunk; return offsets or reject excessive drift from generated_tokens."""
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = cast(list[tuple[int, int]], encoded["offset_mapping"])
        if abs(len(offsets) - generated_tokens) > MAX_CHECKPOINT_TOKEN_DRIFT:
            raise ValueError(f"cannot reconstruct checkpoint reliably: generated {generated_tokens} tokens, retokenized {len(offsets)}; review source trajectory")
        return offsets
