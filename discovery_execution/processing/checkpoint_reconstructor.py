"""Reconstruct finer checkpoints from request text while preserving exact saved boundaries."""

from typing import Any, Literal, cast

from transformers import PreTrainedTokenizerBase

from discovery_execution.audit.answer_grader import AnswerGrader
from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK
from discovery_execution.harness.models import Checkpoint, RunState
from discovery_execution.processing.models import GradedCheckpoint
from discovery_execution.processing.storage import checkpoint_snapshot


class CheckpointReconstructor:
    """Build graded analysis checkpoints with reconstruct; never modify the source state.

    The tokenizer must match generation. Original boundaries keep their reviewed grades. Each original generation block is divided evenly by retokenized assistant text. These intermediate budgets are nominal; text_tokens records each segment's measured size. Prompts never count toward a cut.
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
        subdivisions = OUTPUT_TOKENS_PER_BLOCK // self.interval_tokens
        if subdivisions > 1:
            self._validate_requests(state)
        known_answers = self._reviewed_answers(state, saved, grader) if subdivisions > 1 else {}
        result = []
        previous = Checkpoint(output_tokens=0, message_count=0, last_message_chars=0)
        for endpoint in saved.values():
            token_ends = self._block_token_ends(state, previous, endpoint)
            cuts = self._equal_cuts(token_ends, subdivisions)
            start_token = 0
            for part, end_token in enumerate(cuts, start=1):
                if part == subdivisions:
                    checkpoint = endpoint
                else:
                    message_count, char_end = token_ends[end_token - 1]
                    boundary = Checkpoint(output_tokens=previous.output_tokens + part * self.interval_tokens,
                                          message_count=message_count, last_message_chars=char_end)
                    answer = grader._solver_answer(checkpoint_snapshot(state, boundary)["messages"])
                    if answer not in known_answers:
                        known_answers[answer] = 1 if grader.check(answer) else 0
                    checkpoint = GradedCheckpoint(**boundary.model_dump(), correctness=known_answers[answer],
                                                  boundary_source="reconstructed", text_tokens=0)
                checkpoint.text_tokens = end_token - start_token
                result.append(checkpoint)
                start_token = end_token
            previous = endpoint
        return result

    def _block_token_ends(self, state: RunState, start: Checkpoint, end: Checkpoint) -> list[tuple[int, int]]:
        """Return (message count, character end) for each assistant token between two saved boundaries.

        Each assistant span is tokenized separately, excluding user/system messages. Offsets refer to the unchanged conversation text, so the resulting prefix never contains text after its cut.
        """
        token_ends = []
        for index in range(max(start.message_count - 1, 0), end.message_count):
            message = state.messages[index]
            if message["role"] != "assistant":
                continue
            start_char = start.last_message_chars if index == start.message_count - 1 else 0
            end_char = end.last_message_chars if index == end.message_count - 1 else len(message["content"])
            if not 0 <= start_char <= end_char <= len(message["content"]):
                raise ValueError("saved character boundary is outside its assistant message")
            encoded = self.tokenizer(message["content"][start_char:end_char], add_special_tokens=False,
                                     return_offsets_mapping=True)
            offsets = cast(list[tuple[int, int]], encoded["offset_mapping"])
            token_ends.extend((index + 1, start_char + offset_end) for _, offset_end in offsets)
        return token_ends

    def _equal_cuts(self, token_ends: list[tuple[int, int]], subdivisions: int) -> list[int]:
        """Return cumulative token counts dividing a block equally, cutting only at complete characters.

        Multiple byte tokens can share one character end. Keep that group together and choose the nearest valid boundary; reject text too short to produce distinct segments.
        """
        candidates = [index for index in range(1, len(token_ends)) if token_ends[index - 1] != token_ends[index]]
        if len(candidates) < subdivisions - 1 or not token_ends:
            raise ValueError("not enough distinct text boundaries to subdivide generation block")
        cuts = []
        for part in range(1, subdivisions):
            remaining_cuts = subdivisions - part - 1
            available = candidates[:len(candidates) - remaining_cuts]
            selected = min(available, key=lambda index: abs(index * subdivisions - part * len(token_ends)))
            cuts.append(selected)
            candidates = [index for index in candidates if index > selected]
        return [*cuts, len(token_ends)]

    def _reviewed_answers(self, state: RunState, saved: dict[int, GradedCheckpoint],
                          grader: AnswerGrader) -> dict[str, Literal[0, 1]]:
        """Index reviewed grades by selected answer text, rejecting conflicting labels.

        Args:
            state: Full conversation, sliced only at saved boundaries.
            saved: Original checkpoints with their reviewed grades.
            grader: Selects complete solve/revise boxes without using correctness.

        Returns:
            dict: One consistent binary grade per selected answer string.
        """
        answers: dict[str, Literal[0, 1]] = {}
        for checkpoint in saved.values():
            answer = grader._solver_answer(checkpoint_snapshot(state, checkpoint)["messages"])
            if answer in answers and answers[answer] != checkpoint.correctness:
                raise ValueError("conflicting reviewed grades for the same selected answer")
            answers[answer] = checkpoint.correctness
        return answers

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
        return {item.output_tokens: GradedCheckpoint(**item.model_dump(), boundary_source="saved", text_tokens=0,
                correctness=audit["checkpoints"][f"{item.output_tokens // OUTPUT_TOKENS_PER_BLOCK}x"]["correctness"]["score"])
                for item in state.checkpoints}

    def _validate_requests(self, state: RunState) -> None:
        """Check request usage and saved text boundaries agree; raise on corrupted generation metadata."""
        assistants = [index for index, message in enumerate(state.messages) if message["role"] == "assistant"]
        boundaries = {item.output_tokens: item for item in state.checkpoints}
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
                end_char = boundary.last_message_chars
            elif finish == "stop":
                end_char = len(state.messages[message_index]["content"])
            else:
                raise ValueError(f"unsupported request finish reason: {finish}")
            if end_tokens in boundaries:
                saved = boundaries[end_tokens]
                if (saved.message_count, saved.last_message_chars) != (message_index + 1, end_char):
                    raise ValueError("request ending disagrees with saved checkpoint")
            if end_tokens <= start_tokens or end_char < start_char:
                raise ValueError("request token/character boundaries are not increasing")
            start_tokens, start_char, previous_finish = end_tokens, end_char, finish
        if start_tokens != state.output_tokens:
            raise ValueError("requests do not reach the saved final budget")
