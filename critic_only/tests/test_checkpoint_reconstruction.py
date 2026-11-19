"""Check finer boundaries, answer selection, and failure without altering generation data."""

from typing import Any
from unittest.mock import Mock

from math_verify.errors import TimeoutException

import pytest
from transformers import PreTrainedTokenizerFast

from discovery_execution.audit.answer_grader import AnswerGrader
from discovery_execution.audit.constants import CRITIQUE_PROMPT_PATH
from discovery_execution.harness.models import Checkpoint, RunState
from discovery_execution.processing.checkpoint_reconstructor import CheckpointReconstructor
from discovery_execution.processing.storage import checkpoint_snapshot


@pytest.fixture
def finer_checkpoints(tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch) -> CheckpointReconstructor:
    """Use eight-token generation blocks and four-token analysis blocks to test the real algorithm cheaply."""
    monkeypatch.setattr("discovery_execution.processing.checkpoint_reconstructor.OUTPUT_TOKENS_PER_BLOCK", 8)
    return CheckpointReconstructor(tokenizer, 4)


def saved_state(tokenizer: PreTrainedTokenizerFast) -> tuple[RunState, dict[str, Any]]:
    """Build five original blocks in one response, with alternating completed answers."""
    messages = [{"role": "user", "content": "Find the number."}, {"role": "assistant", "content": ""}]
    boundaries, requests = [], []
    for block in range(1, 6):
        text = f"\\boxed{{{block % 2}}} word word word "
        assert len(tokenizer.encode(text, add_special_tokens=False)) == 8
        messages[-1]["content"] += text
        boundaries.append(Checkpoint(output_tokens=block * 8, message_count=2, last_message_chars=len(messages[-1]["content"])))
        requests.append({"output_tokens": block * 8, "usage": {"completion_tokens": 8}, "finish_reason": "length", "injected": False})
    state = RunState(messages=messages, output_tokens=40, phase="solve", injected=False,
                     response_finished=False, requests=requests, checkpoints=boundaries)
    return state, {"checkpoints": {f"{block}x": {"correctness": {"score": block % 2}} for block in range(1, 6)}}


def test_ten_checkpoints_preserve_original_boundaries(finer_checkpoints: CheckpointReconstructor,
    tokenizer: PreTrainedTokenizerFast) -> None:
    """Retain all ten boundaries; intermediate unfinished boxes retain the previous complete answer."""
    state, audit = saved_state(tokenizer)
    original = state.model_dump_json()
    checkpoints = finer_checkpoints.reconstruct(state, audit, AnswerGrader("1"))
    assert [item.output_tokens for item in checkpoints] == list(range(4, 41, 4))
    assert [item.correctness for item in checkpoints] == [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
    for index, saved in enumerate(state.checkpoints):
        assert checkpoint_snapshot(state, checkpoints[2 * index + 1]) == checkpoint_snapshot(state, saved)
        assert checkpoints[2 * index + 1].boundary_source == "saved"
    assert [item.text_tokens for item in checkpoints] == [4] * 10
    assert state.model_dump_json() == original
    first = checkpoint_snapshot(state, checkpoints[0])["messages"][-1]["content"]
    assert first == r"\boxed{1" and "word" not in first


def test_request_stop_and_critique_boxes(finer_checkpoints: CheckpointReconstructor,
    tokenizer: PreTrainedTokenizerFast) -> None:
    """A natural response ending at 4 tokens is exact; subsequent critique boxes never replace solver answers."""
    messages = [{"role": "user", "content": "Find the number."},
                {"role": "assistant", "content": "word word word word"},
                {"role": "user", "content": CRITIQUE_PROMPT_PATH.read_text().strip()},
                {"role": "assistant", "content": r"\boxed{1} word word word word word word word"}]
    encoded = tokenizer(messages[-1]["content"], add_special_tokens=False, return_offsets_mapping=True)
    end_at_eight = encoded["offset_mapping"][3][1]
    state = RunState(messages=messages, output_tokens=16, phase="critique", injected=False, response_finished=False,
        requests=[{"output_tokens": 4, "usage": {"completion_tokens": 4}, "finish_reason": "stop"},
                  {"output_tokens": 8, "usage": {"completion_tokens": 4}, "finish_reason": "length"},
                  {"output_tokens": 16, "usage": {"completion_tokens": 8}, "finish_reason": "length"}],
        checkpoints=[Checkpoint(output_tokens=8, message_count=4, last_message_chars=end_at_eight),
                     Checkpoint(output_tokens=16, message_count=4, last_message_chars=len(messages[-1]["content"]))])
    audit = {"checkpoints": {"1x": {"correctness": {"score": 0}}, "2x": {"correctness": {"score": 0}}}}
    checkpoints = finer_checkpoints.reconstruct(state, audit, AnswerGrader("1"))
    assert [item.correctness for item in checkpoints] == [0, 0, 0, 0]
    assert checkpoints[0].boundary_source == "reconstructed"
    assert checkpoints[0].message_count == 2


@pytest.mark.parametrize("damage", ["usage", "duplicate", "missing", "finish"])
def test_unrecoverable_source_fails(finer_checkpoints: CheckpointReconstructor,
    tokenizer: PreTrainedTokenizerFast, damage: str) -> None:
    """Corrupt or unreconstructable sources fail instead of silently moving a cut or dropping a seed."""
    state, audit = saved_state(tokenizer)
    if damage == "usage":
        state.requests[0]["usage"]["completion_tokens"] = 7
    elif damage == "duplicate":
        state.checkpoints.append(state.checkpoints[0])
    elif damage == "missing":
        state.requests = state.requests[1:]
    else:
        state.requests[0]["finish_reason"] = "invalid"
    with pytest.raises(ValueError):
        finer_checkpoints.reconstruct(state, audit, AnswerGrader("1"))


def test_new_answer_timeout_is_not_incorrect(tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch) -> None:
    """A newly observed answer that times out aborts reconstruction instead of receiving label zero."""
    monkeypatch.setattr("discovery_execution.processing.checkpoint_reconstructor.OUTPUT_TOKENS_PER_BLOCK", 16)
    text = r"\boxed{2} word word word \boxed{1} word word word "
    state = RunState(messages=[{"role": "user", "content": "Find the number."}, {"role": "assistant", "content": text}],
        output_tokens=16, phase="solve", injected=False, response_finished=False,
        requests=[{"output_tokens": 16, "usage": {"completion_tokens": 16}, "finish_reason": "length"}],
        checkpoints=[Checkpoint(output_tokens=16, message_count=2, last_message_chars=len(text))])
    grader = Mock(wraps=AnswerGrader("1"))
    grader.check.side_effect = TimeoutException("timed out")
    with pytest.raises(TimeoutException):
        CheckpointReconstructor(tokenizer, 8).reconstruct(state, {"checkpoints": {"1x": {"correctness": {"score": 1}}}}, grader)


def test_retokenization_drift_uses_balanced_text_parts(tokenizer: PreTrainedTokenizerFast,
    monkeypatch: pytest.MonkeyPatch) -> None:
    """Compressed text is divided evenly across messages, excluding the intervening user prompt."""
    monkeypatch.setattr("discovery_execution.processing.checkpoint_reconstructor.OUTPUT_TOKENS_PER_BLOCK", 32)
    messages = [{"role": "user", "content": "Find the number."}, {"role": "assistant", "content": "word"},
                {"role": "user", "content": "Revise."}, {"role": "assistant", "content": "word " * 16}]
    state = RunState(messages=messages, output_tokens=32, phase="revise", injected=False, response_finished=False,
        requests=[{"output_tokens": 16, "usage": {"completion_tokens": 16}, "finish_reason": "stop"},
                  {"output_tokens": 32, "usage": {"completion_tokens": 16}, "finish_reason": "length"}],
        checkpoints=[Checkpoint(output_tokens=32, message_count=4, last_message_chars=len(messages[-1]["content"]))])
    boundaries = CheckpointReconstructor(tokenizer, 16).reconstruct(state, {"checkpoints": {"1x": {"correctness": {"score": 0}}}}, AnswerGrader("1"))
    assert [boundary.text_tokens for boundary in boundaries] == [8, 9]
    assert boundaries[0].boundary_source == "reconstructed"
    assert boundaries[0].message_count == 4
    assert checkpoint_snapshot(state, boundaries[0])["messages"][-1]["content"] == "word " * 6 + "word"
    assert checkpoint_snapshot(state, boundaries[1])["messages"] == messages


def test_conflicting_reviewed_grades_rejected(finer_checkpoints: CheckpointReconstructor,
    tokenizer: PreTrainedTokenizerFast) -> None:
    """Identical selected answer text cannot silently acquire whichever saved grade appears last."""
    state, audit = saved_state(tokenizer)
    audit["checkpoints"]["5x"]["correctness"]["score"] = 0
    with pytest.raises(ValueError, match="conflicting.*grade"):
        finer_checkpoints.reconstruct(state, audit, AnswerGrader("1"))


def test_natural_stop_must_match_saved_boundary(finer_checkpoints: CheckpointReconstructor,
    tokenizer: PreTrainedTokenizerFast) -> None:
    """A saved request ending at a checkpoint cannot disagree with that checkpoint's character boundary."""
    text = r"\boxed{1} word word word extra"
    state = RunState(messages=[{"role": "user", "content": "Find the number."}, {"role": "assistant", "content": text}],
        output_tokens=8, phase="solve", injected=False, response_finished=True,
        requests=[{"output_tokens": 8, "usage": {"completion_tokens": 8}, "finish_reason": "stop"}],
        checkpoints=[Checkpoint(output_tokens=8, message_count=2, last_message_chars=len(text) - len(" extra"))])
    audit = {"checkpoints": {"1x": {"correctness": {"score": 1}}}}
    with pytest.raises(ValueError, match="request.*saved checkpoint"):
        finer_checkpoints.reconstruct(state, audit, AnswerGrader("1"))


@pytest.mark.parametrize("token_ends,parts,expected", [
    ([(2, 1), (2, 1), (2, 1), (2, 2)], 2, [3, 4]),
    ([(2, 1), (2, 1), (2, 2), (2, 3)], 2, [2, 4]),
    ([(2, index) for index in range(1, 18)], 2, [8, 17]),
    ([(2, index) for index in range(1, 9)], 4, [2, 4, 6, 8]),
])
def test_character_safe_equal_cuts(finer_checkpoints: CheckpointReconstructor,
    token_ends: list[tuple[int, int]], parts: int, expected: list[int]) -> None:
    """Keep byte tokens sharing a character together; odd lengths and finer subdivisions remain balanced."""
    assert finer_checkpoints._equal_cuts(token_ends, parts) == expected


def test_too_short_block_rejected(finer_checkpoints: CheckpointReconstructor) -> None:
    """A single character cannot become two distinct nonempty text segments."""
    with pytest.raises(ValueError, match="distinct text boundaries"):
        finer_checkpoints._equal_cuts([(2, 1), (2, 1)], 2)
