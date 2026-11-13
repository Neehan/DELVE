"""Verify final-box grading and preserve raw checkpoint regressions."""

import pytest

from discovery_execution.audit.answer_grader import AnswerGrader
from discovery_execution.audit.constants import GRADING_RULE
from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK, PROMPTS_DIRECTORY


@pytest.mark.parametrize("text,expected", [
    (r"\boxed{42}", True),
    (r"\boxed{0} then \boxed{42}", True),
    (r"\boxed{42} then \boxed{0}", False),
    (r"\boxed{42} then \boxed{", True),
    (r"\boxed{42} then \boxed{\frac{1}{", True),
    (r"\boxed{42} then \boxed{0} then \boxed{", False),
    (r"\boxed{42} then \boxed{\text{No error found}}", False),
    ("The answer is 42.", False),
    (r"The answer is $42$.", False),
    (r"\boxed{\frac{84}{2}}", True),
    (r"\boxed{} The final answer is $42$.", False),
    (r"\boxed{0} The final answer is $42$. I hope", False),
    (r"\boxed{42} The final answer is $0$. I hope", True),
    (r"\boxed{42} then \boxed{}", False),
    (r"\boxed{\frac{84}{2}} followed by $0$.", True),
])
def test_final_box_only(text: str, expected: bool) -> None:
    """Ignore earlier guesses and require a parsable final boxed answer."""
    assert AnswerGrader("42").check(text) is expected


def test_checkpoint_regressions_are_preserved() -> None:
    """A wrong revision stays wrong in the audit, independently of earlier correct checkpoints."""
    snapshots = [
        {"output_tokens": OUTPUT_TOKENS_PER_BLOCK, "messages": [
            {"role": "user", "content": r"Distractor \boxed{0}"},
            {"role": "assistant", "content": r"\boxed{42}"},
        ]},
        {"output_tokens": 2 * OUTPUT_TOKENS_PER_BLOCK, "messages": [
            {"role": "assistant", "content": r"\boxed{42}"},
            {"role": "assistant", "content": r"Revision: \boxed{0}"},
        ]},
    ]
    result = AnswerGrader("42").grade_checkpoints(snapshots, 2)
    assert result["complete"]
    assert [row["correctness"]["score"] for row in result["checkpoints"].values()] == [1, 0]


@pytest.mark.parametrize("revision,expected", [
    ("Still working", 1),
    (r"\boxed{", 1),
    (r"\boxed{0}", 0),
    (r"\boxed{}", 0),
    (r"\boxed{\text{No error found}}", 0),
])
def test_critique_does_not_replace_solver_answer(revision: str, expected: int) -> None:
    """Ignore complete and partial critique boxes, but accept a later submitted revision even if wrong."""
    messages = [
        {"role": "user", "content": "Solve the problem"},
        {"role": "assistant", "content": r"\boxed{42}"},
        {"role": "user", "content": (PROMPTS_DIRECTORY / "critique.md").read_text().strip()},
        {"role": "assistant", "content": r"\boxed{\text{No error found}}"},
        {"role": "assistant", "content": r"More critique: \boxed{0} then \boxed{"},
    ]
    snapshots = [{"output_tokens": OUTPUT_TOKENS_PER_BLOCK, "messages": list(messages)}]
    messages += [{"role": "user", "content": (PROMPTS_DIRECTORY / "revise.md").read_text().strip()},
                 {"role": "assistant", "content": revision}]
    snapshots.append({"output_tokens": 2 * OUTPUT_TOKENS_PER_BLOCK, "messages": messages})
    result = AnswerGrader("42").grade_checkpoints(snapshots, 2)
    assert result["grading_rule"] == GRADING_RULE
    assert [row["correctness"]["score"] for row in result["checkpoints"].values()] == [1, expected]


def test_oracle_instruction_exits_critique() -> None:
    """An injected sketch starts a solver response even if it interrupted a critique."""
    messages = [
        {"role": "assistant", "content": r"\boxed{0}"},
        {"role": "user", "content": (PROMPTS_DIRECTORY / "critique.md").read_text().strip()},
        {"role": "assistant", "content": r"\boxed{42}"},
    ]
    snapshots = [{"output_tokens": OUTPUT_TOKENS_PER_BLOCK, "messages": list(messages)}]
    messages += [{"role": "user", "content": "Follow this verified sketch."},
                 {"role": "assistant", "content": r"\boxed{42}"}]
    snapshots.append({"output_tokens": 2 * OUTPUT_TOKENS_PER_BLOCK, "messages": messages})
    result = AnswerGrader("42").grade_checkpoints(snapshots, 2)
    assert [row["correctness"]["score"] for row in result["checkpoints"].values()] == [0, 1]


def test_incomplete_boxes_cannot_close_across_messages() -> None:
    """Select boxes within individual responses rather than joining unrelated braces across turns."""
    snapshot = {"output_tokens": OUTPUT_TOKENS_PER_BLOCK, "messages": [
        {"role": "assistant", "content": r"\boxed{0} then \boxed{42"},
        {"role": "user", "content": "Revise"},
        {"role": "assistant", "content": "}"},
    ]}
    result = AnswerGrader("42").grade_checkpoints([snapshot], 1)
    assert result["checkpoints"]["1x"]["correctness"]["score"] == 0
