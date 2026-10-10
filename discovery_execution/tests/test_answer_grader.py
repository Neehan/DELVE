"""Verify final-box grading and preserve raw checkpoint regressions."""

import pytest

from discovery_execution.audit.answer_grader import AnswerGrader
from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK


@pytest.mark.parametrize("text,expected", [
    (r"\boxed{42}", True),
    (r"\boxed{0} then \boxed{42}", True),
    (r"\boxed{42} then \boxed{0}", False),
    (r"\boxed{42} then \boxed{", False),
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
