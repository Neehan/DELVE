"""Grade the latest complete solve/revise answer, excluding critique boxes and unfinished boxes."""

from typing import Any

from math_verify import LatexExtractionConfig, parse, verify

from discovery_execution.audit.constants import BOX_MARKER, CRITIQUE_PROMPT_PATH, GRADING_RULE, PARSING_TIMEOUT, VERIFICATION_TIMEOUT
from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK


def validate_grading_rule(audit: dict[str, Any], source: str) -> None:
    """Reject an audit from another answer-selection rule before consuming its scores.

    audit is the saved JSON object; source identifies it in the error. Returns None on success. Regrade legacy artifacts before combining them with new labels.
    """
    if audit.get("grading_rule") != GRADING_RULE:
        raise ValueError(f"regrade with {GRADING_RULE} before using audit: {source}")


class AnswerGrader:
    """Check one final answer per prefix; cumulative success belongs to framework analysis."""

    def __init__(self, final_answer: str) -> None:
        """Parse the reference once and reject an unparseable gold answer."""
        self._extraction = LatexExtractionConfig(boxed_match_priority=0, try_extract_without_anchor=False)
        self._critique_instruction = CRITIQUE_PROMPT_PATH.read_text(encoding="utf-8").strip()
        self._gold = self._parse(f"{BOX_MARKER}{{{final_answer}}}")
        if not self._gold:
            raise ValueError("final_answer cannot be parsed by Math-Verify")

    def check(self, text: str) -> bool:
        """Return whether text's last complete box matches the gold answer.

        An unfinished box leaves the preceding complete answer in force. A complete but empty, unparseable, or wrong box replaces it and scores zero.
        """
        answer = self._last_complete_box(text)
        parsed = self._parse(answer) if answer is not None else []
        return bool(parsed) and verify(self._gold, parsed, timeout_seconds=VERIFICATION_TIMEOUT, raise_on_error=True)

    def _last_complete_box(self, text: str) -> str | None:
        """Return the last brace-balanced box in text, or None if none is complete; never select by correctness."""
        for suffix in reversed(text.split(BOX_MARKER)[1:]):
            answer = suffix.lstrip()
            if not answer.startswith("{"):
                continue
            depth = 0
            for index, character in enumerate(answer):
                if index > 0 and answer[index - 1] == "\\":
                    continue
                if character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth == 0:
                        return BOX_MARKER + answer[:index + 1]
        return None

    def _solver_answer(self, messages: list[dict[str, str]]) -> str:
        """Return the latest complete solver box from saved messages, ignoring user text and critique responses.

        Snapshots preserve the harness instructions rather than per-message phases. Match the exact critique instruction; a new solve, revision, or oracle instruction ends that phase. Scan messages separately so unfinished boxes cannot close across turns.
        """
        answer = ""
        critique = False
        for message in messages:
            if message["role"] == "user":
                critique = message["content"].strip() == self._critique_instruction
            elif message["role"] == "assistant" and not critique:
                candidate = self._last_complete_box(message["content"])
                if candidate is not None:
                    answer = candidate
        return answer

    def grade_checkpoints(self, snapshots: list[dict[str, Any]], compute_multiplier_k: int) -> dict[str, Any]:
        """Return versioned grades for snapshots up to compute_multiplier_k, preserving wrong-answer revisions."""
        checkpoints: dict[str, Any] = {}
        output_tokens = 0
        for snapshot in snapshots:
            text = self._solver_answer(snapshot["messages"])
            output_tokens = snapshot["output_tokens"]
            block = output_tokens // OUTPUT_TOKENS_PER_BLOCK
            checkpoints[f"{block}x"] = {"correctness": {"score": int(self.check(text))}}
        return {"grading_rule": GRADING_RULE, "output_tokens": output_tokens,
                "complete": output_tokens == compute_multiplier_k * OUTPUT_TOKENS_PER_BLOCK,
                "checkpoints": checkpoints}

    def _parse(self, text: str) -> list[Any]:
        """Use the library's mathematical normalization without free-text fallbacks."""
        return parse(text, extraction_config=[self._extraction], extraction_mode="first_match",
                     fallback_mode="no_fallback", parsing_timeout=PARSING_TIMEOUT)
