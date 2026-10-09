"""Grade the final boxed answer at each checkpoint independently with Math-Verify."""

from typing import Any

from math_verify import LatexExtractionConfig, parse, verify

from discovery_execution.audit.constants import BOX_MARKER, PARSING_TIMEOUT, VERIFICATION_TIMEOUT
from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK


class AnswerGrader:
    """Check one final answer per prefix; cumulative success belongs to framework analysis."""

    def __init__(self, final_answer: str) -> None:
        """Parse the reference once and reject an unparseable gold answer."""
        self._extraction = LatexExtractionConfig(boxed_match_priority=0, try_extract_without_anchor=False)
        self._gold = self._parse(f"{BOX_MARKER}{{{final_answer}}}")
        if not self._gold:
            raise ValueError("final_answer cannot be parsed by Math-Verify")

    def check(self, text: str) -> bool:
        """Select the last box before parsing so multiple guesses cannot become a set of answers."""
        _, marker, answer = text.rpartition(BOX_MARKER)
        if not marker:
            return False
        parsed = self._parse(marker + answer)
        return bool(parsed) and verify(self._gold, parsed, timeout_seconds=VERIFICATION_TIMEOUT, raise_on_error=True)

    def grade_checkpoints(self, snapshots: list[dict[str, Any]], compute_multiplier_k: int) -> dict[str, Any]:
        """Keep raw checkpoint correctness, including regressions after an earlier correct answer."""
        checkpoints: dict[str, Any] = {}
        output_tokens = 0
        for snapshot in snapshots:
            text = "\n".join(message["content"] for message in snapshot["messages"] if message["role"] == "assistant")
            output_tokens = snapshot["output_tokens"]
            block = output_tokens // OUTPUT_TOKENS_PER_BLOCK
            checkpoints[f"{block}x"] = {"correctness": {"score": int(self.check(text))}}
        return {"output_tokens": output_tokens,
                "complete": output_tokens == compute_multiplier_k * OUTPUT_TOKENS_PER_BLOCK,
                "checkpoints": checkpoints}

    def _parse(self, text: str) -> list[Any]:
        """Use the library's mathematical normalization without free-text fallbacks."""
        return parse(text, extraction_config=[self._extraction], extraction_mode="first_match",
                     fallback_mode="no_fallback", parsing_timeout=PARSING_TIMEOUT)
