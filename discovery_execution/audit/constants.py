"""Final-answer extraction and Math-Verify settings."""

from pathlib import Path

from discovery_execution.harness.constants import PROMPTS_DIRECTORY

BOX_MARKER = r"\boxed"
GRADING_RULE = "last-complete-solve-revise-box-v1"
CRITIQUE_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / PROMPTS_DIRECTORY / "critique.md"
PARSING_TIMEOUT = 10
VERIFICATION_TIMEOUT = 10
COMPILED_AUDIT_FILENAME = "audit.jsonl"
