"""Validate a completed DELVE JSONL partition without changing any files."""

import json
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from constants import MAX_SKETCH_WORDS, MIN_SKETCH_WORDS

from discovery_execution.audit.answer_grader import AnswerGrader
from discovery_execution.audit.constants import BOX_MARKER
from discovery_execution.harness.models import Problem


def validate_partition(path: Path) -> dict[str, object]:
    """Check schema, identities, sketches, and final boxes; return collection counts."""
    problems: dict[str, Problem] = {}
    partitions: set[tuple[str, int]] = set()
    word_counts: Counter[int] = Counter()
    unboxed_solutions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        problem = Problem.model_validate_json(line)
        if problem.problem_id in problems:
            raise ValueError(f"duplicate problem ID: {problem.problem_id}")
        problems[problem.problem_id] = problem
        partitions.add((problem.competition, problem.year))
        if len(partitions) > 1:
            raise ValueError("file contains more than one competition/year partition")
        grader = AnswerGrader(problem.final_answer)
        proofs: set[str] = set()
        for index, solution in enumerate(problem.solutions):
            identity = f"{problem.problem_id}, solution {index}"
            proof = solution.solution.strip()
            if proof in proofs:
                raise ValueError(f"duplicate solution text: {identity}")
            proofs.add(proof)
            words = len(solution.sketch.split())
            if not MIN_SKETCH_WORDS <= words <= MAX_SKETCH_WORDS:
                raise ValueError(f"sketch has {words} words: {identity}")
            if BOX_MARKER in solution.sketch:
                raise ValueError(f"sketch contains a boxed answer: {identity}")
            if BOX_MARKER not in proof:
                unboxed_solutions.append(identity)
            elif not grader.check(proof):
                raise ValueError(f"final solution box does not match gold: {identity}")
            word_counts[words] += 1
    if not problems:
        raise ValueError("partition has no problems")
    return {"path": str(path), "problems": len(problems),
            "solutions": word_counts.total(), "alternatives": word_counts.total() - len(problems),
            "sketch_word_counts": dict(sorted(word_counts.items())),
            "solution_boxes_checked": word_counts.total() - len(unboxed_solutions),
            "unboxed_solutions_requiring_manual_answer_review": unboxed_solutions}


def main() -> None:
    """Validate the supplied staged or published partition and print its counts."""
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("partition", type=Path, help="Completed competition/year JSONL file")
    arguments = parser.parse_args()
    print(json.dumps(validate_partition(arguments.partition), indent=2))


if __name__ == "__main__":
    main()
