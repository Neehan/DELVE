---
name: curate-math-dataset
description: Import or enrich DELVE competition datasets with checked solutions, genuinely distinct alternatives, and consistent 12–15-word oracle sketches. Use for dataset curation, solution review, or sketch generation.
---

# Curate a math dataset

Produce a checked JSONL partition at `data/datasets/<competition>/<year>.jsonl`, using the existing `Problem` and `Solution` schemas in `discovery_execution/harness/models.py`. Keep dataset work separate from harness changes.

## Import and preserve

Read the requested sources and inspect representative existing rows before writing. Confirm the competition, year, editions, expected problem count, and source fields. Resolve missing or ambiguous editions rather than silently selecting them.

Preserve statements, including mathematical notation and diagrams. Prefer explicit source answer fields; store `final_answer` as a string. If a reference requires answer extraction, record the extraction method and independently verify it. Never derive the gold answer from a newly generated solution. Keep source URLs and any source revision in the working review report; do not add fields to the dataset schema.

Each row has `problem_id`, `competition`, `year`, `statement`, `final_answer`, and `solutions`. Each solution has `source` (`reference` or `ai_generated`), `solution`, and `sketch`. Follow existing problem-ID conventions and preserve source order. Mark newly written proofs `ai_generated`; use `reference` only for sourced reference material.

Stage unfinished work outside the live partition. A temporary `sketch: null` is permitted in staging only; every published solution needs a nonempty sketch. Preserve existing rows, solutions, and ordering unless the user explicitly requests their revision.

## Complete and review solutions

Divide problems among author agents with disjoint ownership. Give each agent the statement, gold answer, existing solutions, schema, and a separate staging destination. Require each newly written proof to end with the final answer in `\boxed{...}`; preserve already approved source proofs when only adding sketches.

Add alternate solutions when they use a materially different mathematical argument or reduction. Rewording, changing coordinates without changing the argument, or repeating the same calculation is not an independent alternative. Do not force an alternative count.

Assign proofs to reviewers who did not write them. Review the reasoning before consulting the expected answer: check case coverage, extraneous roots, geometry branches, boundary conditions, and any finite enumeration's exhaustiveness. Use exact arithmetic or small reproducible checks where they substantiate the argument. Agreement of the final box with gold alone does not establish correctness.

Track review results by `(problem_id, solution_index)`, including issues and their resolution. Resolve every failed proof before starting sketches for that batch. If the request only concerns sketches, use the existing checked solutions and preserve them.

## Write sketches with fresh agents

After solution review, delegate sketch writing to a fresh set of agents who did not author those proofs. Give them the complete statement and approved solution, plus representative existing sketches. Keep their ownership disjoint and their output limited to `(problem_id, solution_index, sketch)`.

Use 12–15 whitespace-separated words per sketch, with a hard maximum of 15; aim for 14–15 to match current AIME sketches. Use neutral imperative mathematical prose, usually a method clause and a decisive-reduction clause separated by a semicolon. Maintain comparable style, length, mathematical density, and strategy-level information across the batch.

Preserve the load-bearing step or surprising insight and any essential branch-selection condition. Target the actual solution's method, including distinctions between alternatives. Avoid generic instructions, filler added to reach the word count, and compressed formulas or artificial hyphenation used to evade the limit.

Do not reveal the final answer, disguise it, include evaluated intermediate totals, or supply a ready-to-evaluate expression for the answer. Structural constants from the problem or necessary case distinctions can remain. For easy problems, state the general reduction without carrying out the calculation.

Assign sketches to independent reviewers who did not write them. Review against the full proof, not just the word count. Check fidelity, the decisive insight, answer leakage, clarity, and information density relative to other hints. Re-review revised hints and check every solution has exactly one accepted sketch.

## Validate and publish locally

Merge only approved material. Compare identities, statements, answers, source labels, solution text, and ordering against the source and the approved staging artifacts. Verify expected problem and solution coverage; compare untouched data with its pre-edit snapshot.

From the repository root, validate the completed staged JSONL using the project's environment:

```bash
PYTHONPATH=. uv run python .agents/skills/curate-math-dataset/scripts/validate_dataset.py /path/to/staged.jsonl
```

This checks the existing schema, a single partition, unique IDs and proof texts, sketch lengths, and agreement of available final boxes with gold using the harness's Math-Verify grader. It reports unboxed proofs separately without guessing answers from prose. Newly written proofs must all have checked boxes; preserved unboxed proofs require explicit manual answer review. The script does **not** certify proofs, alternate-method independence, answer-leakage absence, or uniform hint quality; those require the reviews above.

Install the complete file atomically. For a new partition, refuse to replace an existing destination. For an explicitly requested update, preserve a recoverable copy and verify the authorized diff before replacing it. Load the final partition through `discovery_execution.dataset.Dataset` and confirm counts and content.

Report the local path, sources, problem count, total solutions and alternatives, word-count distribution, and completed checks. HF upload and Git commit/push require the user's authorization; this skill does not grant it. The workflow and acceptance criteria are reproducible, while generated wording may vary.
