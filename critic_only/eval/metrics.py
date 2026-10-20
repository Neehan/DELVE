"""Proper scoring rules against individual binary outcomes, including finite-sample target noise."""

from typing import Any, cast

import numpy as np
from scipy.special import expit
from transformers import EvalPrediction

from critic_only.constants import PROBABILITY_EPSILON, SUCCESS_COUNT_INDEX, TRIAL_COUNT_INDEX


def outcome_scores(probability: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return binary NLL and Brier for counts [..., success/trial], normalized over recorded trials."""
    probability = np.clip(probability, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    successes, trials = counts[..., SUCCESS_COUNT_INDEX], counts[..., TRIAL_COUNT_INDEX]
    nll = -(successes * np.log(probability) + (trials - successes) * np.log1p(-probability))
    brier = successes * (1 - probability) ** 2 + (trials - successes) * probability ** 2
    return nll / np.maximum(trials, 1), brier / np.maximum(trials, 1)


def terminal_scores(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select the last observed ordinary checkpoint, not cumulative ever-correct success."""
    counts = labels
    last = (counts[..., TRIAL_COUNT_INDEX] > 0).sum(axis=-1) - 1
    if np.any(last < 0):
        raise ValueError("evaluation requires ordinary continuation labels")
    rows = np.arange(len(last))
    return outcome_scores(expit(logits[rows, last]), counts[rows, last])


def compute_metrics(prediction: EvalPrediction, problem_keys: list[str]) -> dict[str, float]:
    """Select on problem-macro unprivileged terminal NLL, using dataset-order problem identities."""
    logits, privileged_logits, available = cast(tuple[np.ndarray, np.ndarray, np.ndarray], prediction.predictions)
    labels = np.asarray(prediction.label_ids)
    report = evaluation_report(np.asarray(logits), labels, problem_keys)
    metrics = {"terminal_nll": report["terminal_nll"], "terminal_brier": report["terminal_brier"]}
    mask = np.asarray(available, dtype=bool)
    if mask.any():
        privileged_keys = [key for key, present in zip(problem_keys, mask) if present]
        report = evaluation_report(np.asarray(privileged_logits)[mask], labels[mask], privileged_keys)
        metrics.update(privileged_terminal_nll=report["terminal_nll"], privileged_terminal_brier=report["terminal_brier"])
    return metrics


def evaluation_report(logits: np.ndarray, labels: np.ndarray, problem_keys: list[str]) -> dict[str, Any]:
    """Report problem-macro terminal scores and horizon-specific scores with available counts."""
    if len(problem_keys) != len(logits) or len(labels) != len(logits):
        raise ValueError("evaluation problem identities, predictions, and labels must have equal lengths")
    nll, brier = terminal_scores(logits, labels)
    problems = sorted(set(problem_keys))
    per_problem = []
    identities = np.asarray(problem_keys)
    for key in problems:
        mask = identities == key
        per_problem.append({"problem_key": key, "nll": float(nll[mask].mean()), "brier": float(brier[mask].mean())})
    by_remaining_blocks = {}
    for index in range(logits.shape[-1]):
        counts = labels[:, index]
        mask = counts[:, TRIAL_COUNT_INDEX] > 0
        if mask.any():
            horizon_nll, horizon_brier = outcome_scores(expit(logits[mask, index]), counts[mask])
            by_remaining_blocks[str(index + 1)] = {"prefixes": int(mask.sum()), "nll": float(horizon_nll.mean()),
                                                  "brier": float(horizon_brier.mean())}
    return {"problems": len(problems), "prefixes": len(problem_keys),
            "terminal_nll": float(np.mean([item["nll"] for item in per_problem])),
            "terminal_brier": float(np.mean([item["brier"] for item in per_problem])),
            "by_remaining_blocks": by_remaining_blocks, "per_problem": per_problem}
