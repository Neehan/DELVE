"""Proper scoring rules against individual binary outcomes, including finite-sample target noise."""

from typing import Any, cast

import numpy as np
from scipy.special import expit
from transformers import EvalPrediction

from critic_only.constants import OUTCOME_COUNT_FIELDS, PROBABILITY_EPSILON, SUCCESS_COUNT_INDEX, TRIAL_COUNT_INDEX


def outcome_scores(probability: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return binary NLL and Brier for counts [..., success/trial], normalized over recorded trials."""
    probability = np.clip(probability, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    successes, trials = counts[..., SUCCESS_COUNT_INDEX], counts[..., TRIAL_COUNT_INDEX]
    nll = -(successes * np.log(probability) + (trials - successes) * np.log1p(-probability))
    brier = successes * (1 - probability) ** 2 + (trials - successes) * probability ** 2
    return nll / np.maximum(trials, 1), brier / np.maximum(trials, 1)


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
    """Score scalar terminal predictions, averaging prefixes within problems before averaging problems."""
    if len(problem_keys) != len(logits) or len(labels) != len(logits):
        raise ValueError("evaluation problem identities, predictions, and labels must have equal lengths")
    if logits.ndim != 1 or labels.shape != (len(logits), OUTCOME_COUNT_FIELDS) or np.any(labels[:, TRIAL_COUNT_INDEX] <= 0):
        raise ValueError("evaluation requires scalar logits and nonempty terminal success/trial counts")
    nll, brier = outcome_scores(expit(logits), labels)
    problems = sorted(set(problem_keys))
    per_problem = []
    identities = np.asarray(problem_keys)
    for key in problems:
        mask = identities == key
        per_problem.append({"problem_key": key, "nll": float(nll[mask].mean()), "brier": float(brier[mask].mean())})
    return {"problems": len(problems), "prefixes": len(problem_keys),
            "terminal_nll": float(np.mean([item["nll"] for item in per_problem])),
            "terminal_brier": float(np.mean([item["brier"] for item in per_problem])),
            "per_problem": per_problem}
