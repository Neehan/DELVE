"""Score final correctness predictions, giving each problem equal weight regardless of its trajectory count."""

from typing import Any, cast

import numpy as np
from scipy.special import expit
from transformers import EvalPrediction


def outcome_scores(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Measure each prediction's error against its final correct/incorrect label.

    NLL penalizes confident wrong predictions. Brier is squared error between the predicted probability and the binary label.

    Args:
        logits: Predicted success logits [N].
        labels: Binary final correctness labels [N].

    Returns:
        tuple[np.ndarray, np.ndarray]: Negative log-likelihood and Brier scores, each [N]; lower is better.
    """
    nll = labels * np.logaddexp(0, -logits) + (1 - labels) * np.logaddexp(0, logits)
    return nll, (expit(logits) - labels) ** 2


def compute_metrics(prediction: EvalPrediction, problem_keys: list[str]) -> dict[str, float]:
    """Report validation error separately for inputs with and without reference solutions.

    Args:
        prediction: Trainer outputs containing both sets of logits, reference availability, and final labels.
        problem_keys: Problem ID for each example, in prediction order.

    Returns:
        dict[str, float]: Problem-averaged NLL and Brier scores. terminal_nll, measured without references, selects the best checkpoint.
    """
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
    """Average errors within each problem, then average the problem scores equally.

    This prevents problems with more saved trajectories from receiving more weight.

    Args:
        logits: Final-success logits [N].
        labels: Final binary correctness [N], aligned with logits.
        problem_keys: Problem IDs [N] in the same order.

    Returns:
        dict[str, Any]: Problem/example counts, mean NLL and Brier, and a per-problem breakdown.
    """
    if len(problem_keys) != len(logits) or len(labels) != len(logits):
        raise ValueError("evaluation problem identities, predictions, and labels must have equal lengths")
    if logits.ndim != 1 or labels.shape != logits.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("evaluation requires scalar logits and binary final labels")
    nll, brier = outcome_scores(logits, labels)
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
