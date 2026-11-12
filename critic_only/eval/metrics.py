"""Score success probabilities against grouped continuation outcomes, weighting problems equally."""

from typing import Any

import numpy as np
from scipy.special import expit
from transformers import EvalPrediction


def outcome_scores(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score a prediction against the observed fraction of successful continuations.

    NLL and Brier equal the average scores against the individual binary outcomes. Brier includes the observed outcome variance, so it is not merely squared error against the fraction.

    Args:
        logits: Predicted success logits [N].
        labels: Observed success fractions [N], each in [0, 1].

    Returns:
        tuple[np.ndarray, np.ndarray]: Negative log-likelihood and Brier scores, each [N]; lower is better.
    """
    nll = labels * np.logaddexp(0, -logits) + (1 - labels) * np.logaddexp(0, logits)
    return nll, (expit(logits) - labels) ** 2 + labels * (1 - labels)


def compute_metrics(prediction: EvalPrediction, problem_keys: list[str]) -> dict[str, float]:
    """Report reference-conditioned validation metrics, averaging problems equally.

    Args:
        prediction: Trainer outputs containing scalar logits and success-fraction labels.
        problem_keys: Problem ID for each example in prediction order.

    Returns:
        dict[str, float]: NLL, outcome Brier, and value MSE. terminal_nll selects the best checkpoint.
    """
    report = evaluation_report(np.asarray(prediction.predictions), np.asarray(prediction.label_ids), problem_keys)
    return {"terminal_nll": report["terminal_nll"], "terminal_brier": report["terminal_brier"], "value_mse": report["value_mse"]}


def evaluation_report(logits: np.ndarray, labels: np.ndarray, problem_keys: list[str]) -> dict[str, Any]:
    """Average errors within each problem, then average the problem scores equally.

    This prevents problems with more saved trajectories from receiving more weight.

    Args:
        logits: Final-success logits [N].
        labels: Observed success fractions [N], aligned with logits.
        problem_keys: Problem IDs [N] in the same order.

    Returns:
        dict[str, Any]: Problem/example counts, mean NLL and Brier, and a per-problem breakdown.
    """
    if len(problem_keys) != len(logits) or len(labels) != len(logits):
        raise ValueError("evaluation problem identities, predictions, and labels must have equal lengths")
    if (logits.ndim != 1 or labels.shape != logits.shape or not np.isfinite(logits).all()
            or not np.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any()):
        raise ValueError("evaluation requires finite scalar logits and success fractions in [0, 1]")
    nll, brier = outcome_scores(logits, labels)
    value_mse = (expit(logits) - labels) ** 2
    problems = sorted(set(problem_keys))
    per_problem = []
    identities = np.asarray(problem_keys)
    for key in problems:
        mask = identities == key
        per_problem.append({"problem_key": key, "nll": float(nll[mask].mean()), "brier": float(brier[mask].mean()),
                            "value_mse": float(value_mse[mask].mean())})
    return {"problems": len(problems), "prefixes": len(problem_keys),
            "terminal_nll": float(np.mean([item["nll"] for item in per_problem])),
            "terminal_brier": float(np.mean([item["brier"] for item in per_problem])),
            "value_mse": float(np.mean([item["value_mse"] for item in per_problem])),
            "per_problem": per_problem}
