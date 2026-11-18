"""Score exact future-checkpoint curves at the input endpoint, weighting problems equally."""

from typing import Any

import numpy as np
from scipy.special import expit
from transformers import EvalPrediction


def outcome_scores(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return elementwise stable NLL and Brier for checkpoint predictions.

    Binary labels give standard outcome scores. Fractional labels are also supported: the variance term makes Brier equal the mean score against the underlying binary outcomes.

    Args:
        logits: Predicted success logits with arbitrary matching dimensions.
        labels: Matching correctness targets, each in [0, 1].

    Returns:
        tuple[np.ndarray, np.ndarray]: Negative log-likelihood and Brier scores with the input shape; lower is better.
    """
    nll = labels * np.logaddexp(0, -logits) + (1 - labels) * np.logaddexp(0, logits)
    return nll, (expit(logits) - labels) ** 2 + labels * (1 - labels)


def compute_metrics(prediction: EvalPrediction, problem_keys: list[str]) -> dict[str, float]:
    """Report reference-conditioned validation metrics, averaging problems equally.

    Args:
        prediction: Trainer outputs containing endpoint curves [N,H-1] and matching checkpoint grades.
        problem_keys: Problem ID for each example in prediction order.

    Returns:
        dict[str, float]: Final-checkpoint and curve-average NLL/Brier. terminal_nll selects the best checkpoint.
    """
    report = evaluation_report(np.asarray(prediction.predictions), np.asarray(prediction.label_ids), problem_keys)
    return {key: report[key] for key in ("terminal_nll", "terminal_brier", "curve_nll", "curve_brier")}


def evaluation_report(logits: np.ndarray, labels: np.ndarray, problem_keys: list[str]) -> dict[str, Any]:
    """Average errors within each problem, then average the problem scores equally.

    This prevents problems with more saved trajectories from receiving more weight.

    Args:
        logits: Checkpoint logits [N,H-1] at each 1× input endpoint.
        labels: Exact checkpoint grades [N,H-1], aligned with logits.
        problem_keys: Problem IDs [N] in the same order.

    Returns:
        dict[str, Any]: Problem/example counts, mean NLL and Brier, and a per-problem breakdown.
    """
    if len(problem_keys) != len(logits) or len(labels) != len(logits):
        raise ValueError("evaluation problem identities, predictions, and labels must have equal lengths")
    if (logits.ndim != 2 or logits.shape[1] == 0 or len(logits) == 0 or labels.shape != logits.shape or not np.isfinite(logits).all()
            or not np.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any()):
        raise ValueError("evaluation requires finite checkpoint curves and targets in [0, 1]")
    nll, brier = outcome_scores(logits, labels)
    identities = np.asarray(problem_keys)
    per_problem = []
    for key in sorted(set(problem_keys)):
        mask = identities == key
        per_problem.append({"problem_key": key, "nll": nll[mask].mean(axis=0).tolist(),
                            "brier": brier[mask].mean(axis=0).tolist()})
    mean_nll = np.asarray([item["nll"] for item in per_problem]).mean(axis=0)
    mean_brier = np.asarray([item["brier"] for item in per_problem]).mean(axis=0)
    return {"problems": len(per_problem), "prefixes": len(problem_keys),
            "terminal_nll": float(mean_nll[-1]), "terminal_brier": float(mean_brier[-1]),
            "curve_nll": float(mean_nll.mean()), "curve_brier": float(mean_brier.mean()),
            "checkpoints": [{"block": index + 2, "nll": float(nll_score), "brier": float(brier_score)}
                            for index, (nll_score, brier_score) in enumerate(zip(mean_nll, mean_brier))],
            "per_problem": per_problem}
