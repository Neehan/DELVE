"""Proper scoring rules against individual binary outcomes, including finite-sample target noise."""

from typing import Any

import numpy as np
from scipy.special import expit
from transformers import EvalPrediction

from critic_only.constants import ORDINARY, PROBABILITY_EPSILON, SUCCESS, TRIALS


def outcome_scores(probability: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute expected binary NLL and Brier over recorded outcomes, not squared error to s/n."""
    probability = np.clip(probability, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    successes, trials = counts[..., SUCCESS], counts[..., TRIALS]
    nll = -(successes * np.log(probability) + (trials - successes) * np.log1p(-probability))
    brier = successes * (1 - probability) ** 2 + (trials - successes) * probability ** 2
    return nll / np.maximum(trials, 1), brier / np.maximum(trials, 1)


def terminal_scores(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select the last observed ordinary checkpoint, not cumulative ever-correct success."""
    counts = labels[:, ORDINARY]
    last = (counts[..., TRIALS] > 0).sum(axis=-1) - 1
    if np.any(last < 0):
        raise ValueError("evaluation requires ordinary continuation labels")
    rows = np.arange(len(last))
    return outcome_scores(expit(logits[:, ORDINARY][rows, last]), counts[rows, last])


def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    """Choose checkpoints using ordinary terminal value quality alone, without oracle test targets."""
    nll, brier = terminal_scores(np.asarray(prediction.predictions), np.asarray(prediction.label_ids))
    return {"terminal_nll": float(nll.mean()), "terminal_brier": float(brier.mean())}


def evaluation_report(logits: np.ndarray, labels: np.ndarray, problem_keys: list[str]) -> dict[str, Any]:
    """Report problem-macro terminal scores and horizon-specific scores with available counts."""
    nll, brier = terminal_scores(logits, labels)
    problems = sorted(set(problem_keys))
    per_problem = []
    identities = np.asarray(problem_keys)
    for key in problems:
        mask = identities == key
        per_problem.append({"problem_key": key, "nll": float(nll[mask].mean()), "brier": float(brier[mask].mean())})
    future = {}
    for index in range(logits.shape[-1]):
        counts = labels[:, ORDINARY, index]
        mask = counts[:, TRIALS] > 0
        if mask.any():
            n, b = outcome_scores(expit(logits[mask, ORDINARY, index]), counts[mask])
            future[str(index + 1)] = {"prefixes": int(mask.sum()), "nll": float(n.mean()), "brier": float(b.mean())}
    return {"problems": len(problems), "prefixes": len(problem_keys),
            "terminal_nll": float(np.mean([item["nll"] for item in per_problem])),
            "terminal_brier": float(np.mean([item["brier"] for item in per_problem])),
            "by_remaining_blocks": future, "per_problem": per_problem}
