"""Verify checkpoint-curve budgets, exact labels, horizon weighting and endpoint reporting."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from transformers import PreTrainedTokenizerFast

from critic_only.data.critic_batcher import CriticBatcher
from critic_only.eval.metrics import evaluation_report
from critic_only.models.loader import create_critic
from critic_only.models.models import CriticConfig


def test_token_budgets_at_endpoint(tokenizer: PreTrainedTokenizerFast,
    reference_token_rows: list[dict[str, Any]]) -> None:
    """Interior tokens round remaining budgets up; actual endpoints use exact budgets despite padding."""
    rows = [{**reference_token_rows[0], "privileged_input_ids": [2, 3], "privileged_solver_mask": [False, True]},
            reference_token_rows[1]]
    live = CriticBatcher(tokenizer)(rows)
    assert live["remaining_blocks"].tolist() == [[3, 2, 2, 2], [3, 3, 3, 2]]
    positions = live["inputs"]["attention_mask"].sum(-1) - 1
    torch.testing.assert_close(live["remaining_blocks"][torch.arange(len(rows)), positions], torch.tensor([row["remaining_blocks"] for row in rows]))
    assert not live["inputs"]["solver_mask"][0, 2:].any()


@pytest.mark.parametrize("labels", [[], [0.5, 1.], [float("nan"), 0.], [1., float("inf")], [0.], 1.])
def test_invalid_curve_labels_rejected(labels: Any, tokenizer: PreTrainedTokenizerFast, reference_token_rows: list[dict[str, Any]]) -> None:
    """Old scalar labels, missing horizons and nonbinary grades fail before a model runs."""
    with pytest.raises(ValueError, match="curve labels"):
        CriticBatcher(tokenizer)([{**reference_token_rows[0], "labels": labels}])


@pytest.mark.parametrize("horizon", [2, 3, 5])
@pytest.mark.parametrize("variant", ["ppo-critic", "ppo-oracle-critic", "de-critic"])
def test_every_token_receives_every_horizon(horizon: int, variant: str, critic_settings: Callable[[str], CriticConfig],
    tokenizer: PreTrainedTokenizerFast, reference_token_rows: list[dict[str, Any]]) -> None:
    """All variants return the same token/curve axes; no checkpoint is replaced by a cumulative label."""
    settings = critic_settings(variant).model_copy(update={"training_mode": "head", "horizon": horizon})
    model = create_critic(settings, True)
    rows = [{**row, "remaining_blocks": horizon - 1,
             "labels": [float((index + offset + 1) % 2) for offset in range(horizon - 1)]}
            for index, row in enumerate(reference_token_rows)]
    batch = CriticBatcher(tokenizer)(rows)
    logits = model(batch["inputs"], batch["oracle"], batch["remaining_blocks"])
    assert logits.shape == (2, 4, horizon - 1)
    labels = batch["labels"]
    assert labels.tolist() == [row["labels"] for row in rows]
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[:, None, :].expand_as(logits), reduction="none")
    torch.testing.assert_close(model.loss(logits, labels, batch["oracle"]), expected)


def test_metrics_keep_final_selection_separate_from_curve() -> None:
    """Report each exact checkpoint and the curve average while selecting by final-checkpoint NLL."""
    logits = np.array([[10., -10.], [-10., 10.], [0., 0.]])
    labels = np.array([[1., 0.], [0., 1.], [1., 0.]])
    report = evaluation_report(logits, labels, ["A", "A", "B"])
    assert report["terminal_nll"] == report["checkpoints"][-1]["nll"]
    assert [item["block"] for item in report["checkpoints"]] == [2, 3]
    assert report["curve_nll"] == pytest.approx((np.logaddexp(0, -10.) + np.log(2.)) / 2)
    changed = logits.copy()
    changed[:, 0] *= -1
    other = evaluation_report(changed, labels, ["A", "A", "B"])
    assert other["terminal_nll"] == report["terminal_nll"]
    assert other["curve_nll"] > report["curve_nll"]


@pytest.mark.parametrize("logits,labels", [(np.zeros((0, 2)), np.zeros((0, 2))),
    (np.zeros((1, 0)), np.zeros((1, 0))), (np.zeros((1, 2)), np.zeros(1)),
    (np.array([[float("nan")]]), np.zeros((1, 1))),
    (np.zeros((1, 2)), np.array([[0.5, 1.]])),
    (np.zeros((1, 2)), np.array([[float("nan"), 1.]]))])
def test_invalid_curve_metrics_fail(logits: np.ndarray, labels: np.ndarray) -> None:
    """Empty or mismatched curve reports must fail rather than silently select a checkpoint."""
    with pytest.raises(ValueError, match="checkpoint curves"):
        evaluation_report(logits, labels, ["problem"] * len(logits))
