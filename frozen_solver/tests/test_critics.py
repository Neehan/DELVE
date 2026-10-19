"""Exercise real PEFT adapters, proper losses, and the DE prior's limiting cases."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from transformers import PreTrainedTokenizerFast

from frozen_solver.data.collator import PrefixCollator
from frozen_solver.eval.metrics import outcome_scores, terminal_scores
from frozen_solver.models.checkpoint import build_critic, load_critic_weights, save_critic
from frozen_solver.models.de_critic import DECritic
from frozen_solver.models.models import CriticConfig


@pytest.mark.parametrize("variant", ["direct", "auxiliary", "de"])
def test_lora_backward_and_restore(variant: str, critic_settings: Callable[[str], CriticConfig],
                                   tokenizer: PreTrainedTokenizerFast, training_rows: list[dict[str, Any]], tmp_path: Path) -> None:
    """Only adapters and heads train; compact checkpoints reproduce predictions exactly."""
    model = build_critic(critic_settings(variant), torch.float32)
    batch = PrefixCollator(tokenizer)(training_rows)
    counts = batch.pop("labels")
    model.train()
    prediction = model(**batch)
    loss = model.loss(prediction, counts)
    loss.backward()
    assert torch.isfinite(loss)
    assert any("lora_" in name and value.grad is not None for name, value in model.named_parameters())
    assert all(not value.requires_grad for name, value in model.backbone.named_parameters() if "lora_" not in name)
    directory = tmp_path / "checkpoint"
    model.eval()
    expected = model(**batch)["logits"].detach()
    save_critic(model, directory)
    restored = build_critic(critic_settings(variant), torch.float32)
    load_critic_weights(restored, directory)
    restored.eval()
    torch.testing.assert_close(restored(**batch)["logits"], expected)
    assert (directory / "critic.safetensors").stat().st_size < (Path(model.settings.backbone) / "model.safetensors").stat().st_size


def test_count_loss_matches_binary_bce(critic_settings: Callable[[str], CriticConfig]) -> None:
    """Aggregated labels use the same likelihood as their individual Bernoulli observations."""
    model = build_critic(critic_settings("direct"), torch.float32)
    logits = torch.tensor([[[0.7, -0.3], [1.0, 2.0]]], requires_grad=True)
    counts = torch.tensor([[[2, 3], [1, 2]], [[0, 0], [0, 0]]]).unsqueeze(0).float()
    loss = model.loss({"logits": logits}, counts)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.7, 0.7, 0.7, -0.3, -0.3]), torch.tensor([1., 1., 0., 1., 0.]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad[:, 1], torch.zeros_like(logits.grad[:, 1]))


def test_de_allows_regression(critic_settings: Callable[[str], CriticConfig]) -> None:
    """When the sketch is already acquired, DE reduces to the nonmonotonic execution curve."""
    model = build_critic(critic_settings("de"), torch.float32)
    assert isinstance(model, DECritic)
    with torch.no_grad():
        model.acquisition.weight.zero_()
        model.acquisition.bias.fill_(30)
        model.execution.weight.zero_()
        model.execution.bias.copy_(torch.tensor([2.0, -2.0]))
    output = model.predict(torch.zeros(1, model.hidden_size))
    torch.testing.assert_close(output["logits"][:, 0], output["logits"][:, 1])
    assert output["logits"][0, 0, 1] < output["logits"][0, 0, 0]


def test_metrics_score_outcomes_not_noisy_means() -> None:
    """A perfectly matched empirical mean still has nonzero binary Brier loss."""
    nll, brier = outcome_scores(np.array([0.5]), np.array([[1, 2]]))
    np.testing.assert_allclose(nll, np.log(2))
    np.testing.assert_allclose(brier, 0.25)
    labels = np.array([[[[2, 2], [0, 2]], [[0, 0], [0, 0]]]])
    _, terminal = terminal_scores(np.array([[[0., 2.], [0., 0.]]]), labels)
    assert terminal[0] > 0.7
