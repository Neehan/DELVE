"""Exercise real PEFT adapters, proper losses, and the DE prior's limiting cases."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from scipy.special import logit
from transformers import EvalPrediction, PreTrainedTokenizerFast

from critic_only.constants import MODEL_FILENAME, PROBABILITY_EPSILON
from critic_only.data.tokenizer import PrefixTokenizer
from critic_only.eval.metrics import compute_metrics, evaluation_report, outcome_scores, terminal_scores
from critic_only.models.de_critic import DECritic
from critic_only.models.ordinary import Ordinary
from critic_only.models.loader import create_critic_from_pretrained
from critic_only.models.models import CriticConfig


@pytest.mark.parametrize("variant", ["ordinary", "oracle-supervised", "de"])
def test_lora_backward_and_restore(variant: str, critic_settings: Callable[[str], CriticConfig],
                                   tokenizer: PreTrainedTokenizerFast, training_rows: list[dict[str, Any]], tmp_path: Path) -> None:
    """Only adapters and heads train; compact checkpoints reproduce predictions exactly."""
    model = create_critic_from_pretrained(critic_settings(variant), torch.float32)
    assert type(model).__name__ == {"ordinary": "Ordinary", "oracle-supervised": "OracleSupervised", "de": "DECritic"}[variant]
    batch = PrefixTokenizer(tokenizer).pad_batch(training_rows)
    counts = batch.pop("labels")
    batch = {key: batch[key][:len(counts)] for key in ("input_ids", "attention_mask", "oracle")}
    model.train()
    prediction = model(**batch)
    loss = model.loss(prediction, counts, batch["oracle"]).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any("lora_" in name and value.grad is not None for name, value in model.named_parameters())
    assert all(not value.requires_grad for name, value in model.backbone.named_parameters() if "lora_" not in name)
    directory = tmp_path / "checkpoint"
    model.eval()
    expected = model(**batch).detach()
    model.save(directory)
    restored = create_critic_from_pretrained(critic_settings(variant), torch.float32)
    restored.load_from_weights(directory)
    restored.eval()
    torch.testing.assert_close(restored(**batch), expected)
    assert (directory / "critic.safetensors").stat().st_size < (Path(model.settings.backbone) / "model.safetensors").stat().st_size


def test_checkpoint_shape_failure_preserves_model(critic_settings: Callable[[str], CriticConfig], tmp_path: Path) -> None:
    """Reject broadcastable checkpoint tensors before modifying any model parameter."""
    model = Ordinary.create_from_pretrained(critic_settings("ordinary"), torch.float32)
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    directory = tmp_path / "checkpoint"
    model.save(directory)
    weights = {name: tensor + 1 for name, tensor in load_file(str(directory / MODEL_FILENAME)).items()}
    weights["head.bias"] = torch.ones(1)
    save_file(weights, str(directory / MODEL_FILENAME))
    with pytest.raises(ValueError, match="shape differs for head.bias"):
        model.load_from_weights(directory)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)


def test_count_loss_matches_binary_bce(critic_settings: Callable[[str], CriticConfig]) -> None:
    """Aggregated labels use the same likelihood as their individual Bernoulli observations."""
    model = Ordinary.create_from_pretrained(critic_settings("ordinary"), torch.float32)
    logits = torch.tensor([[0.7, -0.3]], requires_grad=True)
    counts = torch.tensor([[[2, 3], [1, 2]]]).float()
    loss = model.loss(logits, counts, torch.tensor([False])).mean()
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.7, 0.7, 0.7, -0.3, -0.3]), torch.tensor([1., 1., 0., 1., 0.]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None


def test_de_allows_regression(critic_settings: Callable[[str], CriticConfig]) -> None:
    """When the sketch is already acquired, DE reduces to the nonmonotonic execution curve."""
    model = DECritic.create_from_pretrained(critic_settings("de"), torch.float32)
    assert isinstance(model, DECritic)
    with torch.no_grad():
        model.acquisition.weight.zero_()
        model.acquisition.bias.fill_(30)
        model.execution.weight.zero_()
        model.execution.bias.copy_(torch.tensor([2.0, -2.0]))
    output = model.predict(torch.zeros(1, model.hidden_size), torch.tensor([False]))
    torch.testing.assert_close(output, torch.tensor([[2.0, -2.0]]))
    assert output[0, 1] < output[0, 0]


def test_direct_head_is_shared_across_history_types(critic_settings: Callable[[str], CriticConfig]) -> None:
    """Condition metadata must not select a separate direct-model head; the actual history provides context."""
    model = create_critic_from_pretrained(critic_settings("oracle-supervised"), torch.float32)
    hidden = torch.randn(1, model.hidden_size).expand(2, -1)
    prediction = model.predict(hidden, torch.tensor([False, True]))
    assert prediction.shape == (2, model.horizons)
    torch.testing.assert_close(prediction[0], prediction[1])


def test_oracle_de_uses_execution_head(critic_settings: Callable[[str], CriticConfig]) -> None:
    """Sketch-injected histories supervise execution directly without training acquisition parameters."""
    model = DECritic.create_from_pretrained(critic_settings("de"), torch.float32)
    hidden = torch.randn(2, model.hidden_size)
    oracle = torch.tensor([True, True])
    prediction = model.predict(hidden, oracle)
    torch.testing.assert_close(prediction, model.execution(hidden))
    counts = torch.tensor([[[1., 2.], [0., 2.]], [[0., 2.], [2., 2.]]])
    model.loss(prediction, counts, oracle).mean().backward()
    for parameter in model.acquisition.parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
    assert model.execution.weight.grad is not None and model.execution.weight.grad.abs().sum() > 0


def test_metrics_score_outcomes_not_noisy_means() -> None:
    """A perfectly matched empirical mean still has nonzero binary Brier loss."""
    nll, brier = outcome_scores(np.array([0.5]), np.array([[1, 2]]))
    np.testing.assert_allclose(nll, np.log(2))
    np.testing.assert_allclose(brier, 0.25)
    labels = np.array([[[2, 2], [0, 2]]])
    _, terminal = terminal_scores(np.array([[0., 2.]]), labels)
    assert terminal[0] > 0.7


def test_checkpoint_selection_matches_problem_macro_report() -> None:
    """Unequal prefix counts must not make checkpoint selection prefer the worse problem-level critic."""
    labels = np.zeros((4, 1, 2))
    labels[:, 0, :] = 1
    problem_keys = ["problem-A"] * 3 + ["problem-B"]
    available = np.array([True, False, True, True])
    selected_scores = []
    for probabilities in ([0.9, 0.9, 0.9, 0.1], [0.5, 0.5, 0.5, 0.5]):
        logits = np.zeros((4, 1))
        logits[:, 0] = logit(probabilities)
        metrics = compute_metrics(EvalPrediction(cast(Any, (logits, logits, available)), labels), problem_keys)
        report = evaluation_report(logits, labels, problem_keys)
        assert metrics["terminal_nll"] == report["terminal_nll"]
        assert metrics["terminal_brier"] == report["terminal_brier"]
        privileged_report = evaluation_report(logits[available], labels[available],
                                              [key for key, present in zip(problem_keys, available) if present])
        assert metrics["privileged_terminal_nll"] == privileged_report["terminal_nll"]
        selected_scores.append(metrics["terminal_nll"])
    assert selected_scores[1] < selected_scores[0]


@pytest.mark.parametrize("confidence", [-30.0, 30.0])
def test_de_bfloat16_extremes_remain_finite(confidence: float, critic_settings: Callable[[str], CriticConfig]) -> None:
    """BF16 saturation must not create infinite logits or NaN gradients in the DE prior."""
    model = DECritic.create_from_pretrained(critic_settings("de"), torch.float32)
    with torch.no_grad():
        model.acquisition.weight.zero_()
        model.acquisition.bias.fill_(confidence)
        model.execution.weight.zero_()
        model.execution.bias.fill_(confidence)
    counts = torch.tensor([[[1., 2.], [1., 2.]]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = model.predict(torch.zeros(1, model.hidden_size), torch.tensor([False]))
        loss = model.loss(prediction, counts, torch.tensor([False])).mean()
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


@pytest.mark.parametrize("horizon", [2, 3, 5, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vectorized_de_matches_convolution(horizon: int, dtype: torch.dtype,
                                           critic_settings: Callable[[str], CriticConfig]) -> None:
    """Preserve the original discrete convolution and its gradients across budgets and autocast modes."""
    settings = critic_settings("de").model_copy(update={"horizon": horizon})
    model = DECritic.create_from_pretrained(settings, torch.float32)
    hidden = torch.randn(3, model.hidden_size, requires_grad=True)
    with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
        prediction = model.predict(hidden, torch.zeros(len(hidden), dtype=torch.bool))
        discovered, alpha = model.acquisition(hidden).float().sigmoid().unbind(dim=-1)
        epsilon = model.execution(hidden).float().sigmoid()
    curve = []
    for end in range(model.horizons):
        value = discovered * epsilon[:, end]
        for delay in range(end + 1):
            value = value + (1 - discovered) * alpha * (1 - alpha).pow(delay) * epsilon[:, end - delay]
        curve.append(value)
    prior = torch.stack(curve, dim=-1).clamp(PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    expected = torch.logit(prior)
    actual = prediction
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    parameters = (hidden, *model.acquisition.parameters(), *model.execution.parameters())
    actual_gradients = torch.autograd.grad(actual.sum(), parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
