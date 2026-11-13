"""Exercise cached-feature MLPs, binary losses, and the DE convolution's limiting cases."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from scipy.special import logit
from transformers import EvalPrediction, PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from critic_only.constants import CONFIG_FILENAME, MODEL_FILENAME
from critic_only.data.critic_batcher import CriticBatcher
from critic_only.eval.metrics import compute_metrics, evaluation_report, outcome_scores
from critic_only.models.de_critic import DECritic
from critic_only.models.ppo_critic import PPOCritic
from critic_only.models.loader import create_critic
from critic_only.models.models import CriticConfig


@pytest.mark.parametrize("variant", ["ppo-critic", "ppo-oracle-critic", "de-critic"])
def test_mlp_backward_and_restore(variant: str, critic_settings: Callable[[str], CriticConfig],
                                   tokenizer: PreTrainedTokenizerFast, training_rows: list[dict[str, Any]], tmp_path: Path) -> None:
    """All MLP and head parameters train; compact checkpoints reproduce predictions exactly."""
    model = create_critic(critic_settings(variant), True)
    assert type(model).__name__ == {"ppo-critic": "PPOCritic", "ppo-oracle-critic": "PPOOracleCritic", "de-critic": "DECritic"}[variant]
    batch = CriticBatcher(None)(training_rows)
    labels = batch.pop("labels")
    model.train()
    prediction = model(**batch)
    assert prediction.shape == (len(labels),)
    loss = model.loss(prediction, labels, batch["oracle"]).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(value.requires_grad and value.grad is not None for value in model.parameters())
    directory = tmp_path / "checkpoint"
    model.eval()
    expected = model(**batch).detach()
    model.save(directory)
    restored = create_critic(critic_settings(variant), True)
    restored.load_from_weights(directory)
    restored.eval()
    torch.testing.assert_close(restored(**batch), expected)
    assert (directory / "critic.safetensors").is_file()


def test_checkpoint_shape_failure_preserves_model(critic_settings: Callable[[str], CriticConfig], tmp_path: Path) -> None:
    """Reject broadcastable checkpoint tensors before modifying any model parameter."""
    model = PPOCritic(critic_settings("ppo-critic"))
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    directory = tmp_path / "checkpoint"
    model.save(directory)
    weights = {name: tensor + 1 for name, tensor in load_file(str(directory / MODEL_FILENAME)).items()}
    weights["prediction_head.bias"] = torch.ones(2)
    save_file(weights, str(directory / MODEL_FILENAME))
    with pytest.raises(ValueError, match="shape differs for prediction_head.bias"):
        model.load_from_weights(directory)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)


def test_binary_loss_weights_oracle_examples(critic_settings: Callable[[str], CriticConfig]) -> None:
    """Apply the oracle weight to individual binary losses without changing unaided examples."""
    settings = critic_settings("ppo-critic").model_copy(update={"oracle_weight": 0.5})
    model = PPOCritic(settings)
    logits = torch.zeros(2, requires_grad=True)
    labels = torch.tensor([0., 1.])
    loss = model.loss(logits, labels, torch.tensor([False, True]))
    expected = torch.log(torch.tensor(2.)) * torch.tensor([1., 0.5])
    torch.testing.assert_close(loss, expected)
    loss.mean().backward()
    assert logits.grad is not None


def test_de_allows_regression(critic_settings: Callable[[str], CriticConfig]) -> None:
    """A discovered strategy uses execution even when future discovery is unlikely; revisions can regress."""
    model = DECritic(critic_settings("de-critic"))
    assert isinstance(model, DECritic)
    with torch.no_grad():
        model.strategy_discovered.weight.zero_()
        model.strategy_discovered.bias.fill_(30)
        model.discovery.weight.zero_()
        model.discovery.bias.fill_(-30)
        model.execution.weight.zero_()
        model.execution.bias.copy_(torch.tensor([2.0, -2.0]))
    output = model.predict(torch.zeros(2, model.settings.hidden_size), torch.tensor([False, False]), torch.tensor([1, 2]))
    torch.testing.assert_close(output, torch.tensor([2.0, -2.0]))
    assert output[1] < output[0]


def test_direct_head_is_shared_across_history_types(critic_settings: Callable[[str], CriticConfig]) -> None:
    """Condition metadata must not select a separate direct-model head; the actual history provides context."""
    model = create_critic(critic_settings("ppo-oracle-critic"), True)
    hidden = torch.randn(1, model.settings.hidden_size).expand(2, -1)
    prediction = model.predict(hidden, torch.tensor([False, True]), torch.tensor([1, 1]))
    assert prediction.shape == (2,)
    torch.testing.assert_close(prediction[0], prediction[1])


def test_oracle_de_uses_execution_head(critic_settings: Callable[[str], CriticConfig]) -> None:
    """Sketch-injected histories supervise execution directly without training discovery parameters."""
    model = DECritic(critic_settings("de-critic"))
    hidden = torch.randn(2, model.settings.hidden_size)
    oracle = torch.tensor([True, True])
    prediction = model.predict(hidden, oracle, torch.tensor([1, 2]))
    torch.testing.assert_close(prediction, model.execution(hidden)[torch.arange(2), torch.tensor([0, 1])])
    labels = torch.tensor([0., 1.])
    model.loss(prediction, labels, oracle).mean().backward()
    for parameter in (*model.discovery.parameters(), *model.strategy_discovered.parameters()):
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
    assert model.execution.weight.grad is not None and model.execution.weight.grad.abs().sum() > 0


def test_metrics_score_individual_binary_outcomes() -> None:
    """Both outcomes have Brier 0.25 at a predicted probability of one half."""
    nll, brier = outcome_scores(np.array([0., 0.]), np.array([0, 1]))
    np.testing.assert_allclose(nll, np.log(2))
    np.testing.assert_allclose(brier, 0.25)
    report = evaluation_report(np.array([2.]), np.array([0]), ["problem"])
    assert report["terminal_brier"] > 0.7


def test_metrics_preserve_confident_error_penalties() -> None:
    """Evaluation must match binary training loss even when sigmoid rounds to zero or one."""
    logits = np.array([-1000., 1000., -100., 100.])
    labels = np.array([1, 0, 0, 1])
    nll, brier = outcome_scores(logits, labels)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits), torch.from_numpy(labels).double(), reduction="none")
    np.testing.assert_allclose(nll, expected.numpy(), atol=1e-12)
    np.testing.assert_allclose(brier, [1., 1., 0., 0.], atol=1e-12)


def test_checkpoint_selection_matches_problem_macro_report() -> None:
    """Unequal prefix counts must not make checkpoint selection prefer the worse problem-level critic."""
    labels = np.ones(4)
    problem_keys = ["problem-A"] * 3 + ["problem-B"]
    selected_scores = []
    for probabilities in ([0.9, 0.9, 0.9, 0.1], [0.5, 0.5, 0.5, 0.5]):
        logits = logit(probabilities)
        metrics = compute_metrics(EvalPrediction(logits, labels), problem_keys)
        report = evaluation_report(logits, labels, problem_keys)
        assert metrics["terminal_nll"] == report["terminal_nll"]
        assert metrics["terminal_brier"] == report["terminal_brier"]
        selected_scores.append(metrics["terminal_nll"])
    assert selected_scores[1] < selected_scores[0]


@pytest.mark.parametrize("confidence", [-1000.0, -30.0, 30.0, 1000.0])
@pytest.mark.parametrize("oracle", [False, True])
@pytest.mark.parametrize("remaining", [1, 2])
def test_de_bfloat16_extremes_remain_finite(confidence: float, oracle: bool, remaining: int,
                                          critic_settings: Callable[[str], CriticConfig]) -> None:
    """Extreme finite logits must retain learning signals without probability clamps, for both history types."""
    model = DECritic(critic_settings("de-critic"))
    with torch.no_grad():
        model.strategy_discovered.weight.zero_()
        model.strategy_discovered.bias.fill_(confidence)
        model.discovery.weight.zero_()
        model.discovery.bias.fill_(confidence)
        model.execution.weight.zero_()
        model.execution.bias.fill_(confidence)
    labels = torch.tensor([1. if confidence < 0 else 0.])
    condition = torch.tensor([oracle])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = model.predict(torch.zeros(1, model.settings.hidden_size), condition, torch.tensor([remaining]))
        loss = model.loss(prediction, labels, condition).mean()
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    assert model.execution.bias.grad is not None and model.execution.bias.grad.abs().sum() > 0
    if oracle:
        torch.testing.assert_close(prediction, torch.tensor([confidence]))


@pytest.mark.parametrize("horizon", [2, 3, 5, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vectorized_de_matches_convolution(horizon: int, dtype: torch.dtype,
                                           critic_settings: Callable[[str], CriticConfig]) -> None:
    """Preserve the original discrete convolution and its gradients across budgets and autocast modes."""
    settings = critic_settings("de-critic").model_copy(update={"horizon": horizon})
    model = DECritic(settings)
    remaining = torch.arange(1, horizon)
    hidden = torch.randn(len(remaining), model.settings.hidden_size, requires_grad=True)
    with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
        prediction = model.predict(hidden, torch.zeros(len(hidden), dtype=torch.bool), remaining)
        strategy_discovered = model.strategy_discovered(hidden).float().sigmoid().squeeze(-1)
        alpha = model.discovery(hidden).float().sigmoid().squeeze(-1)
        epsilon = model.execution(hidden).float().sigmoid()
    curve = []
    for end in range(model.horizons):
        value = strategy_discovered * epsilon[:, end]
        for delay in range(end + 1):
            value = value + (1 - strategy_discovered) * alpha * (1 - alpha).pow(delay) * epsilon[:, end - delay]
        curve.append(value)
    probabilities = torch.stack(curve, dim=-1)
    expected = torch.logit(probabilities)[torch.arange(len(remaining)), remaining - 1]
    actual = prediction
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    parameters = (hidden, *model.strategy_discovered.parameters(), *model.discovery.parameters(), *model.execution.parameters())
    actual_gradients = torch.autograd.grad(actual.sum(), parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_lora_updates_adapters_only(critic_settings: Callable[[str], CriticConfig],
                                  tokenizer: PreTrainedTokenizerFast, reference_token_rows: list[dict[str, Any]]) -> None:
    """Token inputs train adapters and heads while the pretrained Qwen parameters remain frozen."""
    settings = critic_settings("ppo-critic").model_copy(update={"training_mode": "lora", "lora_rank": 2, "lora_alpha": 4, "trainable_layers": 64})
    model = create_critic(settings, True)
    batch = CriticBatcher(tokenizer)(reference_token_rows)
    logits = model(batch["inputs"], batch["oracle"], batch["remaining_blocks"])
    labels = batch["labels"]
    model.loss(logits, labels, batch["oracle"]).mean().backward()
    assert model.language_model is not None
    parameters = dict(model.language_model.named_parameters())
    adapters = [parameter for name, parameter in parameters.items() if "lora_" in name]
    assert adapters and all(parameter.requires_grad and parameter.grad is not None for parameter in adapters)
    assert any(torch.count_nonzero(cast(torch.Tensor, parameter.grad)).item() > 0 for parameter in adapters)
    assert all(not parameter.requires_grad and parameter.grad is None
               for name, parameter in parameters.items() if "lora_" not in name)


@pytest.mark.parametrize("layer_count,expected_layers", [(1, {2}), (2, {1, 2}), (3, {0, 1, 2}),
                                                       (10, {0, 1, 2})])
def test_lora_layer_selection(layer_count: int, expected_layers: set[int],
                              critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
                              reference_token_rows: list[dict[str, Any]], tmp_path: Path) -> None:
    """Only selected final layers receive adapters; gradients and checkpoint restoration remain correct."""
    settings = critic_settings("ppo-critic")
    config = Qwen3Config.from_pretrained(settings.model)
    config.num_hidden_layers = 3
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    directory = tmp_path / "three-layer-model"
    Qwen3Model(config).save_pretrained(directory)
    settings = settings.model_copy(update={"model": str(directory), "training_mode": "lora",
                                           "lora_rank": 2, "lora_alpha": 4, "trainable_layers": layer_count})
    model = create_critic(settings, True)
    batch = CriticBatcher(tokenizer)(reference_token_rows)
    model(batch["inputs"], batch["oracle"], batch["remaining_blocks"]).sum().backward()
    assert model.language_model is not None
    embedding_layer = model.language_model.get_submodule("base_model.model.embed_tokens")
    embeddings = embedding_layer(batch["inputs"]["input_ids"])
    assert not embeddings.requires_grad
    parameters = dict(model.language_model.named_parameters())
    adapted_layers = {int(name.split("layers.")[1].split(".")[0])
                      for name in parameters if "lora_" in name}
    assert adapted_layers == expected_layers
    for name, parameter in parameters.items():
        assert parameter.requires_grad == ("lora_" in name)
        assert (parameter.grad is not None) == ("lora_" in name)
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    model.eval()
    expected = model(batch["inputs"], batch["oracle"], batch["remaining_blocks"])
    checkpoint = tmp_path / "checkpoint"
    model.save(checkpoint)
    saved_settings = CriticConfig.model_validate_json((checkpoint / CONFIG_FILENAME).read_text())
    assert saved_settings.trainable_layers == layer_count
    restored = create_critic(saved_settings, True)
    restored.load_from_weights(checkpoint)
    restored.eval()
    actual = restored(batch["inputs"], batch["oracle"], batch["remaining_blocks"])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("layer_count,expected_layers", [(1, {2}), (2, {1, 2}), (3, {0, 1, 2}), (64, {0, 1, 2})])
def test_full_layer_selection_updates_and_restores(layer_count: int, expected_layers: set[int], critic_settings: Callable[[str], CriticConfig],
    tokenizer: PreTrainedTokenizerFast, reference_token_rows: list[dict[str, Any]], tmp_path: Path) -> None:
    """Full tuning changes only selected final blocks and head; a fresh checkpoint reload reproduces its logits."""
    settings = critic_settings("ppo-oracle-critic")
    config = Qwen3Config.from_pretrained(settings.model)
    config.num_hidden_layers = 3
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    directory = tmp_path / "three-layer-model"
    Qwen3Model(config).save_pretrained(directory)
    settings = settings.model_copy(update={"model": str(directory), "training_mode": "full", "trainable_layers": layer_count})
    model = create_critic(settings, True)
    batch = CriticBatcher(tokenizer)(reference_token_rows)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert all("lora_" not in name for name in before)
    assert trainable == {name for name in before if not name.startswith("language_model.")
                         or any(name.startswith(f"language_model.layers.{index}.") for index in expected_layers)}
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters() if parameter.requires_grad)
    model.train()
    logits = model(batch["inputs"], batch["oracle"], batch["remaining_blocks"])
    model.loss(logits, batch["labels"], batch["oracle"]).mean().backward()
    assert all((parameter.grad is not None) == (name in trainable) for name, parameter in model.named_parameters())
    torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4).step()
    changed = {name for name, parameter in model.named_parameters() if not torch.equal(parameter, before[name])}
    assert changed <= trainable
    assert all(any(name.startswith(f"language_model.layers.{index}.") for name in changed) for index in expected_layers)
    model.eval()
    expected = model(batch["inputs"], batch["oracle"], batch["remaining_blocks"]).detach()
    checkpoint = tmp_path / "full-layer-checkpoint"
    model.save(checkpoint)
    restored = create_critic(CriticConfig.model_validate_json((checkpoint / CONFIG_FILENAME).read_text()), True)
    restored.load_from_weights(checkpoint)
    restored.eval()
    torch.testing.assert_close(restored(batch["inputs"], batch["oracle"], batch["remaining_blocks"]), expected, rtol=0, atol=0)


@pytest.mark.parametrize("successes,trials", [(0, 8), (3, 8), (7, 16), (16, 16)])
def test_grouped_loss_matches_individual_outcomes(successes: int, trials: int,
    critic_settings: Callable[[str], CriticConfig]) -> None:
    """Fraction-target BCE and its gradient equal the mean across independent binary outcomes."""
    model = PPOCritic(critic_settings("ppo-critic"))
    logits = torch.tensor([0.7], requires_grad=True)
    label = torch.tensor([successes / trials])
    grouped = model.loss(logits, label, torch.tensor([False])).mean()
    outcomes = torch.tensor([1.] * successes + [0.] * (trials - successes))
    individual = torch.nn.functional.binary_cross_entropy_with_logits(logits.expand(trials), outcomes)
    torch.testing.assert_close(grouped, individual)
    torch.testing.assert_close(torch.autograd.grad(grouped, logits)[0], torch.autograd.grad(individual, logits)[0])
    nll, brier = outcome_scores(logits.detach().numpy(), label.numpy())
    individual_nll, individual_brier = outcome_scores(np.full(trials, 0.7), outcomes.numpy())
    np.testing.assert_allclose(nll, individual_nll.mean(), rtol=1e-6)
    np.testing.assert_allclose(brier, individual_brier.mean(), rtol=1e-6)
    report = evaluation_report(logits.detach().numpy(), label.numpy(), ["problem"])
    assert report["value_mse"] == pytest.approx((torch.sigmoid(logits).item() - successes / trials) ** 2)
