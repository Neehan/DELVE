"""Check DE inheritance, residual regularization, masked gradients, and checkpoint recovery."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast

from critic_only.cli import train
from critic_only.constants import DEFAULT_RESIDUAL_PENALTY
from critic_only.data.critic_batcher import CriticBatcher
from critic_only.models.de_critic import DECritic
from critic_only.models.de_residual_critic import DEResidualCritic
from critic_only.models.loader import create_critic
from critic_only.models.models import CriticConfig
from critic_only.tests.test_trainer import make_trainer
from critic_only.trainer.de_residual_trainer import DEResidualTrainer


@pytest.mark.parametrize('horizon', [2, 3, 5, 8])
def test_zero_residual_matches_strict_de(horizon: int, critic_settings: Callable[[str], CriticConfig]) -> None:
    """Matched DE weights give exactly the same predictions for both conditions and interior/end budgets."""
    settings = critic_settings('de-residual-critic').model_copy(update={'horizon': horizon})
    torch.manual_seed(17)
    strict = DECritic(settings.model_copy(update={'variant': 'de-critic', 'residual_penalty': None}))
    torch.manual_seed(17)
    model = DEResidualCritic(settings)
    hidden = torch.randn(2, 3, settings.hidden_size)
    oracle = torch.tensor([False, True])
    budgets = torch.tensor([[horizon, horizon, horizon - 1]]).expand(2, -1)
    expected = strict.predict(hidden, oracle, budgets)['logits']
    output = model.predict(hidden, oracle, budgets)
    torch.testing.assert_close(output['logits'], expected, rtol=0, atol=0)
    assert output['residual'].shape == expected.shape
    assert output['residual'].count_nonzero() == 0
    with torch.no_grad():
        model.residual_head.bias.fill_(-0.7)
    shifted = model.predict(hidden, oracle, budgets)
    torch.testing.assert_close(shifted['logits'], expected - 0.7)


@pytest.mark.parametrize('penalty', [0., 0.2])
@pytest.mark.parametrize('oracle_weight', [0., 0.4, 1.])
def test_residual_penalty_matches_masked_objective(penalty: float, oracle_weight: float,
    critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
    reference_token_rows: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """BCE and penalty use identical masks/weights; instructions, padding and zero-weight rows have zero gradient."""
    settings = critic_settings('de-residual-critic').model_copy(update={'residual_penalty': penalty, 'oracle_weight': oracle_weight})
    rows = [{**reference_token_rows[0], 'oracle': False}, {**reference_token_rows[1], 'oracle': True,
            'privileged_input_ids': [2, 3], 'privileged_solver_mask': [False, True], 'length': 2}]
    trainer = make_trainer(tmp_path / 'run', settings, tokenizer, rows, [])
    assert isinstance(trainer, DEResidualTrainer)
    model = cast(DEResidualCritic, trainer.model)
    batch = CriticBatcher(tokenizer)(rows)
    logits = torch.randn(2, 4, 2, requires_grad=True)
    residual = torch.tensor([[[1000., 1000.], [.2, -.3], [.4, -.5], [.6, -.7]],
                             [[1000., 1000.], [.8, -.9], [1000., 1000.], [1000., 1000.]]], requires_grad=True)
    forward = Mock(return_value={'logits': logits, 'residual': residual})
    monkeypatch.setattr(model, 'forward', forward)
    actual, outputs = trainer.compute_loss(model, batch, return_outputs=True)
    forward.assert_called_once()
    first = F.binary_cross_entropy_with_logits(logits[0, 1:], batch['labels'][0].expand(3, 2))
    second = F.binary_cross_entropy_with_logits(logits[1, 1], batch['labels'][1])
    expected = (first + penalty * residual[0, 1:].square().mean()
                + oracle_weight * (second + penalty * residual[1, 1].square().mean())) / 2
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(torch.autograd.grad(actual, (logits, residual), retain_graph=True),
                                                  torch.autograd.grad(expected, (logits, residual))):
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert actual_gradient[~batch['inputs']['solver_mask']].count_nonzero() == 0
        if oracle_weight == 0:
            assert actual_gradient[1].count_nonzero() == 0
    assert set(outputs) == {'logits'}
    torch.testing.assert_close(outputs['logits'], torch.stack([logits[0, 3], logits[1, 1]]))


@pytest.mark.parametrize('mode', ['head', 'lora', 'full'])
def test_residual_updates_and_checkpoint_restore(mode: str, critic_settings: Callable[[str], CriticConfig],
    tokenizer: PreTrainedTokenizerFast, reference_token_rows: list[dict[str, Any]], tmp_path: Path) -> None:
    """The same pass trains DE and correction heads; nonzero corrections and the penalty survive checkpoint reload."""
    settings = critic_settings('de-residual-critic').model_copy(update={'training_mode': mode,
        'trainable_layers': None if mode == 'head' else 1, 'lora_rank': 2 if mode == 'lora' else None,
        'lora_alpha': 4 if mode == 'lora' else None, 'residual_penalty': 0.2})
    rows = [reference_token_rows[0], {**reference_token_rows[1], 'oracle': True}]
    trainer = make_trainer(tmp_path / 'run', settings, tokenizer, rows, [])
    model = cast(DEResidualCritic, trainer.model)
    batch = CriticBatcher(tokenizer)(rows)
    trainer.compute_loss(model, batch).backward()
    for head in (model.discovery, model.execution, model.residual_head):
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    model.eval()
    expected = model(batch['inputs'], batch['oracle'], batch['remaining_blocks'])
    assert expected['residual'].count_nonzero() > 0
    checkpoint = tmp_path / 'checkpoint'
    model.save(checkpoint)
    restored = create_critic(settings, True).eval()
    restored.load_from_weights(checkpoint)
    actual = restored(batch['inputs'], batch['oracle'], batch['remaining_blocks'])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    changed = create_critic(settings.model_copy(update={'residual_penalty': 0.3}), True)
    with pytest.raises(ValueError, match='configuration differs'):
        changed.load_from_weights(checkpoint)


def test_oracle_residual_does_not_train_discovery(critic_settings: Callable[[str], CriticConfig]) -> None:
    """Oracle rows correct execution logits while discovery remains excluded from their prediction."""
    model = DEResidualCritic(critic_settings('de-residual-critic'))
    with torch.no_grad():
        model.residual_head.bias.fill_(0.4)
    hidden = torch.randn(2, 3, model.settings.hidden_size)
    output = model.predict(hidden, torch.ones(2, dtype=torch.bool), torch.full((2, 3), 2))
    torch.testing.assert_close(output['logits'], model.execution(hidden)[..., :2] + 0.4)
    output['logits'].sum().backward()
    assert model.discovery.weight.grad is not None
    assert model.discovery.weight.grad.count_nonzero() == 0


@pytest.mark.parametrize('oracle', [False, True])
@pytest.mark.parametrize('logit', [-1000., 1000.])
def test_residual_bfloat16_extremes_remain_finite(oracle: bool, logit: float,
    critic_settings: Callable[[str], CriticConfig]) -> None:
    """Logit addition and the FP32 squared penalty stay finite under BF16 autocast."""
    model = DEResidualCritic(critic_settings('de-residual-critic'))
    with torch.no_grad():
        for head in (model.discovery, model.execution, model.residual_head):
            head.weight.zero_()
            head.bias.fill_(logit)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        outputs = model.predict(torch.zeros(1, 2, model.settings.hidden_size), torch.tensor([oracle]), torch.tensor([[3, 2]]))
        bce = model.loss(outputs['logits'], torch.tensor([[0., 1.]]), torch.tensor([oracle]))
        loss = (bce + DEFAULT_RESIDUAL_PENALTY * outputs['residual'].float().square()).mean()
    assert outputs['logits'].dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


@pytest.mark.parametrize('coefficient', ['-0.1', 'nan', 'inf'])
def test_invalid_residual_penalty_fails_early(coefficient: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid regularization fails at CLI parsing before loading a dataset or model."""
    monkeypatch.setattr('sys.argv', ['train', '--data', '/missing', '--variant', 'de-residual-critic', '--residual-penalty', coefficient])
    with pytest.raises(ValueError, match='nonnegative and finite'):
        train.parse_arguments()


@pytest.mark.parametrize('variant', ['ppo-critic', 'ppo-oracle-critic', 'de-critic'])
def test_other_variants_reject_residual_penalty(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A penalty option must not be silently ignored by a baseline critic."""
    monkeypatch.setattr('sys.argv', ['train', '--data', '/missing', '--variant', variant, '--residual-penalty', '0'])
    with pytest.raises(ValueError, match='requires --variant'):
        train.parse_arguments()


def test_residual_penalty_default_and_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """The residual variant has an explicit initial coefficient; zero is a valid unregularized ablation."""
    arguments = ['train', '--data', '/missing', '--variant', 'de-residual-critic']
    monkeypatch.setattr('sys.argv', arguments)
    assert train.parse_arguments().residual_penalty == DEFAULT_RESIDUAL_PENALTY
    monkeypatch.setattr('sys.argv', [*arguments, '--residual-penalty', '0'])
    assert train.parse_arguments().residual_penalty == 0


@pytest.mark.parametrize('coefficient', [None, -1, float('nan'), float('inf')])
def test_saved_residual_config_is_validated(coefficient: float | None, critic_settings: Callable[[str], CriticConfig]) -> None:
    """Direct construction and checkpoint JSON reject missing or invalid coefficients, not just the CLI."""
    values = critic_settings('de-residual-critic').model_dump()
    values['residual_penalty'] = coefficient
    with pytest.raises(ValueError):
        CriticConfig.model_validate(values)
