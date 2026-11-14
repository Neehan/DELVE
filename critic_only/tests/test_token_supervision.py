"""Check causal token values, exact supervision masks and trajectory-weighted Monte Carlo loss."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from critic_only.data.solver_tokens import conversation_spans, solver_token_mask
from critic_only.data.tokenizer import PrefixTokenizer
from critic_only.data.storage import CriticInputStorage
from critic_only.models.base_critic import BaseCritic
from critic_only.models.loader import create_critic
from critic_only.models.models import CriticConfig
from critic_only.tests.test_data import compile_fixture
from critic_only.tests.test_trainer import make_trainer
from critic_only.tests.models import CollectionFixture
from discovery_execution.processing.storage import write_json_atomic


def test_masks_exclude_reference_hint_and_delimiters() -> None:
    """Repeated reference/hint text and role delimiters cannot become solver supervision positions."""
    messages = [{"role": "user", "content": "hint α"}, {"role": "assistant", "content": "solve α"},
                {"role": "user", "content": "revise"}, {"role": "assistant", "content": "answer β"}]
    conversation, spans = conversation_spans(messages)
    rendered = "reference: solve α; answer β; hint α\n<user>" + conversation + "</user>"
    offsets = [(index, index + 1) for index in range(len(rendered))]
    mask = solver_token_mask(rendered, conversation, spans, offsets)
    assert "".join(char for char, included in zip(rendered, mask) if included) == "solve αanswer β"
    assert not any(mask[:rendered.index(conversation)])
    with pytest.raises(ValueError, match="exactly once"):
        solver_token_mask(rendered + conversation, conversation, spans, offsets)


def test_reference_prompt_and_causal_metadata(prepared_inputs: CollectionFixture, tokenizer: PreTrainedTokenizerFast,
    encoding_settings: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """References precede history, while future grades and endpoint-only metadata never enter either prompt."""
    processor = compile_fixture(prepared_inputs)
    for path in prepared_inputs.datasets_directory.rglob("*.jsonl"):
        path.write_text("changed after setup")
    row = next(row for row in processor.iter_inputs() if row["oracle"])
    captured = []
    encoder = PrefixTokenizer(tokenizer)
    original = encoder._encode_messages

    def capture(messages: list[dict[str, str]], spans: list[tuple[int, int]], maximum: int,
                identity: str) -> tuple[list[int], list[bool]]:
        """Record rendered message inputs while preserving the real fast-tokenizer path."""
        captured.append(messages)
        return original(messages, spans, maximum, identity)

    monkeypatch.setattr(encoder, "_encode_messages", capture)
    encoded = encoder.encode(row, encoding_settings)
    assert "The answer is one." in captured[1][0]["content"]
    assert "The answer is one." not in captured[0][0]["content"]
    assert "not available to the solver" in captured[1][0]["content"]
    assert "Use symmetry." in captured[1][1]["content"]
    assert "Thinking" in captured[1][1]["content"]
    assert "FINAL FUTURE" not in str(captured)
    assert "Remaining budget" not in str(captured) and "Current phase" not in str(captured)
    assert encoded["solver_mask"][-1] and encoded["privileged_solver_mask"][-1]
    assert sum(encoded["solver_mask"]) == sum(encoded["privileged_solver_mask"])
    with pytest.raises(ValueError, match="truncation"):
        encoder.encode(row, {**encoding_settings, "max_length": 1})


def test_stale_objective_rejected(tmp_path: Path) -> None:
    """Old shared-prefix targets must not silently train under the new objective, including frozen mode."""
    write_json_atomic(tmp_path / "manifest.json", {"tokenization": {"settings": {"target": "shared-prefix-final-success"}}})
    with pytest.raises(ValueError, match="rerun setup"):
        CriticInputStorage(tmp_path).load_metadata()


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_token_values_are_causal(mode: str, critic_settings: Callable[[str], CriticConfig]) -> None:
    """A value at token t equals a separate forward stopped at t, and later tokens cannot change it."""
    settings = critic_settings("ppo-critic").model_copy(update={"training_mode": mode, "trainable_layers": 1,
        "lora_rank": 2 if mode == "lora" else None, "lora_alpha": 4 if mode == "lora" else None})
    model = create_critic(settings, True).eval()
    ids = torch.tensor([[2, 3, 4, 5]])
    flags, budget = torch.tensor([False]), torch.tensor([2])
    values = model({"input_ids": ids, "attention_mask": torch.ones_like(ids)}, flags, budget)
    assert values.shape == ids.shape
    for endpoint in range(1, ids.shape[1] + 1):
        prefix = ids[:, :endpoint]
        value = model({"input_ids": prefix, "attention_mask": torch.ones_like(prefix)}, flags, budget)
        torch.testing.assert_close(values[:, endpoint - 1], value[:, -1], atol=1e-6, rtol=1e-5)
    changed = ids.clone()
    changed[:, 2:] = 7
    other = model({"input_ids": changed, "attention_mask": torch.ones_like(changed)}, flags, budget)
    torch.testing.assert_close(values[:, :2], other[:, :2], atol=0, rtol=0)


def test_loss_masks_and_endpoint_selection(critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
    reference_token_rows: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only selected positions receive BCE gradients; rows have equal weight and evaluation returns endpoints."""
    settings = critic_settings("ppo-critic").model_copy(update={"training_mode": "full", "trainable_layers": 1})
    rows = [reference_token_rows[0], {**reference_token_rows[1], "privileged_input_ids": [2, 3],
                                   "privileged_solver_mask": [False, True], "length": 2}]
    trainer = make_trainer(tmp_path / "run", settings, tokenizer, rows, [])
    batch = trainer.data_collator(rows)
    logits = torch.tensor([[9., .1, -.3, .7], [8., -.5, 10., 10.]], requires_grad=True)
    model = cast(BaseCritic, trainer.model)
    monkeypatch.setattr(model, "forward", lambda **kwargs: logits)
    loss, output = trainer.compute_loss(model, batch, return_outputs=True)
    first = torch.nn.functional.binary_cross_entropy_with_logits(logits[0, 1:], batch["labels"][0].expand(3))
    second = torch.nn.functional.binary_cross_entropy_with_logits(logits[1, 1], batch["labels"][1])
    torch.testing.assert_close(loss, (first + second) / 2)
    torch.testing.assert_close(output["logits"], torch.stack([logits[0, 3], logits[1, 1]]))
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[~batch["inputs"]["solver_mask"]].count_nonzero() == 0
    assert logits.grad[batch["inputs"]["solver_mask"]].count_nonzero() == 4
