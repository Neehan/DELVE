"""Check causal token values, exact supervision masks and trajectory-weighted Monte Carlo loss."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from critic_only.data.solver_tokens import conversation_spans, solver_token_mask
from critic_only.data.critic_batcher import CriticBatcher
from critic_only.data.tokenizer import PrefixTokenizer
from critic_only.data.storage import CriticInputStorage
from critic_only.models.base_critic import BaseCritic
from critic_only.models.loader import create_critic
from critic_only.models.models import CriticConfig
from critic_only.tests.test_data import compile_fixture
from critic_only.tests.test_trainer import make_trainer
from critic_only.tests.models import CollectionFixture
from discovery_execution.processing.storage import write_json_atomic


@pytest.fixture
def byte_tokenizer(tokenizer: PreTrainedTokenizerFast) -> PreTrainedTokenizerFast:
    """Exercise byte-level Unicode and whitespace offsets without downloading a tokenizer during tests."""
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(["assistant:\n\nαβ", "user:\nproblem", "answer\\boxed{2}\n", " reference "],
        trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=["[PAD]", "<|im_start|>", "<|im_end|>"]))
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", padding_side="right",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"], chat_template=tokenizer.chat_template)


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


@pytest.mark.parametrize("ending", [[], [{"role": "user", "content": "Revise"}],
    [{"role": "assistant", "content": ""}], [{"role": "assistant", "content": " \n\t"}]])
def test_history_requires_nonempty_solver_endpoint(ending: list[dict[str, str]]) -> None:
    """An empty or non-solver endpoint must not silently select an earlier assistant's hidden state."""
    history = [{"role": "assistant", "content": "Earlier reasoning"}] + ending if ending else []
    with pytest.raises(ValueError, match="end"):
        conversation_spans(history)


def test_token_crossing_boundary_is_excluded() -> None:
    """Offset spans exclude merged delimiter/content tokens and zero-length special-token offsets."""
    conversation, spans = conversation_spans([{"role": "assistant", "content": "αβ"}])
    start, end = spans[0]
    offsets = [(0, 0), (start - 1, start + 1), (start + 1, end), (end, end + 1)]
    assert solver_token_mask(conversation + "!", conversation, spans, offsets) == [False, False, True, False]
    with pytest.raises(ValueError, match="tokens"):
        solver_token_mask(conversation, conversation, spans, [(0, 0), (start - 1, end)])
    conversation, spans = conversation_spans([{"role": "assistant", "content": "earlier"},
                                             {"role": "assistant", "content": "β"}])
    with pytest.raises(ValueError, match="final solver"):
        solver_token_mask(conversation, conversation, spans, [spans[0], (spans[1][0] - 1, spans[1][1])])


@pytest.mark.parametrize("content", ["αβ → ² 😀", "<think>reasoning without closing tag", r"unfinished \\boxed{",
    "answer\n\n", "a", "assistant:\nnot a role change"])
def test_byte_tokenization_preserves_endpoint(content: str, byte_tokenizer: PreTrainedTokenizerFast) -> None:
    """Unicode, reasoning markers and unfinished boxes survive tokenization through the exact final solver token."""
    conversation, spans = conversation_spans([{"role": "user", "content": "problem"},
                                             {"role": "assistant", "content": content}])
    messages = [{"role": "system", "content": "Reference: " + content}, {"role": "user", "content": conversation}]
    formatter = PrefixTokenizer(byte_tokenizer)
    ids, mask = formatter._encode_messages(messages, spans, 1000, "example")
    assert cast(str, byte_tokenizer.decode(ids)).endswith(content)
    assert byte_tokenizer.decode([token for token, included in zip(ids, mask) if included]) == content
    assert mask[-1] and len(mask) == len(ids)
    assert formatter._encode_messages(messages, spans, len(ids), "example") == (ids, mask)
    with pytest.raises(ValueError, match="truncation"):
        formatter._encode_messages(messages, spans, len(ids) - 1, "example")


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
    """Old shared-prefix targets must not silently train under the new objective."""
    write_json_atomic(tmp_path / "manifest.json", {"tokenization": {"settings": {"target": "shared-prefix-final-success"}}})
    with pytest.raises(ValueError, match="rerun setup"):
        CriticInputStorage(tmp_path).load_metadata()


@pytest.mark.parametrize("variant", ["ppo-critic", "ppo-oracle-critic", "de-critic", "de-residual-critic"])
@pytest.mark.parametrize("mode", ["head", "lora", "full"])
def test_token_values_are_causal(variant: str, mode: str, critic_settings: Callable[[str], CriticConfig]) -> None:
    """A value at token t equals a separate forward stopped at t, and later tokens cannot change it."""
    settings = critic_settings(variant).model_copy(update={"training_mode": mode, "trainable_layers": None if mode == "head" else 1,
        "lora_rank": 2 if mode == "lora" else None, "lora_alpha": 4 if mode == "lora" else None})
    model = create_critic(settings, True).eval()
    ids = torch.tensor([[2, 3, 4, 5]])
    flags, budget = torch.tensor([False]), torch.tensor([[2, 2, 2, 2]])
    values = model({"input_ids": ids, "attention_mask": torch.ones_like(ids)}, flags, budget)["logits"]
    assert values.shape == (*ids.shape, settings.horizon - 1)
    for endpoint in range(1, ids.shape[1] + 1):
        prefix = ids[:, :endpoint]
        value = model({"input_ids": prefix, "attention_mask": torch.ones_like(prefix)}, flags, budget[:, :endpoint])["logits"]
        torch.testing.assert_close(values[:, endpoint - 1], value[:, -1], atol=1e-6, rtol=1e-5)
    changed = ids.clone()
    changed[:, 2:] = 7
    other = model({"input_ids": changed, "attention_mask": torch.ones_like(changed)}, flags, budget)["logits"]
    torch.testing.assert_close(values[:, :2], other[:, :2], atol=0, rtol=0)


def test_loss_masks_and_endpoint_selection(critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
    reference_token_rows: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only selected positions receive BCE gradients; rows have equal weight and evaluation returns endpoints."""
    settings = critic_settings("ppo-critic").model_copy(update={"training_mode": "full", "trainable_layers": 1})
    rows = [reference_token_rows[0], {**reference_token_rows[1], "privileged_input_ids": [2, 3],
                                   "privileged_solver_mask": [False, True], "length": 2}]
    trainer = make_trainer(tmp_path / "run", settings, tokenizer, rows, [])
    batch = trainer.data_collator(rows)
    logits = torch.tensor([[[9., 9.], [.1, -.2], [-.3, .4], [.7, -.6]],
                           [[8., 8.], [-.5, .3], [10., 10.], [10., 10.]]], requires_grad=True)
    model = cast(BaseCritic, trainer.model)
    monkeypatch.setattr(model, "forward", lambda **kwargs: {"logits": logits})
    loss, output = trainer.compute_loss(model, batch, return_outputs=True)
    first = torch.nn.functional.binary_cross_entropy_with_logits(logits[0, 1:], batch["labels"][0].expand(3, 2))
    second = torch.nn.functional.binary_cross_entropy_with_logits(logits[1, 1], batch["labels"][1])
    torch.testing.assert_close(loss, (first + second) / 2)
    torch.testing.assert_close(output["logits"], torch.stack([logits[0, 3], logits[1, 1]]))
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[~batch["inputs"]["solver_mask"]].count_nonzero() == 0
    assert logits.grad[batch["inputs"]["solver_mask"]].count_nonzero() == 8


@pytest.mark.parametrize("mask", [[], [False] * 4, [True], [True, True, True, False]])
def test_invalid_supervision_masks_rejected(mask: list[bool], tokenizer: PreTrainedTokenizerFast,
    reference_token_rows: list[dict[str, Any]]) -> None:
    """Reject masks with no targets, wrong length or an unsupervised terminal position before training."""
    row = {**reference_token_rows[0], "privileged_solver_mask": mask}
    with pytest.raises(ValueError, match="solver mask"):
        CriticBatcher(tokenizer)([row])


@pytest.mark.parametrize("variant", ["ppo-oracle-critic", "de-critic", "de-residual-critic"])
@pytest.mark.parametrize("mode", ["head", "lora", "full"])
def test_padding_cannot_change_values_or_gradients(variant: str, mode: str, critic_settings: Callable[[str], CriticConfig],
    tokenizer: PreTrainedTokenizerFast, reference_token_rows: list[dict[str, Any]]) -> None:
    """A one-token solver span in a padded batch has the same values and gradient as its unpadded forward."""
    settings = critic_settings(variant).model_copy(update={"training_mode": mode,
        "trainable_layers": None if mode == "head" else 1, "lora_rank": 2 if mode == "lora" else None, "lora_alpha": 4 if mode == "lora" else None})
    model = create_critic(settings, True).eval()
    rows = [{**reference_token_rows[0], "privileged_input_ids": [2, 3], "privileged_solver_mask": [False, True]},
            reference_token_rows[1]]
    batch = CriticBatcher(tokenizer)(rows)
    single = CriticBatcher(tokenizer)(rows[:1])
    values = model(batch["inputs"], batch["oracle"], batch["remaining_blocks"])["logits"]
    expected = model(single["inputs"], single["oracle"], single["remaining_blocks"])["logits"]
    torch.testing.assert_close(values[:1, :2], expected)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for actual, target in zip(torch.autograd.grad(values[0, 1].sum(), parameters), torch.autograd.grad(expected[0, 1].sum(), parameters)):
        torch.testing.assert_close(actual, target, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_head_and_finetuning_initial_token_values_match(mode: str, critic_settings: Callable[[str], CriticConfig],
    tokenizer: PreTrainedTokenizerFast, reference_token_rows: list[dict[str, Any]]) -> None:
    """Before updates, matched frozen-Qwen and fine-tuning critics predict identically at every token."""
    settings = critic_settings("ppo-oracle-critic")
    head = create_critic(settings, True).eval()
    live = create_critic(settings.model_copy(update={"training_mode": mode, "trainable_layers": 1,
        "lora_rank": 2 if mode == "lora" else None, "lora_alpha": 4 if mode == "lora" else None}), True).eval()
    live.hidden_state_projection.load_state_dict(head.hidden_state_projection.state_dict())
    live.get_submodule("prediction_head").load_state_dict(head.get_submodule("prediction_head").state_dict())
    rows = [reference_token_rows[0], {**reference_token_rows[1], "privileged_input_ids": [2, 3],
                                    "privileged_solver_mask": [False, True]}]
    batch = CriticBatcher(tokenizer)(rows)
    with torch.no_grad():
        expected = head(batch["inputs"], batch["oracle"], batch["remaining_blocks"])["logits"]
        actual = live(batch["inputs"], batch["oracle"], batch["remaining_blocks"])["logits"]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("label", [0., 1.])
def test_extreme_token_logits_stay_finite(label: float, critic_settings: Callable[[str], CriticConfig]) -> None:
    """Correct and incorrect saturated predictions retain finite BCE and gradients, including zero oracle weight."""
    model = create_critic(critic_settings("ppo-oracle-critic").model_copy(update={"oracle_weight": 0}), True)
    logits = torch.tensor([[[-1000., 1000.]], [[1000., -1000.]]], requires_grad=True)
    loss = model.loss(logits, torch.tensor([[label, label], [label, label]]), torch.tensor([False, True]))
    assert torch.isfinite(loss).all() and loss[1].count_nonzero() == 0
    loss.mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[1].count_nonzero() == 0
