# Critic-only experiments

Learn to predict whether a frozen solver's final answer will be correct after using its remaining self-refinement budget. These experiments train only the critic; they generate no new solver trajectories.

1. **Setup:** Read saved `continue` and `continue-oracle` branches and their audits. For example, a prefix at 2× with eight branches continued to 5× becomes one example with three blocks remaining. If three branches finish correctly, its label is `[3 successes, 8 trials]`. Earlier correct answers do not count if the final answer is wrong. Ordinary and sketch-injected histories become separate examples; future branch text never enters the input.
2. **Input and prediction:** The model's chat template places critic instructions in the system message and the saved trajectory in the user message. One transformer forward reads the whole input; its last non-padding hidden state produces one scalar probability of final-budget success. The critic does not generate an answer as text.
3. **Training:** Initialize from the solver's model and train LoRA adapters plus prediction heads using binary outcome likelihood from the counts. When a matching reference solution exists, also encode a second view with that solution in the system message, explicitly unavailable to the solver. Average the two losses, not the predictions. The reference follows the solution selection recorded during collection.

| Variant | Training histories | Prediction |
|---|---|---|
| `ordinary` | Ordinary only | Direct scalar head |
| `oracle-supervised` | Ordinary and sketch-injected | Same direct scalar head |
| `de` | Ordinary and sketch-injected | Discovery–execution formula evaluated at the remaining budget |

DE learns discovery and execution parameters internally, but only its final scalar is supervised. Compare it with `oracle-supervised` to test the structural prior under the same supervision.

Setup saves Parquet and pretokenized Arrow under `data/critic_only/`; training reuses those tokens each epoch. For local outputs, replace `--outputs-repo ... --revision ...` with `--outputs-directory <path>`. Problems from 2026 onward are test-only. Older problems split approximately 80/20 into train/validation with seed 1234, keeping every example of a problem together. `--oracle-supervision-fraction` limits the fraction of training problems contributing oracle examples.

Install separately from vLLM, process once with matching local problems in `data/datasets/`, then train on two GPUs:

```bash
uv sync --locked --extra frozen
uv run --extra frozen python -m critic_only.cli.setup --outputs-repo notadib/DELVE-Outputs --revision <outputs-commit> --solver-model qwen3-4b-instruct-2507
for variant in ordinary oracle-supervised de; do
  uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.train --variant "$variant" --output "output/critic_only/$variant"
done
```

Defaults use BF16, LoRA rank 32, one prefix per GPU, gradient accumulation 8, and gradient checkpointing. Adjust `--batch-size` and `--accumulation-steps` to fit memory. Training logs metrics and saves resumable epoch checkpoints; use `--resume <checkpoint-directory>` after interruption.

Every epoch evaluates ordinary validation histories. The best checkpoint minimizes final-outcome NLL without references, averaged per problem, and is saved under `best/`. After choosing settings on validation, evaluate the held-out test set:

```bash
uv run --extra frozen python -m critic_only.cli.evaluate --checkpoint output/critic_only/de/best --split test --destination output/critic_only/de/test
```

Evaluation saves `metrics.json` and `predictions.parquet`, reporting scalar predictions and NLL/Brier scores separately with and without reference solutions.
