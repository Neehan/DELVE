# Critic-only experiments

Learn to predict whether a frozen solver's final answer will be correct after using its remaining self-refinement budget. These experiments train only the critic; they generate no new solver trajectories.

1. **Setup:** Each branch becomes a separate example after 1× of continuation. A fork at 2×, optional sketch injection, and 1× of solver reasoning gives an input at 3× with two blocks remaining in a 5× run. Eight branches give eight inputs, each labeled by its own final correct/incorrect answer. Full-budget `unaided` and `oracle-execution` runs injected at 0× supply the problem-only fork; `continue` and `continue-oracle` supply later forks. For a 5× budget, use forks at 0×–3×; the next block after a 4× fork is already terminal and is excluded.
2. **Input and prediction:** The model's chat template places critic instructions in the system message and the saved history in the user message, including the sketch and the solver's response when present. One forward pass maps the last non-padding hidden state to one probability of final-budget correctness. Later text and audit grades never enter the input; a correct intermediate answer can still finish incorrectly.
3. **Training:** Initialize from the solver's model and train LoRA adapters plus prediction heads with binary cross-entropy. Each label is a single binary final outcome; sibling outcomes are not pooled. When a matching reference solution exists, also encode a privileged view with that solution in the system message, explicitly unavailable to the solver. Average the two losses, not predictions. References follow the saved oracle solution selection or the collection's seed-based round robin.

| Variant / class | Training histories | Prediction |
|---|---|---|
| `ppo-critic` / `PPOCritic` | Unaided branch histories | Direct scalar head |
| `ppo-oracle-critic` / `PPOOracleCritic` | Unaided and oracle branch histories | Same direct scalar head |
| `de-critic` / `DECritic` | Unaided and oracle branch histories | Discovery–execution formula evaluated at the remaining budget |

DE reads the actual continuation and learns discovery and execution parameters internally. Oracle histories use the execution prediction; unaided histories use the geometric discovery–execution convolution. With strategy-discovered probability $s$ (`strategy_discovered`) and remaining budget $r$, unaided success is $s\epsilon_r + (1-s)\sum_{j=1}^{r} \alpha(1-\alpha)^{j-1}\epsilon_{r-j+1}$; oracle success is $\epsilon_r$. This means the solver already has a viable strategy; $\alpha$ describes discovery while still searching. Only final correctness supervises the scalar; there is no Bayesian update. Compare PPOCritic with PPOOracleCritic for oracle supervision, then PPOOracleCritic with DECritic for DE structure on the same data.

Setup saves Parquet and pretokenized Arrow under `data/critic_only/`; training reuses those tokens each epoch. Rebuild older pooled-prefix or count-label caches into a new `--destination` before using this setup. For local outputs, replace `--outputs-repo ... --revision ...` with `--outputs-directory <path>`. Problems from 2026 onward are test-only. Older problems split approximately 80/20 into train/validation with seed 1234, keeping every example of a problem together. `--oracle-supervision-fraction` limits the fraction of training problems contributing oracle examples.

Install separately from vLLM, process once with matching local problems in `data/datasets/`, then train on two GPUs:

```bash
uv sync --locked --extra frozen
uv run --extra frozen python -m critic_only.cli.setup --outputs-repo notadib/DELVE-Outputs --revision <outputs-commit> --solver-model qwen3-4b-instruct-2507
for variant in ppo-critic ppo-oracle-critic de-critic; do
  uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.train --variant "$variant" --output "output/critic_only/$variant"
done
```

GPU training uses FlashAttention-2 and BF16; install a compatible FlashAttention-2 kernel in the training environment. Defaults use LoRA rank 32, one branch history per GPU, gradient accumulation 8, and gradient checkpointing. Adjust `--batch-size` and `--accumulation-steps` to fit memory. Training logs metrics and saves resumable epoch checkpoints; use `--resume <checkpoint-directory>` after interruption.

Main experiments use all training problems. For ablations, train with `--training-problem-fraction 0.5` to retain half the problems and their available histories (unaided only for PPOCritic). Subsets are nested and fixed with seed 1234, rounding problem counts down; validation/test stay unchanged. No new setup cache is needed.

Every epoch evaluates unaided validation histories. The best checkpoint minimizes final-outcome NLL without references, averaged per problem, and is saved under `best/`. After choosing settings on validation, evaluate the held-out test set:

```bash
uv run --extra frozen python -m critic_only.cli.evaluate --checkpoint output/critic_only/de-critic/best --split test --destination output/critic_only/de-critic/test
```

Evaluation saves `metrics.json` and `predictions.parquet`, reporting scalar predictions and NLL/Brier scores separately with and without reference solutions.
