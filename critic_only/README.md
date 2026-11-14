# Critic-only experiments

Predict each trajectory’s final answer correctness at 5× from its first 1× of reasoning. Setup selects unaided and oracle-at-zero root runs, seeds 0–7; each row keeps its own history and verified binary outcome. With complete coverage, 1,000 problems give 8,000 unaided and 8,000 oracle rows. Continuation arms do not supply critic inputs.

PPO and OraclePPO apply the same MLP/value head to every solver token’s final-layer hidden state. Each causal prefix predicts that trajectory’s final reward using binary cross-entropy. Average losses over solver tokens within each trajectory, then average trajectories. References, instructions, role delimiters and padding receive no loss. This is Monte Carlo supervision, without GAE or hidden-state pooling.

Training and evaluation use the matching reference solution. The system prompt states the total budget; endpoint phase/completion metadata is omitted because earlier positions must not see it. Validation `eval_loss` measures token-averaged loss; `terminal_nll` scores only the 1× endpoint and selects the best checkpoint. Metrics weight problems equally.

| Variant | Training histories | Supervision |
|---|---|---|
| `ppo-critic` | Unaided | Every solver token |
| `ppo-oracle-critic` | Unaided + oracle | Every solver token |
| `de-critic` | Unaided + oracle | Endpoint only; existing block-based model |

**Workflow:** setup → train → evaluate. LoRA/full modes train on tokens. `--trainable-layers N` selects the last N transformer blocks. Frozen mode remains an endpoint-only control: first run `cache_hidden_states`, then train the width-512 GELU MLP on cached final-solver-token vectors. No full token-state cache is written.

```bash
uv sync --locked --extra frozen
uv run --extra frozen python -m critic_only.cli.setup --outputs-directory data/outputs --solver-model qwen3-4b-instruct-2507 --destination data/outputs/processed-token-values
DATA=data/outputs/processed-token-values/qwen3-4b-instruct-2507
RUN=output/critic_only/token-ppo
uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.train --data "$DATA" --variant ppo-critic --training-mode full --trainable-layers 1 --learning-rate 3e-6 --output "$RUN"
uv run --extra frozen python -m critic_only.cli.evaluate --data "$DATA" --checkpoint "$RUN/best" --split validation --destination "$RUN/validation"
```

GPU execution requires compatible FlashAttention-2 and a separate environment from vLLM. Training saves resumable checkpoints and metrics; use `--resume` to continue. Keep seed-1234 problem splits: 1,000 pre-2026 training problems, remaining older problems for validation, and 2026 onward for test. Select settings on validation before evaluating test.

Rerun setup into a new destination, preserving the existing `splits.parquet`. Old shared-prefix token/vector caches are incompatible with the new histories, prompt and masks; do not reuse them or overwrite previous results. Grades still use the last complete solve/revise box and exclude critique/unfinished boxes. Legacy grading rules remain rejected.
