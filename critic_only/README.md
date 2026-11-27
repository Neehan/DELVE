# Critic-only experiments

Predict correctness at 2×–5× from an approximately 1× solver prefix, where 1× = 4,096 tokens. Setup uses seeds 0–7 from unaided and oracle-at-zero runs: 1,000 training problems give 16,000 histories. Validation uses the remaining pre-2026 problems; test uses 2026 onward. Labels grade the last complete solve/revise box, excluding critique. The matching reference solution is visible only to the critic; future reasoning and labels never enter its input.

All critics apply a width-512 GELU MLP to every solver token’s final-layer hidden state. Training averages BCE over four checkpoints and solver tokens, then across trajectories. Instructions, references and padding receive no loss. Validation measures the 1× endpoint on unaided histories; 5× `terminal_nll` selects the best checkpoint.

PPO predicts four unrestricted logits. DE combines geometric discovery with an execution curve; oracle histories use execution alone. DE rounds remaining budgets up to whole blocks, and execution probabilities may decrease across checkpoints.

`de-residual-critic` inherits DE and adds a zero-initialized correction to each checkpoint logit. Its trainer adds `λ × residual²` to BCE using the same token masks and trajectory weights. `--residual-penalty` sets λ (untuned default `0.01`; `0` disables the penalty). Reported NLL/Brier and checkpoint selection use corrected predictions without adding the penalty.

Start from exported trajectory Parquet files (`discovery_execution.scripts.export_outputs`), or use `--outputs-repo REPO --revision COMMIT` instead of `--outputs-directory`. HF downloads are temporary; setup retains split trajectory shards for reprocessing and separate tokenized training inputs. User-supplied local exports are untouched.

```bash
uv sync --locked --extra frozen
uv run --extra frozen python -m critic_only.cli.setup --outputs-directory data/outputs/published --solver-model qwen3-4b-instruct-2507 --destination data/outputs/processed-curves
DATA=data/outputs/processed-curves/qwen3-4b-instruct-2507
RUN=output/critic_only/curve-full-8
uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.train --data "$DATA" --output "$RUN" --variant ppo-oracle-critic
uv run --extra frozen python -m critic_only.cli.evaluate --data "$DATA" --checkpoint "$RUN/best" --split validation --destination "$RUN/validation"
```

Defaults: one epoch, random head, full tuning of eight Qwen blocks, LR 1e-5, batch 1 per GPU and accumulation 16. Two GPUs give effective batch 32 and 500 updates, with validation/checkpoints every 125 updates. AdamW uses 5% warm-up, linear decay and weight decay 0.01. GPU runs require FlashAttention-2. Resume with `--resume RUN/checkpoint-N`.

`ppo-critic` trains on unaided histories; the other variants also use oracle histories. All supervise the same solver tokens and checkpoint labels. Training modes: `head` freezes Qwen, `lora` trains adapters, and `full` trains selected Qwen layers.

Generation still uses 8,192-token blocks. Setup splits each into roughly equal halves of retokenized text, preserving original boundaries and reviewed grades; all ten analysis checkpoints are saved. Export does not re-grade legacy audits. To rebuild derived inputs, stop training and run setup with `--reprocess --destination data/outputs/processed-curves --solver-model qwen3-4b-instruct-2507`. Splits and source trajectories are retained; changed inputs require a new training run.
