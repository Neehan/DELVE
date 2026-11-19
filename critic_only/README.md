# Critic-only experiments

Predict exact answer correctness at 2×, 3×, 4× and 5× compute from the first 1× of a frozen solver’s reasoning. Setup selects unaided and oracle-at-zero trajectories, seeds 0–7: 1,000 training problems give 16,000 histories. Each history has four binary checkpoint labels; correctness may fall after a revision. Grades use the last complete solve/revise box, excluding critique and unfinished boxes. The matching reference solution is visible only to the critic; future reasoning and grades never enter its input.

All critics apply the same width-512 GELU MLP to every solver token’s final-layer hidden state. PPO predicts four unrestricted logits; DE couples them through discovery and execution. Training averages BCE over the four horizons and valid solver tokens within each trajectory, then across trajectories. Instructions, references and padding receive no loss. Validation reports endpoint metrics for each deadline and the whole curve on unaided histories; final-5× `terminal_nll` still selects the best checkpoint. There is no pooling or GAE.

DE remains a discrete block model: interior first-block tokens use rounded-up remaining budgets 2–5; the exact 1× endpoint uses budgets 1–4. This is a coarse within-block approximation, not reconstructed generation-token timing. Its execution probabilities can decrease with budget. Oracle histories use execution alone; unaided histories use the original geometric discovery–execution convolution, with no extra strategy-discovered head.

```bash
uv sync --locked --extra frozen
uv run --extra frozen python -m critic_only.cli.setup --outputs-directory data/outputs --solver-model qwen3-4b-instruct-2507 --destination data/outputs/processed-curves
DATA=data/outputs/processed-curves/qwen3-4b-instruct-2507
RUN=output/critic_only/curve-full-8
uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.train --data "$DATA" --output "$RUN" --variant ppo-oracle-critic
uv run --extra frozen python -m critic_only.cli.evaluate --data "$DATA" --checkpoint "$RUN/best" --split validation --destination "$RUN/validation"
```

Defaults: random head initialization, full tuning of the last eight Qwen blocks, LR 1e-5 shared by Qwen and the head, one epoch, batch 1 per GPU, accumulation 16, and validation/checkpoints every 125 updates. Two GPUs give effective batch 32 and 500 updates for 16,000 histories. AdamW uses 5% warm-up, linear decay and weight decay 0.01. These are starting settings from the scalar-target experiments, not validated curve results. GPU execution requires FlashAttention-2. Repeat the training command with `--resume RUN/checkpoint-N` to restore weights, optimizer, scheduler and RNG state.

`ppo-critic` uses unaided training histories; `ppo-oracle-critic` and `de-critic` also use oracle histories. Every mode trains the shared MLP and prediction heads with the same all-token loss: `head` freezes Qwen, `lora` trains adapters, and `full` trains selected Qwen layers.

Rebuild setup in a new destination: scalar-target caches and checkpoints are incompatible with these curve targets and the updated prompt. Preserve the existing `splits.parquet` in the new destination to retain identical assignments. Training uses 1,000 pre-2026 problems, validation the remaining older problems, and test 2026 onward. Select settings on validation before evaluating test. Setup never uploads data.
