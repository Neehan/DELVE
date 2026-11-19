# Critic-only experiments

Predict answer correctness at 8,192, 12,288, 16,384 and 20,480 generated tokens from an approximately 4,096-token solver prefix. Here 1× is the shared processing interval `CHECKPOINT_INTERVAL_TOKENS = 4096`; generation still uses 8,192-token blocks. Setup selects the original five-generation-block unaided and oracle-at-zero trajectories, seeds 0–7: 1,000 training problems give 16,000 histories. Each history has four binary checkpoint labels; correctness may fall after a revision. Grades use the last complete solve/revise box, excluding critique and unfinished boxes. The matching reference solution is visible only to the critic; future reasoning and grades never enter its input.

All critics apply the same width-512 GELU MLP to every solver token’s final-layer hidden state. PPO predicts four unrestricted logits; DE couples them through discovery and execution. Training averages BCE over the four horizons and valid solver tokens within each trajectory, then across trajectories. Instructions, references and padding receive no loss. Validation reports endpoint metrics for each deadline and the whole curve on unaided histories; final-5× `terminal_nll` still selects the best checkpoint. There is no pooling or GAE.

DE remains a discrete block model: interior first-block tokens use rounded-up remaining budgets 2–5; the 1× input endpoint uses budgets 1–4. This is a coarse within-block approximation, not reconstructed generation-token timing. Its execution probabilities can decrease with budget. Oracle histories use execution alone; unaided histories use the original geometric discovery–execution convolution, with no extra strategy-discovered head.

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

For an existing processed model, run `python -m critic_only.cli.setup --reprocess --destination data/outputs/processed --solver-model qwen3-4b-instruct-2507` while training is stopped. This keeps the raw trajectory shards and `splits.parquet`, saves all ten 4k checkpoints through 40,960 tokens, and replaces superseded checkpoint/token files after validation. Only the first five checkpoints define this training task. Previous critic caches/checkpoints are incompatible; start a new training run. Training uses 1,000 pre-2026 problems, validation the remaining older problems, and test 2026 onward. Select settings on validation before evaluating test. Setup never uploads data. Saved 8k boundaries retain their reviewed grades; each original block is split into roughly equal halves of retokenized assistant text using the pinned tokenizer. The 4k budgets are nominal: original token IDs were not saved. Actual segment token counts and boundary provenance are saved in the checkpoint table. Corrupt boundaries or grading errors abort preparation with the trajectory ID; no seed is silently dropped.
