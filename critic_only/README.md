# Critic-only experiments

Predict final answer correctness at 5× from a solver’s first 1× of reasoning. Setup selects unaided and oracle-at-zero trajectories, seeds 0–7: 1,000 training problems give 16,000 histories with individual verified binary labels. The matching reference solution is visible only to the critic. Grades use the last complete solve/revise box, excluding critique and unfinished boxes.

PPO critics apply a shared width-512 GELU MLP to every solver token’s final-layer hidden state. Training averages binary cross-entropy over solver tokens within each trajectory, then over trajectories. Instructions, references and padding receive no loss. Validation scores only the 1× endpoint on unaided histories; `terminal_nll` selects the best checkpoint. This is Monte Carlo supervision, without pooling or GAE.

**Two-stage recipe:** train the head once with Qwen frozen, then initialize each fine-tuning run from that same head and pretrained Qwen. Fine-tuning starts with a fresh optimizer and uses one LR for both head and Qwen. The recipe below uses two GPUs, batch 1 per GPU and accumulation 16: effective batch 32, or 500 updates per epoch on 16,000 histories. Both stages use AdamW, 5% warm-up, linear LR decay and weight decay 0.01.

```bash
uv sync --locked --extra frozen
uv run --extra frozen python -m critic_only.cli.setup --outputs-directory data/outputs --solver-model qwen3-4b-instruct-2507 --destination data/outputs/processed-token-values
DATA=data/outputs/processed-token-values/qwen3-4b-instruct-2507
HEAD=output/critic_only/head
RUN=output/critic_only/full-8

uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.train --data "$DATA" --output "$HEAD" --variant ppo-oracle-critic --training-mode head --hidden-size 512 --learning-rate 3e-4 --epochs 1 --batch-size 1 --accumulation-steps 16 --checkpoint-steps 125
uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.train --data "$DATA" --output "$RUN" --variant ppo-oracle-critic --training-mode full --trainable-layers 8 --head-checkpoint "$HEAD/best" --hidden-size 512 --learning-rate 3e-6 --epochs 1 --batch-size 1 --accumulation-steps 16 --checkpoint-steps 125
uv run --extra frozen python -m critic_only.cli.evaluate --data "$DATA" --checkpoint "$RUN/best" --split validation --destination "$RUN/validation"
```

This recipe improved validation NLL from 0.35773 at head initialization to 0.30012 after eight-layer full tuning in one seed; continued head-only training reached 0.35574. Replication and ablations are pending. GPU execution requires compatible FlashAttention-2 in a separate environment from vLLM. Training saves metrics and resumable checkpoints; repeat the original command with `--resume RUN/checkpoint-N` to restore weights, optimizer, scheduler and RNG state.

`ppo-critic` trains on unaided histories; `ppo-oracle-critic` also uses oracle histories. `de-critic` retains its separate endpoint objective. Training modes are `head` (live frozen Qwen), `full` or `lora` (adapt the final N blocks), and `frozen` (the optional cached-endpoint baseline, requiring `cache_hidden_states` first). Head initialization requires matching model, architecture and token-cache identity; cached-endpoint checkpoints are not head-training checkpoints.

Keep the seeded problem split fixed: 1,000 pre-2026 training problems, remaining older problems for validation, and 2026 onward for test. Select settings on validation before evaluating test. Existing token-level caches remain usable; older shared-prefix caches require setup in a new destination with the original `splits.parquet` preserved.
