# Critic-only experiments

Predict whether a frozen solver's last answer will be correct at its final budget. Cache Qwen's final-token representations once, then train small MLP critics without loading Qwen each epoch. The tokenizer cache remains available for later LoRA experiments.

1. **Setup** compiles saved conversations, checkpoint boundaries, metadata, grades, and matching problem references into shared Parquet shards. These artifacts retain their original JSON structure inside string columns. A seed-1234 split assigns exactly 1,000 pre-2026 problems to training, with proportional competition coverage; remaining older problems are validation and 2026 onward is test. Every trajectory of a problem inherits that assignment. Setup writes locally and never uploads.
2. **Extract** selects each branch after one additional block of reasoning and pairs it with that branch's final binary grade. At a 5× budget, forks at 0×–3× give inputs at 1×–4×. The critic system prompt and actual history are tokenized once without truncation. Frozen Qwen reads those tokens; its last non-padding token's final-layer hidden state is saved as a float32 vector. No response is generated. A privileged view adds the reference solution only to the critic's system prompt, using the saved oracle solution index or the corresponding source-seed round robin.
3. **Train** loads selected vectors into CPU tensors once. All variants use the same two-layer GELU MLP (width 256 by default) before their prediction heads. Training uses binary cross-entropy, averaging ordinary/privileged losses per example, not predictions. No sibling outcomes are pooled. Evaluation reports final-answer NLL and Brier score, averaged within problems and then across problems.

| Variant | Training histories | Prediction |
|---|---|---|
| `ppo-critic` | Unaided | Direct scalar logit |
| `ppo-oracle-critic` | Unaided and oracle | Same direct head |
| `de-critic` | Unaided and oracle | Discovery–execution value at the remaining budget |

DE retains the existing log-space formula: with strategy-discovered probability $s$, discovery probability $\alpha$, and remaining budget $r$, unaided success is $s\epsilon_r+(1-s)\sum_{j=1}^{r}\alpha(1-\alpha)^{j-1}\epsilon_{r-j+1}$. Oracle histories supervise $\epsilon_r$. Execution values may decrease with additional refinement. These experiments test frozen representations; they do not adapt Qwen or run PPO updates.

```text
processed/
    splits.parquet
    <model>/
        manifest.json
        train/part-*.parquet
        validation/part-*.parquet
        test/part-*.parquet
        critic_only/
            tokenized/{train,validation,test}/
            hidden_states/{train,validation,test}/part-*.parquet
```

One model-level manifest pins source checksums, tokenizer/model revision, prompt text, precision, and feature extraction identity. Feature rows store IDs, input hashes, vectors, labels, oracle status, and remaining budget together. Changing prompts or extraction settings requires a new processed directory. Rerunning extraction reuses completed shards; GPU ranks own disjoint shards, and the coordinator marks completion only after all shards exist. Do not run separate extraction jobs against the same destination concurrently.

Install separately from vLLM. Setup requires matching local problem files; replace the HF source flags with `--outputs-directory <path>` for local trajectories. Use a GPU with FlashAttention-2 for extraction:

```bash
uv sync --locked --extra frozen
uv run --extra frozen python -m critic_only.cli.setup --outputs-repo notadib/DELVE-Outputs --revision <outputs-commit> --solver-model qwen3-4b-instruct-2507
uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.extract --data data/outputs/processed/qwen3-4b-instruct-2507
for variant in ppo-critic ppo-oracle-critic de-critic; do
  uv run --extra frozen python -m critic_only.cli.train --data data/outputs/processed/qwen3-4b-instruct-2507 --variant "$variant" --output "output/critic_only/$variant"
done
```

MLP training defaults to float32, batch size 256, and no gradient accumulation. Adjust `--hidden-size`, `--batch-size`, `--epochs`, and `--learning-rate`; use `--training-problem-fraction` or `--oracle-supervision-fraction` for nested seed-1234 ablations without rebuilding features. Validation/test assignments stay fixed. Logs persist loss, learning rate, and validation scores; epoch checkpoints include optimizer/RNG state for `--resume <checkpoint-directory>`.

The best checkpoint minimizes unaided validation NLL without a reference. After choosing settings on validation, evaluate test once:

```bash
uv run --extra frozen python -m critic_only.cli.evaluate --data data/outputs/processed/qwen3-4b-instruct-2507 --checkpoint output/critic_only/de-critic/best --split test --destination output/critic_only/de-critic/test
```
