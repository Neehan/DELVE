# Critic-only experiments

Train a critic to predict final-answer correctness at 5× from a shared 1× solver state. Setup groups `continue-at-1x` and `continue-oracle-at-1x` runs separately. The input is the source history plus the optional saved hint, **before any child continuation**. The target is `successful continuations / total continuations`; all available independent seeds are used, so 8 and 16 branches need no configuration change. Different source seeds are separate inputs, and conflicting hints or sampling settings are rejected.

With one source seed per problem, 1,000 problems give 2,000 inputs across the two conditions. Each has versions with and without the matching critic-only reference solution. Their BCE losses are averaged, then inputs receive equal weight regardless of branch count. This equals averaging binary BCE within each branch group. Parquet rows retain branch seeds, paths, and outcomes for auditing. Evaluation reports mean outcome NLL/Brier and `value_mse` against the observed success fraction.

| Variant | Training histories | Prediction |
|---|---|---|
| `ppo-critic` | Unaided | Direct scalar |
| `ppo-oracle-critic` | Unaided + oracle | Direct scalar |
| `de-critic` | Unaided + oracle | Discovery–execution decomposition |

**Workflow:** setup → cache hidden states → train → evaluate. Setup writes reusable Parquet chunks and tokenizes both input versions. The seed-1234 problem split reserves 1,000 pre-2026 problems for training, remaining older problems for validation, and 2026 onward for test.

Frozen mode trains a two-layer, width-512 GELU MLP on cached Qwen vectors. For LoRA, skip caching and pass `--training-mode lora --lora-layers N` (last N layers; default 64). Set `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce CUDA memory fragmentation. The solver stays fixed in both modes.

Run from the repository root with matching problem files in `data/datasets`. Use a separate environment from vLLM; GPU encoding and LoRA require a compatible `flash-attn` installation in addition to the dependencies below.

```bash
uv sync --locked --extra frozen
uv run --extra frozen python -m critic_only.cli.setup --outputs-repo notadib/DELVE-Outputs --revision <outputs-commit> --solver-model qwen3-4b-instruct-2507 --destination data/outputs/processed-shared-prefix

DATA=data/outputs/processed-shared-prefix/qwen3-4b-instruct-2507
RUN=output/critic_only/de-critic
uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.cache_hidden_states --data "$DATA"
uv run --extra frozen python -m critic_only.cli.train --data "$DATA" --variant de-critic --output "$RUN"
uv run --extra frozen python -m critic_only.cli.evaluate --data "$DATA" --checkpoint "$RUN/best" --split test --destination "$RUN/test"
```

Setup also accepts `--outputs-directory` for local trajectories. Hidden-state caching resumes completed chunks. Training saves metrics and epoch checkpoints; resume with `--resume <checkpoint-directory>`. Select settings on validation, then evaluate test once. Use each command's `--help` for training and ablation options.

For this target change, keep the original outputs and processed data. Copy the existing `splits.parquet` into the new destination before setup to preserve exact assignments, then rebuild tokens and hidden states there: the critic now sees a different history. Setup refuses to overwrite an existing model directory. Rebuild into another destination when adding seeds; do not resume an old training checkpoint against changed labels.
