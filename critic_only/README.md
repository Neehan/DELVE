# Critic-only experiments

Train a critic to predict final-answer correctness from a partial solver trajectory. Each branch supplies its own input and binary label: for example, observe through 3× and predict correctness at 5×. Oracle histories include the injected sketch and subsequent reasoning. Inputs with and without the matching reference solution share the same label; their binary cross-entropy losses are averaged.

| Variant | Training histories | Prediction |
|---|---|---|
| `ppo-critic` | Unaided | Direct scalar |
| `ppo-oracle-critic` | Unaided + oracle | Direct scalar |
| `de-critic` | Unaided + oracle | Discovery–execution decomposition |

**Workflow:** setup → cache hidden states → train → evaluate. Setup writes reusable Parquet chunks and tokenizes both input versions. The seed-1234 problem split reserves 1,000 pre-2026 problems for training, remaining older problems for validation, and 2026 onward for test.

Frozen mode trains a two-layer, width-512 GELU MLP on cached Qwen vectors. For LoRA, skip hidden-state caching and pass `--training-mode lora` to train. The solver stays fixed in both modes.

Run from the repository root with matching problem files in `data/datasets`. Use a separate environment from vLLM; GPU encoding and LoRA require a compatible `flash-attn` installation in addition to the dependencies below.

```bash
uv sync --locked --extra frozen
uv run --extra frozen python -m critic_only.cli.setup --outputs-repo notadib/DELVE-Outputs --revision <outputs-commit> --solver-model qwen3-4b-instruct-2507

DATA=data/outputs/processed/qwen3-4b-instruct-2507
RUN=output/critic_only/de-critic
uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.cache_hidden_states --data "$DATA"
uv run --extra frozen python -m critic_only.cli.train --data "$DATA" --variant de-critic --output "$RUN"
uv run --extra frozen python -m critic_only.cli.evaluate --data "$DATA" --checkpoint "$RUN/best" --split test --destination "$RUN/test"
```

Setup also accepts `--outputs-directory` for local trajectories. Hidden-state caching resumes completed chunks. Training saves metrics and epoch checkpoints; resume with `--resume <checkpoint-directory>`. Select settings on validation, then evaluate test once. Use each command's `--help` for training and ablation options.
