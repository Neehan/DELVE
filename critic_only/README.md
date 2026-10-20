# Critic-only experiments

Train critics on saved DELVE continuation branches with a fixed solver. Compare `ordinary` (ordinary outcomes), `oracle-supervised` (ordinary and oracle outcomes), and `de` (the same supervision with a discovery–execution formula). All use BF16, all-linear LoRA, and trainable prediction heads.

Install separately from the vLLM serving environment:

```bash
uv sync --locked --extra frozen
```

Process once, using matching local problem datasets and pinned outputs/model revisions:

```bash
uv run --extra frozen python -m critic_only.cli.setup --outputs-repo notadib/DELVE-Outputs --revision <outputs-commit> --solver-model qwen3-4b-instruct-2507 --backbone Qwen/Qwen3-4B-Instruct-2507 --backbone-revision <model-commit>
```

This saves Parquet and tokenized Arrow under `data/critic_only/`. Each row contains an actual solver history and future branch success/trial counts. Oracle rows include the exact saved sketch-injection message; neither history includes generated continuations. Labels grade the final answer at that checkpoint, not whether an earlier answer was correct. References match the collection's solution selection. All variants average losses with and without a reference when available.

Problems from 2026 onward are test-only; older problems split 80/20 into train/validation with seed 1234. All prefixes and both history types of a problem stay together. Ordinary trains only on ordinary histories; the other variants also train on oracle histories. Use `--oracle-supervision-fraction` and `--branch-limit` during setup to limit training supervision.

Train each variant on two GPUs:

```bash
for variant in ordinary oracle-supervised de; do
  uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.train --variant "$variant" --output "output/critic_only/$variant"
done
```

Defaults: LoRA rank 32, one prefix per GPU, accumulation 8, and gradient checkpointing. References add a second input view; long-prefix GPU memory still needs measurement. Adjust `--batch-size` and `--accumulation-steps`; resume with `--resume <checkpoint-directory>`. Training saves settings, metric history, and epoch checkpoints, with plain-text progress logs.

Evaluate the best checkpoint:

```bash
uv run --extra frozen python -m critic_only.cli.evaluate --checkpoint output/critic_only/de/best --split test --destination output/critic_only/de/test
```

All variants are selected and evaluated on ordinary histories. Evaluation saves NLL/Brier metrics and prediction Parquet separately with and without references. Checkpoint selection uses validation terminal NLL without references, averaged per problem. Compare `de` against `oracle-supervised` to isolate the structural prior's benefit. These experiments evaluate critics; they do not train the solver.
