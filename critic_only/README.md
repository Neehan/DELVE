# Critic-only experiments

Train critics on saved DELVE continuations while keeping the solver fixed. Compare `ordinary` (ordinary labels), `oracle-supervised` (direct prediction plus oracle supervision), and `de` (a soft discovery–execution prior plus a learned correction). All use the same BF16 backbone, all-linear LoRA, and fully trained prediction heads. This is a critic-only experiment, not RL.

Install the locked training extra in the project environment, separately from the vLLM serving environment:

```bash
uv sync --locked --extra frozen
```

Run the complete CPU test suite with `uv run --extra frozen pytest discovery_execution/tests critic_only/tests` and type-check with `uv run --extra frozen pyright`. The default pytest selection remains the collection harness so its environment does not require training dependencies.

Process once from a pinned HF outputs revision, using the corresponding local problem datasets:

```bash
uv run --extra frozen python -m critic_only.cli.setup --outputs-repo notadib/DELVE-Outputs --revision <outputs-commit> --solver-model qwen3-4b-instruct-2507 --backbone Qwen/Qwen3-4B-Instruct-2507 --backbone-revision <model-commit>
```

The command saves `prepared/` Parquet and a `tokenized/` Arrow cache under `data/critic_only/` (`--destination` changes this root). Training reads the cache directly. Alternatively, process with `--outputs-directory output --revision <collection-name>`. Preparation records input checksums and refuses incomplete audits, changed source prefixes, mismatched sampling settings, or reference statements/sketches that differ from the saved inputs. It reads source conversations to recover the phase at the fork. It never regenerates trajectories or modifies collection files.

Prepared Parquet has one row per exact prefix: problem/source identity, messages, phase, remaining blocks, selected solution index, matching reference solution/sketch, and each branch's seed and raw correctness vector. The pair follows the saved oracle selection, or source-seed round robin when oracle branches are absent. Tokenized Arrow caches contain token IDs and success/trial counts for each remaining horizon. No terminal answer, sketch, branch continuation, or branch outcome enters the default critic input. Tokens are cached, not hidden states; LoRA changes representations during training. Overlong inputs fail instead of being truncated.

By default, 2026 and later are test-only; older problems use a seeded 80/20 train/validation split. Every seed and prefix of a problem stays together. `setup --oracle-supervision-fraction 0.25` retains oracle supervision for approximately a deterministic quarter of training problems; missing oracle labels are masked. `--branch-limit 2` uses at most two training continuations per prefix and condition; validation and test retain all available branches. These selections use separate seeded hash streams from the train/validation split. `--reference-mode solution` creates a separate privileged-reference experiment for every variant. It supplies a reference at both training and evaluation; it is not reference dropout or a claim of reference-free transfer. Neither references nor test labels should guide tuning.

Train the three variants into separate directories on two GPUs:

```bash
uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.train --variant ordinary --output output/critic_only/ordinary
uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.train --variant oracle-supervised --output output/critic_only/oracle-supervised
uv run --extra frozen torchrun --standalone --nproc_per_node=2 -m critic_only.cli.train --variant de --output output/critic_only/de
```

Defaults are rank 32, one prefix per GPU, gradient accumulation 8, SDPA attention, and backbone gradient checkpointing. `--attention flash_attention_2` requires a compatible FlashAttention installation. Trainer/Accelerate manage DDP, mixed precision, optimizer state, and resume; custom FSDP/DeepSpeed paths are not implemented. Resume with the same arguments plus `--resume output/critic_only/de/checkpoint-<step>`. Existing run directories are never silently reused. The longest-prefix GPU memory and throughput still need measurement on the target Slurm allocation.

Slurm logs use plain text, without progress bars. Training prints step, epoch, loss, learning rate, and gradient norm at the first step and every 10 optimizer steps (`--logging-steps`). `trainer_state.json` saves metric history at every log and in each epoch checkpoint, including validation terminal NLL/Brier and final runtime/throughput. `training.json` records run/data settings; checkpoints also contain resolved `training_args.json`. Compare variants using held-out NLL/Brier, not their differently weighted total training losses.

```bash
uv run --extra frozen python -m critic_only.cli.evaluate --checkpoint output/critic_only/de/best --split test --destination output/critic_only/de/test
```

The checkpoint is selected using validation terminal NLL. Evaluation writes probabilities/counts to Parquet and binary-outcome NLL/Brier scores, including per-problem and remaining-horizon results. The primary report averages terminal scores within each problem, then across problems. Bootstrap or compare these problem-level results, not individual correlated branches. Labels are final-box correctness at each checkpoint, not ever-correct success.

For the DE variant, `d` estimates an already-acquired component, `alpha` is a geometric acquisition parameter, and `epsilon[h]` predicts oracle-conditioned correctness after `h` additional blocks. The prior is `u[h] = d * epsilon[h] + (1-d) * sum(alpha * (1-alpha)^(t-1) * epsilon[h-t+1], t=1..h)`. A separate head adds `delta[h]` to `logit(u[h])`, with a squared correction penalty. Neither epsilon nor the final curve is forced monotonic. These are predictive latent quantities, not identified strategy-acquisition labels; oracle execution is an imperfect proxy. This initial model does not implement a beta-binomial prior or short-branch evidence as input.

Training uses binomial NLL, normalized per prefix and condition, with oracle loss only where labels exist. Multiple checkpoint labels from one branch are correlated auxiliary targets, not additional independent rollouts. Tune oracle/correction weights on validation and compare `de` against `oracle-supervised` to separate structural benefits from simply receiving more supervision. Offline gains do not establish online RL gains.

To share prepared data, upload `data/critic_only/prepared/` to a separate HF dataset repository with `hf upload <repo> data/critic_only/prepared . --repo-type dataset`. Upload the tokenizer-specific cache separately if desired. No CLI uploads automatically.
