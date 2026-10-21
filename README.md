# DELVE

Discovery-Execution Long-horizon Value Estimation. The `discovery_execution/` harness collects math-solving trajectories through a vLLM OpenAI-compatible endpoint; RL training comes later.

Each problem/seed runs solve → critique → revise until its budget is exhausted, saving a checkpoint every 8,192 output tokens. Both unaided and oracle runs default to k=5, with explicit budgets up to k=8 supported. Oracle sketches can be injected at any block boundary before the chosen budget. Finishing an answer does not stop generation.

The frozen-solver critic takes the partial conversation, current phase, and remaining budget, and predicts the probability that the last boxed answer will be correct when that budget ends.

Training labels come from continuing the same self-refinement harness and grading the last boxed answer: 1 if correct, 0 otherwise; earlier correct answers do not override the final grade.

[Critic-only experiments](critic_only/README.md) prepare cached training data and compare PPOCritic, PPOOracleCritic, and DECritic LoRA models before online RL.

**Environment**

Run commands from the repository root. Copy `.env.example` to `.env` and configure:

```dotenv
VLLM_BASE_URL=http://localhost:8000/v1
HF_DATASETS_REPO=notadib/DELVE-Datasets
HF_OUTPUTS_REPO=notadib/DELVE-Outputs
HF_TOKEN=...       # Write access to outputs; read access for private datasets.
# VLLM_API_KEY=...  # Only for an authenticated model server.
```

The endpoint must end in `/v1`. `HF_DATASETS_REPO` is needed only for partitions missing locally; `HF_OUTPUTS_REPO` and `HF_TOKEN` are required for uploads. Sampling and timeout settings live in `discovery_execution/harness/constants.py`.

**Dataset**

Use one Hugging Face **dataset repository**, with JSONL files at `<competition>/<year>.jsonl`:

```text
aime/2024.jsonl
aime/2025.jsonl
hmmt/2026.jsonl
```

Competitions are `aime`, `hmmt`, `brumo`, `smt`, `bmt`, and `cmimc`; any positive year is accepted. Requested partitions are cached under `data/datasets/` with the same layout.

Each file contains one problem object per line. This toy example is expanded for readability:

```json
{
  "problem_id": "aime-2024-example",
  "competition": "aime",
  "year": 2024,
  "statement": "Compute 6 × 7.",
  "final_answer": "42",
  "solutions": [
    {
      "source": "reference",
      "solution": "Multiplying 6 by 7 gives 42.",
      "sketch": "Use multiplication."
    }
  ]
}
```

All shown fields are required; extra fields are rejected. Answers are strings, `solutions` is nonempty, and `source` is `reference` or `ai_generated`. IDs must be unique within a partition and safe folder names. Competition/year must match the file path.

**Run and audit**

Install dependencies and prepare a partition:

```bash
bash discovery_execution/scripts/setup.sh --competition aime --year 2024
```

Supported names are defined in `SUPPORTED_MODELS` in `discovery_execution/harness/constants.py`:

| Name used by the harness and vLLM | Hugging Face repository |
| --- | --- |
| `qwen3-4b-instruct-2507` | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| `qwen3-1.7b` | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) |
| `ministral-3-3b-instruct-2512` | [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) |
| `deepseek-r1-distill-qwen-1.5b` | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |

With vLLM 0.26.0 installed, start a server on two GPUs:

```bash
uv run python -m discovery_execution.vllm.serve --model qwen3-4b-instruct-2507
```

The server must report at least 131,072 context tokens, including inputs and generated output; unusually long inputs or many refinement turns may require more.

Serving uses a 131,072-token context and 92% GPU memory utilization. For long trajectories on 2× L40S, use:

| Model | Parallelism | KV cache | Default batch (5×) | Measured long-context tokens/sec |
| --- | --- | --- | ---: | ---: |
| Qwen3 4B | Tensor, 2 GPUs | FP8 E4M3 | 24 | 450 |
| Qwen3 1.7B | Tensor, 2 GPUs | FP8 E4M3 | 32 | 615 |
| Ministral 3B | Tensor, 2 GPUs | BF16 (`auto`) | 16 | 342 |
| DeepSeek 1.5B | Data, 2 replicas | BF16 (`auto`) | 32 | 978 |

Throughput measurements are from the previous synthetic growing-context benchmark up to 70,000 tokens, not a benchmark of the new five-block runs.

Qwen3 4B uses 8,192 batched prefill tokens. A follow-up screen on 2× L40S compared 2,048 versus 8,192 prefill tokens at batches 24, 32, and 40, using 8,192-, 24,576-, and 40,960-token inputs and mixed input lengths. It generated 512 tokens per request with prefix caching disabled. Larger prefill chunks gave a modest gain at long contexts; larger batches gave almost no gain near 40,960 tokens, so the five-block default remains 24. These measurements include prefill and do not estimate full trajectory completion time.

Use that same name for `--model`; requests, saved settings, and output folders retain it:

```bash
uv run python -m discovery_execution.cli.run --competition aime --year 2024 \
  --model qwen3-4b-instruct-2507 --experiment unaided --compute-multiplier-k 1 \
  --seeds {0..15}

uv run python -m discovery_execution.cli.run --competition aime --year 2024 \
  --model qwen3-4b-instruct-2507 --experiment oracle-execution \
  --seeds 0 1 2 3

uv run python -m discovery_execution.cli.audit --competition aime --year 2024 --model qwen3-4b-instruct-2507
```

Both experiments default to 5× (40,960 output tokens). Set `--compute-multiplier-k` to an integer from 1 to 8 to override the budget. Optional flags: `--batch-size` (defaults above scale by 5/k, with a minimum of 1 and caps of 128 for Qwen/Ministral or 256 for DeepSeek), `--inject-sketch-at` (default 0; must be less than k), and `--solution-index` (override the default round-robin selection, `seed % len(solutions)`). A seed selects the same sketch at every injection time; `meta.json` records the actual index, which is preserved on resume. Rerunning resumes saved responses and skips completed trajectories. Transient failures retry three times with 1/2/4-second waits; partial failed responses are discarded.

Results go in `output/`, logs in `logs/`. Each trajectory stores `meta.json` (settings), `conversation.json.zst` (resume state), one snapshot per boundary (`checkpoint_1x.json.zst`, `checkpoint_2x.json.zst`, …), and `audit.json` (grades). Auditing requires the local dataset and also writes a combined `audit.jsonl` per arm.

Branch from an existing unaided checkpoint with the same model and sampling settings:

```bash
uv run python -m discovery_execution.cli.run --competition aime --year 2024 --model qwen3-1.7b \
  --experiment continue --source-arm unaided-5x --source-seed 0 --continue-at 1 --seeds 0 1 2 3
```

Use `--experiment continue-oracle` to inject a sketch at that same prefix; its default solution is selected by the source seed, shared across branches. The 5× default budget includes the prefix. Outputs use `<arm>/<problem>/seed_<source-seed>/branch_<seed>/`, with arms `continue-at-1x` or `continue-oracle-at-1x`. Only post-fork checkpoints are saved, on the original timeline. Audit accepts the same source selectors; continuation arms are excluded from DE fitting. Reusing a branch directory with different source settings or budget fails.

After generation and auditing finish, run `bash discovery_execution/scripts/upload_outputs.sh` from a network-enabled shell. It uploads these artifacts to `HF_OUTPUTS_REPO`, preserving paths below `output/` and excluding locks and temporary files. A missing repository is created private; existing visibility is unchanged. Rerunning updates matching paths without deleting remote-only files.

On MIT ORCD, the Slurm script starts vLLM, generates, audits, then stops the server. Prepare the dataset and cache weights first; the script uses the existing ORCD installation in offline mode. It requests 2× L40S for 6 hours. `--batch-size N` overrides both server and harness concurrency.

```bash
mkdir -p logs/slurm && sbatch discovery_execution/scripts/experiment.sbatch --model qwen3-1.7b --competition aime --year 2024 --experiment unaided --compute-multiplier-k 1 --seeds {0..15}
```

Grades use DE's `correctness.score` field, restricted to 0 or 1. Each checkpoint's last boxed answer is checked with Math-Verify, following [Qwen's final-box convention](https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/parser.py). Missing or unparseable answers score 0. Raw audits preserve regressions; only the framework turns scores `1, 0, 0` into cumulative success `1, 1, 1`.

**Fit the frameworks**

SG, DE, R-SG, and R-DE reuse DE-Framework's estimators. Fit from audited `unaided-1x` and `oracle-at-0x-execution` arms under the selected model:

```bash
uv run python -m discovery_execution.cli.frameworks --competition aime --year 2024 \
  --model qwen3-4b-instruct-2507
```

Use `--frameworks sg de` to select methods; all four run by default. Fitting requires at least 16 complete short seeds and 3 complete five-block oracle seeds per problem. R-DE additionally needs both first-block successes and later/unsolved oracle outcomes. Long unaided and delayed-sketch arms are excluded from fitting.

Outputs are `frameworks/<method>/fit.json` and `predictions.jsonl` under the model directory. Multiple competitions and years select their cross product and share regularized priors. Evaluate forecasts against separately collected unaided 2×–5× arms.
