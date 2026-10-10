# DELVE

Discovery-Execution Long-horizon Value Estimation. The `discovery_execution/` harness collects math-solving trajectories through a vLLM OpenAI-compatible endpoint; RL training comes later.

Each problem/seed runs solve → critique → revise until its budget is exhausted, saving a checkpoint every 8,192 output tokens. Unaided runs choose k=1–8; oracle runs always use k=8 and inject a sketch at 0×–7×. Finishing an answer does not stop generation.

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

Competitions are `aime`, `hmmt`, `brumo`, and `smt`; any positive year is accepted. Requested partitions are cached under `data/datasets/` with the same layout.

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

The server must report at least 100,000 context tokens; long inputs or many refinement turns may require more.

Serving uses a 131,072-token context and 92% GPU memory utilization. For long trajectories on 2× L40S, use:

| Model | Parallelism | KV cache | Batch for 65,536 output tokens | Long-context output tokens/sec |
| --- | --- | --- | ---: | ---: |
| Qwen3 4B | Tensor, 2 GPUs | FP8 E4M3 | 24 | 450 |
| Qwen3 1.7B | Tensor, 2 GPUs | FP8 E4M3 | 32 | 615 |
| Ministral 3B | Tensor, 2 GPUs | BF16 (`auto`) | 16 | 342 |
| DeepSeek 1.5B | Data, 2 replicas | BF16 (`auto`) | 32 | 978 |

Throughput is aggregate across both GPUs in synthetic growing-context workloads up to 70,000 tokens.

Use that same name for `--model`; requests, saved settings, and output folders retain it:

```bash
uv run python -m discovery_execution.cli.run --competition aime --year 2024 \
  --model qwen3-4b-instruct-2507 --experiment unaided --compute-multiplier-k 1 \
  --seeds {0..23}

uv run python -m discovery_execution.cli.run --competition aime --year 2024 \
  --model qwen3-4b-instruct-2507 --experiment oracle-execution \
  --seeds 0 1 2 3

uv run python -m discovery_execution.cli.audit --competition aime --year 2024 --model qwen3-4b-instruct-2507
```

Unaided requires `--compute-multiplier-k 1..8`; oracle fixes k=8. Optional flags: `--batch-size` (the k=8 defaults are shown above; smaller budgets scale by 8/k, capped at 128 for Qwen/Ministral and 256 for DeepSeek), `--inject-sketch-at` (default 0), and `--solution-index` (default 0). Rerunning resumes saved responses and skips completed trajectories. Transient failures retry three times with 1/2/4-second waits; partial failed responses are discarded.

Results go in `output/`, logs in `logs/`. Each trajectory stores `meta.json` (settings), `conversation.json.zst` (resume state), one snapshot per boundary (`checkpoint_1x.json.zst`, `checkpoint_2x.json.zst`, …), and `audit.json` (grades). Auditing requires the local dataset and also writes a combined `audit.jsonl` per arm.

After generation and auditing finish, run `bash discovery_execution/scripts/upload_outputs.sh` from a network-enabled shell. It uploads these artifacts to `HF_OUTPUTS_REPO`, preserving paths below `output/` and excluding locks and temporary files. A missing repository is created private; existing visibility is unchanged. Rerunning updates matching paths without deleting remote-only files.

On MIT ORCD, the Slurm script starts vLLM, generates, audits, then stops the server. Prepare the dataset and cache weights first; the script uses the existing ORCD installation in offline mode. It requests 2× L40S for 6 hours. `--batch-size N` overrides both server and harness concurrency.

```bash
mkdir -p logs/slurm && sbatch discovery_execution/scripts/experiment.sbatch --model qwen3-1.7b --competition aime --year 2024 --experiment unaided --compute-multiplier-k 1 --seeds {0..23}
```

Grades use DE's `correctness.score` field, restricted to 0 or 1. Each checkpoint's last boxed answer is checked with Math-Verify, following [Qwen's final-box convention](https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/parser.py). Missing or unparseable answers score 0. Raw audits preserve regressions; only the framework turns scores `1, 0, 0` into cumulative success `1, 1, 1`.

**Fit the frameworks**

SG, DE, R-SG, and R-DE reuse DE-Framework's estimators. Fit from audited `unaided-1x` and `oracle-at-0x-execution` arms under the selected model:

```bash
uv run python -m discovery_execution.cli.frameworks --competition aime --year 2024 \
  --model qwen3-4b-instruct-2507
```

Use `--frameworks sg de` to select methods; all four run by default. Fitting requires at least 24 complete short seeds and 3 complete oracle seeds per problem. R-DE additionally needs both first-block successes and later/unsolved oracle outcomes. Long unaided and delayed-sketch arms are excluded from fitting.

Outputs are `frameworks/<method>/fit.json` and `predictions.jsonl` under the model directory. Multiple competitions and years select their cross product and share regularized priors. Evaluate forecasts against separately collected unaided 2×–8× arms.
