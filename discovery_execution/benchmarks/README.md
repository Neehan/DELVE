# L40S serving checks

The production target is an 8-block trajectory: 65,536 generated tokens, plus the
problem and refinement instructions. Settings retain a 131,072-token server
context, prefix caching, chunked prefill, asynchronous scheduling, and CUDA graphs.

`claude-throughput.jsonl` preserves the previous agent's October 10, 2026 sweeps
on the allocated 2× L40S node with vLLM 0.26.0. The selected long-context results
are Qwen4B TP2/batch24: 449.5 output tokens/sec, Qwen1.7B TP2/batch32: 615.1,
and Ministral TP2/batch32: 631.6. All three reported zero preemptions. Short-context
rows are historical evidence; they did not determine the long-run parallelism.

DeepSeek's FP8 rows are rejected: the previous agent's AIME quality pilot showed
severe repetition and 0/32 correct for uncalibrated FP8, versus 16/32 for BF16.
The installed vLLM's deprecated `--calculate-kv-scales` flag does not dynamically
calculate scales, so it cannot repair this setup. DeepSeek uses BF16 KV and two
independent replicas; the long batch is reduced to 32 to account for the doubled
cache footprint.

The replacement BF16 pilot measured **977.7 output tokens/sec at batch 32**,
with zero preemptions, 31.9 average running requests, and no waiting queue.
It used a 100-second warmup followed by a 60-second measurement. The server's
sequence ceiling was 256 in the pilot; production serving now defaults to the
long batch of 32 and Slurm explicitly passes the harness batch.

`steady.py` reproduces the synthetic workload: independent workers start at
uniformly distributed context lengths from 600 to 70,000 tokens, request 2,048
output tokens with EOS ignored, then grow their contexts. It measures the vLLM
output-token counter after every worker has completed a first request. Random
stand-in tokens extend the context; this is a hardware throughput check, not a
quality evaluation or a complete 65,536-token trajectory per worker. KV metrics
are summed over data-parallel engines (a sum above 1 is not necessarily full).
Long-run performance depends on the distribution of active context lengths;
reduce `--batch-size` if real jobs encounter sustained preemption.

For an existing server, reproduce the selected batch with:

```bash
.venv/bin/python -m discovery_execution.benchmarks.steady \
  --model qwen3-4b-instruct-2507 --concurrency 24 --max-ctx 70000 \
  --chunk 2048 --window 60 --out logs/benchmarks/throughput.jsonl
```

`check.py` launches the production model commands, checks simple math output,
streaming usage, and unfinished-assistant continuation, and exercises the actual
harness for an 8,192-token functional smoke trajectory. It only reruns long
throughput for DeepSeek, whose KV dtype changed. Run it inside a GPU allocation
with the module, CUDA, library path, offline Hugging Face cache, and API key
exports from `discovery_execution/scripts/experiment.sbatch`:

```bash
.venv/bin/python -m discovery_execution.benchmarks.check
```

Server logs, generated trajectories, and raw functional results stay under
`logs/benchmarks/`; selected measurements are recorded alongside this file.
The current interactive allocation has one CPU core (two SMT threads), whereas
production jobs request eight cores. Treat pilot throughput as indicative rather
than a guarantee for the production CPU allocation. `.env` is local and ignored;
its values are copied from `.env.example`.
