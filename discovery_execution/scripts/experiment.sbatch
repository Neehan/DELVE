#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=logs/slurm/discovery_execution_%j.log
#SBATCH --error=logs/slurm/discovery_execution_%j.log
#SBATCH --job-name=discovery_execution

# Submit from the repository root after setup.sh and mkdir -p logs/slurm.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
module load miniforge

# Use the installed module-Python vLLM, not the hollow conda environment.
export CUDA_HOME=/orcd/software/core/001/pkg/cuda/13.0.1
export PATH="/home/notadib/.local/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/orcd/software/core/001/pkg/miniforge/25.11.0-0/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME=/orcd/pool/008/notadib/hf_cache
export HF_HUB_OFFLINE=1
export VLLM_API_KEY=glm-local-key

PYTHON=.venv/bin/python
READY_TIMEOUT_SECONDS=900
POLL_SECONDS=2
SERVER_PID=
WORKER_PID=

for argument in "$@"; do
    if [[ "$argument" == --help || "$argument" == -h ]]; then
        exec "$PYTHON" -m discovery_execution.cli.run --help
    fi
done

cleanup() {
    # Terminate only this job's children; Slurm also reaps the allocation's processes.
    for pid in "$WORKER_PID" "$SERVER_PID"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Reuse CLI validation and choose a free local port for jobs sharing a node.
SELECTION=$("$PYTHON" - "$@" <<'PY'
import socket
import sys

from discovery_execution.cli.run import parse_arguments
from discovery_execution.harness.constants import ORACLE_EXECUTION_EXPERIMENT
from discovery_execution.vllm.constants import HOST

config = parse_arguments(sys.argv[1:])
audit_arguments = ["--competition", config.competition, "--year", str(config.year), "--model", config.model,
                   "--experiment", config.experiment, "--compute-multiplier-k", str(config.compute_multiplier_k)]
if config.experiment == ORACLE_EXECUTION_EXPERIMENT:
    audit_arguments.extend(["--inject-sketch-at", str(config.inject_sketch_at)])
print(config.model)
print(*audit_arguments, "--seeds", *config.seeds)
print(config.batch_size)
with socket.socket() as listener:
    listener.bind((HOST, 0))
    print(listener.getsockname()[1])
PY
)
{
    read -r MODEL
    read -r -a AUDIT_ARGUMENTS
    read -r BATCH_SIZE
    read -r PORT
} <<< "$SELECTION"
export VLLM_BASE_URL="http://127.0.0.1:$PORT/v1"
mkdir -p logs/vllm
SERVER_LOG="logs/vllm/${MODEL}_${SLURM_JOB_ID}.log"
"$PYTHON" -m discovery_execution.vllm.serve --model "$MODEL" --port "$PORT" \
    --max-num-seqs "$BATCH_SIZE" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

DEADLINE=$((SECONDS + READY_TIMEOUT_SECONDS))
while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM exited during startup; see $SERVER_LOG" >&2
        exit 1
    fi
    if curl --fail --silent --max-time "$POLL_SECONDS" \
        -H "Authorization: Bearer $VLLM_API_KEY" "$VLLM_BASE_URL/models" >/dev/null; then
        break
    fi
    if (( SECONDS >= DEADLINE )); then
        echo "vLLM readiness timed out; see $SERVER_LOG" >&2
        exit 1
    fi
    sleep "$POLL_SECONDS"
done

"$PYTHON" -m discovery_execution.cli.run "$@" &
WORKER_PID=$!
wait "$WORKER_PID"
WORKER_PID=
"$PYTHON" -m discovery_execution.cli.audit "${AUDIT_ARGUMENTS[@]}" &
WORKER_PID=$!
wait "$WORKER_PID"
WORKER_PID=
