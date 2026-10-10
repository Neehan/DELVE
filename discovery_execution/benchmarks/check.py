"""Start production serving commands and check quality, continuation, and throughput.

Run with the environment exports from experiment.sbatch on an allocated GPU node:
  .venv/bin/python -m discovery_execution.benchmarks.check
Optional positional model names restrict the run. Results and server logs go in logs/.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from discovery_execution.cli.run import backend_configuration
from discovery_execution.harness.constants import DEFAULT_BATCH_SIZES, SUPPORTED_MODELS
from discovery_execution.harness.models import GenerationRequest
from discovery_execution.harness.openai_client import OpenAIClient
from discovery_execution.vllm.serve import build_command

DIRECTORY = Path("logs/benchmarks")
PORT = 8017
URL = f"http://127.0.0.1:{PORT}/v1"


async def quality(model: str) -> dict:
    headers = {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}
    async with httpx.AsyncClient(headers=headers, timeout=1200) as transport:
        client = OpenAIClient(transport, backend_configuration({"VLLM_BASE_URL": URL}), asyncio.sleep)
        server = await client.inspect(model)
        prompts = ["Compute 6 * 7. Explain briefly and put the answer in \\boxed{}.",
                   "Solve 3x + 7 = 22. Explain briefly and put x in \\boxed{}."]
        results = await asyncio.gather(*[
            client.generate(GenerationRequest(model, [{"role": "user", "content": prompt}], 512, seed, False))
            for prompt in prompts for seed in range(2)
        ])
        samples = [r.text for r in results]
        repeats = []
        for sample in samples:
            words = sample.split()
            repeats.append(sum(a == b for a, b in zip(words, words[1:])) / max(1, len(words)))
        # Functional smoke check; long-context throughput is measured separately.
        from discovery_execution.harness.models import RunConfig, Problem
        from discovery_execution.harness.self_refine import SelfRefine
        problem = Problem.model_validate(json.loads(Path("data/datasets/aime/2024.jsonl").read_text().splitlines()[0]))
        config = RunConfig(competition="aime", year=2024, model=model, experiment="unaided",
                           compute_multiplier_k=1, seeds=[987654], batch_size=1,
                           inject_sketch_at=0, solution_index=0, hf_repo=None,
                           datasets_directory=Path("data/datasets"), output_directory=DIRECTORY / "trajectories")
        trajectory = SelfRefine(config, problem, 987654, client)
        start = time.monotonic()
        await trajectory.run()
        # Explicitly exercise the unfinished-assistant path regardless of natural EOS.
        messages = [{"role": "user", "content": "Explain how to solve x squared equals 4."}]
        first = await client.generate(GenerationRequest(model, messages, 16, 3, False))
        messages.append({"role": "assistant", "content": first.text})
        continued = await client.generate(GenerationRequest(model, messages, 32, 4, True))
        return {"model": model, "server": server, "samples": samples,
                "adjacent_repeat_rate": sum(repeats) / len(repeats),
                "trajectory_tokens": trajectory.state.output_tokens,
                "trajectory_seconds": round(time.monotonic() - start, 1),
                "continuation": continued.text}


def main() -> None:
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for model in sys.argv[1:] or SUPPORTED_MODELS:
        command = build_command(model, PORT, None)
        log_path = DIRECTORY / f"{model}-server.log"
        with log_path.open("w") as log:
            print(json.dumps({"starting": model, "command": command}), flush=True)
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 900
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"server failed; see {log_path}")
                    try:
                        response = httpx.get(f"{URL}/models", headers={"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}, timeout=2)
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"server startup timed out: {log_path}")
                    time.sleep(2)
                result = asyncio.run(quality(model))
                print(json.dumps(result), flush=True)
                with (DIRECTORY / "quality.jsonl").open("a") as output:
                    output.write(json.dumps(result) + "\n")
                workloads = [(70000, DEFAULT_BATCH_SIZES[model][0])] if model == "deepseek-r1-distill-qwen-1.5b" else []
                for context, batch in workloads:
                    subprocess.run([sys.executable, "-m", "discovery_execution.benchmarks.steady",
                                    "--url", URL, "--model", model, "--concurrency", str(batch),
                                    "--max-ctx", str(context), "--chunk", "2048", "--window", "60",
                                    "--out", str(DIRECTORY / "throughput.jsonl"), "--tag", "production"], check=True)
            finally:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
                time.sleep(5)


if __name__ == "__main__":
    main()
