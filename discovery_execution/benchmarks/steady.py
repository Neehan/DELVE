"""Steady-state decode benchmark that mimics DELVE trajectories.

B workers each own a growing token context (prefix-cache friendly, like the harness
re-sending the whole conversation). Each worker starts at a random point of a trajectory
so context lengths are ~uniform in [prompt, max_ctx] from the start, then appends
`chunk` generated tokens per request (ignore_eos) and restarts when it passes max_ctx.
Throughput is read from vLLM's generation_tokens counter over the measurement window.
"""

import argparse
import asyncio
import json
import os
import random
import time

import httpx


def counter(text: str, name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith(name + "_created"):
            total += float(line.rsplit(" ", 1)[1])
    return total


async def metrics(client: httpx.AsyncClient, root: str) -> dict:
    response = await client.get(f"{root}/metrics")
    response.raise_for_status()
    text = response.text
    return {
        "gen": counter(text, "vllm:generation_tokens_total"),
        "prompt": counter(text, "vllm:prompt_tokens_total"),
        "preempt": counter(text, "vllm:num_preemptions_total"),
        "running": counter(text, "vllm:num_requests_running"),
        "waiting": counter(text, "vllm:num_requests_waiting"),
        "kv": counter(text, "vllm:kv_cache_usage_perc"),
    }


DONE: set[int] = set()


async def worker(client, url, model, rng, args, stop, index):
    prompt_len = 600
    ctx = [rng.randrange(1000, 30000) for _ in range(rng.randrange(prompt_len, args.max_ctx))]
    while not stop.is_set():
        if len(ctx) + args.chunk > args.max_ctx:
            ctx = [rng.randrange(1000, 30000) for _ in range(prompt_len)]
        body = {"model": model, "prompt": ctx, "max_tokens": args.chunk, "min_tokens": args.chunk,
                "ignore_eos": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "seed": rng.randrange(2**31)}
        try:
            r = await client.post(f"{url}/completions", json=body)
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            if stop.is_set():
                return
            raise RuntimeError(f"benchmark worker {index} failed") from e
        DONE.add(index)
        # Real generated tokens would be prefix-cached; random stand-ins cost only a small prefill.
        ctx = ctx + [rng.randrange(1000, 30000) for _ in range(args.chunk)]


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--max-ctx", type=int, default=70000)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--warmup", type=float, default=400)
    p.add_argument("--window", type=float, default=120)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    root = args.url.removesuffix("/v1")
    key = os.environ.get("VLLM_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(timeout=3600, headers=headers,
                                 limits=httpx.Limits(max_connections=args.concurrency + 4)) as client:
        stop = asyncio.Event()
        tasks = [asyncio.create_task(worker(client, args.url, args.model, random.Random(i), args, stop, i))
                 for i in range(args.concurrency)]
        start = time.monotonic()
        # Warm up until every worker finished its first (prefill-heavy) request.
        while len(DONE) < args.concurrency and time.monotonic() - start < args.warmup:
            for task in tasks:
                if task.done():
                    task.result()
            await asyncio.sleep(2)
        if len(DONE) < args.concurrency:
            raise RuntimeError("not every worker warmed up; refuse a misleading measurement")
        warm = round(time.monotonic() - start)
        m0, t0 = await metrics(client, root), time.monotonic()
        samples = []
        while time.monotonic() - t0 < args.window:
            await asyncio.sleep(5)
            for task in tasks:
                if task.done():
                    task.result()
            samples.append(await metrics(client, root))
        m1, t1 = await metrics(client, root), time.monotonic()
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    dt = t1 - t0
    res = {
        "tag": args.tag, "warmup_s": warm, "model": args.model, "concurrency": args.concurrency, "max_ctx": args.max_ctx,
        "gen_tok_s": round((m1["gen"] - m0["gen"]) / dt, 1),
        "prompt_tok_s": round((m1["prompt"] - m0["prompt"]) / dt, 1),
        "preemptions": m1["preempt"] - m0["preempt"],
        "avg_running": round(sum(s["running"] for s in samples) / len(samples), 1),
        "avg_waiting": round(sum(s["waiting"] for s in samples) / len(samples), 1),
        "avg_kv": round(sum(s["kv"] for s in samples) / len(samples), 3),
    }
    print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
