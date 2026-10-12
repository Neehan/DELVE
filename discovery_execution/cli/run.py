"""CLI entrypoint for frozen-policy vLLM trajectory collection."""

import asyncio
import logging
import os
import sys
from argparse import ArgumentParser
from collections.abc import Mapping
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

from discovery_execution.cli.arguments import add_selection_arguments
from discovery_execution.harness.runner import Runner
from discovery_execution.harness.constants import (
    DATASETS_DIRECTORY, DEFAULT_BATCH_SIZES, DEFAULT_SOLUTION_INDEX,
    ENV_FILE, FAILURE_EXIT_CODE, LOG_FORMAT, LOGS_DIRECTORY, OUTPUT_DIRECTORY, SUCCESS_EXIT_CODE,
    TEMPERATURE, TOP_K, TOP_P, DEFAULT_BLOCKS, VLLM_TIMEOUT_SECONDS,
)
from discovery_execution.harness.models import BackendConfig, RunConfig
from discovery_execution.harness.openai_client import OpenAIClient


def parse_arguments(arguments: list[str]) -> RunConfig:
    """Parse and validate the generation selection."""
    parser = ArgumentParser(allow_abbrev=False)
    add_selection_arguments(parser, collecting=True)
    parser.add_argument("--solution-index", type=int, default=DEFAULT_SOLUTION_INDEX,
                        help="Oracle solution index; defaults to seed modulo the problem's solution count")
    parser.add_argument("--batch-size", type=int,
                        help=f"Maximum concurrent trajectories (model defaults: {DEFAULT_BATCH_SIZES})")
    args = vars(parser.parse_args(arguments))
    if args["compute_multiplier_k"] < 1:
        parser.error("compute-multiplier-k must be positive")
    if args["batch_size"] is None:
        long_batch, short_batch = DEFAULT_BATCH_SIZES[args["model"]]
        args["batch_size"] = max(1, min(short_batch, long_batch * DEFAULT_BLOCKS // args["compute_multiplier_k"]))
    try:
        return RunConfig(
            **args, datasets_directory=DATASETS_DIRECTORY, output_directory=OUTPUT_DIRECTORY,
            hf_repo=os.environ.get("HF_DATASETS_REPO"),
        )
    except ValueError as error:
        parser.error(str(error))


def backend_configuration(environment: Mapping[str, str]) -> BackendConfig:
    """Read the required endpoint and use fixed sampling and timeout constants."""
    return BackendConfig(
        base_url=environment["VLLM_BASE_URL"].rstrip("/"),
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        timeout=VLLM_TIMEOUT_SECONDS,
    )


async def main() -> None:
    """Load environment and report a nonzero exit code if any requested seed failed."""
    load_dotenv(ENV_FILE, override=False)
    config = parse_arguments(sys.argv[1:])
    LOGS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[
        logging.StreamHandler(), logging.FileHandler(LOGS_DIRECTORY / f"{config.model}-{timestamp}.log"),
    ])
    try:
        backend_config = backend_configuration(os.environ)
        api_key = os.environ.get("VLLM_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        limits = httpx.Limits(max_connections=config.batch_size)
        async with httpx.AsyncClient(headers=headers, timeout=backend_config.timeout, limits=limits) as transport:
            client = OpenAIClient(transport, backend_config, asyncio.sleep)
            runner = Runner(config, client)
            succeeded = await runner.run()
    except Exception:
        logging.exception("Run setup failed")
        raise SystemExit(FAILURE_EXIT_CODE)
    raise SystemExit(SUCCESS_EXIT_CODE if succeeded else FAILURE_EXIT_CODE)


if __name__ == "__main__":
    asyncio.run(main())
