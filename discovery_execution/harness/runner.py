"""Load a problem partition and schedule independent self-refinement trajectories."""

import asyncio
import logging
from collections.abc import Iterator

from discovery_execution.dataset import Dataset
from discovery_execution.harness.constants import ORACLE_EXPERIMENTS
from discovery_execution.harness.models import Problem, RunConfig
from discovery_execution.harness.openai_client import OpenAIClient
from discovery_execution.harness.self_refine import SelfRefine


class Runner:
    """Schedule one SelfRefine instance per problem/seed, up to the configured batch limit."""

    def __init__(self, config: RunConfig, client: OpenAIClient) -> None:
        """Keep shared configuration, client, and the iterable dataset."""
        self.config = config
        self.client = client
        self.dataset = Dataset(config)

    async def run(self) -> bool:
        """Validate sketch availability and the model, then process every problem/seed."""
        if self.config.experiment in ORACLE_EXPERIMENTS and self.config.solution_index is not None:
            for problem in self.dataset:
                if self.config.solution_index >= len(problem.solutions):
                    raise ValueError(f"missing solution index for {problem.problem_id}")
        await self.client.inspect(self.config.model)
        jobs = ((problem, seed) for problem in self.dataset for seed in self.config.seeds)
        results = await asyncio.gather(*[self._worker(jobs) for _ in range(self.config.batch_size)])
        return all(results)

    async def _worker(self, jobs: Iterator[tuple[Problem, int]]) -> bool:
        """Keep one batch slot busy and continue processing after individual trajectory failures."""
        succeeded = True
        for problem, seed in jobs:
            try:
                refinement = SelfRefine(self.config, problem, seed, self.client)
                await refinement.run()
            except Exception:
                logging.exception("Failed %s seed=%s", problem.problem_id, seed)
                succeeded = False
        return succeeded
