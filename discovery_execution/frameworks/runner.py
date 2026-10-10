"""Fit DE's estimators from binary audits and save parameters and predictions."""

import fcntl
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from discovery_execution.frameworks.discovery_execution.discovery_execution import DiscoveryExecution
from discovery_execution.frameworks.discovery_execution.regularized_discovery_execution import RegularizedDiscoveryExecution
from discovery_execution.frameworks.geometric.regularized_simple_geometric import RegularizedSimpleGeometric
from discovery_execution.frameworks.geometric.simple_geometric import SimpleGeometric
from discovery_execution.frameworks.models import BasePrediction, InputSelection
from discovery_execution.frameworks.utils.constants import FIT_FILENAME, FRAMEWORKS_DIRECTORY, JSON_INDENT, MIN_AUDIT_SCORE, PREDICTIONS_FILENAME
from discovery_execution.frameworks.utils.dataloader import DataLoader
from discovery_execution.harness.constants import LOCK_FILENAME


class FrameworkRunner:
    """Fit selected estimators on shared observations; save one result directory per partition and estimator."""

    def __init__(self, data_loader: DataLoader, results: Path) -> None:
        """Inject the audited-data loader and output root."""
        self.data_loader = data_loader
        self.results = results

    def run(self, selection: InputSelection) -> list[Path]:
        """Fit across selected partitions, sharing regularized priors, then save by partition."""
        observations = self.data_loader.load(selection)
        if not observations:
            raise ValueError("no problems have enough complete unaided-1x and oracle-at-0x-execution audits")
        directories = []
        for framework in [SimpleGeometric(), DiscoveryExecution(), RegularizedSimpleGeometric(), RegularizedDiscoveryExecution()]:
            if framework.name not in selection.frameworks:
                continue
            framework.fit(observations)
            grouped: dict[str, list[BasePrediction]] = {}
            for prediction in framework.predict():
                grouped.setdefault(prediction.dataset, []).append(prediction)
            for dataset, predictions in grouped.items():
                directory = self.results / dataset / selection.model / FRAMEWORKS_DIRECTORY / framework.name
                fitted = {"framework": framework.name, "model": selection.model, "dataset": dataset,
                          "passing_score": MIN_AUDIT_SCORE,
                          "fit_datasets": sorted({observation.dataset for observation in observations}),
                          **framework.get_parameters(dataset)}
                self._save(directory, fitted, predictions)
                directories.append(directory)
        return directories

    def _save(self, directory: Path, fitted: dict[str, object], predictions: list[BasePrediction]) -> None:
        """Serialize both outputs before publishing under a framework-directory lock."""
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / LOCK_FILENAME).open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            with TemporaryDirectory(dir=directory) as temporary:
                staged = Path(temporary)
                (staged / FIT_FILENAME).write_text(json.dumps(fitted, indent=JSON_INDENT, allow_nan=False) + "\n", encoding="utf-8")
                content = "".join(json.dumps(asdict(prediction), allow_nan=False) + "\n" for prediction in predictions)
                (staged / PREDICTIONS_FILENAME).write_text(content, encoding="utf-8")
                for filename in (FIT_FILENAME, PREDICTIONS_FILENAME):
                    os.replace(staged / filename, directory / filename)
