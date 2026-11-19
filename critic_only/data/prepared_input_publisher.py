"""Publish a validated derived dataset, retaining raw trajectory shards and removing superseded caches."""

from pathlib import Path
from typing import Any

from discovery_execution.processing.constants import MANIFEST_FILENAME
from discovery_execution.processing.storage import file_digest, read_manifest, write_json_atomic


class PreparedInputPublisher:
    """Use publish to replace derived files after preparation succeeds in a temporary directory.

    Content-addressed filenames keep old readers valid until the new manifest is published. Only old manifest-owned derived files are removed; raw trajectories and split assignments are never replaced. Run setup while training readers are stopped.
    """

    def __init__(self, directory: Path) -> None:
        """Store directory, the existing processed model root to update."""
        self.directory = directory

    def publish(self, staging: Path) -> None:
        """Validate staged files, publish their manifest, then remove superseded derived files.

        Args:
            staging: Fully prepared token/checkpoint files with a manifest referencing the unchanged raw shards.

        Returns:
            None. A failed preparation never reaches this method or alters the current manifest.
        """
        old, new = read_manifest(self.directory), read_manifest(staging)
        previous = self._derived_files(old)
        current = self._derived_files(new)
        for name, digest in previous.items():
            if file_digest(self.directory / name) != digest:
                raise ValueError(f"existing derived file changed: {name}")
        for name, digest in current.items():
            if file_digest(staging / name) != digest:
                raise ValueError(f"staged derived file changed: {name}")
        if old["trajectories"] != new["trajectories"] or old["split_sha256"] != new["split_sha256"]:
            raise ValueError("reprocessing must preserve raw shards and problem splits")
        files = [path for path in staging.rglob("*") if path.is_file() and path.name != MANIFEST_FILENAME]
        for path in files:
            destination = self.directory / path.relative_to(staging)
            if destination.exists() and file_digest(destination) != file_digest(path):
                raise ValueError(f"refusing to overwrite a different existing file: {destination}")
        for folder in staging.rglob("*"):
            if folder.is_dir():
                (self.directory / folder.relative_to(staging)).mkdir(parents=True, exist_ok=True)
        for path in files:
            destination = self.directory / path.relative_to(staging)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                path.rename(destination)
        write_json_atomic(self.directory / MANIFEST_FILENAME, new)
        for name in previous.keys() - current.keys():
            (self.directory / name).unlink()

    def _derived_files(self, manifest: dict[str, Any]) -> dict[str, str]:
        """Return manifest-owned token/checkpoint files, rejecting paths outside the dataset root."""
        files = dict(manifest["tokenization"]["shards"]) if "tokenization" in manifest else {}
        if "analysis_checkpoints" in manifest:
            section = manifest["analysis_checkpoints"]
            files[section["path"]] = section["sha256"]
        for name in files:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".parquet":
                raise ValueError(f"invalid derived file path: {name}")
            if name in manifest["trajectories"]:
                raise ValueError(f"derived file aliases a raw trajectory shard: {name}")
        return files
