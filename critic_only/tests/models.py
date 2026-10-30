"""Paths for mutable collection fixtures before shared processing."""

from pathlib import Path

from pydantic import BaseModel


class CollectionFixture(BaseModel):
    """Locate raw trajectories, reference problems, and disposable processed output."""

    outputs_directory: Path
    datasets_directory: Path
    destination: Path
    solver_model: str
    horizon: int
