#!/usr/bin/env bash
# Upload saved trajectories and audits using the repository's locked environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec uv run --locked python -m discovery_execution.scripts.upload_outputs
