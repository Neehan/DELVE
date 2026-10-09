#!/usr/bin/env bash
# Install the locked environment and validate or download selected dataset partitions.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
uv sync --locked
exec uv run --locked python -m discovery_execution.scripts.setup "$@"
