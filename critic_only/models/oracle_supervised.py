"""Direct value prediction with auxiliary supervision from oracle continuations."""

from critic_only.models.ordinary import Ordinary


class OracleSupervised(Ordinary):
    """Train the same direct head on ordinary histories and histories containing the oracle sketch."""
