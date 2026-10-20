"""Direct value prediction with auxiliary supervision from oracle continuations."""

from critic_only.models.ordinary import Ordinary


class OracleSupervised(Ordinary):
    """Reuse Ordinary's heads; the oracle-supervised configuration enables the oracle loss."""
