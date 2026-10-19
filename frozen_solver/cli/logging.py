"""Keep dataset, download, and model-loading progress bars out of Slurm logs."""

from datasets.utils import disable_progress_bars as disable_dataset_progress
from huggingface_hub.utils.tqdm import disable_progress_bars as disable_hub_progress
from transformers.utils.logging import disable_progress_bar as disable_model_progress


def disable_progress_bars() -> None:
    """Disable library progress bars; Trainer emits periodic text metrics separately."""
    disable_dataset_progress()
    disable_hub_progress()
    disable_model_progress()
