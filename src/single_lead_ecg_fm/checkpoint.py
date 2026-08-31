"""Optional helpers for obtaining and loading the upstream ECG-FM checkpoint."""

from __future__ import annotations

from os import PathLike
from pathlib import Path


DEFAULT_REPO_ID = "wanglab/ecg-fm"
DEFAULT_FILENAME = "mimic_iv_ecg_physionet_pretrained.pt"


def download_pretrained_checkpoint(
    cache_dir: str | PathLike[str] | None = None,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    revision: str = "main",
) -> Path:
    """Download the official checkpoint through Hugging Face Hub.

    The dependency is optional so importing this package never initiates a
    download or requires network-related packages.
    """

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Checkpoint download requires huggingface-hub. "
            "Install this package with the 'download' extra."
        ) from exc

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    return Path(path)


def load_ecg_fm_checkpoint(checkpoint_path: str | PathLike[str]):
    """Load an ECG-FM model with the upstream fairseq-signals loader."""

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"ECG-FM checkpoint not found: {path}")

    try:
        from fairseq_signals.models import build_model_from_checkpoint
    except ImportError as exc:
        raise ImportError(
            "Loading ECG-FM requires the official fairseq-signals package. "
            "See the installation instructions in README.md."
        ) from exc

    return build_model_from_checkpoint(str(path))
