"""Single-lead adaptation and fine-tuning utilities for ECG-FM."""

from .checkpoint import download_pretrained_checkpoint, load_ecg_fm_checkpoint
from .model import (
    ECGFMClassifier,
    adapt_first_convolution_to_single_lead,
    build_single_lead_classifier,
)
from .training import (
    FineTuningConfig,
    balanced_class_weights,
    fit_two_stage,
    freeze_backbone,
    train_epoch,
    unfreeze_backbone,
)

__all__ = [
    "ECGFMClassifier",
    "FineTuningConfig",
    "adapt_first_convolution_to_single_lead",
    "balanced_class_weights",
    "build_single_lead_classifier",
    "download_pretrained_checkpoint",
    "fit_two_stage",
    "freeze_backbone",
    "load_ecg_fm_checkpoint",
    "train_epoch",
    "unfreeze_backbone",
]
