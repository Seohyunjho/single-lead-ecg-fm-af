"""Dataset-agnostic two-stage fine-tuning for the single-lead classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from .model import ECGFMClassifier


Batch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class FineTuningConfig:
    """Optimization settings for head-only and full-model training."""

    head_epochs: int = 5
    full_epochs: int = 10
    head_learning_rate: float = 1e-3
    full_learning_rate: float = 1e-5
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if self.head_epochs < 0 or self.full_epochs < 0:
            raise ValueError("Epoch counts cannot be negative.")
        if self.head_epochs + self.full_epochs == 0:
            raise ValueError("At least one training epoch is required.")
        if self.head_learning_rate <= 0 or self.full_learning_rate <= 0:
            raise ValueError("Learning rates must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")


def freeze_backbone(model: ECGFMClassifier) -> None:
    """Freeze pretrained encoder parameters for head-only training."""

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False


def unfreeze_backbone(model: ECGFMClassifier) -> None:
    """Unfreeze pretrained encoder parameters for end-to-end fine-tuning."""

    for parameter in model.backbone.parameters():
        parameter.requires_grad = True


def balanced_class_weights(
    class_counts: Sequence[int],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return inverse-frequency weights N / (C * n_c)."""

    if len(class_counts) < 2:
        raise ValueError("class_counts must contain at least two classes.")
    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=device)
    if not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Every class count must be a positive finite value.")
    return counts.sum() / (counts.numel() * counts)


def train_epoch(
    model: ECGFMClassifier,
    train_loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: torch.device | str,
) -> float:
    """Run one training epoch and return mean loss per sample."""

    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("train_loader produced no samples.")
    return total_loss / total_samples


def _fit_stage(
    *,
    model: ECGFMClassifier,
    train_loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    stage: str,
    device: torch.device | str,
) -> list[dict[str, int | float | str]]:
    history: list[dict[str, int | float | str]] = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
        )
        history.append({"stage": stage, "epoch": epoch, "loss": loss})
    return history


def fit_two_stage(
    model: ECGFMClassifier,
    train_loader: Iterable[Batch],
    class_counts: Sequence[int],
    *,
    config: FineTuningConfig | None = None,
    device: torch.device | str = "cpu",
) -> list[dict[str, int | float | str]]:
    """Train the classifier head, then fine-tune the full model.

    Only optimization loss is returned. Evaluation and performance reporting
    intentionally remain outside this public, dataset-agnostic implementation.
    """

    config = config or FineTuningConfig()
    model.to(device)
    weights = balanced_class_weights(class_counts, device=device)
    if weights.numel() != model.classifier.out_features:
        raise ValueError("class_counts length must match the classifier output size.")
    criterion = nn.CrossEntropyLoss(weight=weights)
    history: list[dict[str, int | float | str]] = []

    if config.head_epochs:
        freeze_backbone(model)
        head_optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=config.head_learning_rate,
            weight_decay=config.weight_decay,
        )
        history.extend(
            _fit_stage(
                model=model,
                train_loader=train_loader,
                optimizer=head_optimizer,
                criterion=criterion,
                epochs=config.head_epochs,
                stage="head",
                device=device,
            )
        )

    if config.full_epochs:
        unfreeze_backbone(model)
        full_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.full_learning_rate,
            weight_decay=config.weight_decay,
        )
        history.extend(
            _fit_stage(
                model=model,
                train_loader=train_loader,
                optimizer=full_optimizer,
                criterion=criterion,
                epochs=config.full_epochs,
                stage="full",
                device=device,
            )
        )

    return history
