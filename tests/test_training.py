from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from single_lead_ecg_fm import (
    ECGFMClassifier,
    FineTuningConfig,
    balanced_class_weights,
    fit_two_stage,
    freeze_backbone,
    unfreeze_backbone,
)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(encoder_embed_dim=3)
        self.projection = nn.Conv1d(1, 3, kernel_size=1)

    def extract_features(self, source, padding_mask=None, mask=False):
        assert mask is False
        return {"x": self.projection(source).transpose(1, 2)}


def test_balanced_class_weights() -> None:
    weights = balanced_class_weights([3, 1])
    torch.testing.assert_close(weights, torch.tensor([2 / 3, 2.0]))


def test_freeze_and_unfreeze_backbone() -> None:
    model = ECGFMClassifier(TinyBackbone(), dropout=0.0)
    freeze_backbone(model)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())

    unfreeze_backbone(model)
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_two_stage_training_returns_loss_only() -> None:
    torch.manual_seed(0)
    model = ECGFMClassifier(TinyBackbone(), dropout=0.0)
    loader = DataLoader(
        TensorDataset(torch.randn(6, 1, 8), torch.tensor([0, 1, 0, 1, 0, 1])),
        batch_size=3,
        shuffle=False,
    )
    config = FineTuningConfig(head_epochs=1, full_epochs=1)

    history = fit_two_stage(model, loader, [3, 3], config=config)

    assert [entry["stage"] for entry in history] == ["head", "full"]
    assert all(set(entry) == {"stage", "epoch", "loss"} for entry in history)
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
