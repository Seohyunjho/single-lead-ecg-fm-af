from types import SimpleNamespace

import pytest
import torch
from torch import nn

from single_lead_ecg_fm import (
    ECGFMClassifier,
    adapt_first_convolution_to_single_lead,
)


class MockBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(encoder_embed_dim=4)
        self.feature_extractor = nn.Module()
        self.feature_extractor.conv_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(12, 4, kernel_size=2, stride=2, bias=False))]
        )

    def extract_features(self, source, padding_mask=None, mask=False):
        assert mask is False
        features = self.feature_extractor.conv_layers[0](source).transpose(1, 2)
        return {"x": features, "padding_mask": None}


def test_single_lead_adaptation_copies_selected_channel() -> None:
    backbone = MockBackbone()
    old_weight = backbone.feature_extractor.conv_layers[0][0].weight.detach().clone()

    returned = adapt_first_convolution_to_single_lead(backbone, source_lead_index=1)

    new_conv = backbone.feature_extractor.conv_layers[0][0]
    assert returned is backbone
    assert new_conv.in_channels == 1
    torch.testing.assert_close(new_conv.weight, old_weight[:, 1:2])


def test_classifier_accepts_single_lead_batch() -> None:
    backbone = adapt_first_convolution_to_single_lead(
        MockBackbone(), source_lead_index=1
    )
    model = ECGFMClassifier(backbone, num_classes=2, dropout=0.0)

    logits = model(torch.randn(3, 1, 20))

    assert logits.shape == (3, 2)


def test_classifier_rejects_multilead_input() -> None:
    model = ECGFMClassifier(MockBackbone(), embedding_dim=4)
    with pytest.raises(ValueError, match="batch, 1, time"):
        model(torch.randn(2, 12, 20))


def test_masked_temporal_mean_ignores_padding() -> None:
    features = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    padding_mask = torch.tensor([[False, False, True]])

    pooled = ECGFMClassifier._temporal_mean(features, padding_mask)

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0]]))
