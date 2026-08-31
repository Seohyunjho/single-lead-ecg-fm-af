"""ECG-FM single-lead adaptation and classification model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .checkpoint import load_ecg_fm_checkpoint


def _first_convolution(backbone: nn.Module) -> tuple[nn.Sequential, nn.Conv1d]:
    try:
        block = backbone.feature_extractor.conv_layers[0]
        convolution = block[0]
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            "The backbone does not expose ECG-FM's first convolution at "
            "feature_extractor.conv_layers[0][0]."
        ) from exc

    if not isinstance(block, nn.Sequential) or not isinstance(convolution, nn.Conv1d):
        raise TypeError("ECG-FM's first feature-extractor block must start with Conv1d.")
    return block, convolution


def adapt_first_convolution_to_single_lead(
    backbone: nn.Module,
    *,
    source_lead_index: int = 1,
) -> nn.Module:
    """Adapt ECG-FM in place from multi-lead to single-lead input.

    The new convolution retains the chosen input channel's pretrained weights.
    Lead indices are zero-based, so index 1 corresponds to Lead II in the
    standard 12-lead ordering used by ECG-FM.
    """

    block, old_conv = _first_convolution(backbone)
    if old_conv.in_channels == 1:
        if source_lead_index != 0:
            raise ValueError("An already single-lead convolution only supports index 0.")
        return backbone
    if source_lead_index < 0 or source_lead_index >= old_conv.in_channels:
        raise ValueError(
            f"source_lead_index must be between 0 and {old_conv.in_channels - 1}."
        )
    if old_conv.groups != 1:
        raise ValueError("Grouped input convolutions are not supported for lead selection.")

    new_conv = nn.Conv1d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=1,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight[:, source_lead_index : source_lead_index + 1])
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    block[0] = new_conv
    return backbone


def _infer_embedding_dim(backbone: nn.Module) -> int:
    candidates: tuple[Any, ...] = (
        getattr(getattr(backbone, "cfg", None), "encoder_embed_dim", None),
        getattr(backbone, "encoder_embed_dim", None),
        getattr(getattr(backbone, "encoder", None), "embedding_dim", None),
    )
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError(
        "Could not infer the encoder embedding dimension; pass embedding_dim explicitly."
    )


class ECGFMClassifier(nn.Module):
    """Single-lead ECG-FM encoder with temporal pooling and a linear head."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        embedding_dim: int | None = None,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.backbone = backbone
        self.embedding_dim = embedding_dim or _infer_embedding_dim(backbone)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    @staticmethod
    def _unpack_backbone_output(output: Any) -> tuple[Tensor, Tensor | None]:
        if isinstance(output, Tensor):
            return output, None
        if isinstance(output, dict) and isinstance(output.get("x"), Tensor):
            padding_mask = output.get("padding_mask")
            if padding_mask is not None and not isinstance(padding_mask, Tensor):
                raise TypeError("padding_mask returned by the backbone must be a Tensor.")
            return output["x"], padding_mask
        raise TypeError("The ECG-FM backbone must return a Tensor or a dictionary containing 'x'.")

    @staticmethod
    def _temporal_mean(features: Tensor, padding_mask: Tensor | None) -> Tensor:
        if features.ndim != 3:
            raise ValueError("Backbone features must have shape [batch, time, embedding].")
        if padding_mask is None:
            return features.mean(dim=1)
        if padding_mask.shape != features.shape[:2]:
            raise ValueError("padding_mask must have shape [batch, time].")

        valid = (~padding_mask.bool()).unsqueeze(-1)
        counts = valid.sum(dim=1).clamp_min(1)
        return (features * valid).sum(dim=1) / counts

    def forward(
        self,
        source: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if source.ndim != 3 or source.shape[1] != 1:
            raise ValueError("source must have shape [batch, 1, time].")

        extract_features = getattr(self.backbone, "extract_features", None)
        if not callable(extract_features):
            raise TypeError("The ECG-FM backbone must provide extract_features().")
        output = extract_features(source, padding_mask=padding_mask, mask=False)
        features, output_padding_mask = self._unpack_backbone_output(output)
        pooled = self._temporal_mean(features, output_padding_mask)
        if pooled.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, got {pooled.shape[-1]}."
            )
        return self.classifier(self.dropout(pooled))


def build_single_lead_classifier(
    checkpoint_path: str,
    *,
    source_lead_index: int = 1,
    embedding_dim: int | None = None,
    num_classes: int = 2,
    dropout: float = 0.3,
) -> ECGFMClassifier:
    """Load ECG-FM, adapt its first convolution, and attach a classifier."""

    backbone = load_ecg_fm_checkpoint(checkpoint_path)
    adapt_first_convolution_to_single_lead(
        backbone,
        source_lead_index=source_lead_index,
    )
    return ECGFMClassifier(
        backbone,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        dropout=dropout,
    )
