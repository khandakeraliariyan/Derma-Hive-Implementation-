"""Phase 4 MobileNetV3-Large SkinNet-HDA clean-room implementation.

This is our proposed lightweight variant. Phase 1, 2, and 3 code is unchanged.

REQUIRED BACKBONE-INTERFACE ADAPTATION — not an additional proposed module:
torchvision MobileNetV3-Large produces [B,960,7,7] at 224x224, rather than
ConvNeXt-Tiny's [B,768,7,7]. The SE channels, Transformer d_model/FFN, position
embedding, pooled visual vector, and classifier input therefore use 960.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from skinnet_hda import FocalLoss, SEChannelAttention, SpatialAttention

NUM_CLASSES = 7
FEATURE_CHANNELS = 960
FEATURE_HEIGHT = 7
FEATURE_WIDTH = 7
TOKEN_COUNT = FEATURE_HEIGHT * FEATURE_WIDTH


class MobileNetHDA(nn.Module):
    """MobileNetV3-Large -> dual attention -> Transformer -> metadata fusion."""

    def __init__(
        self, metadata_dim: int, num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if metadata_dim <= 0:
            raise ValueError("metadata_dim must be positive")
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.backbone = backbone.features
        self.channel_attention = SEChannelAttention(FEATURE_CHANNELS, reduction=16)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, TOKEN_COUNT, FEATURE_CHANNELS)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FEATURE_CHANNELS,
            nhead=8,
            dim_feedforward=FEATURE_CHANNELS * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Linear(FEATURE_CHANNELS + metadata_dim, num_classes)
        self.metadata_dim = metadata_dim
        self.num_classes = num_classes
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self, images: torch.Tensor, metadata: torch.Tensor,
        return_shapes: bool = False,
    ):
        if metadata.ndim != 2 or metadata.shape[1] != self.metadata_dim:
            raise ValueError(
                f"Expected metadata [B,{self.metadata_dim}], got {tuple(metadata.shape)}"
            )
        shapes: dict[str, tuple[int, ...]] = {"image": tuple(images.shape)}
        features = self.backbone(images)
        shapes["mobilenet"] = tuple(features.shape)
        expected = (FEATURE_CHANNELS, FEATURE_HEIGHT, FEATURE_WIDTH)
        if features.shape[1:] != expected:
            raise RuntimeError(
                f"Expected MobileNet features [B,{expected}], got {tuple(features.shape)}"
            )
        features = self.channel_attention(features)
        shapes["se"] = tuple(features.shape)
        features = self.spatial_attention(features)
        shapes["spatial_attention"] = tuple(features.shape)
        tokens = features.flatten(2).transpose(1, 2)
        shapes["tokens"] = tuple(tokens.shape)
        tokens = self.transformer(tokens + self.position_embedding)
        shapes["transformer"] = tuple(tokens.shape)
        image_features = tokens.mean(dim=1)
        shapes["pooled_image_feature"] = tuple(image_features.shape)
        shapes["metadata"] = tuple(metadata.shape)
        fused = torch.cat((image_features, metadata), dim=1)
        shapes["fused_feature"] = tuple(fused.shape)
        logits = self.classifier(fused)
        shapes["logits"] = tuple(logits.shape)
        return (logits, shapes) if return_shapes else logits


__all__ = ["FocalLoss", "MobileNetHDA"]

