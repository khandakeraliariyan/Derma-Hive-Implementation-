"""Phase 3 SkinNet-HDA clean-room reconstruction.

This is our own implementation based on the paper's high-level sequence. The
paper does not specify the SE ratio, spatial kernel, Transformer configuration,
positional encoding, or exact ConvNeXt feature stage; those choices are marked
below as clean-room implementation decisions and are not claimed by the paper.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

NUM_CLASSES = 7


class SEChannelAttention(nn.Module):
    """Paper-specified SE formulation; reduction=16 is our decision."""

    def __init__(self, channels: int = 768, reduction: int = 16) -> None:
        super().__init__()
        if channels % reduction != 0:
            raise ValueError("channels must be divisible by reduction")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.excitation(self.pool(features))


class SpatialAttention(nn.Module):
    """Paper-specified average/max pooling; kernel_size=7 is our decision."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Spatial-attention kernel size must be odd")
        self.gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        average = features.mean(dim=1, keepdim=True)
        maximum = features.amax(dim=1, keepdim=True)
        return features * self.gate(torch.cat((average, maximum), dim=1))


class SkinNetHDA(nn.Module):
    """ConvNeXt -> SE -> spatial attention -> Transformer -> metadata fusion."""

    def __init__(
        self,
        metadata_dim: int,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        embedding_dim: int = 768,
        token_count: int = 49,
    ) -> None:
        super().__init__()
        if metadata_dim <= 0:
            raise ValueError("metadata_dim must be positive")
        if embedding_dim != 768 or token_count != 49:
            raise ValueError(
                "This reconstruction expects ConvNeXt-Tiny's [B,768,7,7] output"
            )
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        # OUR DECISION: final convolutional feature map before backbone pooling/head.
        self.backbone = backbone.features
        self.channel_attention = SEChannelAttention(embedding_dim, reduction=16)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, token_count, embedding_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Linear(embedding_dim + metadata_dim, num_classes)
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
        shapes["convnext"] = tuple(features.shape)
        if features.shape[1:] != (768, 7, 7):
            raise RuntimeError(
                "Expected ConvNeXt output [B,768,7,7], got " + str(tuple(features.shape))
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


class FocalLoss(nn.Module):
    """Multiclass focal loss operating on logits, with per-class alpha."""

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        if alpha.ndim != 1:
            raise ValueError("alpha must be a one-dimensional class-weight tensor")
        self.register_buffer("alpha", alpha.float())
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = torch.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()
        indices = targets.unsqueeze(1)
        log_pt = log_probabilities.gather(1, indices).squeeze(1)
        pt = probabilities.gather(1, indices).squeeze(1)
        alpha_t = self.alpha[targets]
        return (-alpha_t * (1.0 - pt).pow(self.gamma) * log_pt).mean()

