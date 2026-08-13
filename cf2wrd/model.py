from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    efficientnet_b0,
    mobilenet_v2,
)


class DualBranchEncoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        global_dim: int = 512,
        local_dim: int = 1024,
        freeze_mobile_blocks: int = 3,
        freeze_efficient_blocks: int = 5,
    ):
        super().__init__()
        mobile_weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        efficient_weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None

        self.mobile = mobilenet_v2(weights=mobile_weights)
        self.efficient = efficientnet_b0(weights=efficient_weights)

        self.mobile.classifier = nn.Identity()
        self.efficient.classifier = nn.Identity()

        # TorchVision MobileNetV2/EfficientNet-B0 both expose 1280-D pooled features.
        self.global_proj = nn.Linear(1280, global_dim)
        self.local_proj = nn.Linear(1280, local_dim)

        self._freeze_prefix(self.mobile.features, freeze_mobile_blocks)
        self._freeze_prefix(self.efficient.features, freeze_efficient_blocks)

    @staticmethod
    def _freeze_prefix(features: nn.Sequential, n: int) -> None:
        for block in list(features.children())[: max(0, n)]:
            for p in block.parameters():
                p.requires_grad = False

    def forward(self, global_x: torch.Tensor, local_x: torch.Tensor):
        gmap = self.mobile.features(global_x)
        g = F.adaptive_avg_pool2d(gmap, 1)
        g = torch.flatten(g, 1)
        g = self.global_proj(g)

        lmap = self.efficient.features(local_x)
        l = F.adaptive_avg_pool2d(lmap, 1)
        l = torch.flatten(l, 1)
        l = self.local_proj(l)
        return g, l

    def feature_maps(self, global_x: torch.Tensor, local_x: torch.Tensor):
        """Return final feature maps for Grad-CAM hooks/debugging."""
        return self.mobile.features(global_x), self.efficient.features(local_x)


class ACFFM(nn.Module):
    """Adaptive Complementary Feature Fusion Module at vector level.

    The local alignment is always parameterized as 1024 -> 512. Pearson
    selection changes which local input dimensions and corresponding weight
    columns participate in the forward computation, but it does not change
    the stored architecture or parameter count.
    """

    def __init__(self, global_dim: int, local_dim: int, fusion_dim: int):
        super().__init__()
        self.global_align = nn.Linear(global_dim, fusion_dim)
        self.local_align = nn.Linear(local_dim, fusion_dim)
        self.global_score = nn.Linear(fusion_dim, 1)
        self.local_score = nn.Linear(fusion_dim, 1)

    def forward(
        self,
        g: torch.Tensor,
        l_full: torch.Tensor,
        selected_local_indices: torch.Tensor,
    ):
        g = self.global_align(g)

        # Sparse local projection with fixed stored parameters:
        # use only active local dimensions and the matching columns of the
        # full 1024->512 weight matrix. Bias is unchanged.
        idx = selected_local_indices.to(l_full.device)
        l_active = torch.index_select(l_full, 1, idx)
        w_active = torch.index_select(self.local_align.weight, 1, idx)
        l = F.linear(l_active, w_active, self.local_align.bias)

        scores = torch.cat([self.global_score(g), self.local_score(l)], dim=1)
        weights = torch.softmax(scores, dim=1)
        fused = weights[:, 0:1] * g + weights[:, 1:2] * l
        return fused, weights, l_active


class CF2WRD(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        global_dim: int = 512,
        local_dim: int = 1024,
        fusion_dim: int = 512,
        dropout: float = 0.25,
        freeze_mobile_blocks: int = 3,
        freeze_efficient_blocks: int = 5,
        selected_local_indices: Optional[Iterable[int]] = None,
    ):
        super().__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.fusion_dim = fusion_dim

        self.encoder = DualBranchEncoder(
            pretrained=pretrained,
            global_dim=global_dim,
            local_dim=local_dim,
            freeze_mobile_blocks=freeze_mobile_blocks,
            freeze_efficient_blocks=freeze_efficient_blocks,
        )

        self.register_buffer(
            "selected_local_indices",
            torch.empty(0, dtype=torch.long),
            persistent=True,
        )
        # IMPORTANT: ACFFM is always created with the full local_dim so the
        # model has a fixed parameter count independent of Pearson activity.
        self.acffm = ACFFM(
            global_dim=self.global_dim,
            local_dim=self.local_dim,
            fusion_dim=self.fusion_dim,
        )
        self.bn = nn.BatchNorm1d(fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)

        if selected_local_indices is not None:
            self.set_selected_local_indices(selected_local_indices)

    def set_selected_local_indices(self, indices: Iterable[int]):
        indices = torch.as_tensor(list(indices), dtype=torch.long)
        if indices.numel() == 0:
            raise ValueError("selected_local_indices must not be empty")
        if indices.min() < 0 or indices.max() >= self.local_dim:
            raise ValueError("selected_local_indices out of range")
        self.selected_local_indices = indices

    @property
    def local_activity_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.local_dim, dtype=torch.bool, device=self.selected_local_indices.device)
        if self.selected_local_indices.numel() > 0:
            mask[self.selected_local_indices] = True
        return mask

    def forward(
        self,
        global_x: torch.Tensor,
        local_x: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        if local_x is None:
            local_x = global_x
        if self.selected_local_indices.numel() == 0:
            raise RuntimeError(
                "Pearson activity mask not initialized. Fit it on the training split "
                "or construct the model with selected_local_indices."
            )

        g, l_full = self.encoder(global_x, local_x)
        fused, weights, l_active = self.acffm(
            g, l_full, self.selected_local_indices
        )
        logits = self.classifier(self.dropout(self.bn(fused)))

        if return_features:
            mask = self.local_activity_mask.to(l_full.device)
            l_masked = l_full * mask.to(l_full.dtype).unsqueeze(0)
            return logits, {
                "global_feature": g,
                "local_feature_full": l_full,
                "local_feature_active": l_active,
                "local_feature_masked": l_masked,
                "fused_feature": fused,
                "branch_weights": weights,
            }
        return logits


def build_model(cfg: dict, selected_local_indices=None, pretrained_override=None) -> CF2WRD:
    m = cfg["model"]
    pretrained = m.get("pretrained", True)
    if pretrained_override is not None:
        pretrained = pretrained_override
    return CF2WRD(
        pretrained=pretrained,
        num_classes=m.get("num_classes", 2),
        global_dim=m.get("global_dim", 512),
        local_dim=m.get("local_dim", 1024),
        fusion_dim=m.get("fusion_dim", 512),
        dropout=m.get("dropout", 0.25),
        freeze_mobile_blocks=m.get("freeze_mobile_blocks", 3),
        freeze_efficient_blocks=m.get("freeze_efficient_blocks", 5),
        selected_local_indices=selected_local_indices,
    )
