from __future__ import annotations

from typing import Callable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_3tuple(value: int | Sequence[int]) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    items = tuple(int(v) for v in value)
    if len(items) != 3:
        raise ValueError(f"Expected 3 values, got {items}")
    return items


def _same_padding(kernel_size: int | Sequence[int]) -> Tuple[int, int, int]:
    kz, ky, kx = _to_3tuple(kernel_size)
    return kz // 2, ky // 2, kx // 2


class ConvINLReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        bias: bool = False,
        negative_slope: float = 0.01,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=_to_3tuple(kernel_size),
            stride=_to_3tuple(stride),
            padding=_same_padding(kernel_size),
            bias=bias,
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class SeparableConv(nn.Module):
    """Spatially separable 3D convolution: 3x1x1 followed by 1x3x3."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        negative_slope: float = 0.01,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvINLReLU(
                in_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                negative_slope=negative_slope,
                dropout=dropout,
            ),
            ConvINLReLU(
                out_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                negative_slope=negative_slope,
                dropout=dropout,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StandardConvBlock(nn.Module):
    """3x3x3 convolutional block with IN + LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        negative_slope: float = 0.01,
        dropout: float = 0.0,
        **_: int,
    ) -> None:
        super().__init__()
        self.block = ConvINLReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            negative_slope=negative_slope,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SeparableConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        negative_slope: float = 0.01,
        dropout: float = 0.0,
        **_: int,
    ) -> None:
        super().__init__()
        self.block = SeparableConv(
            in_channels,
            out_channels,
            negative_slope=negative_slope,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SumFusionConv(nn.Module):
    """Ablation-friendly parallel fusion without adaptive weighting."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        negative_slope: float = 0.01,
        dropout: float = 0.0,
        **_: int,
    ) -> None:
        super().__init__()
        self.anisotropic_branch = SeparableConv(
            in_channels,
            out_channels,
            negative_slope=negative_slope,
            dropout=dropout,
        )
        self.isotropic_branch = StandardConvBlock(
            in_channels,
            out_channels,
            negative_slope=negative_slope,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.anisotropic_branch(x) + self.isotropic_branch(x)


class AAsConv(nn.Module):
    """Adaptive anisotropic convolution from the paper.

    Branch A:
        3x1x1 -> 1x3x3
    Branch B:
        3x3x3
    The two branch outputs are fused by softmax-normalized channel weights
    produced from a shared squeeze + branch-specific excitation pathway.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 4,
        negative_slope: float = 0.01,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        attn_channels = max(out_channels // reduction, 8)

        self.anisotropic_branch = SeparableConv(
            in_channels,
            out_channels,
            negative_slope=negative_slope,
            dropout=dropout,
        )
        self.isotropic_branch = StandardConvBlock(
            in_channels,
            out_channels,
            negative_slope=negative_slope,
            dropout=dropout,
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.squeeze = nn.Sequential(
            nn.Conv3d(out_channels, attn_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.excitation_aniso = nn.Conv3d(attn_channels, out_channels, kernel_size=1, bias=True)
        self.excitation_iso = nn.Conv3d(attn_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xa = self.anisotropic_branch(x)
        xb = self.isotropic_branch(x)

        xu = xa + xb
        descriptor = self.pool(xu)
        hidden = self.squeeze(descriptor)

        logits_a = self.excitation_aniso(hidden)
        logits_b = self.excitation_iso(hidden)

        weights = torch.stack((logits_a, logits_b), dim=1)
        weights = torch.softmax(weights, dim=1)

        out = weights[:, 0] * xa + weights[:, 1] * xb
        return out


BlockFactory = Callable[[int, int, int, float, float], nn.Module]


def build_conv_block(
    mode: str,
    in_channels: int,
    out_channels: int,
    reduction: int = 4,
    negative_slope: float = 0.01,
    dropout: float = 0.0,
) -> nn.Module:
    mode = mode.lower()
    if mode in {"aas", "aasconv", "adaptive_anisotropic"}:
        return AAsConv(
            in_channels,
            out_channels,
            reduction=reduction,
            negative_slope=negative_slope,
            dropout=dropout,
        )
    if mode in {"standard", "conv3d", "isotropic"}:
        return StandardConvBlock(
            in_channels,
            out_channels,
            negative_slope=negative_slope,
            dropout=dropout,
        )
    if mode in {"separable", "anisotropic", "sep"}:
        return SeparableConvBlock(
            in_channels,
            out_channels,
            negative_slope=negative_slope,
            dropout=dropout,
        )
    if mode in {"sum", "sumfusion", "parallel_sum"}:
        return SumFusionConv(
            in_channels,
            out_channels,
            negative_slope=negative_slope,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported conv mode: {mode}")


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        downsample_stride: Sequence[int] = (1, 1, 1),
        apply_downsample: bool = False,
        conv_mode: str = "aas",
        reduction: int = 4,
        negative_slope: float = 0.01,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.downsample = None
        if apply_downsample:
            self.downsample = ConvINLReLU(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=downsample_stride,
                negative_slope=negative_slope,
                dropout=dropout,
            )
            stage_in_channels = out_channels
        else:
            stage_in_channels = in_channels

        self.block1 = build_conv_block(
            conv_mode,
            stage_in_channels,
            out_channels,
            reduction=reduction,
            negative_slope=negative_slope,
            dropout=dropout,
        )
        self.block2 = build_conv_block(
            conv_mode,
            out_channels,
            out_channels,
            reduction=reduction,
            negative_slope=negative_slope,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class CSFFProjector(nn.Module):
    """Cross-Scale Feature Fusion projector.

    En1 is adaptively pooled to the target spatial size, projected by a 1x1x1 conv,
    and added to the target encoder feature map.
    """

    def __init__(self, detail_channels: int, target_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(detail_channels, target_channels, kernel_size=1, bias=False)

    def forward(self, detail: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        resized = F.adaptive_avg_pool3d(detail, output_size=target.shape[2:])
        projected = self.proj(resized)
        return target + projected


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        upsample_stride: Sequence[int],
        num_classes: int,
        conv_mode: str = "aas",
        reduction: int = 4,
        negative_slope: float = 0.01,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        stride = _to_3tuple(upsample_stride)
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=stride, stride=stride)
        self.block = build_conv_block(
            conv_mode,
            out_channels + skip_channels,
            out_channels,
            reduction=reduction,
            negative_slope=negative_slope,
            dropout=dropout,
        )
        self.head = SegmentationHead(out_channels, num_classes)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x, self.head(x)


def kaiming_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.InstanceNorm3d):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
