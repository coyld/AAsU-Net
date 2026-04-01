from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from .layers import CSFFProjector, DecoderStage, EncoderStage, kaiming_init


class AAsUNet(nn.Module):
    """Paper-faithful implementation of AAsU-Net.

    Default layout follows the paper:
    - 6 encoder stages with two conv blocks each
    - 5 decoder stages with one conv block each
    - optional CSFF in the encoder
    - deep supervision heads at all decoder stages

    Additional switches are exposed to support ablations reported in the paper,
    while keeping the full AAsU-Net as the default configuration.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        encoder_channels: Sequence[int] = (24, 48, 96, 192, 320, 320),
        conv_mode: str = "aas",
        use_csff: bool = True,
        reduction: int = 4,
        leakiness: float = 0.01,
        deep_supervision: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 6:
            raise ValueError("AAsUNet expects exactly 6 encoder channel definitions.")

        enc = list(encoder_channels)
        dec = [enc[4], enc[3], enc[2], enc[1], enc[0]]
        self.deep_supervision = deep_supervision
        self.use_csff = use_csff
        self.conv_mode = conv_mode
        self.encoder_channels = enc
        self.decoder_channels = dec

        downsample_strides = [
            (1, 1, 1),
            (1, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        ]

        self.encoder = nn.ModuleList(
            [
                EncoderStage(
                    in_channels,
                    enc[0],
                    apply_downsample=False,
                    downsample_stride=downsample_strides[0],
                    conv_mode=conv_mode,
                    reduction=reduction,
                    negative_slope=leakiness,
                    dropout=dropout,
                ),
                EncoderStage(
                    enc[0],
                    enc[1],
                    apply_downsample=True,
                    downsample_stride=downsample_strides[1],
                    conv_mode=conv_mode,
                    reduction=reduction,
                    negative_slope=leakiness,
                    dropout=dropout,
                ),
                EncoderStage(
                    enc[1],
                    enc[2],
                    apply_downsample=True,
                    downsample_stride=downsample_strides[2],
                    conv_mode=conv_mode,
                    reduction=reduction,
                    negative_slope=leakiness,
                    dropout=dropout,
                ),
                EncoderStage(
                    enc[2],
                    enc[3],
                    apply_downsample=True,
                    downsample_stride=downsample_strides[3],
                    conv_mode=conv_mode,
                    reduction=reduction,
                    negative_slope=leakiness,
                    dropout=dropout,
                ),
                EncoderStage(
                    enc[3],
                    enc[4],
                    apply_downsample=True,
                    downsample_stride=downsample_strides[4],
                    conv_mode=conv_mode,
                    reduction=reduction,
                    negative_slope=leakiness,
                    dropout=dropout,
                ),
                EncoderStage(
                    enc[4],
                    enc[5],
                    apply_downsample=True,
                    downsample_stride=downsample_strides[5],
                    conv_mode=conv_mode,
                    reduction=reduction,
                    negative_slope=leakiness,
                    dropout=dropout,
                ),
            ]
        )

        self.csff = nn.ModuleList([CSFFProjector(enc[0], ch) for ch in enc[1:]])

        self.decoder = nn.ModuleList(
            [
                DecoderStage(
                    enc[5],
                    enc[4],
                    dec[0],
                    upsample_stride=(2, 2, 2),
                    num_classes=out_channels,
                    conv_mode=conv_mode,
                    reduction=reduction,
                    negative_slope=leakiness,
                    dropout=dropout,
                ),
                DecoderStage(
                    dec[0],
                    enc[3],
                    dec[1],
                    upsample_stride=(2, 2, 2),
                    num_classes=out_channels,
                    conv_mode=conv_mode,
                    reduction=reduction,
                    negative_slope=leakiness,
                    dropout=dropout,
                ),
                DecoderStage(
                    dec[1],
                    enc[2],
                    dec[2],
                    upsample_stride=(2, 2, 2),
                    num_classes=out_channels,
                    conv_mode=conv_mode,
                    reduction=reduction,
                    negative_slope=leakiness,
                    dropout=dropout,
                ),
                DecoderStage(
                    dec[2],
                    enc[1],
                    dec[3],
                    upsample_stride=(2, 2, 2),
                    num_classes=out_channels,
                    conv_mode=conv_mode,
                    reduction=reduction,
                    negative_slope=leakiness,
                    dropout=dropout,
                ),
                DecoderStage(
                    dec[3],
                    enc[0],
                    dec[4],
                    upsample_stride=(1, 2, 2),
                    num_classes=out_channels,
                    conv_mode=conv_mode,
                    reduction=reduction,
                    negative_slope=leakiness,
                    dropout=dropout,
                ),
            ]
        )

        self.apply(kaiming_init)

    @property
    def min_spatial_shape(self) -> tuple[int, int, int]:
        return (16, 64, 64)

    def _maybe_csff(self, detail: torch.Tensor, target: torch.Tensor, idx: int) -> torch.Tensor:
        if not self.use_csff:
            return target
        return self.csff[idx](detail, target)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        e1 = self.encoder[0](x)
        e2 = self._maybe_csff(e1, self.encoder[1](e1), 0)
        e3 = self._maybe_csff(e1, self.encoder[2](e2), 1)
        e4 = self._maybe_csff(e1, self.encoder[3](e3), 2)
        e5 = self._maybe_csff(e1, self.encoder[4](e4), 3)
        e6 = self._maybe_csff(e1, self.encoder[5](e5), 4)
        return [e1, e2, e3, e4, e5, e6]

    def decode(self, features: List[torch.Tensor]) -> tuple[torch.Tensor, List[torch.Tensor]]:
        e1, e2, e3, e4, e5, e6 = features
        d5, o5 = self.decoder[0](e6, e5)
        d4, o4 = self.decoder[1](d5, e4)
        d3, o3 = self.decoder[2](d4, e3)
        d2, o2 = self.decoder[3](d3, e2)
        d1, o1 = self.decoder[4](d2, e1)
        return d1, [o1, o2, o3, o4, o5]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        features = self.encode(x)
        _, outputs = self.decode(features)
        primary = outputs[0]
        return {
            "logits": primary,
            "deep_supervision": outputs if self.deep_supervision else [primary],
            "encoder_features": features,
        }
