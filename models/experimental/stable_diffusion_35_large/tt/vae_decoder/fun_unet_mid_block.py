# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .fun_resnet_block import resnet_block, TtResnetBlock2DParameters
from .fun_attention import vae_attention, TtAttentionParameters
from ..parallel_config import StableDiffusionParallelManager

if TYPE_CHECKING:
    import torch

## Parameters


@dataclass
class TtUNetMidBlock2DParameters:
    attentions: list[TtAttentionParameters]
    resnets: list[TtResnetBlock2DParameters]

    @classmethod
    def from_torch(
        cls,
        unet_mid_block: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        core_grid,
        device,
    ) -> TtUNetMidBlock2DParameters:
        return cls(
            attentions=[
                TtAttentionParameters.from_torch(attention, dtype=dtype, core_grid=core_grid, device=device)
                for attention in unet_mid_block.attentions
            ],
            resnets=[
                TtResnetBlock2DParameters.from_torch(resnet_block, dtype=dtype, device=device, core_grid=core_grid)
                for resnet_block in unet_mid_block.resnets
            ],
        )


def unet_mid_block(
    x: ttnn.Tensor, parameters: TtUNetMidBlock2DParameters, parallel_manager: StableDiffusionParallelManager | None
) -> ttnn.Tensor:
    x = resnet_block(x, parameters.resnets[0], None)
    x = vae_attention(x, parameters.attentions[0], None)
    x = resnet_block(x, parameters.resnets[1], None)

    return x
