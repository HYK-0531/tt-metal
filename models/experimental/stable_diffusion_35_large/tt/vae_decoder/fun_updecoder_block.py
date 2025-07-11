# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

# from .conv2d import TtConv2d, TtConv2dParameters
from .fun_resnet_block import resnet_block, TtResnetBlock2DParameters
from .fun_upsample2d import vae_upsample2d, TtUpsample2DParameters
from ..parallel_config import StableDiffusionParallelManager

if TYPE_CHECKING:
    import torch

## Parameters


@dataclass
class TtUpDecoderBlock2DParameters:
    resnets: list[TtResnetBlock2DParameters]
    upsamplers: list[TtUpsample2DParameters]

    @classmethod
    def from_torch(
        cls,
        updecoder_block: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        core_grid,
        device,
    ) -> TtUpDecoderBlock2DParameters:
        return cls(
            resnets=[
                TtResnetBlock2DParameters.from_torch(resnet_block, dtype=dtype, core_grid=core_grid, device=device)
                for resnet_block in updecoder_block.resnets
            ],
            upsamplers=[
                TtUpsample2DParameters.from_torch(torch_upsample, dtype=dtype, device=device)
                for torch_upsample in updecoder_block.upsamplers
            ],
        )


def updecoder_block(
    x: ttnn.Tensor, parameters: TtUpDecoderBlock2DParameters, parallel_manager: StableDiffusionParallelManager | None
) -> ttnn.Tensor:
    for resnet_params in parameters.resnets:
        x = resnet_block(x, resnet_params, None)

    for upsample_params in parameters.upsamplers:
        x = vae_upsample2d(x, upsample_params)

    return x
