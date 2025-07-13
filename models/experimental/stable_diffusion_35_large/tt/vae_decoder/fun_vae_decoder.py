# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

# from .conv2d import TtConv2d, TtConv2dParameters
from .fun_conv2d import vae_conv2d, TtConv2dParameters
from .fun_unet_mid_block import unet_mid_block, TtUNetMidBlock2DParameters
from .fun_updecoder_block import updecoder_block, TtUpDecoderBlock2DParameters
from .fun_group_norm import vae_group_norm, TtGroupNormParameters
from ..parallel_config import StableDiffusionParallelManager
from loguru import logger

if TYPE_CHECKING:
    import torch

## Parameters


@dataclass
class TtVaeDecoderParameters:
    conv_in: TtConv2dParameters
    mid_block: TtUNetMidBlock2DParameters
    up_blocks: list[TtUpDecoderBlock2DParameters]
    conv_norm_out: TtGroupNormParameters
    conv_out: TtConv2dParameters

    @classmethod
    def from_torch(
        cls,
        torch_vae_decoder: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        device,
        core_grid,
    ) -> TtVaeDecoderParameters:
        return cls(
            conv_in=TtConv2dParameters.from_torch(
                torch_vae_decoder.conv_in, dtype=dtype, device=device
            ),  # TODO: Add Silu, Add act_block_h
            mid_block=TtUNetMidBlock2DParameters.from_torch(
                torch_vae_decoder.mid_block, dtype=dtype, device=device, core_grid=core_grid
            ),
            up_blocks=[
                TtUpDecoderBlock2DParameters.from_torch(up_block, dtype=dtype, core_grid=core_grid, device=device)
                for up_block in (torch_vae_decoder.up_blocks or [])
            ],
            conv_norm_out=TtGroupNormParameters.from_torch(
                torch_vae_decoder.conv_norm_out, device=device, core_grid=core_grid
            ),
            conv_out=TtConv2dParameters.from_torch(
                torch_vae_decoder.conv_out, dtype=dtype, device=device
            ),  # TODO: Add Silu, Add act_block_h
        )


# TODO: Verify upscale_dtype from reference code
def sd_vae_decode(
    x: ttnn.Tensor,
    parameters: TtVaeDecoderParameters,
    parallel_manager: StableDiffusionParallelManager,
) -> ttnn.Tensor:
    logger.info("vae_conv2d", x.shape)
    x = vae_conv2d(x, parameters.conv_in)
    logger.info("unet_mid_block")
    x = unet_mid_block(x, parameters.mid_block, None)

    logger.info("updecoder_block")
    for up_block_params in parameters.up_blocks:
        x = updecoder_block(x, up_block_params, None)

    logger.info("vae_group_norm")
    x = vae_group_norm(x, parameters.conv_norm_out)
    logger.info("vae_conv2d")
    x = ttnn.silu(x)
    x = vae_conv2d(x, parameters.conv_out)

    return x
