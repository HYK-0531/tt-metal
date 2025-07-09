# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

# from .conv2d import TtConv2d, TtConv2dParameters
from .fun_conv2d import vae_conv2d_, TtConv2dParameters
from .group_norm import TtGroupNormParameters
from ..parallel_config import StableDiffusionParallelManager

if TYPE_CHECKING:
    import torch

## Parameters


@dataclass
class TtResnetBlock2DParameters:
    norm1: TtGroupNormParameters
    norm2: TtGroupNormParameters
    conv1: TtConv2dParameters
    conv2: TtConv2dParameters
    conv_shortcut: TtConv2dParameters | None

    @classmethod
    def from_torch(
        cls,
        resnet_block: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        num_out_blocks: int,
        core_grid,
        device: ttnn.MeshDevice,
    ) -> TtResnetBlock2DParameters:
        return cls(
            norm1=None,  # TtGroupNormParameters.from_torch(resnet_block.norm1.state_dict(), device=device,num_channels=resnet_block.norm1.num_channels,num_groups=resnet_block.norm1.num_groups,core_grid=core_grid,num_out_blocks=num_out_blocks),
            norm2=None,  # TtGroupNormParameters.from_torch(resnet_block.norm2.state_dict(), device=device,num_channels=resnet_block.norm2.num_channels,num_groups=resnet_block.norm2.num_groups,core_grid=core_grid,num_out_blocks=num_out_blocks),
            conv1=TtConv2dParameters.from_torch(resnet_block.conv1, dtype=dtype, device=device),
            conv2=None,  # TtConv2dParameters.from_torch(resnet_block.conv2.state_dict(), dtype=dtype, device=device),
            conv_shortcut=None,  # TtConv2dParameters.from_torch(substate(resnet_block.state_dict(), "conv_shortcut"), dtype=dtype, device=device) if has_substate(resnet_block.state_dict(), "conv_shortcut") else None,
        )

    @property
    def in_channels(self) -> int:
        return self.conv1.in_channels


def resnet_block(
    x_in: ttnn.Tensor, parameters: TtResnetBlock2DParameters, parallel_manager: StableDiffusionParallelManager | None
) -> ttnn.Tensor:
    # x = group_norm(x_in, parameters.norm1)
    # x = ttnn.silu(x)
    x = vae_conv2d_(x_in, parameters.conv1)

    return x
