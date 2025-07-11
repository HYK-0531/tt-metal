# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn


if TYPE_CHECKING:
    pass


# TODO: Pass in dtype. Parameterize hardcoded values. Add epsilon as parameter so we can just from torch groupnorm
@dataclass
class TtGroupNormParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor
    mask: ttnn.Tensor
    memory_config: ttnn.MemoryConfig
    core_grid: ttnn.CoreGrid
    # num_out_blocks: int
    inplace: bool
    num_channels: int
    num_groups: int

    @classmethod
    def from_torch(
        cls,
        torch_groupnorm,
        *,
        core_grid: ttnn.CoreGrid | None = None,
        device: ttnn.MeshDevice,
    ) -> TtGroupNormParameters:
        inplace = False  # input_width * input_height <= k_device  # a heuristic

        # TODO: Update with parallel information
        if not core_grid:  # get core grid from the mesh device.
            core_grid = ttnn.CoreGrid(y=4, x=4)
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        num_channels = torch_groupnorm.num_channels
        num_groups = torch_groupnorm.num_groups
        torch_weight = ttnn.create_group_norm_weight_bias_rm(
            torch_groupnorm.state_dict()["weight"], num_channels, core_grid.y
        )
        torch_bias = ttnn.create_group_norm_weight_bias_rm(
            torch_groupnorm.state_dict()["bias"], num_channels, core_grid.y
        )
        torch_mask = ttnn.create_group_norm_input_mask(num_channels, num_groups, core_grid.y)

        return cls(
            weight=ttnn.from_torch(
                torch_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=memory_config,
            ),
            bias=ttnn.from_torch(
                torch_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=memory_config,
            ),
            mask=ttnn.from_torch(
                torch_mask,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=memory_config,
            ),
            memory_config=memory_config,
            core_grid=core_grid,
            # num_out_blocks=num_out_blocks,
            inplace=inplace,
            num_channels=num_channels,
            num_groups=num_groups,
        )


def vae_group_norm(x_in, parameters: TtGroupNormParameters, eps=1e-6):
    [batch_size, height, width, channels] = list(x_in.shape)

    # TODO: Compute optimal output blocks
    num_out_blocks = -(-width // 32)
    x = ttnn.to_memory_config(x_in, ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = x.reshape([batch_size, 1, width * height, channels])
    x = ttnn.tilize_with_zero_padding(x, use_multicore=True)
    x = ttnn.group_norm(
        x,
        weight=parameters.weight,
        bias=parameters.bias,
        input_mask=parameters.mask,
        num_groups=parameters.num_groups,
        epsilon=eps,
        core_grid=parameters.core_grid,
        # memory_config=parameters.memory_config,
        inplace=False,
        num_out_blocks=num_out_blocks,
        output_layout=ttnn.TILE_LAYOUT,
    )
    # x = ttnn.to_layout(x, x_in.layout)
    return x.reshape([batch_size, height, width, channels])
