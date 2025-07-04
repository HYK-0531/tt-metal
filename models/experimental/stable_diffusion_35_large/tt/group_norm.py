# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn


if TYPE_CHECKING:
    import torch


@dataclass
class TtGroupNormParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor
    mask: ttnn.Tensor
    memory_config: ttnn.MemoryConfig
    core_grid: ttnn.CoreGrid
    num_out_blocks: int
    inplace: bool
    input_width: int
    input_height: int
    num_channels: int
    num_groups: int

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        input_width: int,
        input_height: int,
        num_channels: int,
        num_groups: int,
        num_out_blocks: int,
        core_grid: ttnn.CoreGrid,
        device: ttnn.MeshDevice,
    ) -> TtGroupNormParameters:
        inplace = False  # input_width * input_height <= k_device  # a heuristic

        memory_config = ttnn.DRAM_MEMORY_CONFIG
        torch_weight = ttnn.create_group_norm_weight_bias_rm(state["weight"], num_channels, core_grid.y)
        torch_bias = ttnn.create_group_norm_weight_bias_rm(state["bias"], num_channels, core_grid.y)
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
            num_out_blocks=num_out_blocks,
            inplace=inplace,
            input_width=input_width,
            input_height=input_height,
            num_channels=num_channels,
            num_groups=num_groups,
        )


class TtGroupNorm:
    def __init__(self, parameters: TtGroupNormParameters, *, eps: float) -> None:
        self._eps = eps
        self._parameters = parameters

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        [batch_size, height, width, channels] = list(x.shape)

        parameters = self._parameters

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = x.reshape([batch_size, 1, width * height, channels])
        x = ttnn.tilize_with_zero_padding(x, use_multicore=True)

        x = ttnn.group_norm(
            x,
            weight=parameters.weight,
            bias=parameters.bias,
            input_mask=parameters.mask,
            num_groups=parameters.num_groups,
            # epsilon=self._eps,
            core_grid=parameters.core_grid,
            memory_config=parameters.memory_config,
            inplace=False,  # parameters.inplace,
            num_out_blocks=parameters.num_out_blocks,
            output_layout=ttnn.TILE_LAYOUT,
        )

        # to_layout does noxt work with block sharded tensors
        # if memory_config is None:
        #   memory_config = ttnn.DRAM_MEMORY_CONFIG
        # x = ttnn.to_memory_config(x, parameters.memory_config)
        # x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        return x.reshape([batch_size, height, width, channels])
