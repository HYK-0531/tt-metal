# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .utils import from_torch_fast

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
        batch_size: int,
        input_width: int,
        input_height: int,
        num_channels: int,
        num_groups: int,
        device: ttnn.Device,
    ) -> TtGroupNormParameters:
        k_device = 256 * device.core_grid.x * device.core_grid.y
        inplace = input_width * input_height <= k_device  # a heuristic

        if inplace:
            memory_config, core_grid = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
                device=device,
                num_channels=num_channels,
                num_groups=num_groups,
                input_nhw=batch_size * input_height * input_width,
                is_height_sharded=False,
            )
            num_out_blocks = 1

            # if input_memory_config.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            #     grid_y = self.group_norm_core_grid.y
            # elif input_memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            #     grid_y = 1
            # else:
            #     grid_y = int(self.group_norm_core_grid.x * self.group_norm_core_grid.y)
        else:
            # https://github.com/tenstorrent/tt-metal/issues/22149#issuecomment-2884093864
            h = num_channels // num_groups * num_groups
            assert h % 32 == 0
            grid_y = device.core_grid.y
            while h // grid_y % 32 != 0:
                grid_y -= 1

            core_grid = ttnn.CoreGrid(x=device.core_grid.x, y=grid_y)
            k_grid = 256 * core_grid.x * core_grid.y
            num_out_blocks = -(-input_width * input_height // k_grid)  # a heuristic
            memory_config = ttnn.DRAM_MEMORY_CONFIG

        torch_weight = ttnn.create_group_norm_weight_bias_rm(
            state["weight"],
            num_channels,
            core_grid.y,
        )
        torch_bias = ttnn.create_group_norm_weight_bias_rm(
            state["bias"],
            num_channels,
            core_grid.y,
        )
        torch_mask = ttnn.create_group_norm_input_mask(
            num_channels,
            num_groups,
            core_grid.y,
        )

        return cls(
            weight=from_torch_fast(
                torch_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            ),
            bias=from_torch_fast(
                torch_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            ),
            mask=from_torch_fast(
                torch_mask,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
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

        if parameters.inplace:
            x = ttnn.to_memory_config(x, parameters.memory_config)
            x = ttnn.reallocate(x)
        else:
            x = ttnn.tilize_with_zero_padding(x, use_multicore=True)

        x = ttnn.group_norm(
            x,
            weight=parameters.weight,
            bias=parameters.bias,
            input_mask=parameters.mask,
            num_groups=parameters.num_groups,
            epsilon=self._eps,
            core_grid=parameters.core_grid,
            # memory_config=memory_config if memory_config is not None else prep.memory_config,
            inplace=parameters.inplace,
            num_out_blocks=parameters.num_out_blocks,
        )

        # to_layout does not work with block sharded tensors
        # if memory_config is None:
        #   memory_config = ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.to_memory_config(x, parameters.memory_config)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        return x.reshape([batch_size, height, width, channels])
