# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn


@dataclass
class TtLinearParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None
    compute_config: ttnn.DeviceComputeKernelConfig
    core_grid: ttnn.CoreGrid

    @classmethod
    def from_torch(
        cls,
        torch_linear: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        device,
        core_grid: None = None,
        is_conv=False,
    ) -> TtLinearParameters:
        if not len(torch_linear.state_dict().keys()):
            breakpoint()
        weight = torch_linear.state_dict()["weight"]
        if is_conv:
            assert (
                1 == weight.shape[-1] == weight.shape[-2]
            ), f"Weight of shape {weight.shape} cannot be converted to linear"
            weight = torch.squeeze(weight, (-2, -1))
        bias = torch_linear.state_dict()["bias"]

        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        return cls(
            weight=ttnn.from_torch(weight.permute(-1, -2), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT),
            bias=ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT),
            compute_config=compute_config,
            core_grid=(core_grid if core_grid is not None else device.core_grid),
        )


def vae_linear(x, parameters):
    in_layout = x.layout
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.linear(
        input_tensor_a=x,
        input_tensor_b=parameters.weight,
        bias=parameters.bias,
        core_grid=parameters.core_grid,
        compute_kernel_config=parameters.compute_config,
    )
    output_tensor = ttnn.to_layout(output_tensor, in_layout)
    return output_tensor
