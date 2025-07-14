# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn


@dataclass
class TtConv2dParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int] | tuple[int, int, int, int]
    stride: tuple[int, int]
    dilation: tuple[int, int]
    device: ttnn.Device
    activation: str
    compute_config: ttnn.DeviceComputeKernelConfig
    conv_config: ttnn.Conv2dConfig

    @classmethod
    def from_torch(
        cls, torch_conv: torch.nn.Module, *, dtype: ttnn.DataType | None = None, device, activation="", act_block_h=32
    ) -> TtConv2dParameters:
        weight = torch_conv.state_dict()["weight"]
        bias = torch_conv.state_dict()["bias"]
        conv_config = ttnn.Conv2dConfig(
            dtype=dtype,
            weights_dtype=dtype,
            activation=activation,
            shard_layout=(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                # if torch_conv.in_channels <= 512
                # else ttnn.TensorMemoryLayout.BLOCK_SHARDED  # Prevent asserts. TODO: Add correct optimization
            ),
            reshard_if_not_optimal=True,
            deallocate_activation=True,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            reallocate_halo_output=False,
        )

        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        if act_block_h is not None:
            conv_config.act_block_h_override = act_block_h

        return cls(
            weight=ttnn.from_torch(weight, dtype=dtype, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0)),
            # bias=ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=dtype,mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0)),
            bias=ttnn.from_torch(
                bias.reshape((1, 1, 1, -1)), dtype=dtype, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1)
            ),
            out_channels=torch_conv.out_channels,
            in_channels=torch_conv.in_channels,
            kernel_size=torch_conv.kernel_size,
            padding=torch_conv.padding,
            stride=torch_conv.stride,
            dilation=torch_conv.dilation,
            device=device,
            activation=activation,
            compute_config=compute_config,
            conv_config=conv_config,
        )


def vae_conv2d(x, parameters):
    b, h, w, c = x.shape
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(parameters.device)):
        x = ttnn.to_device(x, parameters.device)

    # TODO: compute optimal slice config per height or width.
    # slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceWidth, num_slices=w // 2)
    o_c, _, _, _ = parameters.weight.shape
    slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceWidth, num_slices=min(256, w // 2))
    conv_config = parameters.conv_config
    # conv_config.act_block_h_override = 16

    # ttnn.visualize_mesh_device(parameters.device, tensor=parameters.weight)
    output_tensor, [_out_height, _out_width] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=parameters.weight,
        bias_tensor=parameters.bias,
        in_channels=c,
        out_channels=o_c,  # parameters.out_channels//4,
        device=parameters.device,
        kernel_size=parameters.kernel_size,
        stride=parameters.stride,
        padding=parameters.padding,
        batch_size=b,
        input_height=h,
        input_width=w,
        conv_config=conv_config,
        compute_config=parameters.compute_config,
        # memory_config=ttnn.DRAM_MEMORY_CONFIG,
        slice_config=slice_config,
        return_output_dim=True,
    )

    output_tensor = ttnn.reshape(output_tensor, (x.shape[0], _out_height, _out_width, output_tensor.shape[3]))
    torch_tensor = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(parameters.device, dim=-1))
    output_tensor = ttnn.from_torch(torch_tensor, dtype=output_tensor.dtype, device=parameters.device)
    return output_tensor
