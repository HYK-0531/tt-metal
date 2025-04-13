# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn._ttnn
from models.utility_functions import comp_allclose_and_pcc, comp_pcc
from loguru import logger
from enum import Enum

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    get_lib_dtype,
)


class DeinterleaveMode(Enum):
    DeinterleaveBatch = 1
    DeinterleaveLocal = 2


def torch_deinterleave_to_batch(torch_input_nhwc, stride_hw):
    torch_deinterleaved_to_batch = torch.zeros(
        torch_input_nhwc.shape[0] * stride_hw[0] * stride_hw[1],
        torch_input_nhwc.shape[1] // stride_hw[0],
        torch_input_nhwc.shape[2] // stride_hw[1],
        torch_input_nhwc.shape[3],
    )

    print(f"torch_deinterleaved_to_batch shape: {torch_deinterleaved_to_batch.shape}")
    print(f"torch_input_nhwc shape: {torch_input_nhwc.shape}")
    for src_batch in range(torch_input_nhwc.shape[0]):
        for split_h in range(stride_hw[0]):
            for split_w in range(stride_hw[1]):
                batch_idx = src_batch * stride_hw[0] * stride_hw[1] + split_h * stride_hw[1] + split_w
                torch_deinterleaved_to_batch[batch_idx, :, :, :] = torch_input_nhwc[
                    src_batch,
                    split_h :: stride_hw[0],
                    split_w :: stride_hw[1],
                    :,
                ]
    return torch_deinterleaved_to_batch


def torch_deinterleave_local(torch_input_nhwc, stride_hw, num_of_cores):
    return
    # torch_deinterleaved_local = torch.zeros(
    #     num_of_cores,
    #     torch_input_nhwc.shape[0] * stride_hw[0] * stride_hw[1],
    #     torch_input_nhwc.shape[1] // stride_hw[0],
    #     torch_input_nhwc.shape[2] // stride_hw[1],
    #     torch_input_nhwc.shape[3],
    # )
    # assert(torch_input_nhwc.shape[0] == 1)

    # height_per_core = torch_input_nhwc.shape[1] // num_of_cores[0]

    # for core in range(num_of_cores):
    #     for split_h in range(stride_hw[0]):
    #         for split_w in range(stride_hw[1]):

    #             torch_deinterleaved_to_batch[1,
    #                                          core*height_per_core:core*height_per_core+height_per_core,
    #                                          split_h::stride_hw[0],
    #                                          split_w::stride_hw[1],
    #                                          :] = torch_input_nhwc[

    #             batch_idx = src_batch * stride_hw[0] * stride_hw[1] + split_h * stride_hw[1] + split_w
    #             torch_deinterleaved_to_batch[batch_idx, :, :, :] = torch_input_nhwc[
    #                 src_batch,
    #                 split_h :: stride_hw[0],
    #                 split_w :: stride_hw[1],
    #                 :,
    #             ]
    # return torch_deinterleaved_to_batch


def run_deinterleave(
    device,
    mode: DeinterleaveMode,
    shape_nhwc,
    input_memory_config,
    stride_hw,
    barrier_threshold=0,
):
    input_dtype = "bfloat16"
    torch_input = 2 * torch.rand(size=shape_nhwc, dtype=get_lib_dtype(torch, input_dtype)) - 1
    # torch_input = torch.ones(size=shape_nhwc, dtype=get_lib_dtype(torch, input_dtype))
    # torch_input[:, ::2, ::2, :] = 100# + torch.range(0, shape_nhwc[-1]-1, dtype=get_lib_dtype(torch, input_dtype))
    # torch_input[:, ::2, 1::2, :] = 200
    # torch_input[:, 1::2, ::2, :] = 300
    # torch_input[:, 1::2, 1::2, :] = 400
    # for b in range (shape_nhwc[0]):
    #     for h in range(shape_nhwc[1]):
    #         for w in range(shape_nhwc[2]):
    #             for c in range(shape_nhwc[3]):
    #                 if (w % 2 == 0) and (h % 2 == 0):
    #                     base = 100
    #                 elif (w % 2 == 1) and (h % 2 == 0):
    #                     base = 200
    #                 elif (w % 2 == 0) and (h % 2 == 1):
    #                     base = 300
    #                 elif (w % 2 == 1) and (h % 2 == 1):
    #                     base = 400
    #                 if (c == 0):
    #                     torch_input[b, h, w, c] = base
    #                 else:
    #                     torch_input[b, h, w, c] = base + h * 10 + w

    # move to 1,1, NHW, C as in conv2ds
    torch_input_view = torch_input.reshape(1, 1, shape_nhwc[0] * shape_nhwc[1] * shape_nhwc[2], shape_nhwc[3])
    print(f"torch_input_view.shape={torch_input_view.shape}")
    print(f"torch_input_view={torch_input_view}")

    ttnn_input = ttnn.from_torch(
        torch_input_view,
        device=device,
        dtype=get_lib_dtype(ttnn, input_dtype),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=input_memory_config,
    ).to(device)

    # print(f"Input tensor mem: {ttnn_input.memory_config()}")
    compute_kernel_options = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    print(f"shard_shape {input_memory_config.shard_spec.shape}")
    print(f"shard_spec mode {input_memory_config.shard_spec.mode}")

    if mode == DeinterleaveMode.DeinterleaveBatch:
        golden_output = torch_deinterleave_to_batch(torch_input, stride_hw)

        ttnn_output = ttnn.experimental.deinterleave_to_batch(
            ttnn_input,
            compute_kernel_config=compute_kernel_options,
            stride_hw=stride_hw,
            input_height=shape_nhwc[1],
            input_width=shape_nhwc[2],
            barrier_threshold=barrier_threshold,
        )

        torch_output = ttnn.to_torch(ttnn_output)

        # print(f"ttnn_output shape={ttnn_output.shape}")
        # print(f"torch_output {torch_output[:,:,:,:]}")

        torch_output = torch_output.view(  # TBD where to do this
            shape_nhwc[0] * stride_hw[0] * stride_hw[1],
            shape_nhwc[1] // stride_hw[0],
            shape_nhwc[2] // stride_hw[1],
            shape_nhwc[3],
        )
    else:
        golden_output = torch_deinterleave_local(torch_input, stride_hw)

        ttnn_output = ttnn.experimental.deinterleave_to_batch(
            ttnn_input,
            compute_kernel_config=compute_kernel_options,
            stride_hw=stride_hw,
            input_height=shape_nhwc[1],
            input_width=shape_nhwc[2],
            barrier_threshold=barrier_threshold,
        )

    # print(f"golden={golden_output}")
    # print("============")
    print(f"golden_shape={golden_output.shape}")
    print(f"torch_shape={torch_output.shape}")
    passing, out = comp_allclose_and_pcc(golden_output, torch_output, rtol=0.01, atol=0.01, pcc=0.999)
    logger.info(out)
    assert passing, out


@pytest.mark.parametrize(
    "shape, core_grid, stride_hw",
    [
        ([1, 4 * 32, 32, 32], ttnn.CoreGrid(x=4, y=1), [2, 2]),
        # ([1, 16 * 32, 32, 32], ttnn.CoreGrid(x=8, y=2), [2, 2]),
        ([1, 256, 1024, 32], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        ([1, 256, 1024, 64], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        ([1, 1024, 256, 32], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        ([1, 1024, 256, 64], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        ([1, 1024, 128, 32], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        ([1, 1024, 128, 64], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        ([1, 1024, 128, 48], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        # # ([1, 1024, 128, 56], ttnn.CoreGrid(x=8, y=8), [2, 2]), # PCC 0.003550328966433327
        # # ([1, 256, 1024, 32], ttnn.CoreGrid(x=8, y=8), [4, 4]), # RuntimeError: TT_FATAL @ /localdev/mbezulj/tt-metal/ttnn/cpp/ttnn/operations/experimental/deinterleave/device/deinterleave_device_operation.cpp:29: per_core_height >= 2 * operation_attributes.stride_hw[0]
        # # ([1, 256, 1024, 64], ttnn.CoreGrid(x=8, y=8), [4, 4]), # RuntimeError: TT_FATAL @ /localdev/mbezulj/tt-metal/ttnn/cpp/ttnn/operations/experimental/deinterleave/device/deinterleave_device_operation.cpp:29: per_core_height >= 2 * operation_attributes.stride_hw[0]
        # ([1, 1024, 256, 32], ttnn.CoreGrid(x=8, y=8), [4, 4]),
        # ([1, 1024, 256, 64], ttnn.CoreGrid(x=8, y=8), [4, 4]),
        # ([1, 1024, 128, 32], ttnn.CoreGrid(x=8, y=8), [4, 4]),
        # ([1, 1024, 128, 64], ttnn.CoreGrid(x=8, y=8), [4, 4]),
        # ([1, 1024, 128, 48], ttnn.CoreGrid(x=8, y=8), [4, 4]),
        # ([1, 1024, 256, 32], ttnn.CoreGrid(x=8, y=8), [8, 8]),
        # ([1, 1024, 256, 64], ttnn.CoreGrid(x=8, y=8), [8, 8]),
        # ([1, 1024, 128, 32], ttnn.CoreGrid(x=8, y=8), [8, 8]),
        # ([1, 1024, 128, 64], ttnn.CoreGrid(x=8, y=8), [8, 8]),
        # ([1, 1024, 128, 48], ttnn.CoreGrid(x=8, y=8), [8, 8]),
        # ([1, 256, 1024, 32], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        # ([1, 256, 1024, 48], ttnn.CoreGrid(x=8, y=8), [4, 4]),
        # ([1, 256, 1024, 56], ttnn.CoreGrid(x=8, y=8), [8, 8]), # RuntimeError: TT_FATAL @ /localdev/mbezulj/tt-metal/ttnn/cpp/ttnn/operations/experimental/deinterleave/device/deinterleave_device_operation.cpp:29: per_core_height >= operation_attributes.stride_hw[0]
        # ([1, 1024, 256, 32], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        # ([1, 1024, 256, 64], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        ([1, 1024, 256, 64], ttnn.CoreGrid(x=8, y=8), [4, 4]),
        ([1, 1024, 256, 64], ttnn.CoreGrid(x=8, y=8), [8, 8]),
        # ([1, 1024, 256, 32], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        # ([1, 1024, 256, 64], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        ([1, 1024, 4, 64 * 32], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        # ([1, 1024, 4, 64*64], ttnn.CoreGrid(x=8, y=8), [2, 2]),
    ],
)
# @pytest.mark.parametrize("barrier_threshold", [1,2,4,8,16,32,64,128,256])
def test_deinterleave_shape(
    device,
    shape,
    core_grid,
    stride_hw,
    barrier_threshold=0,
):
    torch.manual_seed(2025)

    memory_config = ttnn.create_sharded_memory_config_(
        shape=[shape[0] * shape[1] * shape[2], shape[3]],
        core_grid=core_grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        strategy=ttnn.ShardStrategy.HEIGHT,
    )

    print(f"Memory config: {memory_config}")

    run_deinterleave(
        device,
        DeinterleaveMode.DeinterleaveBatch,
        shape,
        memory_config,
        stride_hw,
        barrier_threshold,
    )
