# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_allclose
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
    check_dim,
    compute_kernel_ids,
    compute_kernel_options,
    create_ttnn_tilized_tensor,
    get_compute_kernel_options,
)

TILE_WIDTH = 32
TILE_HEIGHT = 32


def grid_size():
    return (8, 7)


def div_up(x, y):
    """Divide x by y and round up."""
    return (x + y - 1) // y


def round_up(x, y):
    """Round x up to the nearest multiple of y."""
    return ((x + y - 1) // y) * y


def nd_shard_spec(tensor_shape, shard_shape, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    if len(tensor_shape) < 2:
        tensor_shape = [1] * (2 - len(tensor_shape)) + tensor_shape

    assert len(shard_shape) >= 2
    assert all(s == 1 for s in shard_shape[:-2])
    assert shard_shape[-1] % TILE_WIDTH == 0
    assert shard_shape[-2] % TILE_HEIGHT == 0

    assert len(tensor_shape) == len(shard_shape)

    sub_grid_x = div_up(round_up(tensor_shape[-1], TILE_WIDTH), shard_shape[-1])
    sub_grid_y = div_up(round_up(tensor_shape[-2], TILE_HEIGHT), shard_shape[-2])
    if orientation == ttnn.ShardOrientation.COL_MAJOR:
        sub_grid_x, sub_grid_y = sub_grid_y, sub_grid_x
    assert sub_grid_x > 0 and sub_grid_y > 0
    grid_x, grid_y = grid_size()
    num_sub_grid_x = grid_x // sub_grid_x
    num_sub_grid_y = grid_y // sub_grid_y
    assert num_sub_grid_x > 0 and num_sub_grid_y > 0
    num_batches = math.prod(tensor_shape[:-2])

    core_ranges = []
    cnt = 0
    if orientation == ttnn.ShardOrientation.ROW_MAJOR:
        sub_grid_idxes = ((j, i) for j in range(num_sub_grid_y) for i in range(num_sub_grid_x))
    else:
        sub_grid_idxes = ((j, i) for i in range(num_sub_grid_x) for j in range(num_sub_grid_y))
    for j, i in sub_grid_idxes:
        x1 = i * sub_grid_x
        y1 = j * sub_grid_y
        x2 = (i + 1) * sub_grid_x - 1
        y2 = (j + 1) * sub_grid_y - 1
        core_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x1, y1), ttnn.CoreCoord(x2, y2)))
        cnt += 1
        if cnt >= num_batches:
            break

    return ttnn.NdShardSpec(shard_shape, ttnn.CoreRangeSet(core_ranges), orientation)


def test_moreh_bug_report_rank_crta(device):
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # shape = [1, 1, 32 * 1, 32 * 1]
    shape = [56, 3, 32 * 6, 32 * 2]
    input_shard_shape = [1, 1, 32 * 1, 32 * 2]
    other_shard_shape = [1, 1, 32 * 3, 32 * 1]
    output_shard_shape = [1, 1, 32 * 2, 32 * 2]

    input_dtype = ttnn.bfloat16
    other_dtype = ttnn.bfloat16
    output_dtype = ttnn.bfloat16
    cpu_dtype = torch.bfloat16

    # torch_input = torch.ones(shape, dtype=cpu_dtype)
    # torch_other = torch.ones(shape, dtype=cpu_dtype)

    torch_input = torch.rand(shape, dtype=cpu_dtype)
    torch_other = torch.rand(shape, dtype=cpu_dtype)
    torch_output = torch.empty(shape, dtype=cpu_dtype)

    input_shard_spec = nd_shard_spec(shape, input_shard_shape)
    other_shard_spec = nd_shard_spec(shape, other_shard_shape)
    output_shard_spec = nd_shard_spec(shape, output_shard_shape)

    input_memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, input_shard_spec)
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=input_dtype,
        layout=ttnn.TILE_LAYOUT,
        pad_value=float("nan") if input_dtype is not ttnn.bfloat8_b else float("0"),
        memory_config=input_memory_config,
    )

    other_memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, other_shard_spec)
    ttnn_other = ttnn.from_torch(
        torch_other,
        device=device,
        dtype=other_dtype,
        layout=ttnn.TILE_LAYOUT,
        pad_value=float("nan") if other_dtype is not ttnn.bfloat8_b else float("0"),
        memory_config=other_memory_config,
    )

    output_memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, output_shard_spec)
    ttnn_output = ttnn.from_torch(
        torch_output,
        device=device,
        dtype=output_dtype,
        layout=ttnn.TILE_LAYOUT,
        pad_value=float("nan") if output_dtype is not ttnn.bfloat8_b else float("0"),
        memory_config=output_memory_config,
    )

    ttnn_output = ttnn.operations.moreh.bug_report_rank_crta(
        ttnn_input,
        ttnn_other,
        output=ttnn_output,
        compute_kernel_config=compute_kernel_config,
    )

    torch.add(torch_input, torch_other, out=torch_output)
    ttnn_cpu_output = ttnn.to_torch(ttnn_output)
    passing, out = comp_allclose(
        torch_output,
        ttnn_cpu_output,
        rtol=1e-2,
        atol=1e-3,
    )
    print(f"torch_output={torch_output}")
    print(f"ttnn_cpu_output={ttnn_cpu_output}")
    logger.info(f"passing={passing}")
    logger.info(f"out={out}")
    assert passing
