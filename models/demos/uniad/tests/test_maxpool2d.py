# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Tuple

import pytest

import ttnn
from models.utility_functions import is_blackhole
from tests.sweep_framework.sweep_utils.max_pool2d_common import run_max_pool2d

parameters = {
    "max_pool2d_short_sweep_suite": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [6, 64, 464, 800, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 256, 200, 200, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 100, 100, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
        ],
    },
    "test_run_max_pool": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [6, 64, 464, 800, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 256, 200, 200, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 100, 100, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
        ],
    },
    "test_run_max_pool_width_shard": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "in_place": [True, False],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [6, 64, 464, 800, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 256, 200, 200, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 100, 100, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
        ],
    },
    "test_run_max_pool_height_shard": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "in_place": [True, False],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [6, 64, 464, 800, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 256, 200, 200, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 100, 100, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
        ],
    },
    "test_run_max_pool_block_shard": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "in_place": [True, False],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [6, 64, 464, 800, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 256, 200, 200, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 100, 100, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
        ],
    },
    "test_run_max_pool_mem_config": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [6, 64, 464, 800, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 256, 200, 200, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 100, 100, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 50, 50, 2, 2, 2, 2, 0, 0, 1, 1, False],
        ],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None


def run(
    input_specs,
    dtype,
    *,
    device,
):
    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_specs
    sharding = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    return run_max_pool2d(
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding,
        ceil_mode,
    )


import pytest


@pytest.mark.parametrize("input_spec", parameters["max_pool2d_short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["max_pool2d_short_sweep_suite"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_localrun(device, dtype, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool_height_shard"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool_height_shard"]["dtype"])
@pytest.mark.parametrize("in_place", parameters["test_run_max_pool_height_shard"]["in_place"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_localrun(device, dtype, in_place, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    if (kernel_height > 5 or kernel_width > 5) and in_place and dtype == ttnn.bfloat8_b:
        pytest.skip("this case runs out of memory due to combination of large remote temp CB and large untilize out CB")
    if input_spec[:4] == [1, 512, 10, 10] and in_place and dtype == ttnn.bfloat8_b and is_blackhole():
        pytest.skip(
            "this case runs out of memory on blackhole due to large remote temp CB, this is only an issue on blackhole since the larger number of cores results in a smaller nhe per core which results in more remote references and hence a larger remote temp CB"
        )
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
        in_place=in_place,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool(device, dtype, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool_width_shard"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool_width_shard"]["dtype"])
@pytest.mark.parametrize("in_place", parameters["test_run_max_pool_width_shard"]["in_place"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool_width_shard(device, dtype, in_place, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ceil_mode=ceil_mode,
        in_place=in_place,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool_block_shard"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool_block_shard"]["dtype"])
@pytest.mark.parametrize("in_place", parameters["test_run_max_pool_block_shard"]["in_place"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool_block_shard(device, dtype, in_place, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ceil_mode=ceil_mode,
        in_place=in_place,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool_mem_config"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool_mem_config"]["dtype"])
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool_mem_config(device, dtype, input_spec, memory_config):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        ceil_mode=ceil_mode,
        memory_config=memory_config,
    )
