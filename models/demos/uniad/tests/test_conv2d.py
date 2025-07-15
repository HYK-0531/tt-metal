# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv

# @pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
# @pytest.mark.parametrize("stride", [2])
# @pytest.mark.parametrize("batch_size", [2])
# @pytest.mark.parametrize(
#     "output_channels, input_channels, input_height, input_width, shard_layout, config",
#     (
#         (353, 384, 8, 8, WS, None),
#         (128, 128, 32, 32, BS, None),
#         (16, 16, 256, 256, HS, {"act_block_h": 32}),
#     ),
# )
# @pytest.mark.parametrize(
#     "weights_dtype",
#     [None, ttnn.bfloat16],
# )
# @pytest.mark.parametrize(
#     "output_dtype",
#     [ttnn.bfloat8_b, ttnn.bfloat16],
# )
# @pytest.mark.parametrize(
#     "input_dtype",
#     [ttnn.bfloat8_b,  ttnn.float32],
# )
# @pytest.mark.parametrize(
#     "fp32_accum",
#     [True, False],
# )
# @pytest.mark.parametrize(
#     "packer_l1_acc",
#     [False],
# )
# @pytest.mark.parametrize(
#     "filter, padding",
#     [
#         [3, (1, 2, 2, 3)],
#         [1, 0],
#         [5, (2, 4, 3, 5)],
#     ],
# )
# @pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
# @pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
# def test_conv_features(
#     device,
#     torch_tensor_map,
#     use_program_cache,
#     math_fidelity,
#     output_dtype,
#     weights_dtype,
#     batch_size,
#     output_channels,
#     input_channels,
#     input_height,
#     input_width,
#     shard_layout,
#     config,
#     filter,
#     stride,
#     padding,
#     output_layout,
#     fp32_accum,
#     packer_l1_acc,
#     input_dtype,
# ):
#     if output_layout == ttnn.ROW_MAJOR_LAYOUT and shard_layout == WS:
#         pytest.skip("Bug in Width Sharded Row Major Tensor Creation when height%32!=0. #19408")

#     if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat8_b:
#         pytest.skip("Row major layout not compatible with bfloat8_b")

#     if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat16 and packer_l1_acc and fp32_accum:
#         pytest.skip("skipping due to pack_untilize_dst issue!")

#     run_conv(
#         device,
#         torch_tensor_map,
#         math_fidelity,
#         output_dtype,
#         weights_dtype,
#         batch_size,
#         output_channels,
#         input_channels,
#         input_height,
#         input_width,
#         filter,
#         filter,
#         stride,
#         stride,
#         padding,
#         config,
#         shard_layout=shard_layout,
#         output_layout=output_layout,
#         has_bias=True,
#         fp32_accum=fp32_accum,
#         packer_l1_acc=packer_l1_acc,
#         preprocess_weights_on_device=True,
#         run_twice=True,
#         input_layout=ttnn.TILE_LAYOUT if input_dtype == ttnn.bfloat8_b else None,
#         input_dtype=input_dtype,
#     )


SliceHeight = ttnn.Conv2dSliceHeight
SliceWidth = ttnn.Conv2dSliceWidth


@pytest.mark.parametrize(
    "input_layout, dtype",
    [[ttnn.TILE_LAYOUT, ttnn.bfloat8_b]],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "slice_type, num_slices",
    [
        (SliceHeight, 4),
        (SliceHeight, 8),
        (SliceHeight, 16),
        (SliceHeight, 32),
        (SliceWidth, 4),
        (SliceWidth, 8),
        (SliceWidth, 16),
        (SliceWidth, 32),
    ],
    ids=[
        "SliceHeight-4",
        "SliceHeight-8",
        "SliceHeight-16",
        "SliceHeight-32",
        "SliceWidth-4",
        "SliceWidth-8",
        "SliceWidth-16",
        "SliceWidth-32",
    ],
)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, weights_dtype, kernel, stride, padding, dilation, act_block_h_override,  math_fidelity",
    # fmt: off
    # input tensor is of type NCHW
    (
        (6, 3, 64, 928, 1600, ttnn.bfloat8_b, (7, 7), (2, 2), (3, 3), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 64, 64, 232, 400, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 64, 64, 232, 400, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 64, 256, 232, 400, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 64, 256, 232, 400, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 64, 232, 400, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 64, 64, 232, 400, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 64, 256, 232, 400, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 64, 232, 400, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 64, 64, 232, 400, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 64, 256, 232, 400, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 128, 232, 400, ttnn.bfloat8_b, (1, 1), (2, 2), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 128, 128, 116, 200, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 128, 512, 116, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 512, 232, 400, ttnn.bfloat8_b, (1, 1), (2, 2), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 512, 128, 116, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 128, 128, 116, 200, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 512, 128, 116, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 128, 128, 116, 200, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 128, 512, 116, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 128, 512, 116, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 512, 128, 116, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 128, 128, 116, 200, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 128, 512, 116, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 512, 256, 116, 200, ttnn.bfloat8_b, (1, 1), (2, 2), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),    # Failing
        (6, 512, 1024, 116, 200, ttnn.bfloat8_b, (1, 1), (2, 2), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),      # Failing
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),      # Failing
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),      # Failing
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),      # Failing
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),     # Failing
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 27, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 1024, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 512, 58, 100, ttnn.bfloat8_b, (1, 1), (2, 2), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 512, 27, 29, 50, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 512, 2048, 29, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 2048, 58, 100, ttnn.bfloat8_b, (1, 1), (2, 2), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 2048, 512, 29, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 512, 27, 29, 50, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 512, 2048, 29, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 2048, 512, 29, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 512, 27, 29, 50, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 512, 2048, 29, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 512, 256, 116, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 1024, 256, 58, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 2048, 256, 29, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 256, 116, 200, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 256, 58, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 256, 29, 50, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (6, 256, 256, 29, 50, ttnn.bfloat8_b, (3, 3), (2, 2), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 200, 200, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 200, 200, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 200, 200, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 200, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 128, 200, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 128, 200, 200, ttnn.bfloat8_b, (3, 3), (2, 2), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 256, 100, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 100, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 128, 100, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 128, 100, 100, ttnn.bfloat8_b, (3, 3), (2, 2), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 256, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 128, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 128, 50, 50, ttnn.bfloat8_b, (3, 3), (2, 2), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 256, 25, 25, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 25, 25, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 128, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 128, 50, 50, ttnn.bfloat8_b, (3, 3), (2, 2), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 256, 25, 25, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 25, 25, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 128, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 128, 50, 50, ttnn.bfloat8_b, (3, 3), (2, 2), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 256, 25, 25, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 25, 25, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 128, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 128, 50, 50, ttnn.bfloat8_b, (3, 3), (2, 2), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 256, 25, 25, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 25, 25, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 128, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 128, 50, 50, ttnn.bfloat8_b, (3, 3), (2, 2), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 256, 25, 25, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 25, 25, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 256, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (5, 256, 128, 100, 100, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (5, 128, 256, 100, 100, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (5, 256, 256, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (5, 256, 128, 200, 200, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (5, 128, 256, 200, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (5, 256, 256, 50, 50, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 128, 200, 200, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 256, 200, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 128, 200, 200, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 256, 200, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 256, 128, 200, 200, ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 0, ttnn.MathFidelity.LoFi),
        (1, 128, 256, 200, 200, ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1), 0, ttnn.MathFidelity.LoFi),
    )
    # fmt: on
)
@pytest.mark.parametrize(
    "has_bias, fp32_accum, packer_l1_acc",
    [[True, True, False]],
)
def test_conv_dram(
    device,
    torch_tensor_map,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    has_bias,
    weights_dtype,
    dtype,
    slice_type,
    num_slices,
    kernel,
    stride,
    padding,
    dilation,
    act_block_h_override,
    math_fidelity,
    fp32_accum,
    input_layout,
    packer_l1_acc,
):
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")
    config = {
        "act_block_h": act_block_h_override,
    }

    # Force width sharding when using SliceWidth to ensure compatibility with DRAM slicing
    # This is needed otherwise it does height sharding and throws this error:
    # RuntimeError: TT_FATAL @ /home/ttuser/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp:50: input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED
    # info: Input tensor must be width sharded
    # shard_layout = None
    # if slice_type == SliceWidth:
    #     shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    # elif slice_type == SliceHeight:
    #     shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding,
        config,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        input_dtype=dtype,
        input_layout=input_layout,
        output_layout=input_layout,
        run_twice=True,
        fast_compare=True,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=slice_type,
            num_slices=num_slices,
        ),
    )
