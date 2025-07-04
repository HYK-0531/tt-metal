# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import torch

import ttnn


def test_example(device):
    shape = [1 * 32, 32]
    # shape = [2 * 32, 32]

    torch_input = torch.ones(shape, dtype=torch.bfloat16) * -1.7014118e38
    torch_other = torch.zeros(shape, dtype=torch.bfloat16) * 0

    expected_output = torch_input * torch_other

    # There is a cumulative effect, if we run the two kernels inside the inner loop multiple times
    # at a certain iteration, which will introduce a specific number of nops at the top of the second
    # kernel's unpacker code, the output will be wrong. If we just run one iteration with that many
    # nops on top of the unpacker code, it will not fault. The number of nops required ranges between 19-28.
    # Starting from 17 empirically optimizes the probability of encountering the error.
    max_nops_required = 28
    optimal_start_point = 17
    iteration_count = 0
    for y in range(optimal_start_point, max_nops_required):
        for x in range(y, max_nops_required):
            print("iteration_count ", iteration_count)
            print("nops ", x)
            os.environ["UNOPS"] = str(x)
            tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
            tt_other = ttnn.from_torch(torch_other, layout=ttnn.TILE_LAYOUT, device=device)

            tt_output1 = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            tt_output = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            ttnn.prim.example(tt_other, tt_output1)
            ttnn.prim.example_multiple_return(tt_input, tt_output1, output=tt_output)

            actual_output = ttnn.to_torch(tt_output)

            ret = torch.allclose(actual_output, expected_output, atol=0, rtol=0)
            if not ret:
                print("FAILED")
                print("actual_output", actual_output.to(torch.float32).flatten().tolist())
                return

            iteration_count = iteration_count + 1
