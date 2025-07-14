# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


def test_example(device):
    torch.manual_seed(0)

    shape = [56 * 32, 32]

    torch_input = torch.ones(shape, dtype=torch.bfloat16) * -1.7014118e38
    torch_other = torch.zeros(shape, dtype=torch.bfloat16) * 0

    expected_output = torch_input * torch_other

    for x in range(50000):
        print("iter", x)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
        tt_other = ttnn.from_torch(torch_other, layout=ttnn.TILE_LAYOUT, device=device)

        tt_output1 = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn.prim.example(tt_other, tt_output1)
        ttnn.prim.example_multiple_return(tt_input, tt_output1, output=tt_output)

        actual_output = ttnn.to_torch(tt_output)

        ret = torch.allclose(actual_output, expected_output, atol=0, rtol=0)
        if not ret:
            print("actual_output", actual_output[:,0])
            print("err iter ", x)
            break
        assert torch.allclose(actual_output, expected_output, atol=0, rtol=0)