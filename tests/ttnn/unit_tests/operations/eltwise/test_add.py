# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("hw", [(32, 64)])
def test_add_2D_tensors(mesh_device, hw):
    torch_input_tensor_a = torch.rand(hw, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(hw, dtype=torch.bfloat16)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    for i in range(10000):
        ttnn.add(input_tensor_a, input_tensor_b)
        ttnn.synchronize_device(mesh_device)
        if i % 1000 == 0:
            print(f"Iteration {i} completed")


# @pytest.mark.parametrize("hw", [(32, 64)])
# @pytest.mark.parametrize("scalar", [0.42])
# def test_add_scalar(device, hw, scalar):
#     torch_input_tensor_a = torch.rand(hw, dtype=torch.bfloat16)
#     torch_output_tensor = scalar + torch_input_tensor_a

#     input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
#     output = input_tensor_a + scalar
#     output = ttnn.to_torch(output)

#     assert_with_pcc(torch_output_tensor, output, 0.9999)
