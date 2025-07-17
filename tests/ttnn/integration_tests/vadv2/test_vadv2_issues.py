# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import disable_persistent_kernel_cache, profiler


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_nonzero(
    device,
    reset_seeds,
):
    torch_input = torch.tensor([[[[0, 4, 0, 2, 4, 0, 3]]]])

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device)
    for i in range(2):
        output_indices, output_tensor = ttnn.nonzero(ttnn_input)
        ttnn.deallocate(output_indices)
        ttnn.deallocate(output_tensor)
