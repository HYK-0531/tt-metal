# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def test(device):
    a = torch.tensor([[100, 100]], dtype=torch.long, device="cpu")
    b = ttnn.from_torch(a, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    bs = 2
    num_heads = 8
    embed_dims = 256

    value_l_ = ttnn.from_torch(
        torch.randn(2, 256, 10000), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    for level, (H_, W_) in enumerate(b):
        reshaped = ttnn.reshape(value_l_, [bs * num_heads, embed_dims, H_, W_], device=device)
        print(f"Reshaped tensor shape: {ttnn.shape(reshaped)}")
