# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
import os

from models.experimental.stable_diffusion_xl_base.tt.tt_geglu import TtGEGLU
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


class TtFeedForward(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
    ):
        super().__init__()

        self.device = device
        self.tt_geglu = TtGEGLU(device, state_dict, f"{module_path}.net.0", model_config)

        weights = state_dict[f"{module_path}.net.2.weight"].unsqueeze(0).unsqueeze(0)
        bias = state_dict[f"{module_path}.net.2.bias"]

        self.tt_weights, self.tt_bias = prepare_linear_params(device, weights, bias, model_config.ff_weights_dtype)
        self.ff2_model_config = model_config.get_matmul_config(f"{module_path}.net.2")
        assert self.ff2_model_config is not None, "ff2_model_config should not be None"
        self.default_compute_kernel_config = model_config.get_mm_compute_config(module_path)

    def forward(self, hidden_states):
        print("FF sync begin")
        ttnn.synchronize_device(self.device)
        print("FF sync end")

        if hidden_states.shape[2] == 1024:
            print("Setting env variable")
            os.environ["TT_MM_THROTTLE_PERF"] = "5"

        hidden_states = self.tt_geglu(hidden_states)

        print("Post GEGLU sync begin")
        ttnn.synchronize_device(self.device)
        print("Post GEGLU sync end")

        print(f"Feedforward begin linear, shapes: {hidden_states.shape} x {self.tt_weights.shape}")
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_weights,
            bias=self.tt_bias,
            program_config=self.ff2_model_config,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )
        print(f"Feedforward linear sync begin")
        ttnn.synchronize_device(self.device)
        print(f"Feedforward linear sync end")
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

        if hidden_states.shape[2] == 1024:
            print("Deleting env variable")
            del os.environ["TT_MM_THROTTLE_PERF"]

        return hidden_states
