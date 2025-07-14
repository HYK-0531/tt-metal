import torch.nn as nn

import ttnn


class TT_MLP_Expert(nn.Module):
    def __init__(self, hf_config, state_dict, mesh_device, batch_size=32):
        super().__init__()
        self.hf_config = hf_config
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.batch_size = batch_size
        self.topk_experts = hf_config.num_experts_per_tok
        self.topk_group = hf_config.topk_group

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.tt_gate_proj_weights = ttnn.from_torch(
            state_dict["gate_proj.weight"],
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.tt_up_proj_weights = ttnn.from_torch(
            state_dict["up_proj.weight"],
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.tt_down_proj_weights = ttnn.from_torch(
            state_dict["down_proj.weight"],
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

    def forward(self, tt_input):
        tt_gate_proj = ttnn.linear(
            tt_input,
            self.tt_gate_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        tt_up_proj = ttnn.linear(
            tt_input,
            self.tt_up_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        activated_gate_proj = ttnn.mul(
            tt_gate_proj,
            tt_up_proj,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(tt_gate_proj)
        ttnn.deallocate(tt_up_proj)

        tt_down_proj = ttnn.linear(
            activated_gate_proj,
            self.tt_down_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        return tt_down_proj
