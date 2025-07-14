import torch
import torch.nn as nn

import ttnn


class TT_MoE_Gate(nn.Module):
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

        self.tt_linear_weights = ttnn.from_torch(
            state_dict["weight"].T.unsqueeze(0).unsqueeze(0),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.tt_bias_correction_weights = ttnn.from_torch(
            state_dict["e_score_correction_bias"].repeat(self.batch_size, 1).unsqueeze(0).unsqueeze(0),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.tt_norm_eps = ttnn.from_torch(
            torch.tensor([1e-20]).repeat(self.batch_size, self.topk_experts).unsqueeze(0).unsqueeze(0),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.tt_expert_scale = ttnn.from_torch(
            torch.tensor([hf_config.routed_scaling_factor])
            .repeat(self.batch_size, self.topk_experts)
            .unsqueeze(0)
            .unsqueeze(0),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

    def forward(self, tt_input):
        tt_logits = ttnn.linear(
            tt_input,
            self.tt_linear_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        # tt_scores = ttnn.sigmoid(tt_logits, vector_mode=4, fast_and_approximate_mode=False)
        tt_scores = ttnn.softmax(tt_logits, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        tt_logits.deallocate()

        tt_scores_with_bias = ttnn.add(
            tt_scores, self.tt_bias_correction_weights, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )
        tt_scores_grouped = ttnn.reshape(tt_scores_with_bias, (1, self.batch_size, 8, 32))
        tt_scores_grouped_padded = ttnn.pad(
            tt_scores_grouped, [(0, 0), (0, 0), (0, 0), (0, 64 - 32)], value=-float("inf")
        )
        ttnn.deallocate(tt_scores_grouped)
        ttnn_top2_values, ttnn_top2_indices = ttnn.topk(tt_scores_grouped_padded, 2, dim=3, largest=True, sorted=True)
        ttnn.deallocate(tt_scores_grouped_padded)
        ttnn.deallocate(ttnn_top2_indices)
        ttnn_group_scores = ttnn.sum(ttnn_top2_values, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ttnn_top2_values)
        ttnn_group_scores = ttnn.reshape(ttnn_group_scores, (1, 1, self.batch_size, 8))
        ttnn_group_scores = ttnn.pad(ttnn_group_scores, [(0, 0), (0, 0), (0, 0), (0, 64 - 8)], value=-float("inf"))
        ttnn_group_top4_values, ttnn_group_top4_indices = ttnn.topk(
            ttnn_group_scores, k=4, dim=3, largest=True, sorted=True
        )
        ttnn.deallocate(ttnn_group_scores)
        ttnn.deallocate(ttnn_group_top4_values)

        torch_inf_mask = torch.full((1, 1, self.batch_size, 8), -float("inf"))
        torch_ones_tensor = torch.ones((1, 1, self.batch_size, 4))
        tt_group_mask = ttnn.from_torch(
            torch_inf_mask,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_ones_tensor = ttnn.from_torch(
            torch_ones_tensor,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        tt_group_mask = ttnn.experimental.scatter(tt_group_mask, 3, ttnn_group_top4_indices, tt_ones_tensor)
        ttnn.deallocate(tt_ones_tensor)
        ttnn.deallocate(ttnn_group_top4_indices)

        tt_group_mask = ttnn.reshape(tt_group_mask, (1, self.batch_size, 8, 1))
        tt_scores_mask = ttnn.repeat(tt_group_mask, ttnn.Shape((1, 1, 1, 32)))
        ttnn.deallocate(tt_group_mask)
        tt_scores_mask = ttnn.reshape(tt_scores_mask, (1, 1, self.batch_size, 256))
        tt_scores_with_bias = ttnn.mul(tt_scores_with_bias, tt_scores_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tt_scores_mask)
        tt_top8_temp_values, tt_top8_experts_indices = ttnn.topk(
            tt_scores_with_bias, k=self.topk_experts, dim=3, largest=True, sorted=True
        )
        ttnn.deallocate(tt_scores_with_bias)
        ttnn.deallocate(tt_top8_temp_values)
        tt_top8_experts_weights = ttnn.experimental.gather(tt_scores, dim=3, index=tt_top8_experts_indices)
        ttnn.deallocate(tt_scores)

        denominator = ttnn.sum(tt_top8_experts_weights, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG, keepdim=True)
        tt_top8_experts_weights = ttnn.div(tt_top8_experts_weights, denominator)
        ttnn.deallocate(denominator)

        tt_top8_experts_weights = ttnn.add(
            tt_top8_experts_weights, self.tt_norm_eps, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )
        tt_top8_experts_weights = ttnn.mul(
            tt_top8_experts_weights, self.tt_expert_scale, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )

        return tt_top8_experts_weights, tt_top8_experts_indices
