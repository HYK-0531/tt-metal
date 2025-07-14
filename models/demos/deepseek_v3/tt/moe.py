import torch
import torch.nn as nn

import ttnn
from models.demos.deepseek_v3.tt.mlp_expert import TT_MLP_Expert
from models.demos.deepseek_v3.tt.moe_gate import TT_MoE_Gate


class TT_MoE(nn.Module):
    def __init__(self, hf_config, state_dict, mesh_device, batch_size=32):
        super().__init__()
        self.hf_config = hf_config
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.batch_size = batch_size
        self.topk_experts = hf_config.num_experts_per_tok
        self.topk_group = hf_config.topk_group
        self.experts_per_device = 8
        self.num_experts = hf_config.n_routed_experts

        # Extract only gate-related weights from the full state_dict
        gate_state_dict = {
            key.replace("gate.", ""): value for key, value in state_dict.items() if key.startswith("gate.")
        }
        self.tt_moe_gate = TT_MoE_Gate(hf_config, gate_state_dict, mesh_device, batch_size)

        expert_state_dict = {
            key.replace("experts.0.", ""): value.T.repeat(self.experts_per_device, 1, 1).unsqueeze(0)
            for key, value in state_dict.items()
            if key.startswith("experts.0.")
        }
        self.tt_experts = TT_MLP_Expert(hf_config, expert_state_dict, mesh_device, batch_size)

        self.all_to_all_dispatch_output_tensors = ttnn.from_torch(
            torch.zeros([1, self.batch_size * 4, 1, 7168]),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.all_to_all_dispatch_metadata_tensors = ttnn.from_torch(
            torch.zeros([1, self.batch_size * 4, 1, self.experts_per_device], dtype=torch.int32),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        # Create a one hot vector of shape (self.batch_size*4, self.experts_per_device)
        self.expert_mapping_tensors = ttnn.from_torch(
            torch.eye(self.experts_per_device, dtype=torch.int32)
            .repeat_interleave(self.num_experts, dim=0)
            .unsqueeze(0)
            .unsqueeze(0),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.all_to_all_combine_output_tensors = ttnn.from_torch(
            torch.zeros([self.topk_experts, self.batch_size, 1, 7168]),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        compute_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
        subdevice_shard_cores_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(compute_grid[0] - 1, compute_grid[1] - 1),
                ),
            }
        )
        self.all_to_all_semaphore_handles = ttnn.create_global_semaphore(mesh_device, subdevice_shard_cores_grid, 0)
        self.combine_semaphore_handles = ttnn.create_global_semaphore(mesh_device, subdevice_shard_cores_grid, 0)

    def forward(self, tt_input):
        tt_top8_experts_weights, tt_top8_experts_indices = self.tt_moe_gate(tt_input)

        tt_row_major_input = ttnn.to_layout(tt_input, ttnn.ROW_MAJOR_LAYOUT)
        tt_row_major_input = ttnn.reshape(tt_row_major_input, (self.batch_size, 1, 1, 7168))  # Shape([32, 1, 1, 7168])

        tt_row_major_top8_experts_indices = ttnn.to_layout(tt_top8_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
        tt_row_major_top8_experts_indices = ttnn.reshape(
            tt_row_major_top8_experts_indices, (self.batch_size, 1, 1, self.topk_experts)
        )  # Shape([32, 1, 1, 8])
        ttnn.all_to_all_dispatch(
            tt_row_major_input,
            tt_row_major_top8_experts_indices,
            self.expert_mapping_tensors,
            cluster_axis=0,
            num_links=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            global_semaphore=self.all_to_all_semaphore_handles,
            subdevice_id=None,
            output_tensors=[self.all_to_all_dispatch_output_tensors, self.all_to_all_dispatch_metadata_tensors],
        )

        post_all_to_all_dispatch_output = ttnn.reshape(
            self.all_to_all_dispatch_output_tensors, (1, 1, self.batch_size * 4, 7168)
        )
        post_all_to_all_dispatch_output = ttnn.to_layout(post_all_to_all_dispatch_output, ttnn.TILE_LAYOUT)
        post_all_to_all_dispatch_output = ttnn.repeat(
            post_all_to_all_dispatch_output, ttnn.Shape((1, self.experts_per_device, 1, 1))
        )
        experts_output = self.tt_experts(post_all_to_all_dispatch_output)
        ttnn.deallocate(post_all_to_all_dispatch_output)
        experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
        experts_output = ttnn.reshape(experts_output, (self.experts_per_device, self.batch_size * 4, 1, 7168))
        # Combine the experts output Shape([8, 32, 1, 7168])
        before_combine = ttnn.to_torch(ttnn.get_device_tensors(experts_output)[0])
        if torch.any(before_combine > 10000000):
            print("experts_output has values greater than 10000000")
            print(before_combine[torch.where(before_combine > 10000000)])
        combine_out_tensor = ttnn.all_to_all_combine(
            experts_output,
            self.expert_mapping_tensors,
            self.all_to_all_dispatch_metadata_tensors,
            num_links=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            global_semaphore=self.combine_semaphore_handles,
            axis=0,
            optional_output_tensor=self.all_to_all_combine_output_tensors,
        )
        after_combine = ttnn.to_torch(ttnn.get_device_tensors(combine_out_tensor)[0])
        if torch.any(after_combine > 10000000):
            print("all_to_all_combine_output_tensors has values greater than 10000000")
            print(after_combine[torch.where(after_combine > 10000000)])
        combine_out_tensor = ttnn.reshape(
            self.all_to_all_combine_output_tensors, (1, self.experts_per_device, self.batch_size, 7168)
        )
        combine_out_tensor = ttnn.to_layout(combine_out_tensor, ttnn.TILE_LAYOUT)
        moe_output_tensor = ttnn.sum(combine_out_tensor, dim=1, keepdim=True)
        return moe_output_tensor
