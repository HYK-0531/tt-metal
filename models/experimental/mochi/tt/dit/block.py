from typing import Optional, Tuple, Callable

import ttnn
from models.common.lightweightmodule import LightweightModule

from models.experimental.mochi.tt.common import col_parallel_linear
from models.experimental.mochi.tt.dit.attention import AsymmetricAttention
from models.experimental.mochi.tt.dit.mlp import FeedForward
from models.experimental.mochi.tt.dit.norms import modulated_rmsnorm, residual_tanh_gated_rmsnorm


class AsymmetricJointBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        layer_num,
        dtype,
        hidden_size_x: int,
        hidden_size_y: int,
        num_heads: int,
        *,
        mlp_ratio_x: float = 8.0,  # Ratio of hidden size to d_model for MLP for visual tokens
        mlp_ratio_y: float = 4.0,  # Ratio of hidden size to d_model for MLP for text tokens
        update_y: bool = True,  # Whether to update text tokens in this block
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        **block_kwargs,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.state_dict = state_dict
        self.state_dict_prefix = state_dict_prefix
        self.weight_cache_path = weight_cache_path
        self.layer_num = layer_num
        self.dtype = dtype

        self.update_y = update_y
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y

        # Create modulation layers with weights and biases
        self.mod_x, self.mod_x_bias = col_parallel_linear(
            "mod_x",
            bias=True,
            weight_cache_path=weight_cache_path,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            mesh_device=mesh_device,
        )

        self.mod_y, self.mod_y_bias = col_parallel_linear(
            "mod_y",
            bias=True,
            weight_cache_path=weight_cache_path,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            mesh_device=mesh_device,
        )

        # Self-attention
        self.attn = AsymmetricAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}.attn",
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            dim_x=hidden_size_x,
            dim_y=hidden_size_y,
            num_heads=num_heads,
            update_y=update_y,
            **block_kwargs,
        )

        # MLP layers using FeedForward
        mlp_hidden_dim_x = int(hidden_size_x * mlp_ratio_x)
        self.mlp_x = FeedForward(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            in_features=hidden_size_x,
            hidden_size=mlp_hidden_dim_x,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            state_dict_prefix=f"{state_dict_prefix}.mlp_x",
            seq_shard=True,
        )

        if self.update_y:
            mlp_hidden_dim_y = int(hidden_size_y * mlp_ratio_y)
            self.mlp_y = FeedForward(
                mesh_device=mesh_device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=layer_num,
                dtype=dtype,
                in_features=hidden_size_y,
                hidden_size=mlp_hidden_dim_y,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
                state_dict_prefix=f"{state_dict_prefix}.mlp_y",
                seq_shard=False,
            )

    def dealloc(self):
        ttnn.deallocate(self.mod_x)
        if self.mod_x_bias is not None:
            ttnn.deallocate(self.mod_x_bias)
        ttnn.deallocate(self.mod_y)
        if self.mod_y_bias is not None:
            ttnn.deallocate(self.mod_y_bias)
        self.attn.dealloc()
        self.mlp_x.dealloc()
        if self.update_y:
            self.mlp_y.dealloc()

    def prefetch_weights(self, ccl_semaphore_handles, ccl_sub_device_id, topology):
        mlp_weights = self.mlp_x.prefetch_weights(ccl_semaphore_handles, ccl_sub_device_id, topology)
        attn_weights = self.attn.prefetch_weights(ccl_semaphore_handles, ccl_sub_device_id, topology)

        # Synchronize to ensure fabric is not in use when this function returns
        fsdp_event = ttnn.record_event(self.mesh_device, 0, sub_device_ids=[ccl_sub_device_id])
        ttnn.wait_for_event(0, fsdp_event)
        return {**mlp_weights, **attn_weights}

    def ff_block_x(
        self,
        x_1BNX: ttnn.Tensor,
        scale_x_B11X: ttnn.Tensor,
        gate_x_B11X: ttnn.Tensor,
        ccl_semaphore_handles: dict,
        worker_sub_device_id: ttnn.SubDeviceId,
        topology: ttnn.Topology,
        fsdp_weights: dict,
    ) -> ttnn.Tensor:
        """Feed-forward block for visual features.

        Args:
            x_1BNX: Input tensor of shape (1, B, N, X)
            scale_x_B11X: Scale tensor of shape (B, 1, 1, X)
            gate_x_B11X: Gate tensor of shape (B, 1, 1, X)

        Returns:
            Tensor of shape (1, B, N, X)
        """
        x_mod_1BNX = modulated_rmsnorm(x_1BNX, scale_x_B11X)
        x_res_shard_1BNX = self.mlp_x(
            x_mod_1BNX,
            ccl_semaphore_handles=ccl_semaphore_handles,
            worker_sub_device_id=worker_sub_device_id,
            topology=topology,
            fsdp_weights=fsdp_weights,
        )
        x_1BNX = residual_tanh_gated_rmsnorm(x_1BNX, x_res_shard_1BNX, gate_x_B11X)
        return x_1BNX

    def ff_block_y(
        self,
        y_1BLY: ttnn.Tensor,
        scale_y_B11Y: ttnn.Tensor,
        gate_y_B11Y: ttnn.Tensor,
        ccl_semaphore_handles: dict,
        worker_sub_device_id: ttnn.SubDeviceId,
        topology: ttnn.Topology,
    ) -> ttnn.Tensor:
        """Feed-forward block for text features.

        Args:
            y_1BLY: Input tensor of shape (1, B, L, Y)
            scale_y_B11Y: Scale tensor of shape (B, 1, 1, Y)
            gate_y_B11Y: Gate tensor of shape (B, 1, 1, Y)

        Returns:
            Tensor of shape (1, B, L, Y)
        """
        y_mod_1BLY = modulated_rmsnorm(y_1BLY, scale_y_B11Y)
        y_res_shard_1BLY = self.mlp_y(
            y_mod_1BLY,
            ccl_semaphore_handles=ccl_semaphore_handles,
            worker_sub_device_id=worker_sub_device_id,
            topology=topology,
        )
        if self.num_devices > 1:
            # Collect hidden-dim-fractured MLP outputs
            L = y_1BLY.shape[2]
            y_res_1BLY = ttnn.experimental.all_gather_async(
                y_res_shard_1BLY,
                dim=3,
                multi_device_global_semaphore=ccl_semaphore_handles["ff_block_y"],
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                subdevice_id=worker_sub_device_id,
            )
            y_res_1BLY = ttnn.reshape(y_res_1BLY, (1, 1, L, self.hidden_size_y), y_res_1BLY.shape)
        else:
            y_res_1BLY = y_res_shard_1BLY
        y_1BLY = residual_tanh_gated_rmsnorm(y_1BLY, y_res_1BLY, gate_y_B11Y)
        return y_1BLY

    def forward(
        self,
        x: ttnn.Tensor,
        c: ttnn.Tensor,
        y: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        N: int,
        ccl_semaphore_handles: dict,
        worker_sub_device_id: ttnn.SubDeviceId,
        ccl_sub_device_id: ttnn.SubDeviceId,
        topology: ttnn.Topology,
        persistent_buffers: dict,
        fsdp_weights: dict,
        prefetch_weights_fn: Callable,
        uncond: bool = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Forward pass of a block.
        Shape metadata:
            B: batch
            N: vision sequence length
            M: sharded vision sequence length
            L: text sequence length
            H: number of heads
            D: head dim
            X: visual hidden dim
            Y: text hidden dim
            Z: 4 * X
            C: 4 * Y
        """
        x_1BMX = x
        c_B11X = c
        y_1BLY = y
        rope_cos_1HMD = rope_cos
        rope_sin_1HMD = rope_sin

        M = x_1BMX.shape[2]

        # Set up compute kernel config for high-fidelity computations
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Apply modulation
        # c_B11X = ttnn.silu(c_B11X)

        # Apply linear layers with bias
        mod_x_B11Z = ttnn.linear(
            c_B11X,
            self.mod_x,
            bias=self.mod_x_bias,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=7, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # mod_x_B11Z = ttnn.all_gather(mod_x_B11Z, dim=3)
        mod_x_B11Z = ttnn.experimental.all_gather_async(
            mod_x_B11Z,
            dim=3,
            multi_device_global_semaphore=ccl_semaphore_handles["mod_x"],
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            subdevice_id=worker_sub_device_id,
        )

        scale_msa_x_B11X = ttnn.slice(mod_x_B11Z, [0, 0, 0, 0], [1, 1, 1, self.hidden_size_x])
        gate_msa_x_B11X = ttnn.slice(mod_x_B11Z, [0, 0, 0, self.hidden_size_x], [1, 1, 1, 2 * self.hidden_size_x])
        scale_mlp_x_B11X = ttnn.slice(mod_x_B11Z, [0, 0, 0, 2 * self.hidden_size_x], [1, 1, 1, 3 * self.hidden_size_x])
        gate_mlp_x_B11X = ttnn.slice(mod_x_B11Z, [0, 0, 0, 3 * self.hidden_size_x], [1, 1, 1, 4 * self.hidden_size_x])

        scale_msa_y_B11Y = None
        if not uncond:
            mod_y_B11C = ttnn.linear(
                c_B11X,
                self.mod_y,
                bias=self.mod_y_bias,
                compute_kernel_config=compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=7, x=8),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            mod_y_B11C = ttnn.experimental.all_gather_async(
                mod_y_B11C,
                dim=3,
                multi_device_global_semaphore=ccl_semaphore_handles["mod_y"],
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                subdevice_id=worker_sub_device_id,
            )
            if self.update_y:
                scale_msa_y_B11Y = ttnn.slice(mod_y_B11C, [0, 0, 0, 0], [1, 1, 1, self.hidden_size_y])
                gate_msa_y_B11Y = ttnn.slice(
                    mod_y_B11C, [0, 0, 0, self.hidden_size_y], [1, 1, 1, 2 * self.hidden_size_y]
                )
                scale_mlp_y_B11Y = ttnn.slice(
                    mod_y_B11C, [0, 0, 0, 2 * self.hidden_size_y], [1, 1, 1, 3 * self.hidden_size_y]
                )
                gate_mlp_y_B11Y = ttnn.slice(
                    mod_y_B11C, [0, 0, 0, 3 * self.hidden_size_y], [1, 1, 1, 4 * self.hidden_size_y]
                )

            else:
                scale_msa_y_B11Y = mod_y_B11C

        # Self-attention block
        x_attn_1BMX, y_attn_shard_1BLY, next_fsdp_weights = self.attn(
            x_1BMX,
            y_1BLY,
            N=N,
            scale_x=scale_msa_x_B11X,
            scale_y=scale_msa_y_B11Y,
            rope_cos=rope_cos_1HMD,
            rope_sin=rope_sin_1HMD,
            trans_mat=trans_mat,
            uncond=uncond,
            ccl_semaphore_handles=ccl_semaphore_handles,
            worker_sub_device_id=worker_sub_device_id,
            ccl_sub_device_id=ccl_sub_device_id,
            topology=topology,
            persistent_buffers=persistent_buffers,
            fsdp_weights=fsdp_weights,
            prefetch_weights_fn=prefetch_weights_fn,
        )

        assert x_attn_1BMX.shape[2] == M
        x_1BMX = residual_tanh_gated_rmsnorm(x_1BMX, x_attn_1BMX, gate_msa_x_B11X)
        # MLP block
        x_1BMX = self.ff_block_x(
            x_1BMX,
            scale_mlp_x_B11X,
            gate_mlp_x_B11X,
            ccl_semaphore_handles=ccl_semaphore_handles,
            worker_sub_device_id=worker_sub_device_id,
            topology=topology,
            fsdp_weights=fsdp_weights,
        )

        if not uncond:
            if self.num_devices > 1:
                # Collect hidden-dim-fractured attention outputs
                L = y_1BLY.shape[2]
                y_attn_1BLY = ttnn.experimental.all_gather_async(
                    y_attn_shard_1BLY,
                    dim=3,
                    multi_device_global_semaphore=ccl_semaphore_handles["y_attn"],
                    num_links=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=topology,
                    subdevice_id=worker_sub_device_id,
                )
                y_attn_1BLY = ttnn.reshape(y_attn_1BLY, (1, 1, L, self.hidden_size_y), y_attn_1BLY.shape)
            else:
                y_attn_1BLY = y_attn_shard_1BLY

            if self.update_y:
                y_1BLY = residual_tanh_gated_rmsnorm(y_1BLY, y_attn_1BLY, gate_msa_y_B11Y)
                y_1BLY = self.ff_block_y(
                    y_1BLY,
                    scale_mlp_y_B11Y,
                    gate_mlp_y_B11Y,
                    ccl_semaphore_handles=ccl_semaphore_handles,
                    worker_sub_device_id=worker_sub_device_id,
                    topology=topology,
                )
            else:
                y_1BLY = y_attn_1BLY

        return x_1BMX, y_1BLY, next_fsdp_weights
