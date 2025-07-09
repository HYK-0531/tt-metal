import torch
from torch import nn

import ttnn

from models.helper_funcs import Linear as TTLinear

from .mlp import TTMiniMaxM1BlockSparseTop2MLP

BLOCK = 256


class TTMiniMaxM1SparseMoeBlock(nn.Module):
    def __init__(self, config, state_dict, base_address, device=None):
        super(TTMiniMaxM1SparseMoeBlock, self).__init__()

        self.device = device
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise

        # Gating
        self.tt_gate_weight = ttnn.from_torch(state_dict[f"{base_address}.gate.weight"], device=self.device)
        self.tt_gate = TTLinear(
            self.tt_gate_weight.padded_shape[-1], self.tt_gate_weight.padded_shape[-2], self.tt_gate_weight
        )

        self.experts = nn.ModuleList(
            [
                TTMiniMaxM1BlockSparseTop2MLP(config, state_dict, f"{base_address}.experts.{i}")
                for i in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = ttnn.view(hidden_states, (batch_size * seq_len, hidden_dim))

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.tt_gate(hidden_states)

        routing_weights = ttnn.softmax(router_logits, dim=1)
        routing_weights, selected_experts = ttnn.topk(routing_weights, self.top_k, dim=-1)
        # routing_weights /= ttnn.sum(routing_weights, dim=-1, keepdim=True)
        routing_weights = ttnn.div(routing_weights, ttnn.sum(routing_weights, dim=-1))

        final_hidden_states = ttnn.zeros(
            (batch_size * seq_len, hidden_dim), dtype=hidden_states.dtype, device=self.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = ttnn.frmo_torch(
            torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        )

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            idx = ttnn.from_torch(idx)
            top_x = ttnn.from_torch(top_x)

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = ttnn.multiply(expert_layer(current_state), routing_weights[top_x, idx, None])

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.

            # TODO: THIS LINE
            final_hidden_states.index_add_(0, top_x, current_hidden_states)

        final_hidden_states = ttnn.reshape(final_hidden_states, (batch_size, seq_len, hidden_dim))
        return final_hidden_states, router_logits
