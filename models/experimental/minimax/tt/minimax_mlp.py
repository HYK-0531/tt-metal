from torch import nn

import ttnn

from models.helper_funcs import Linear as TTLinear
from models.utility_functions import torch_to_tt_tensor_rm

from .minimax_utils import get_activation_fn


class TTMiniMaxM1MLP(nn.Module):
    def __init__(self, config, state_dict, base_address, device=None):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.device = device

        self.tt_gate_proj_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.gate_proj.weight"], self.device)
        self.tt_gate_proj = TTLinear(
            self.tt_gate_proj_weight.padded_shape[-1],
            self.tt_gate_proj_weight.padded_shape[-2],
            self.tt_gate_proj_weight,
        )

        self.tt_up_proj_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.up_proj.weight"], self.device)
        self.tt_up_proj = TTLinear(
            self.tt_up_proj_weight.padded_shape[-1], self.tt_up_proj_weight.padded_shape[-2], self.tt_up_proj_weight
        )

        self.tt_down_proj_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.down_proj.weight"], self.device)
        self.tt_down_proj = TTLinear(
            self.tt_down_proj_weight.padded_shape[-1],
            self.tt_down_proj_weight.padded_shape[-2],
            self.tt_down_proj_weight,
        )

        self.act_fn = get_activation_fn[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(ttnn.multiply(self.gate_proj(x)), self.up_proj(x)))
        return down_proj


class TTMiniMaxM1BlockSparseTop2MLP(nn.Module):
    def __init__(self, config, state_dict, base_address, device=None):
        super().__init__()

        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.device = device

        self.tt_w1_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.w1.weight"], self.device)
        self.tt_w1 = TTLinear(self.tt_w1_weight.padded_shape[-1], self.tt_w1_weight.padded_shape[-2], self.tt_w1_weight)

        self.tt_w2_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.w2.weight"], self.device)
        self.tt_w2 = TTLinear(self.tt_w2_weight.padded_shape[-1], self.tt_w2_weight.padded_shape[-2], self.tt_w2_weight)

        self.tt_w3_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.w3.weight"], self.device)
        self.tt_w3 = TTLinear(self.tt_w3_weight.padded_shape[-1], self.tt_w3_weight.padded_shape[-2], self.tt_w3_weight)

        self.act_fn = get_activation_fn(config.hidden_act)

    def forward(self, hidden_states):
        current_hidden_states = ttnn.multiply(self.act_fn(self.tt_w1(hidden_states)), self.tt_w3(hidden_states))
        current_hidden_states = self.tt_w2(current_hidden_states)
        return current_hidden_states
