from torch import nn

import ttnn

from models.helper_funcs import Linear as TTLinear
from models.utility_functions import torch_to_tt_tensor_rm


class TTglu(nn.Module):
    def __init__(self, state_dict, base_address, device=None):
        super(TTglu, self).__init__()
        self.device = device

        self.linear_1_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.l1.weight"], self.device)
        self.linear_1_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.l1.bias"], self.device)
        self.linear1 = TTLinear(
            self.linear_1_weight.padded_shape[-1],
            self.linear_1_weight.padded_shape[-2],
            self.linear_1_weight,
            self.linear_1_bias,
        )

        self.linear_2_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.l2.weight"], self.device)
        self.linear_2_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.l2.bias"], self.device)
        self.linear2 = TTLinear(
            self.linear_2_weight.padded_shape[-1],
            self.linear_2_weight.padded_shape[-2],
            self.linear_2_weight,
            self.linear_2_bias,
        )

        self.linear_3_weight = ttnn.from_torch(state_dict[f"{base_address}.l3.weight"], self.device)
        self.linear_3_bias = ttnn.from_torch(state_dict[f"{base_address}.l3.bias"], self.device)
        self.linear3 = TTLinear(
            self.linear_3_weight.padded_shape[-1],
            self.linear_3_weight.padded_shape[-2],
            self.linear_3_weight,
            self.linear_3_bias,
        )

    def forward(self, x):
        o1 = self.linear1(x)
        o2 = self.linear2(x)
        # Element-wise multiplication between 2 output matrices
        output = ttnn.multiply(o1, o2)
        output = self.linear3(output)
        return output
