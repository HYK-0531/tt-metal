import torch
from torch import nn

import ttnn
from typing import Optional, Tuple

from models.helper_funcs import Linear as TTLinear
from models.utility_functions import torch_to_tt_tensor_rm

from .minimax_norm import TTMiniMaxM1RMSNorm
from .minimax_attention import TTMiniMaxM1LightningAttention, TTMiniMaxM1Attention
from .minimax_mlp import TTMiniMaxM1MLP
from .minimax_moe import TTMiniMaxM1SparseMoeBlock

BLOCK = 256


class TTMiniMaxM1DecoderLayer(nn.Module):
    def __init__(self, config, state_dict, base_address, layer_idx, device=None):
        super(TTMiniMaxM1DecoderLayer, self).__init__()
        self.device = device
        self.hidden_size = config.hidden_size

        self.self_attn = self.build_attn(config, layer_idx)
        self.layer_idx = layer_idx

        self.block_sparse_moe = TTMiniMaxM1SparseMoeBlock(
            config, state_dict, f"{base_address}.block_sparse_moe", device=self.device
        )
        self.input_layernorm = TTMiniMaxM1RMSNorm(
            config, state_dict, f"{base_address}.input_layernorm", device=self.device
        )
        self.post_attention_layernorm = TTMiniMaxM1RMSNorm(
            config, state_dict, f"{base_address}.post_attention_layernorm", device=self.device
        )

        self.postnorm = getattr(config, "postnorm", False)
        self.layernorm_attention_alpha = (
            getattr(config, "layernorm_linear_attention_alpha", 1)
            if config.attention_type == 0
            else getattr(config, "layernorm_full_attention_alpha", 1)
        )
        self.layernorm_attention_beta = (
            getattr(config, "layernorm_linear_attention_beta", 1)
            if config.attention_type == 0
            else getattr(config, "layernorm_full_attention_beta", 1)
        )
        self.layernorm_mlp_alpha = getattr(config, "layernorm_mlp_alpha", 1)
        self.layernorm_mlp_beta = getattr(config, "layernorm_mlp_beta", 1)

        shared_intermediate = getattr(config, "shared_intermediate_size", 0)
        self.shared_moe = False

        if shared_intermediate > 0:
            self.shared_moe = True
            self.shared_mlp = TTMiniMaxM1MLP(config, state_dict, f"{base_address}.shared_mlp")

            self.tt_coefficient_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.coefficient.weight"], device)
            self.tt_coefficient = TTLinear(
                self.tt_coefficient_weight.padded_shape[-1],
                self.tt_coefficient_weight.padded_shape[-2],
                self.tt_coefficient_weight,
            )

    def build_attn(self, config, state_dict, base_address, layer_idx):
        if config.attention_type == 0:
            Attention_module = TTMiniMaxM1LightningAttention
        else:
            # Ideally this will be replaced with TTMiniMaxM1FlashAttention2 later
            Attention_module = TTMiniMaxM1Attention
        return Attention_module(
            config, state_dict, f"{base_address}.self_attn", layer_idx=layer_idx, device=self.device
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        slope_rate: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        if self.postnorm:
            residual = hidden_states

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attn_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            slope_rate=slope_rate,
        )

        hidden_states = ttnn.multiply(residual, self.layernorm_attention_alpha) + ttnn.multiply(
            hidden_states, self.layernorm_attention_beta
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.postnorm:
            residual = hidden_states

        moe_hidden_states, router_logits = self.block_sparse_moe(hidden_states)

        if self.shared_moe:
            output_mlp = self.shared_mlp(hidden_states)
            weight_fp32 = self.tt_coefficient_weight.float()

            coef = ttnn.matmul(hidden_states, weight_fp32)
            coef = ttnn.sigmoid(coef)

            hidden_states = ttnn.multiply(moe_hidden_states, (1 - coef)) + ttnn.multiply(output_mlp, coef)
        else:
            hidden_states = moe_hidden_states

        hidden_states = ttnn.multiply(residual, self.layernorm_mlp_alpha) + ttnn.multiply(
            hidden_states, self.layernorm_mlp_beta
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs
