import math
import copy
import torch
from torch import nn

import ttnn
from models.utility_functions import (
    tt_to_torch_tensor,
)

from .minimax_config import TTMiniMaxM1Config
from .minimax_norm import TTMiniMaxM1RMSNorm
from .minimax_decoder import TTMiniMaxM1DecoderLayer

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    MoeModelOutputWithPast,
)


class TTMiniMaxM1PreTrainedModel(PreTrainedModel):
    config_class = TTMiniMaxM1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxM1DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        # This currently doens't do anything, nor would we really care since all we will only load pretrained weights anyway
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class TTMiniMaxM1Model(TTMiniMaxM1PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MiniMaxM1DecoderLayer`]

    Args:
        config: MiniMaxM1Config
    """

    def __init__(self, config, state_dict, base_address, device=None):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.device = device

        # self.embed_tokens = ttnn.embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.tt_embed_tokens_weight = ttnn.from_torch(state_dict[f"{base_address}.embed_tokens.weight"])

        self.attn_type_list = config.attn_type_list
        config_copy = copy.deepcopy(config)

        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            _config = copy.deepcopy(config)
            if self.attn_type_list[i] == 0:
                _config._attn_implementation = "linear_attention"
                _config.attention_type = 0
            else:
                _config._attn_implementation = config_copy._attn_implementation
                _config.attention_type = 1
            self.layers.append(
                TTMiniMaxM1DecoderLayer(
                    _config, state_dict, f"{base_address}.layers.{i}", layer_idx=i, device=self.device
                )
            )

        self._attn_implementation = config_copy._attn_implementation
        self.norm = TTMiniMaxM1RMSNorm(config, state_dict, f"{base_address}.norm", self.device)

        self.gradient_checkpointing = False
        self.slopes = self._build_slope_tensor(config.num_attention_heads)
        # mask
        self._linear_attn_mask = torch.empty(0)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest = 2 ** math.floor(math.log2(n))
                return get_slopes_power_of_2(closest) + get_slopes(2 * closest)[0::2][: n - closest]

        slopes = torch.tensor(get_slopes(n_attention_heads), dtype=torch.float32)
        return ttnn.from_torch(slopes.reshape(n_attention_heads, 1, 1), device=None)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_router_logits=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = ttnn.embedding(input_ids, self.tt_embed_tokens_weight)

        slope_rates = [
            ttnn.multiply(self.slopes, (1 - idx / (len(self.tt_layers) - 1)) + 1e-5)
            for idx in range(len(self.tt_layers))
        ]

        all_hidden_states = [] if output_hidden_states else None
        all_attns = [] if output_attentions else None
        next_cache = [] if use_cache else None

        for idx, layer in enumerate(self.tt_layers):
            if output_hidden_states:
                all_hidden_states.append(tt_to_torch_tensor(tt_hidden))

            past_kv = past_key_values[idx] if past_key_values is not None else None
            tt_hidden, attn, new_kv = layer(
                tt_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                output_attentions=output_attentions,
                use_cache=use_cache,
                slope_rate=slope_rates[idx],
            )

            if use_cache:
                next_cache.append(new_kv)
            if output_attentions:
                all_attns.append(attn)

        tt_hidden = self.tt_norm(tt_hidden)
        hidden_states = tt_to_torch_tensor(tt_hidden)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        next_cache = tuple(next_cache) if use_cache else None
        all_hidden_states = tuple(all_hidden_states) if output_hidden_states else None
        all_attns = tuple(all_attns) if output_attentions else None

        if not return_dict:
            outputs = (hidden_states, next_cache, all_hidden_states, all_attns)
            return tuple(o for o in outputs if o is not None)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns,
            router_logits=None,
        )
