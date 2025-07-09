import copy
import torch
from torch import nn

import ttnn

from .minimax_config import TTMiniMaxM1Config
from .minimax_norm import TTMiniMaxM1RMSNorm
from .minimax_decoder import TTMiniMaxM1DecoderLayer

from transformers.modeling_utils import PreTrainedModel


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
