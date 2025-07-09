import math
import torch
from torch import nn

import ttnn
from typing import Optional, Tuple

from models.helper_funcs import Linear as TTLinear
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from .minimax_utils import get_activation_fn, repeat_kv
from .minimax_norm import TTMiniMaxM1RMSNorm
from .minimax_embeddings import TTMiniMaxM1RotaryEmbedding

BLOCK = 256


class TTMiniMaxM1LightningAttention(nn.Module):
    def __init__(self, config, state_dict, base_address, layer_idx: Optional[int] = None, device=None):
        super(TTMiniMaxM1LightningAttention, self).__init__()

        self.device = self.device
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)

        self.act = get_activation_fn(config.hidden_act)
        self.norm = TTMiniMaxM1RMSNorm(config, state_dict, f"{base_address}.norm", device)

        self.tt_out_proj_weight = ttnn.from_torch(state_dict[f"{base_address}.out_proj.weight"], self.device)
        self.tt_out_proj = TTLinear(
            self.tt_out_proj_weight.padded_shape[-1], self.tt_out_proj_weight.padded_shape[-2], self.tt_out_proj_weight
        )

        # W_qkv: (hidden_size, 3 * head_dim * num_heads)
        self.tt_qkv_proj_weight = ttnn.from_torch(state_dict[f"{base_address}.qkv_proj.weight"], self.device)
        self.tt_qkv_proj = TTLinear(
            self.tt_qkv_proj_weight.padded_shape[-1], self.tt_qkv_proj_weight.padded_shape[-2], self.tt_qkv_proj_weight
        )

        self.tt_output_gate_weight = ttnn.from_torch(state_dict[f"{base_address}.output_gate.weight"], self.device)
        self.tt_output_gate = TTLinear(
            self.tt_output_gate_weight.padded_shape[-1],
            self.tt_output_gate_weight.padded_shape[-2],
            self.tt_output_gate_weight,
        )

        # for inference only
        self.offset = 0
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states,
        attn_mask: Optional[torch.Tensor] = None,  # (b, h, n, m)
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        slope_rate: Optional[torch.Tensor] = None,
    ):
        # hidden_states: (batch, seq, hidden_size)
        b, n, d = hidden_states.shape
        # qkv: (batch, seq, 3 * head_dim * num_heads)
        qkv = self.act(self.tt_qkv_proj(hidden_states))

        # q, k, v = torch.split(qkv, [self.head_dim] * 3, dim=3)
        # q = q.transpose(1, 2)
        # k = k.transpose(1, 2)
        # v = v.transpose(1, 2)

        (q, k, v) = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, memory_config=ttnn.L1_MEMORY_CONFIG, num_heads=self.num_heads, transpose_key=True
        )
        ttnn.deallocate(qkv)

        if past_key_value is None:
            self.offset = q.shape[-2]
        else:
            self.offset += 1

        ratio = ttnn.exp(-slope_rate)
        if past_key_value is None:
            # Prefill, move to forward_prefill for future purposes
            slope_rate = ttnn.from_torch(slope_rate, dtype=ttnn.float32)

            # Causal mask
            if attn_mask is not None:
                v = tt_to_torch_tensor(v).masked_fill((1 - attn_mask).unsqueeze(1).unsqueeze(-1).to(torch.bool), 0)
                v = torch_to_tt_tensor_rm(v, self.device)

            NUM_BLOCK = (n + BLOCK - 1) // BLOCK

            b, h, n, d = q.shape
            e = v.shape[-1]

            # Set up decay schedule; note that array stays as a torch tensor
            array = torch.arange(BLOCK).to(q) + 1

            q_decay = ttnn.exp(ttnn.multiply(-slope_rate, ttnn.from_torch(array.reshape(-1, 1))))
            k_decay = ttnn.exp(ttnn.multiply(-slope_rate, ttnn.from_torch((BLOCK - array.reshape(-1, 1)))))

            index = array[:, None] - array[None, :]
            s_index = (
                tt_to_torch_tensor(slope_rate)
                * index[
                    None,
                    None,
                ]
            )
            s_index = torch.where(index >= 0, -s_index, float("-inf"))

            diag_decay = ttnn.from_torch(torch.exp(s_index))

            # kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
            kv = ttnn.zeros(shape=[b, h, d, e], dtype=ttnn.float32, device=self.device)
            # output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
            output = ttnn.empty(shape=[b, h, n, e], dtype=ttnn.float32, device=self.device)

            for i in range(NUM_BLOCK):
                si = i * BLOCK
                ei = min(si + BLOCK, n)
                m = ei - si

                # qi = q[:, :, si:ei].contiguous()
                # ki = k[:, :, si:ei].contiguous()
                # vi = v[:, :, si:ei].contiguous()
                qi = q[:, :, si:ei]
                ki = k[:, si:ei, :]
                vi = v[:, :, si:ei]

                # qkv_none_diag = torch.matmul(qi * q_decay[:, :m], kv).to(torch.float32)
                qkv_none_diag = ttnn.matmul(ttnn.multiply(qi, q_decay[:, :m]), kv)

                # qk = torch.matmul(qi, ki.transpose(-1, -2)).to(torch.float32) * diag_decay[:, :, :m, :m]
                # qkv_diag = torch.matmul(qk, vi.to(torch.float32))
                qk = ttnn.multiply(ttnn.matmul(qi, ki), diag_decay[:, :, :m, :m])
                qkv_diag = ttnn.matmul(qk, vi)

                block_decay = ttnn.exp(ttnn.multiply(-slope_rate, m))
                output[:, :, si:ei] = qkv_none_diag + qkv_diag

                # kv = block_decay * kv + torch.matmul((ki * k_decay[:, -m:]).transpose(-1, -2).to(vi.dtype), vi)
                kv = ttnn.multiply(block_decay, kv) + ttnn.matmul(ttnn.multiply(ki, k_decay[:, -m:]), vi)

        else:
            # Decode, move to forward_decode for future purposes
            kv = past_key_value
            output = []

            # Autoregressively build decayed kv cache along each of the n (seq_len) tokens
            for i in range(n):
                # kv = ratio * kv + torch.einsum(
                #     "... n d, ... n e -> ... d e",
                #     k[:, :, i : i + 1],
                #     v[:, :, i : i + 1],
                # )

                # Note that k is already transposed from split_query_key_value_and_split_heads
                kv = kv + ttnn.matmul(k[:, :, i : i + 1], v[:, :, i + 1])
                kv = ttnn.multiply(ratio, kv)

                # qkv = torch.einsum("... n e, ... e d -> ... n d", q[:, :, i : i + 1], kv.to(q.dtype))
                qkv = ttnn.matmul(q[:, :, i : i + 1], kv)
                output.append(qkv)

            output = ttnn.concat(output, dim=-2)

        # Reshape: (b, h, n, d) -> (b, n, h*d)
        # output = rearrange(output, "b h n d -> b n (h d)")
        output = ttnn.reshape(output, (b, n, h * d))

        # Normalize
        output = self.norm(output)

        # Gate
        output = ttnn.multiply(ttnn.sigmoid(self.tt_output_gate(hidden_states)), output)

        # Outproj
        output = self.tt_out_proj(output)

        attn_weights = None
        return output, attn_weights, kv

    def forward_prefill(
        self,
        hidden_states,
        attn_mask: Optional[torch.Tensor] = None,  # (b, h, n, m)
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        slope_rate: Optional[torch.Tensor] = None,
    ):
        pass

    def forward_decode(
        self,
        hidden_states,
        attn_mask: Optional[torch.Tensor] = None,  # (b, h, n, m)
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        slope_rate: Optional[torch.Tensor] = None,
    ):
        pass


class TTMiniMaxM1Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(
        self,
        config,
        state_dict,
        base_address,
        layer_idx: Optional[int] = None,
        device=None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout

        # Load and wrap weights for TT projections
        self.tt_q_proj_weight = ttnn.from_torch(state_dict[f"{base_address}.q_proj.weight"], device)
        self.tt_q_proj = TTLinear(
            self.tt_q_proj_weight.padded_shape[-1],
            self.tt_q_proj_weight.padded_shape[-2],
            self.tt_q_proj_weight,
        )

        self.tt_k_proj_weight = ttnn.from_torch(state_dict[f"{base_address}.k_proj.weight"], device)
        self.tt_k_proj = TTLinear(
            self.tt_k_proj_weight.padded_shape[-1],
            self.tt_k_proj_weight.padded_shape[-2],
            self.tt_k_proj_weight,
        )

        self.tt_v_proj_weight = ttnn.from_torch(state_dict[f"{base_address}.v_proj.weight"], device)
        self.tt_v_proj = TTLinear(
            self.tt_v_proj_weight.padded_shape[-1],
            self.tt_v_proj_weight.padded_shape[-2],
            self.tt_v_proj_weight,
        )

        self.tt_o_proj_weight = ttnn.from_torch(state_dict[f"{base_address}.o_proj.weight"], device)
        self.tt_o_proj = TTLinear(
            self.tt_o_proj_weight.padded_shape[-1],
            self.tt_o_proj_weight.padded_shape[-2],
            self.tt_o_proj_weight,
        )

        # Rotary embeddings
        self.rotary_emb = TTMiniMaxM1RotaryEmbedding(
            config.rotary_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2 with Mistral->MiniMaxM1
class TTMiniMaxM1FlashAttention2(TTMiniMaxM1Attention):
    """
    MiniMaxM1 flash attention module. This module inherits from `MiniMaxM1Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Have flash attention default to regular attention for now
