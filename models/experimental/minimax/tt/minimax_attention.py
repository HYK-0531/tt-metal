import math
import torch
from torch import nn

import ttnn
from typing import Optional, Tuple
from loguru import logger

from models.helper_funcs import Linear as TTLinear
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from .minimax_utils import get_activation_fn
from .minimax_norm import TTMiniMaxM1RMSNorm

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

    def __init__(self, config, state_dict, base_address, layer_idx: Optional[int] = None, device=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_dim = getattr(config, "rotary_dim", self.head_dim)

        self.rotary_emb = MiniMaxM1RotaryEmbedding(
            self.rotary_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
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
class MiniMaxM1FlashAttention2(MiniMaxM1Attention):
    """
    MiniMaxM1 flash attention module. This module inherits from `MiniMaxM1Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Union[Cache, Tuple[torch.Tensor]]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reshape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if past_key_value is not None:
            # reuse k, v, for evaluation only
            key_states = torch.cat([past_key_value[0], key_states], dim=-3)
            value_states = torch.cat([past_key_value[1], value_states], dim=-3)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
