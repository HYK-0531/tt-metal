# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from dataclasses import dataclass


@dataclass
class CLIPTextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_heads: int
    max_position_embeddings: int
    layer_norm_eps: float
    attention_dropout: float


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/clip/modeling_clip.py
# ensure tokens can only attend to previous tokens
def _create_4d_causal_attention_mask(input_shape, dtype, device):
    batch_size, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len)


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/clip/modeling_clip.py
class CLIPTextEmbeddings(torch.nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        # initialize embedding lookup tables
        self.token_embedding = torch.nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = torch.nn.Embedding(config.max_position_embeddings, embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        position_ids = self.position_ids[:, :seq_length]

        # embed tokens and add position embeddings
        inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/clip/modeling_clip.py
class CLIPAttention(torch.nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        # if self.head_dim * self.num_heads != self.embed_dim:
        #     raise ValueError(
        #         f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        #     )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.W_k = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.W_v = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.W_q = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.W_o = torch.nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, causal_attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.W_q(hidden_states) * self.scale
        k = self.W_k(hidden_states)
        v = self.W_v(hidden_states)

        q = self._shape(q, seq_len, batch_size)
        k = self._shape(k, seq_len, batch_size)
        v = self._shape(v, seq_len, batch_size)

        # reshape for batch matmul
        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = q.view(*proj_shape)
        key_states = k.view(*proj_shape)
        value_states = v.view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # apply causal mask
        if causal_attention_mask is not None:
            attn_weights = attn_weights.view(batch_size, self.num_heads, seq_len, seq_len) + causal_attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, seq_len, seq_len)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        attn_output = self.W_o(attn_output)
        return attn_output


class CLIPMLP(torch.nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.activation_fn = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = torch.nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(torch.nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = torch.nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = torch.nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, causal_attention_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(torch.nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: torch.Tensor, causal_attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, causal_attention_mask)
        return hidden_states


class CLIPTextTransformer(torch.nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        # self.embeddings)
        self.encoder = CLIPEncoder(config)
        # print(self.encoder)
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids)

        # create causal attention mask
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            causal_attention_mask=causal_attention_mask,
        )

        # sequence embedding output
        last_hidden_state = encoder_outputs
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # pooled embedding output - single vector per sequence
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
        # print("POOLED" + str(pooled_output.shape))
        # print("LAST HIDDEN STATE" + str(last_hidden_state.shape))
        return last_hidden_state, pooled_output


class CLIPTextEncoder(torch.nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.text_model = CLIPTextTransformer(config)

    def encode(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.text_model(input_ids)
