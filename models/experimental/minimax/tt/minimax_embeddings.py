import torch
from torch import nn

import ttnn
from typing import Optional


BLOCK = 256


# Rotary Embedding converted to TT
class TTMiniMaxM1RotaryEmbedding(nn.Module):
    def __init__(
        self,
        mesh_device,
        dim,
        base_url: str,
        layer_num: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        model_config: dict = None,
        tt_cache_path: Optional[str] = None,
        weights_dict: Optional[dict] = None,
    ):
        super().__init__()
        self.max_seq_len_cached = max_position_embeddings
        self.model_config = model_config

        # Frequency calculation
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=mesh_device) / dim))
        t = torch.arange(
            self.max_seq_len_cached,
            device=mesh_device,
            dtype=torch.int64,
        )
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Cache layer names
        layer_name = f"{base_url}.{layer_num}.rotary_embedding_maxseq{max_position_embeddings}"
        cos_str = f"{layer_name}.cos_cached"
        sin_str = f"{layer_name}.sin_cached"

        # Load or write cached TT weights
        self.tt_cos_cached = get_weights_cached(
            mesh_device,
            model_config,
            tt_cache_path,
            cos_str,
            weight_config_str="COS_CACHED_WEIGHTS",
            weights_to_cache=emb.cos()[None, None, :, :],
            weights_dict=weights_dict,
        )
        self.tt_sin_cached = get_weights_cached(
            mesh_device,
            model_config,
            tt_cache_path,
            sin_str,
            weight_config_str="SIN_CACHED_WEIGHTS",
            weights_to_cache=emb.sin()[None, None, :, :],
            weights_dict=weights_dict,
        )

    def forward(
        self,
        layer: ttnn.Tensor,
        token_idx: Optional[int] = None,
    ) -> ttnn.Tensor:
        seq_len = layer.padded_shape[2]
        assert (
            seq_len <= self.max_seq_len_cached
        ), f"seq_len={seq_len} exceeds max_seq_len_cached={self.max_seq_len_cached}!"
        return ttnn.experimental.rotary_embedding(
            layer,
            self.tt_cos_cached,
            self.tt_sin_cached,
            token_idx,
            memory_config=self.model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"],
        )


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return ttnn.concat([-x2, x1], dim=-1)


# Use in TT attention by converting TT->torch, applying, then back to TT
def apply_rotary_pos_emb_torch(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    rot_dim = cos.shape[-1]
    q_, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_, k_pass = k[..., :rot_dim], k[..., rot_dim:]

    cos = ttnn.unsqueeze(cos[position_ids], unsqueeze_dim)
    sin = ttnn.unsqueeze(sin[position_ids], unsqueeze_dim)

    q_embed = (q_ * cos) + (rotate_half(q_) * sin)
    k_embed = (k_ * cos) + (rotate_half(k_) * sin)

    return ttnn.concat((q_embed, q_pass), dim=-1), ttnn.concat((k_embed, k_pass), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->MiniMaxM1
class MiniMaxM1RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=torch.float32)

        return (
            self.cos_cached[:seq_len].to(dtype=torch.float32),
            self.sin_cached[:seq_len].to(dtype=torch.float32),
        )


def get_weights_cached(
    mesh_device,
    model_config,
    tt_cache_path,
    weight_cache_str,
    weight_config_str,
    weights_to_cache,
    tt_layout=ttnn.TILE_LAYOUT,
    weights_dict=None,
    custom_output_shape=None,
):
    """Load weights from weights_dict or cache and duplicate per device. Store if not cached."""
    custom_output_shape_str = ""
    if custom_output_shape is not None:
        custom_output_shape_str = f"_{custom_output_shape[-2]}_{custom_output_shape[-1]}"
    path = tt_cache_path / f"{weight_cache_str}{custom_output_shape_str}"

    def preprocess_weights(weights_to_cache):
        if weights_to_cache is None:
            raise ValueError(f"weights_to_cache is None for {weight_cache_str}")

        if custom_output_shape is not None:
            padding = (
                0,
                custom_output_shape[-1] - weights_to_cache.shape[-1],
                0,
                custom_output_shape[-2] - weights_to_cache.shape[-2],
            )
            weights_to_cache = torch.nn.functional.pad(weights_to_cache, padding, "constant", 0.0)

        return weights_to_cache

    if weights_dict and str(path) in weights_dict.keys():
        weights = weights_dict[str(path)]
    else:
        weights = ttnn.as_tensor(
            weights_to_cache,
            dtype=model_config[f"{weight_config_str}_DTYPE"],
            layout=tt_layout,
            device=mesh_device,
            memory_config=model_config[f"{weight_config_str}_MEMCFG"],
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if type(mesh_device) == ttnn.MeshDevice else None,
            cache_file_name=str(path),
            preprocess=preprocess_weights,
        )

        # Save weights for reuse between prefill/decode
        if weights_dict is not None:
            weights_dict[str(path)] = weights

    return weights
