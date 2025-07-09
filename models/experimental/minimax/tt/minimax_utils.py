import ttnn


# TT repeat_kv: TT-equivalent of torch.repeat_interleave for KV heads
def repeat_kv(hidden_states, n_rep: int):
    """
    Repeats key/value heads in a TT tensor.
    Args:
        hidden_states: TT Tensor with shape [batch, num_kv_heads, seq_len, head_dim]
        n_rep: number of repetitions per KV head
    Returns:
        TT Tensor with shape [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hs = ttnn.reshape(hidden_states, (batch, num_kv_heads, 1, seq_len, head_dim))
    hs = ttnn.concat([hs] * n_rep, dim=2)
    return ttnn.reshape(hs, (batch, num_kv_heads * n_rep, seq_len, head_dim))


def get_activation_fn(activation):
    """
    Returns appropriate ttnn activation function; defaults to identity if requested function not found.
    Code based on: https://github.com/MiniMax-AI/MiniMax-M1/blob/2abb4f45a9df4154b4bde024d51874bd127edcee/modeling_minimax_m1.py#L58C23-L58C33
    """

    if activation == "gelu":
        return ttnn.gelu
    elif activation == "relu":
        return ttnn.relu
    elif activation == "elu":
        return ttnn.elu
    elif activation == "sigmoid":
        return ttnn.sigmoid
    elif activation == "exp":

        def f(x):
            x_max = ttnn.max(x, dim=-1).values
            y = ttnn.exp(x - x_max)

            return y

        return f
    elif activation == "leak":
        return ttnn.leaky_relu
    elif activation == "1+elu":

        def f(x):
            return 1 + ttnn.elu(x)

        return f
    elif activation == "2+elu":

        def f(x):
            return 2 + ttnn.elu(x)

        return f
    elif activation == "silu" or activation == "swish":
        return ttnn.silu
    elif activation == "sine":
        return ttnn.sin
    else:
        return lambda x: x
