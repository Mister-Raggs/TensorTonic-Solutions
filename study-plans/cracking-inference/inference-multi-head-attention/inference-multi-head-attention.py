import torch

def multi_head_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as hidden_states.
    """
    batch, seq, d_model = hidden_states.shape
    d_k = d_model // num_heads

    query = hidden_states @ w_q
    key = hidden_states @ w_k
    value = hidden_states @ w_v

    def split_heads(x: torch.Tensor) -> torch.Tensor:
        return x.view(batch, seq, num_heads, d_k).transpose(1, 2)

    query, key, value = split_heads(query), split_heads(key), split_heads(value)

    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    attn = torch.matmul(weights, value)

    concat = attn.transpose(1, 2).contiguous().view(batch, seq, d_model)
    return concat @ w_o
