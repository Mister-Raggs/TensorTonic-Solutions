import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    max_vals = torch.max(logits, dim=1, keepdim=True).values
    shifted = logits - max_vals
    exps = torch.exp(shifted)
    return exps / exps.sum(dim=1, keepdim=True)
