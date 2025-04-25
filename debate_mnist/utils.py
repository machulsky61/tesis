import torch

def even_logit(logits):
    """Suma los logits de las clases 0,2,4,6,8."""
    idx = torch.tensor([0,2,4,6,8], device=logits.device)
    return logits[:, idx].logsumexp(dim=-1, keepdim=True)   # shape (N,1)

def odd_logit(logits):
    """Suma los logits de las clases 1,3,5,7,9."""
    idx = torch.tensor([1,3,5,7,9], device=logits.device)
    return logits[:, idx].logsumexp(dim=-1, keepdim=True)
