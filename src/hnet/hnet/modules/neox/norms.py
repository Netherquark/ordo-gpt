import torch
from torch.nn import LayerNorm as LayerNorm

# Based on: gpt-neox/megatron/model/norms.py

class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.
    Derived from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/norms.py
    """
    def __init__(self, dim, eps=1e-5, bias=False):
        super().__init__()
        self.eps = eps
        self.d = dim
        self.bias = bias
        self.scale = torch.nn.Parameter(torch.ones(dim))
        
        if self.bias:
            self.offset = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        x_normed = x_normed.to(x.dtype)
        
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed

def get_norm(config: "NeoXConfig"):
    if config.norm == "rmsnorm":
        norm = RMSNorm
        eps = config.layernorm_epsilon
    elif config.norm == "layernorm":
        norm = LayerNorm
        eps = config.layernorm_epsilon
    else:
        raise Exception(f"norm {config.norm} not recognized")
    
    return norm, eps