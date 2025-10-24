import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import get_activation

# Based on: ParallelMLP in gpt-neox/megatron/model/transformer.py

class NeoXMLP(nn.Module):
    def __init__(self, config: "NeoXConfig"):
        super().__init__()
        
        self.activation_func, self.is_gated = get_activation(config)
        self.hidden_dropout = config.hidden_dropout
        
        d_intermediate = config.d_intermediate
        if self.is_gated:
            # Gated activations (GeGLU, SwiGLU) have 2 up-projections
            d_intermediate //= 2
            
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.d_intermediate,
            bias=True
        )
        
        self.dense_4h_to_h = nn.Linear(
            d_intermediate,
            config.hidden_size,
            bias=True
        )
        
    def forward(self, hidden_states):
        
        # Up-projection
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        
        # Activation
        if self.is_gated:
            # intermediate_parallel is [B, S, 2 * d_intermediate]
            intermediate_parallel, gate = intermediate_parallel.chunk(2, dim=-1)
            intermediate_parallel = self.activation_func(gate) * intermediate_parallel
        else:
            # intermediate_parallel is [B, S, d_intermediate]
            intermediate_parallel = self.activation_func(intermediate_parallel)

        # Down-projection
        output = self.dense_4h_to_h(intermediate_parallel)
        
        return output