import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import get_norm
from .attention import NeoXAttention
from .mlp import NeoXMLP

# Based on: ParallelTransformerBlock in gpt-neox/megatron/model/transformer.py

class NeoXBlock(nn.Module):
    def __init__(self, config: "NeoXConfig"):
        super().__init__()
        
        self.use_parallel_residual = config.use_parallel_residual
        self.hidden_dropout = config.hidden_dropout
        
        norm, eps = get_norm(config)
        
        # Pre-normalization
        self.input_layernorm = norm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = norm(config.hidden_size, eps=eps)
        
        # Attention
        self.attention = NeoXAttention(config)
        
        # MLP
        self.mlp = NeoXMLP(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_past=None,
        seq_len_offset=0
    ):
        # hidden_states: [batch_size, seq_len, hidden_size]
        
        residual = hidden_states
        
        # 1. Input Norm
        ln_output = self.input_layernorm(hidden_states)
        
        # 2. Attention
        # attn_output: [batch_size, seq_len, hidden_size]
        # present: [2, batch_size, num_heads, new_seq_len, head_dim]
        attn_output, present = self.attention(
            ln_output,
            attention_mask,
            layer_past=layer_past,
            seq_len_offset=seq_len_offset,
        )

        # 3. Residual
        if self.use_parallel_residual:
            # Parallel residual: x = x + attn(ln(x)) + mlp(ln(x))
            
            # 3a. MLP Norm
            mlp_ln_output = self.post_attention_layernorm(hidden_states)
            
            # 3b. MLP
            mlp_output = self.mlp(mlp_ln_output)
            
            # 3c. Add both residuals
            mlp_output = self.dropout(mlp_output)
            attn_output = self.dropout(attn_output)
            
            output = mlp_output + attn_output + residual
            
        else:
            # Sequential residual: x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            
            # 3a. Add attention residual
            attn_output = self.dropout(attn_output)
            hidden_states = attn_output + residual
            
            # 3b. MLP Norm
            residual = hidden_states
            mlp_ln_output = self.post_attention_layernorm(hidden_states)
            
            # 3c. MLP
            mlp_output = self.mlp(mlp_ln_output)
            
            # 3d. Add MLP residual
            mlp_output = self.dropout(mlp_output)
            output = mlp_output + residual

        return output, present