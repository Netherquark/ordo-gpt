import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .rotary import RotaryEmbedding

# Based on: ParallelSelfAttention in gpt-neox/megatron/model/transformer.py

class NeoXAttention(nn.Module):
    def __init__(self, config: "NeoXConfig"):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.rotary_ndims = config.rotary_emb_dim
        self.use_cache = config.use_cache
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        # Query, Key, Value projections
        self.query_key_value = nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=True
        )
        
        # Output projection
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            base=config.rotary_emb_base,
            precision=torch.float32 if config.precision == 32 else torch.bfloat16,
        )

    def _split_heads(self, qkv):
        """
        Splits [B, S, 3 * H] into [B, S, 3, N, D]
        """
        batch_size, seq_len, _ = qkv.shape
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        return qkv
        
    def _merge_heads(self, x):
        """
        Merges [B, S, N, D] into [B, S, H]
        """
        batch_size, seq_len, _, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return x

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_past=None,
        seq_len_offset=0
    ):
        # hidden_states: [batch_size, seq_len, hidden_size]
        
        # 1. Get QKV
        qkv = self.query_key_value(hidden_states)
        
        # 2. Split heads
        # [batch_size, seq_len, 3, num_heads, head_dim]
        qkv_heads = self._split_heads(qkv)

        # 3. Separate Q, K, V
        # [batch_size, seq_len, num_heads, head_dim]
        query_heads = qkv_heads[:, :, 0, ...]
        key_heads = qkv_heads[:, :, 1, ...]
        value_heads = qkv_heads[:, :, 2, ...]

        # 4. Apply RoPE
        if self.rotary_ndims > 0:
            query_rot, key_rot = self.rotary_emb(
                query_heads[..., : self.rotary_ndims], 
                key_heads[..., : self.rotary_ndims],
                seq_len_offset=seq_len_offset
            )
            query_heads = torch.cat(
                (query_rot, query_heads[..., self.rotary_ndims :]), dim=-1
            )
            key_heads = torch.cat(
                (key_rot, key_heads[..., self.rotary_ndims :]), dim=-1
            )

        # 5. Handle KV Caching
        if layer_past is not None:
            # layer_past: [2, batch_size, num_heads, seq_len, head_dim]
            past_key, past_value = layer_past
            key_heads = torch.cat((past_key, key_heads), dim=2)
            value_heads = torch.cat((past_value, value_heads), dim=2)
            
        if self.use_cache:
            # Transpose to [batch_size, num_heads, seq_len, head_dim] for cache
            present = torch.stack(
                (key_heads.transpose(1, 2), value_heads.transpose(1, 2))
            )
        else:
            present = None
            
        # 6. Compute Attention
        # Transpose Q, K, V to [batch_size, num_heads, seq_len, head_dim]
        query_heads = query_heads.transpose(1, 2)
        key_heads_transpose = key_heads.transpose(1, 2)
        value_heads_transpose = value_heads.transpose(1, 2)

        # K_t: [batch_size, num_heads, head_dim, seq_len]
        key_heads_t = key_heads_transpose.transpose(-1, -2)
        
        # Attn scores: [batch_size, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(query_heads, key_heads_t)
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # Apply attention mask
        # attention_mask: [batch_size, 1, seq_len, seq_len]
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1).to(value_heads.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        # Attn output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, value_heads_transpose)

        # 7. Merge heads
        # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        # [batch_size, seq_len, hidden_size]
        attn_output = self._merge_heads(attn_output)
        
        # 8. Final projection
        attn_output = self.dense(attn_output)

        return attn_output, present