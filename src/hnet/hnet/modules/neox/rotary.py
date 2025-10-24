import torch
import math

# Based on: gpt-neox/megatron/model/positional_embeddings.py

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.float32):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def _load_cache(self, seq_len, device, dtype):
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            if self.precision == torch.bfloat16:
                emb = emb.bfloat16()
            
            # [seq_len, dim]
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
            
            if self.precision == torch.float32:
                 self.cos_cached = self.cos_cached.float()
                 self.sin_cached = self.sin_cached.float()

    def forward(self, q, k, seq_len_offset=0):
        # q, k: [batch_size, seq_len, num_heads, head_dim]
        self._load_cache(
            seq_len=q.shape[1] + seq_len_offset, device=q.device, dtype=q.dtype
        )
        
        cos = self.cos_cached[seq_len_offset : q.shape[1] + seq_len_offset, ...].to(q.dtype)
        sin = self.sin_cached[seq_len_offset : q.shape[1] + seq_len_offset, ...].to(q.dtype)
        
        q_ro = apply_rotary_pos_emb_torch(q, cos, sin)
        k_ro = apply_rotary_pos_emb_torch(k, cos, sin)
        
        return q_ro, k_ro


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_torch(x, cos, sin):
    # x: [batch_size, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, 1, 1, head_dim]
    return (x * cos) + (rotate_half(x) * sin)