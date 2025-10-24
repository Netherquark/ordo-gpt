from dataclasses import dataclass, field
from typing import Union

@dataclass
class NeoXConfig:
    # Model architecture
    num_layers: int = 26
    hidden_size: int = 1536
    num_attention_heads: int = 16
    d_intermediate: int = 6144  # Often 4 * hidden_size
    
    # Normalization
    norm: str = "rmsnorm"  # "rmsnorm" or "layernorm"
    layernorm_epsilon: float = 1e-5
    
    # Activation
    activation: str = "gelu" # "gelu", "geglu", "swiglu" etc.
    
    # RoPE
    rotary_pct: float = 0.25  # Percentage of hidden_size to apply rotary to
    rotary_emb_dim: int = 384 # rotary_pct * (hidden_size // num_attention_heads)
    rotary_emb_base: int = 10000
    
    # Attention & Block
    use_parallel_residual: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    # Initialization
    init_method: str = "normal" # Not used yet, but good to have
    output_layer_init_method: str = "normal" # Not used yet

    # KV Caching for inference
    use_cache: bool = True
    
    # Not used, but for compatibility
    precision: Union[str, int] = 32
    apply_residual_connection_post_layernorm: bool = False
    bias_dropout_fusion: bool = False
    gpt_j_residual: bool = False
    gpt_j_layernorm: bool = False