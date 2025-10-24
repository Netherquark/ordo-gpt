import torch
import torch.nn.functional as F

# Based on: gpt-neox/megatron/model/activations.py

@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

def gelu(x):
    return gelu_impl(x)

def get_activation(config: "NeoXConfig"):
    """
    Retrieves the activation function specified in config and whether or not the activation is gated.
    """
    is_gated = False
    
    if config.activation == "geglu":
        is_gated = True
        activation_func = F.gelu
    elif config.activation == "swiglu":
        is_gated = True
        activation_func = F.silu  # Swish
    elif config.activation == "gelu":
        activation_func = F.gelu
    elif config.activation == "tanh":
        activation_func = F.tanh
    elif config.activation == "relu":
        activation_func = F.relu
    elif config.activation == "softsign":
        activation_func = F.softsign
    else:
        raise Exception(f"activation function {config.activation} not recognized")
        
    return activation_func, is_gated