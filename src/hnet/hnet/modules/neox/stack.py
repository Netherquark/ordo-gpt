import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from dataclasses import dataclass, field  # <-- Import dataclass and field

from .block import NeoXBlock
from .config import NeoXConfig

@dataclass
class NeoXState:
    presents_list: List = field(default_factory=list)

# Based on: ParallelTransformer in gpt-neox/megatron/model/transformer.py

class NeoXStack(nn.Module):
    def __init__(self, config: "NeoXConfig"):
        super().__init__()
        
        self.config = config
        self.use_cache = config.use_cache
        
        self.layers = nn.ModuleList(
            [NeoXBlock(config) for _ in range(config.num_layers)]
        )
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        # NeoX cache is dynamic, but we return the state wrapper
        return NeoXState(presents_list=[None] * self.config.num_layers)

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_past_list=None  # This will be NeoXState during inference
    ):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, 1, seq_len, seq_len]
        
        if layer_past_list is None:
            layer_past_state = NeoXState(presents_list=[None] * len(self.layers))
        else:
            # We expect layer_past_list to be an instance of NeoXState
            layer_past_state = layer_past_list
            
        presents_list = []
        
        seq_len_offset = 0
        # --- CORRECTED CACHE CHECK ---
        # Check if the state's list is populated and the first element is not None
        if layer_past_state.presents_list and layer_past_state.presents_list[0] is not None:
            # Get seq_len from cache: [2, B, N, S, D]
            seq_len_offset = layer_past_state.presents_list[0][0].shape[2]
        # --- END CORRECTION ---

        # --- CORRECTED LOOP ---
        for i, (layer, layer_past) in enumerate(zip(self.layers, layer_past_state.presents_list)):
        # --- END CORRECTION ---
            
            hidden_states, present = layer(
                hidden_states,
                attention_mask,
                layer_past=layer_past,
                seq_len_offset=seq_len_offset
            )
            
            if self.use_cache:
                presents_list.append(present)
        
        # --- CORRECTED RETURN ---
        # Return the hidden states and the new state object
        return hidden_states, NeoXState(presents_list=presents_list)
        # --- END CORRECTION ---

    def step(self, hidden_states, inference_params: NeoXState):
        """
        A single-token step function for autoregressive generation.
        """
        # hidden_states: [batch_size, 1, hidden_size]
        
        # During step, we don't need a causal mask,
        # as seq_len_offset handles the key/value positions.
        attention_mask = None 
        
        # Call the full forward pass, which is optimized for single-token decoding
        # when layer_past_list (inference_params) is provided.
        hidden_states, new_state = self.forward(
            hidden_states,
            attention_mask=attention_mask,
            layer_past_list=inference_params
        )
        
        return hidden_states, new_state