import re
import copy
from dataclasses import dataclass, field
from typing import Optional, Union

import optree

import torch
import torch.nn as nn

from flash_attn.ops.triton.layer_norm import RMSNorm

from hnet.modules.block import create_block
from hnet.modules.utils import get_seq_idx, get_stage_cfg
from hnet.models.config_hnet import HNetConfig

# --- 1. IMPORT NEOX STACK AND STATE ---
from hnet.modules.neox.stack import NeoXStack, NeoXState
# --- END IMPORT ---


@dataclass
class IsotropicInferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()

        optree.tree_map(
            lambda x: x.zero_() if isinstance(x, torch.Tensor) else x,
            self.key_value_memory_dict,
        )


# --- 2. ADD MASK HELPER FUNCTION ---
def _build_causal_4d_mask(
    mask: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> Optional[torch.Tensor]:
    """
    Builds a 4D causal attention mask from a 2D padding mask.

    Arguments:
        mask: (batch_size, seq_len)
        dtype: torch.dtype
        device: torch.device

    Returns:
        (batch_size, 1, seq_len, seq_len) causal mask
    """
    if mask is None:
        return None

    batch_size, seq_len = mask.shape
    
    # Create a 4D causal mask: (1, 1, seq_len, seq_len)
    causal_mask = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
    
    # Create a 4D padding mask: (B, 1, 1, S)
    # Mask values are 1 for padding, 0 for non-padding
    padding_mask_4d = (1.0 - mask.to(dtype)).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
    
    # Combine them. Causal mask is bool, padding mask is float.
    # We want 0 for tokens we attend to, and -inf for tokens we mask out.
    
    # Invert causal mask: 0 for masked, 1 for attended
    causal_mask_float = (~causal_mask).to(dtype)
    
    # Combine with padding mask
    # 1. Start with causal mask (1 for attended, 0 for masked)
    # 2. Multiply by padding mask (1 for attended, 0 for masked)
    combined_mask = causal_mask_float * mask.unsqueeze(1).unsqueeze(2).to(dtype) # [B, 1, S, S]
    
    # Convert to additive mask (0 for attended, -inf for masked)
    additive_mask = (1.0 - combined_mask) * torch.finfo(dtype).min
    
    return additive_mask.to(dtype)
# --- END HELPER FUNCTION ---


class Isotropic(nn.Module):
    def __init__(
        self,
        config: HNetConfig,
        stage_idx: int,
        pos_idx: int,
        device=None,
        dtype=None,
    ):
        """
        config: HNetConfig. The top-level config for the HNet.
        stage_idx: int. The current stage index.
        pos_idx: int. The position index of the current isotropic layer in the architecture layout.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        arch_at_stage = config.arch_layout[stage_idx]
        arch = arch_at_stage[pos_idx]
        self.arch = arch

        current_d_model = config.d_model[stage_idx]
        current_d_intermediate = config.d_intermediate[stage_idx]

        # --- 3. ADD CONDITIONAL LOGIC for NeoX vs Mamba/MHA ---
        if arch.startswith("T"):
            # New NeoX logic
            self.is_neox = True
            
            num_layers = int(arch[1:])
            
            neox_cfg = config.neox_cfg
            neox_cfg.num_layers = num_layers
            neox_cfg.hidden_size = current_d_model
            neox_cfg.d_intermediate = current_d_intermediate
            
            current_num_heads = config.attn_cfg.num_heads[stage_idx]
            current_rotary_dim = config.attn_cfg.rotary_emb_dim[stage_idx]
            
            neox_cfg.num_attention_heads = current_num_heads
            neox_cfg.rotary_emb_dim = current_rotary_dim
            if current_d_model > 0 and current_num_heads > 0:
                head_dim = current_d_model // current_num_heads
                if head_dim == 0:
                    print(f"Warning: d_model {current_d_model} // num_heads {current_num_heads} is 0.")
                    neox_cfg.rotary_pct = 0.0
                else:
                    neox_cfg.rotary_pct = current_rotary_dim / head_dim
            else:
                neox_cfg.rotary_pct = 0.0

            # Instantiate the single NeoXStack module
            self.stack = NeoXStack(config=neox_cfg)
            
            # NeoXStack does not have a final norm, but Isotropic does.
            # We will use Isotropic's existing RMSNorm.
            self.rmsnorm = RMSNorm(
                current_d_model, eps=1e-5, **factory_kwargs
            )
            # We don't need self.layers for NeoX
            self.layers = nn.ModuleList()

        elif arch.startswith("m") or arch.startswith("t"):
            # Original Mamba/MHA logic
            self.is_neox = False

            # we find all "m" or "t" blocks in the arch string
            layers_arch = re.findall(r"([a-z])(\d+)", arch)

            self.layers = nn.ModuleList()
            for i, (block_type, num_layers) in enumerate(layers_arch):
                if num_layers == "0":
                    continue

                for _ in range(int(num_layers)):
                    self.layers.append(
                        create_block(
                            block_type,
                            d_model=current_d_model,
                            d_intermediate=get_stage_cfg(
                                current_d_intermediate, i, len(layers_arch)
                            ),
                            attn_cfg=get_stage_cfg(
                                config.attn_cfg, i, len(layers_arch)
                            ),
                            ssm_cfg=get_stage_cfg(
                                config.ssm_cfg, i, len(layers_arch)
                            ),
                            layer_idx=len(self.layers),
                            stage_idx=stage_idx,
                            **factory_kwargs,
                        )
                    )

            if len(self.layers) == 0:
                self.layers.append(nn.Identity())

            self.rmsnorm = RMSNorm(current_d_model, eps=1e-5, **factory_kwargs)

        else:
            raise ValueError(f"Unknown arch string in Isotropic: {arch}")
        
        self.height = len(self.layers) if not self.is_neox else self.stack.config.num_layers
        # --- END MODIFICATION ---

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        mask=None,
        inference_params=None,
        **mixer_kwargs,
    ):
        
        # --- 4. MODIFY FORWARD ---
        if self.is_neox:
            # Call the NeoXStack
            
            # Create the 4D causal attention mask from the 2D padding mask
            attention_mask = _build_causal_4d_mask(
                mask, dtype=hidden_states.dtype, device=hidden_states.device
            )
            
            # The NeoX stack is self-contained. It does not use the
            # sequential residual from Isotropic.
            residual = None
            
            hidden_states, presents = self.stack.forward(
                hidden_states,
                attention_mask=attention_mask,
                layer_past_list=inference_params  # This will be NeoXState
            )
            
            # Apply the final RMSNorm
            hidden_states = self.rmsnorm(
                hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
            )
            
            if inference_params is not None:
                assert mask.shape[0] == 1, "seqlen_offset handling assumes batch size 1"
                inference_params.seqlen_offset += hidden_states.shape[1]
                
            return hidden_states, presents # Return the new NeoXState

        else:
            # Original Isotropic forward logic
            residual = None
            packed = cu_seqlens is not None
            if packed:
                # This is a packed tensor, we need to add the batch dimension
                hidden_states = hidden_states.unsqueeze(0)

            for i, layer in enumerate(self.layers):
                layer_mixer_kwargs = get_stage_cfg(mixer_kwargs, i, len(self.layers))
                if layer_mixer_kwargs is None:
                    layer_mixer_kwargs = {}

                # currently supporting only Mamba2 and MHA
                if not (
                    hasattr(layer, "mixer")
                    and (
                        layer.mixer.__class__.__name__ == "Mamba2Wrapper"
                        or layer.mixer.__class__.__name__ == "CausalMHA"
                    )
                ):
                    # currently supporting only Mamba2 and MHA
                    raise NotImplementedError

                hidden_states, residual = layer(
                    hidden_states,
                    residual,
                    inference_params=inference_params,
                    mixer_kwargs=layer_mixer_kwargs,
                )

            # Setting prenorm=False ignores the residual
            hidden_states = self.rmsnorm(
                hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
            )

            if hidden_states.dim() == 3 and packed:
                hidden_states = hidden_states.squeeze(0)

            if inference_params is not None:
                # here we also explicitly assume the mask is all True
                assert mask.shape[0] == 1, "seqlen_offset handling assumes batch size 1"
                inference_params.seqlen_offset += hidden_states.shape[1]

            return hidden_states, inference_params # Return the Mamba/MHA state
        # --- END MODIFICATION ---

    def step(self, hidden_states, inference_params: Union[IsotropicInferenceParams, NeoXState]):
        """
        Assumes hidden_states is (B, 1, D). Steps each of the layers in order, and then steps the main model.
        """
        
        # --- 5. MODIFY STEP ---
        if self.is_neox:
            # NeoX step
            residual = None
            hidden_states, new_state = self.stack.step(
                hidden_states,
                inference_params=inference_params
            )
            hidden_states = self.rmsnorm(
                hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
            )
            return hidden_states, new_state
            
        else:
            # Original Isotropic step
            residual = None
            for layer in self.layers:
                hidden_states, residual = layer.step(
                    hidden_states, inference_params, residual=residual
                )

            hidden_states = self.rmsnorm(
                hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
            )
            return hidden_states, inference_params
        # --- END MODIFICATION ---

    def allocate_inference_cache(
        self, batch_size, max_seqlen, dtype=None, **kwargs
    ) -> Union[IsotropicInferenceParams, NeoXState]:
        
        # --- 6. MODIFY ALLOCATE_INFERENCE_CACHE ---
        if self.is_neox:
            # Allocate NeoX state
            return self.stack.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype
            )
            
        else:
            # Original Isotropic/Mamba/MHA cache allocation
            inference_params = IsotropicInferenceParams(
                max_seqlen=max_seqlen,
                max_batch_size=batch_size,
                key_value_memory_dict={},
            )
            for layer in self.layers:
                inference_params.key_value_memory_dict[
                    layer.mixer.layer_idx
                ] = layer.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype, **kwargs
                )
            return inference_params
        # --- END MODIFICATION ---