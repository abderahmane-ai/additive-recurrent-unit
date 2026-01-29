"""
Hierarchical Additive Recurrent Unit (HARU)

A multi-scale extension of ARU that processes information at different temporal
resolutions through stacked ARU layers with specialized persistence characteristics.

Mathematical Formulation:
    For a K-layer HARU with update frequencies {k_0, k_1, ..., k_{K-1}}:
    
    Layer 0 (Fast):
        h^(0)_t = ARU_0(x_t, h^(0)_{t-1})
    
    Layer i > 0 (Slower):
        if t mod k_i == 0:
            h^(i)_t = ARU_i(h^(i-1)_t, h^(i)_{t-k_i})
        else:
            h^(i)_t = h^(i)_{t-1}  (hold previous state)
    
    Where:
        - ARU_i has persistence bias π_init^(i), with π_init^(i) > π_init^(i-1)
        - Each ARU handles its own input projection internally
        - Temporal smoothing comes from ARU's persistence gate
        - Higher layers receive continuously updated lower-layer representations
          but update their own state sparsely

Design Principles:
    1. Fast layers: Low persistence (τ≈2), high reactivity → captures transients
    2. Slow layers: High persistence (τ≈30), low reactivity → maintains context
    3. Information flows bottom-up, each ARU projects inputs internally
    4. Each layer operates on a potentially different clock (update frequency)
    5. Skip connections at output preserve multi-scale information

Author: Abderahmane Ainouche
License: MIT
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from torch import Tensor
from .model import ARU


class HARU(nn.Module):
    """
    Hierarchical Additive Recurrent Unit (HARU).
    
    A pure recurrent multi-timescale backbone that stacks ARU layers with different
    temporal characteristics. Each layer operates on a potentially different clock,
    creating a natural temporal hierarchy.
    
    Args:
        input_size: Input feature dimension or vocabulary size
        hidden_sizes: List of hidden dimensions for each layer [fast, medium, slow]
        persistence_inits: Persistence gate bias for each layer (higher = longer memory)
        accumulation_inits: Accumulation gate bias for each layer
        reset_inits: Reset gate bias for each layer
        num_classes: Output classes for classification (None for encoder)
        dropout: Dropout probability on final output
        use_embedding: Use embedding layer for discrete inputs
        use_skip_connections: Concatenate all layer outputs (vs. only final layer)
        update_frequencies: How often each layer updates (1=every step, 2=every 2 steps)
                           Higher layers can operate on slower clocks.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 128, 256],
        persistence_inits: Optional[List[float]] = None,
        accumulation_inits: Optional[List[float]] = None,
        reset_inits: Optional[List[float]] = None,
        num_classes: Optional[int] = None,
        dropout: float = 0.1,
        use_embedding: bool = False,
        use_skip_connections: bool = True,
        update_frequencies: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.num_classes = num_classes
        self.use_embedding = use_embedding
        self.use_skip_connections = use_skip_connections
        self.dropout_p = dropout
        
        # Default temporal characteristics: fast → slow
        if persistence_inits is None:
            # Fast: τ≈2, Medium: τ≈7, Slow: τ≈30
            persistence_inits = [0.5, 2.0, 3.5][:self.num_layers]
        
        if accumulation_inits is None:
            # Fast: reactive, Medium: balanced, Slow: conservative
            accumulation_inits = [1.0, 0.0, -1.0][:self.num_layers]
        
        if reset_inits is None:
            reset_inits = [2.0] * self.num_layers
        
        if update_frequencies is None:
            # Default: all layers update every timestep
            update_frequencies = [1] * self.num_layers
        
        assert len(update_frequencies) == self.num_layers, \
            f"update_frequencies length ({len(update_frequencies)}) must match num_layers ({self.num_layers})"
        
        self.update_frequencies = update_frequencies
        
        # Build ARU hierarchy
        self.layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            # Determine input configuration for this layer
            if i == 0:
                # First layer takes raw input
                layer_input_size = input_size
                layer_use_embedding = use_embedding
            else:
                # Higher layers take output from previous layer
                # Let ARU handle the projection internally
                layer_input_size = hidden_sizes[i-1]
                layer_use_embedding = False
            
            layer = ARU(
                input_size=layer_input_size,
                hidden_size=hidden_sizes[i],
                reset_init=reset_inits[i],
                persistence_init=persistence_inits[i],
                accumulation_init=accumulation_inits[i],
                num_classes=None,
                dropout=0.0,
                use_embedding=layer_use_embedding,
                use_layer_norm=False,
            )
            self.layers.append(layer)
        
        # Note: No explicit inter-layer projections needed
        # Each ARU handles its own input projection internally
        
        # Output dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        output_size = hidden_sizes[-1]
        if use_skip_connections and self.num_layers > 1:
            output_size = sum(hidden_sizes)
        
        if num_classes is not None:
            self.classifier = nn.Linear(output_size, num_classes)
        else:
            self.classifier = None
    
    def forward(
        self,
        x: Tensor,
        h0: Optional[List[Tensor]] = None,
        return_all_states: bool = False,
    ) -> Tensor:
        """
        Forward pass through hierarchical layers.
        
        Processes the sequence timestep-by-timestep through all layers, respecting
        update frequencies and inter-layer projections.
        
        Classification semantics:
            - return_all_states=False: Sequence-level classification (final timestep)
            - return_all_states=True: Per-timestep classification (tagging/prediction)
        
        Args:
            x: Input tensor (batch, seq_len) or (batch, seq_len, features)
            h0: Initial hidden states for each layer (list of tensors)
            return_all_states: Return hidden states at all timesteps
            
        Returns:
            Output logits or hidden states depending on configuration
        """
        if self.use_embedding:
            batch_size, seq_len = x.shape
        else:
            batch_size, seq_len, _ = x.shape
        
        # Initialize hidden states with correct dtype (match model parameters)
        param_dtype = next(self.parameters()).dtype
        if h0 is None:
            h0 = [
                torch.zeros(batch_size, h_size, device=x.device, dtype=param_dtype)
                for h_size in self.hidden_sizes
            ]
        
        # Storage for all timesteps (if needed)
        if return_all_states:
            all_layer_states = [[] for _ in range(self.num_layers)]
        
        hidden_states = h0
        
        # Core HARU recurrence: timestep-by-timestep processing
        for t in range(seq_len):
            # Extract input at timestep t
            if self.use_embedding:
                x_t = x[:, t]
            else:
                x_t = x[:, t, :]
            
            # Process through hierarchy
            for i in range(self.num_layers):
                # Check if this layer should update at this timestep
                if t % self.update_frequencies[i] == 0:
                    # Determine input to this layer
                    if i == 0:
                        # First layer receives raw input
                        layer_input = x_t
                    else:
                        # Higher layers receive previous layer's hidden state
                        # ARU will handle projection internally
                        layer_input = hidden_states[i-1]
                    
                    # Update hidden state via ARU step
                    hidden_states[i] = self.layers[i].step(layer_input, hidden_states[i])
                
                # Store state if needed
                if return_all_states:
                    all_layer_states[i].append(hidden_states[i])
        
        # Prepare output
        if return_all_states:
            # Stack states: (batch, seq_len, hidden_size) for each layer
            layer_sequences = [torch.stack(states, dim=1) for states in all_layer_states]
            
            # Combine layers
            if self.use_skip_connections and self.num_layers > 1:
                combined = torch.cat(layer_sequences, dim=-1)
            else:
                combined = layer_sequences[-1]
            
            # Classification
            if self.classifier is not None:
                h_flat = combined.reshape(-1, combined.size(-1))
                out = self.classifier(self.dropout(h_flat))
                return out.reshape(batch_size, seq_len, -1)
            
            return combined
        
        else:
            # Return only final state
            if self.use_skip_connections and self.num_layers > 1:
                combined = torch.cat(hidden_states, dim=-1)
            else:
                combined = hidden_states[-1]
            
            # Classification
            if self.classifier is not None:
                return self.classifier(self.dropout(combined))
            
            return combined
    
    def init_hidden(self, batch_size: int, device: torch.device) -> List[Tensor]:
        """Initialize hidden states for all layers."""
        param_dtype = next(self.parameters()).dtype
        return [
            torch.zeros(batch_size, h_size, device=device, dtype=param_dtype)
            for h_size in self.hidden_sizes
        ]
    
    def extra_repr(self) -> str:
        return (
            f"layers={self.num_layers}, hidden_sizes={self.hidden_sizes}, "
            f"num_classes={self.num_classes}, dropout={self.dropout_p}"
        )


def create_haru_classifier(
    vocab_size: int,
    hidden_sizes: List[int] = [64, 128, 256],
    num_classes: int = 2,
    dropout: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> HARU:
    """Create HARU for text classification."""
    return HARU(
        input_size=vocab_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout=dropout,
        use_embedding=True,
    ).to(device)


def create_haru_encoder(
    input_size: int,
    hidden_sizes: List[int] = [64, 128, 256],
    use_embedding: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> HARU:
    """Create HARU encoder for sequence modeling."""
    return HARU(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=None,
        use_embedding=use_embedding,
    ).to(device)
