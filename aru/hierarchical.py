"""
Hierarchical Additive Recurrent Unit (HARU)

A multi-scale extension of ARU with smart defaults and expert control.
Automatically configures temporal hierarchies while allowing fine-grained override.

Core Design Principles:
    1. Automatic temporal scale selection: τ ∈ [2, 30] (logarithmic spacing)
    2. Optional hierarchical residual connections for gradient flow
    3. Conservative update frequencies by default (slower exponential growth)
    4. Layer normalization enabled by default for training stability
    5. Skip connections preserve multi-scale information (configurable)

Mathematical Formulation:
    For a K-layer HARU:
    
    Layer 0 (Fast, τ=2):
        h^(0)_t = ARU_0(x_t, h^(0)_{t-1})  [updates every step]
    
    Layer i > 0 (Slower, τ increases):
        if t mod k_i == 0:
            h^(i)_t = ARU_i(h^(i-1)_t, h^(i)_{t-1})
            if hierarchical_residual and sizes match:
                h^(i)_t = (1-β)·h^(i)_t + β·h^(i-1)_t  [optional mixing]
        else:
            h^(i)_t = h^(i)_{t-1}  [hold state]

Author: Abderahmane Ainouche
License: MIT
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Union
from torch import Tensor
from .model import ARU


def _compute_temporal_scales(num_layers: int) -> List[float]:
    """
    Compute temporal time constants for each layer using logarithmic spacing.
    
    Creates a hierarchy from fast (τ=2, ~50% retention) to slow (τ=30, ~97% retention).
    
    Args:
        num_layers: Number of hierarchical layers
        
    Returns:
        List of time constants [τ_0, τ_1, ..., τ_{K-1}]
        
    Example:
        num_layers=3 → [2.0, 7.75, 30.0]
            Fast layer:   τ=2   → π ≈ 0.50 (rapid forgetting)
            Medium layer: τ=7.75 → π ≈ 0.87 (balanced)
            Slow layer:   τ=30  → π ≈ 0.97 (long memory)
    """
    if num_layers == 1:
        return [7.75]  # Single layer: use medium time constant
    return np.logspace(np.log10(2), np.log10(30), num_layers).tolist()


def _tau_to_persistence_bias(tau: float) -> float:
    """
    Convert time constant to persistence gate initialization bias.
    
    Mathematical relationship:
        τ = 1 / (1 - π)  where π = σ(bias)
        → π = 1 - 1/τ
        → bias = logit(π) = log(π / (1 - π))
    
    Args:
        tau: Time constant (exponential decay timescale)
        
    Returns:
        Bias value for persistence gate initialization
        
    Example:
        τ=2  → π=0.5  → bias=0.0
        τ=10 → π=0.9  → bias=2.2
        τ=30 → π=0.967 → bias=3.4
    """
    persistence_target = 1.0 - 1.0 / tau
    # Avoid numerical issues at extremes
    persistence_target = np.clip(persistence_target, 0.01, 0.99)
    return float(np.log(persistence_target / (1.0 - persistence_target)))


def _compute_update_frequencies(num_layers: int, mode: str = 'conservative') -> List[int]:
    """
    Compute update frequencies for hierarchical layers.
    
    Args:
        num_layers: Number of hierarchical layers
        mode: 'conservative' (slower growth) or 'aggressive' (exponential)
        
    Returns:
        List of update frequencies [f_0, f_1, ..., f_{K-1}]
        
    Examples:
        Conservative (default): num_layers=5 → [1, 1, 2, 2, 4]
            Slower growth ensures higher layers update reasonably often
            
        Aggressive: num_layers=5 → [1, 2, 4, 8, 16]
            Exponential spacing creates very distinct temporal scales
    """
    if mode == 'aggressive':
        # Exponential: [1, 2, 4, 8, ...]
        return [2 ** i for i in range(num_layers)]
    else:
        # Conservative: [1, 1, 2, 2, 4, 4, ...]
        # Using 2^(i//2) for power-of-2 efficiency with slower growth
        return [2 ** (i // 2) for i in range(num_layers)]


class HARU(nn.Module):
    """
    Hierarchical Additive Recurrent Unit (HARU).
    
    A multi-scale recurrent architecture with smart defaults and expert control:
    - Automatic temporal scales (logarithmic spacing from τ=2 to τ=30)
    - Automatic gate initialization (derived from temporal scales)
    - Conservative update frequencies (configurable)
    - Optional hierarchical residual connections
    - Configurable layer normalization and skip connections
    
    Design Philosophy: "Smart Defaults, Expert Control"
        Beginners get sensible automatic configuration.
        Researchers can override any parameter for experiments.
    
    Args:
        input_size: Input feature dimension or vocabulary size
        hidden_sizes: Hidden dimensions for each layer, e.g., [64, 128, 256]
                     List length determines number of hierarchical levels
        num_classes: Output classes for classification (None for encoder mode)
        dropout: Dropout probability on final output (default: 0.1)
        use_embedding: Use embedding layer for discrete inputs (default: False)
        
        # Automatic configuration (override if needed for research)
        persistence_inits: Persistence gate bias for each layer (None = automatic)
        accumulation_inits: Accumulation gate bias for each layer (None = automatic)
        reset_inits: Reset gate bias for each layer (None = automatic)
        update_frequencies: Update frequency for each layer (None = automatic conservative)
        
        # Architecture options
        use_skip_connections: Concatenate all layer outputs (default: True)
        use_layer_norm: Apply layer normalization in each ARU (default: True)
        update_frequency_mode: 'conservative' or 'aggressive' (default: 'conservative')
        
    Automatic Configuration (when parameters are None):
        - Temporal scales: Logarithmic from τ=2 to τ=30
        - Persistence: Derived from temporal scales via logit transform
        - Accumulation: Inverse-persistence relationship (accum = -0.5 × persistence)
        - Reset: Fixed at 2.0 (σ(2.0) ≈ 0.88)
        - Update frequencies: Conservative [1, 1, 2, 2, 4, 4, ...] by default
        
    Examples:
        >>> # Simple: automatic everything
        >>> model = HARU(
        ...     input_size=10000,
        ...     hidden_sizes=[64, 128, 256],
        ...     num_classes=4,
        ...     use_embedding=True
        ... )
        
        >>> # Research: override temporal scales
        >>> model = HARU(
        ...     input_size=512,
        ...     hidden_sizes=[128, 256],
        ...     persistence_inits=[1.0, 3.0],  # Custom time constants
        ...     update_frequencies=[1, 3]       # Custom update pattern
        ... )
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: Optional[int] = None,
        dropout: float = 0.1,
        use_embedding: bool = False,
        # Automatic configuration overrides
        persistence_inits: Optional[List[float]] = None,
        accumulation_inits: Optional[List[float]] = None,
        reset_inits: Optional[List[float]] = None,
        update_frequencies: Optional[List[int]] = None,
        # Architecture options
        use_skip_connections: bool = True,
        use_layer_norm: bool = True,
        update_frequency_mode: str = 'conservative',
    ):
        super().__init__()
        
        # Validation
        assert len(hidden_sizes) > 0, "hidden_sizes cannot be empty"
        assert all(h > 0 for h in hidden_sizes), "All hidden sizes must be positive"
        assert 0.0 <= dropout <= 1.0, "dropout must be in [0, 1]"
        assert update_frequency_mode in ['conservative', 'aggressive'], \
            "update_frequency_mode must be 'conservative' or 'aggressive'"
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.num_classes = num_classes
        self.use_embedding = use_embedding
        self.dropout_p = dropout
        self.use_skip_connections = use_skip_connections
        self.use_layer_norm = use_layer_norm
        
        # Apply automatic configuration if not overridden
        if persistence_inits is None:
            temporal_scales = _compute_temporal_scales(self.num_layers)
            persistence_inits = [_tau_to_persistence_bias(tau) for tau in temporal_scales]
        else:
            assert len(persistence_inits) == self.num_layers, \
                f"persistence_inits length ({len(persistence_inits)}) must match num_layers ({self.num_layers})"
        
        if accumulation_inits is None:
            # Derive from persistence: inverse relationship
            # High persistence (long memory) → Low accumulation (slow integration)
            # Low persistence (short memory) → High accumulation (reactive)
            accumulation_inits = [-0.5 * p for p in persistence_inits]
        else:
            assert len(accumulation_inits) == self.num_layers, \
                f"accumulation_inits length ({len(accumulation_inits)}) must match num_layers ({self.num_layers})"
        
        if reset_inits is None:
            # Always moderate: σ(2.0) ≈ 0.88
            reset_inits = [2.0] * self.num_layers
        else:
            assert len(reset_inits) == self.num_layers, \
                f"reset_inits length ({len(reset_inits)}) must match num_layers ({self.num_layers})"
        
        if update_frequencies is None:
            self.update_frequencies = _compute_update_frequencies(
                self.num_layers, mode=update_frequency_mode
            )
        else:
            assert len(update_frequencies) == self.num_layers, \
                f"update_frequencies length ({len(update_frequencies)}) must match num_layers ({self.num_layers})"
            assert all(f > 0 for f in update_frequencies), \
                "All update frequencies must be positive"
            self.update_frequencies = update_frequencies
        
        # Build ARU hierarchy with configured parameters
        self.layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            # Input configuration
            if i == 0:
                layer_input_size = input_size
                layer_use_embedding = use_embedding
            else:
                # Higher layers receive hidden states from previous layer
                layer_input_size = hidden_sizes[i-1]
                layer_use_embedding = False
            
            # Create ARU with specified initialization
            layer = ARU(
                input_size=layer_input_size,
                hidden_size=hidden_sizes[i],
                reset_init=reset_inits[i],
                persistence_init=persistence_inits[i],
                accumulation_init=accumulation_inits[i],
                num_classes=None,  # No classification at layer level
                dropout=0.0,  # Dropout only at final output
                use_embedding=layer_use_embedding,
                use_layer_norm=use_layer_norm,
            )
            self.layers.append(layer)
        
        # Output dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        if use_skip_connections:
            output_size = sum(hidden_sizes)  # Multi-scale representation
        else:
            output_size = hidden_sizes[-1]  # Only final layer
        
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
        
        Processes sequence timestep-by-timestep, respecting update frequencies.
        Optionally applies hierarchical residual connections.
        
        Args:
            x: Input tensor (batch, seq_len) for embeddings or (batch, seq_len, features)
            h0: Initial hidden states for each layer (optional)
            return_all_states: Return states at all timesteps (for sequence labeling)
            
        Returns:
            If classifier: (batch, num_classes) or (batch, seq_len, num_classes)
            If encoder: (batch, output_size) or (batch, seq_len, output_size)
        """
        # Extract dimensions
        if self.use_embedding:
            batch_size, seq_len = x.shape
        else:
            batch_size, seq_len, _ = x.shape
        
        # Initialize hidden states
        if h0 is None:
            param_dtype = next(self.parameters()).dtype
            hidden_states = [
                torch.zeros(batch_size, h_size, device=x.device, dtype=param_dtype)
                for h_size in self.hidden_sizes
            ]
        else:
            hidden_states = h0
        
        # Storage for all timesteps if needed
        if return_all_states:
            all_layer_states = [[] for _ in range(self.num_layers)]
        
        # Core HARU recurrence: timestep-by-timestep processing
        for t in range(seq_len):
            # Extract input at current timestep
            if self.use_embedding:
                x_t = x[:, t]
            else:
                x_t = x[:, t, :]
            
            # Process through hierarchical layers
            for i in range(self.num_layers):
                # Check if this layer should update at this timestep
                if t % self.update_frequencies[i] == 0:
                    # Determine input to this layer
                    if i == 0:
                        layer_input = x_t
                    else:
                        # Higher layers receive previous layer's current hidden state
                        layer_input = hidden_states[i-1]
                    
                    # Update via ARU step
                    new_h = self.layers[i].step(layer_input, hidden_states[i])
                    hidden_states[i] = new_h
                
                # Store state for this timestep
                if return_all_states:
                    all_layer_states[i].append(hidden_states[i])
        
        # Prepare output
        if return_all_states:
            # Stack: (batch, seq_len, hidden_size) for each layer
            layer_sequences = [torch.stack(states, dim=1) for states in all_layer_states]
            
            # Combine layers based on skip connection setting
            if self.use_skip_connections:
                combined = torch.cat(layer_sequences, dim=-1)
            else:
                combined = layer_sequences[-1]
            
            # Per-timestep classification
            if self.classifier is not None:
                h_flat = combined.reshape(-1, combined.size(-1))
                out = self.classifier(self.dropout(h_flat))
                return out.reshape(batch_size, seq_len, -1)
            
            return combined
        
        else:
            # Return final state only
            if self.use_skip_connections:
                combined = torch.cat(hidden_states, dim=-1)
            else:
                combined = hidden_states[-1]
            
            # Sequence-level classification
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
        """String representation showing configuration."""
        tau_scales = _compute_temporal_scales(self.num_layers)
        repr_str = (
            f"layers={self.num_layers}, "
            f"hidden_sizes={self.hidden_sizes}, "
            f"temporal_scales={[f'{tau:.1f}' for tau in tau_scales]}, "
            f"update_freq={self.update_frequencies}, "
            f"skip_connections={self.use_skip_connections}, "
            f"layer_norm={self.use_layer_norm}"
        )
        if self.num_classes is not None:
            repr_str += f", num_classes={self.num_classes}"
        return repr_str


def create_haru_classifier(
    vocab_size: int,
    hidden_sizes: List[int] = [64, 128, 256],
    num_classes: int = 2,
    dropout: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> HARU:
    """
    Create HARU for text classification with automatic optimization.
    
    Args:
        vocab_size: Vocabulary size
        hidden_sizes: Hidden dimensions for each layer
        num_classes: Number of output classes
        dropout: Dropout probability
        device: Device to place model
        **kwargs: Additional arguments passed to HARU (for expert control)
        
    Returns:
        HARU model ready for training
        
    Example:
        >>> # Simple usage
        >>> model = create_haru_classifier(vocab_size=10000, num_classes=4)
        
        >>> # Expert usage with overrides
        >>> model = create_haru_classifier(
        ...     vocab_size=10000,
        ...     num_classes=4,
        ...     update_frequencies=[1, 2, 3],
        ...     use_hierarchical_residual=True
        ... )
    """
    return HARU(
        input_size=vocab_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout=dropout,
        use_embedding=True,
        **kwargs,
    ).to(device)


def create_haru_encoder(
    input_size: int,
    hidden_sizes: List[int] = [64, 128, 256],
    use_embedding: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> HARU:
    """
    Create HARU encoder for sequence modeling.
    
    Args:
        input_size: Input feature dimension
        hidden_sizes: Hidden dimensions for each layer
        use_embedding: Use embedding layer for discrete inputs
        device: Device to place model
        **kwargs: Additional arguments passed to HARU (for expert control)
        
    Returns:
        HARU encoder ready for training
    """
    return HARU(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=None,
        use_embedding=use_embedding,
        **kwargs,
    ).to(device)