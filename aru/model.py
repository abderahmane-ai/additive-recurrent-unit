"""
Additive Recurrent Unit (ARU) - Three-Gate Architecture.

A recurrent neural network that enables additive information accumulation
through three independent gates: reset (ρ), persistence (π), and accumulation (α).

Core equation: h_t = ρ_t ⊙ (π_t ⊙ h_{t-1} + α_t ⊙ v_t)

Unlike GRU/LSTM which constrain gates, ARU's three independent gates enable
true additive accumulation with selective forgetting.

Author: Abderahmane Ainouche
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch import Tensor


@torch.jit.script
def _aru_recurrence(
    gate_x_seq: Tensor,
    cand_seq: Tensor,
    h: Tensor,
    gate_h_weight: Tensor,
    hidden_size: int,
) -> Tensor:
    """JIT-compiled ARU recurrence for efficient long sequence processing."""
    seq_len = gate_x_seq.size(1)
    for t in range(seq_len):
        gates_x = gate_x_seq[:, t]
        v_t = cand_seq[:, t]
        gates_h = torch.mm(h, gate_h_weight)
        gates = torch.sigmoid(gates_x + gates_h)
        rho = gates[:, :hidden_size]
        pi = gates[:, hidden_size : 2 * hidden_size]
        alpha = gates[:, 2 * hidden_size :]
        h = rho * (pi * h + alpha * v_t)
    return h


@torch.jit.script
def _aru_recurrence_all_states(
    gate_x_seq: Tensor,
    cand_seq: Tensor,
    h: Tensor,
    gate_h_weight: Tensor,
    hidden_size: int,
) -> Tensor:
    """JIT-compiled ARU recurrence returning hidden states at all timesteps."""
    batch_size = gate_x_seq.size(0)
    seq_len = gate_x_seq.size(1)
    all_states = torch.zeros(
        batch_size, seq_len, hidden_size, device=h.device, dtype=h.dtype
    )
    for t in range(seq_len):
        gates_x = gate_x_seq[:, t]
        v_t = cand_seq[:, t]
        gates_h = torch.mm(h, gate_h_weight)
        gates = torch.sigmoid(gates_x + gates_h)
        rho = gates[:, :hidden_size]
        pi = gates[:, hidden_size : 2 * hidden_size]
        alpha = gates[:, 2 * hidden_size :]
        h = rho * (pi * h + alpha * v_t)
        all_states[:, t] = h
    return all_states


class ARU(nn.Module):
    """
    Additive Recurrent Unit (ARU) - Three-Gate Architecture.
    
    h_t = ρ_t ⊙ (π_t ⊙ h_{t-1} + α_t ⊙ v_t)
    
    Gates:
        ρ (reset):       Controls selective erasure. ρ≈0 erases, ρ≈1 keeps.
        π (persistence): Controls retention. π≈0 forgets, π≈1 maintains.
        α (accumulation): Controls input. α≈0 ignores, α≈1 incorporates.
        v (candidate):   New information, v_t = tanh(W·x_t), independent of h.
    
    Key modes:
        ρ≈1, π≈1, α≈1: Pure accumulation (h_t ≈ h_{t-1} + v_t)
        ρ≈0, π≈*, α≈1: Hard reset (h_t ≈ 0)
        ρ≈1, π≈0, α≈1: Replace (h_t ≈ v_t)
        ρ≈1, π≈1, α≈0: Maintain (h_t ≈ h_{t-1})
    
    Gradient flow note:
        The gradient through the hidden state is scaled by rho * pi at each step.
        Over T steps, gradients are scaled by ∏(rho_t * pi_t). When gates are near 1
        (the default initialization), this provides a near-linear gradient path.
        When gates close, gradient attenuation is multiplicative — this is the
        intended behavior (selective forgetting should also forget gradients).

        For comparison: LSTM's cell state provides c_t = f_t * c_{t-1} + i_t * g_t,
        which has a pure additive path when f_t=1, giving gradient scaling of exactly 1
        through the cell state. ARU trades this unconditional gradient highway for fully
        decoupled gates — rho and pi can independently control erasure and retention
        without the coupling constraint of LSTM's f and i gates (which sum to 1 when
        the output gate is omitted).

    Args:
        input_size: Vocabulary size (use_embedding=True) or feature dim.
        hidden_size: Hidden state dimension.
        reset_init: Reset gate bias init. sigmoid(2.0) ≈ 0.88.
        persistence_init: Persistence gate bias init.
        accumulation_init: Accumulation gate bias init. sigmoid(0) = 0.5.
        num_classes: Output classes for classification (None for encoder).
        dropout: Dropout probability.
        use_embedding: Use embedding layer for discrete inputs.
        use_layer_norm: Apply layer norm to candidates.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        reset_init: float = 2.0,
        persistence_init: float = 2.0,
        accumulation_init: float = 0.0,
        num_classes: Optional[int] = None,
        dropout: float = 0.1,
        use_embedding: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_embedding = use_embedding
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.reset_init = reset_init
        self.persistence_init = persistence_init
        self.accumulation_init = accumulation_init
        self.use_layer_norm = use_layer_norm
        
        # Input processing
        self.input_proj = None
        if use_embedding:
            self.embed = nn.Embedding(input_size, hidden_size, padding_idx=0)
            self.gate_input_size = hidden_size
            self.cand_input_size = hidden_size
        else:
            # Skip projection for small inputs (efficiency)
            if input_size < hidden_size // 4:
                self.gate_input_size = input_size
                self.cand_input_size = input_size
            else:
                self.input_proj = nn.Linear(input_size, hidden_size, bias=False)
                self.gate_input_size = hidden_size
                self.cand_input_size = hidden_size
        
        # Candidate projection (input-only, no h-dependency)
        self.cand_proj = nn.Linear(self.cand_input_size, hidden_size)
        
        if use_layer_norm:
            self.cand_norm = nn.LayerNorm(hidden_size)
        
        # Independent gate projections — each gate gets its own transformation
        # This ensures truly decoupled retention, persistence, and accumulation
        self.rho_x_proj = nn.Linear(self.gate_input_size, hidden_size, bias=True)
        self.pi_x_proj = nn.Linear(self.gate_input_size, hidden_size, bias=True)
        self.alpha_x_proj = nn.Linear(self.gate_input_size, hidden_size, bias=True)
        self.rho_h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pi_h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha_h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Gate bias initialization
        with torch.no_grad():
            self.rho_x_proj.bias.fill_(reset_init)
            self.pi_x_proj.bias.fill_(persistence_init)
            self.alpha_x_proj.bias.fill_(accumulation_init)
        
        self.dropout = nn.Dropout(dropout)
        
        if num_classes is not None:
            self.classifier = nn.Linear(hidden_size, num_classes)
        else:
            self.classifier = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal weight initialization for stable training."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'embed' not in name:
                    nn.init.orthogonal_(param)
            elif 'bias' in name and not any(n in name for n in ('rho_x_proj', 'pi_x_proj', 'alpha_x_proj')):
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: Tensor,
        h0: Optional[Tensor] = None,
        return_all_states: bool = False,
        return_final_state: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len) for embeddings, (batch, seq_len, input_size) otherwise.
            h0: Initial hidden state. Defaults to zeros.
            return_all_states: Return states at all timesteps.
            return_final_state: Return (output, final_hidden) tuple.
        
        Returns:
            Logits (batch, num_classes) or hidden states depending on config.
        """
        if self.use_embedding:
            batch_size, seq_len = x.shape
            embeds = self.embed(x)
            gate_input = embeds
            cand_input_raw = embeds
        else:
            batch_size, seq_len, _ = x.shape
            if self.input_proj is not None:
                embeds = self.input_proj(x)
                gate_input = embeds
                cand_input_raw = embeds
            else:
                embeds = x
                gate_input = x
                cand_input_raw = x
        
        # Always apply dropout (nn.Dropout(0.0) is a no-op, so this is safe
        # for benchmarks that use dropout=0.0 and matches baseline behavior)
        embeds = self.dropout(embeds)
        
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=embeds.dtype)
        else:
            h = h0
        
        # Pre-compute input-dependent terms (key optimization)
        # Each gate independently transforms the input — maintains decoupling
        rho_x = self.rho_x_proj(gate_input)
        pi_x = self.pi_x_proj(gate_input)
        alpha_x = self.alpha_x_proj(gate_input)
        gate_x_precomputed = torch.cat([rho_x, pi_x, alpha_x], dim=-1)
        cand_proj_out = self.cand_proj(cand_input_raw)
        if self.use_layer_norm:
            cand_proj_out = self.cand_norm(cand_proj_out)
        cand_precomputed = torch.tanh(cand_proj_out)
        
        # Build concatenated hidden-projection weight for single mm in JIT loop
        gate_h_weight = torch.cat([
            self.rho_h_proj.weight,
            self.pi_h_proj.weight,
            self.alpha_h_proj.weight,
        ], dim=0).t()
        
        if return_all_states:
            all_states = _aru_recurrence_all_states(
                gate_x_precomputed, cand_precomputed, h, gate_h_weight, self.hidden_size
            )
            if self.classifier is not None:
                h_flat = all_states.reshape(-1, self.hidden_size)
                out = self.classifier(self.dropout(h_flat))
                return out.reshape(batch_size, seq_len, -1)
            return all_states
        
        h = _aru_recurrence(
            gate_x_precomputed, cand_precomputed, h, gate_h_weight, self.hidden_size
        )
        
        h_drop = self.dropout(h)
        
        if self.classifier is not None:
            out = self.classifier(h_drop)
            return (out, h) if return_final_state else out
        
        return (h, h) if return_final_state else h
    
    def _step(self, gates_x: Tensor, v_t: Tensor, h: Tensor) -> Tensor:
        """Single timestep with pre-computed inputs. Used by step()."""
        rho_h = self.rho_h_proj(h)
        pi_h = self.pi_h_proj(h)
        alpha_h = self.alpha_h_proj(h)
        gates_h = torch.cat([rho_h, pi_h, alpha_h], dim=-1)
        gates = torch.sigmoid(gates_x + gates_h)
        rho = gates[:, :self.hidden_size]
        pi = gates[:, self.hidden_size:2*self.hidden_size]
        alpha = gates[:, 2*self.hidden_size:]
        return rho * (pi * h + alpha * v_t)
    
    def step(self, x_t: Tensor, h: Tensor) -> Tensor:
        """
        Single timestep for autoregressive generation.
        
        Args:
            x_t: (batch,) for embeddings, (batch, input_size) otherwise.
            h: Previous hidden state (batch, hidden_size).
        
        Returns:
            New hidden state (batch, hidden_size).
        """
        if self.use_embedding:
            e_t = self.embed(x_t)
            gate_input = e_t
            cand_input = e_t
        else:
            if self.input_proj is not None:
                e_t = self.input_proj(x_t)
                gate_input = e_t
                cand_input = e_t
            else:
                gate_input = x_t
                cand_input = x_t
        
        rho_x = self.rho_x_proj(gate_input)
        pi_x = self.pi_x_proj(gate_input)
        alpha_x = self.alpha_x_proj(gate_input)
        gates_x = torch.cat([rho_x, pi_x, alpha_x], dim=-1)
        v_input = self.cand_proj(cand_input)
        if self.use_layer_norm:
            v_input = self.cand_norm(v_input)
        v_t = torch.tanh(v_input)
        
        return self._step(gates_x, v_t, h)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(batch_size, self.hidden_size, device=device)
    
    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_classes={self.num_classes}, dropout={self.dropout_p}"
        )


def create_aru_classifier(
    vocab_size: int,
    hidden_size: int = 128,
    num_classes: int = 2,
    dropout: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> ARU:
    """Create ARU for text classification."""
    return ARU(
        input_size=vocab_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout=dropout,
        use_embedding=True,
    ).to(device)


def create_aru_encoder(
    input_size: int,
    hidden_size: int = 64,
    use_embedding: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> ARU:
    """Create ARU encoder for sequence modeling."""
    return ARU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=None,
        use_embedding=use_embedding,
    ).to(device)


@torch.jit.script
def aru_loss(logits: Tensor, targets: Tensor, label_smoothing: float = 0.0) -> Tensor:
    """Cross-entropy loss with optional label smoothing.

    Note: Current benchmarks use nn.CrossEntropyLoss() directly for chunked computation
    efficiency. This function is available as a convenience for experiments requiring
    label smoothing.

    Args:
        logits: Raw model logits (batch, num_classes).
        targets: Ground-truth class indices (batch,).
        label_smoothing: Smoothing factor (0.0 = standard CE, >0 softens targets).
    """
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
