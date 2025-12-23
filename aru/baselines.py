"""
Baseline RNN implementations (GRU, LSTM, RNN) for fair comparison with ARU.

All models use the same optimization strategy:
- Pre-compute input projections outside the recurrence loop
- JIT-compiled recurrence for efficient long sequence processing
- Orthogonal weight initialization
- Consistent API (forward, step, init_hidden)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor


# =============================================================================
# JIT-COMPILED RECURRENCE FUNCTIONS
# =============================================================================


@torch.jit.script
def _gru_recurrence(
    gate_x_seq: Tensor, cand_x_seq: Tensor, h: Tensor,
    gate_h_weight: Tensor, cand_h_weight: Tensor, hidden_size: int,
) -> Tensor:
    """JIT-compiled GRU recurrence."""
    for t in range(gate_x_seq.size(1)):
        gate_x_t = gate_x_seq[:, t]
        cand_x_t = cand_x_seq[:, t]
        gate_h_t = torch.mm(h, gate_h_weight)
        gates = torch.sigmoid(gate_x_t + gate_h_t)
        z = gates[:, :hidden_size]
        r = gates[:, hidden_size:]
        cand_h_t = torch.mm(r * h, cand_h_weight)
        h_tilde = torch.tanh(cand_x_t + cand_h_t)
        h = (1 - z) * h_tilde + z * h
    return h


@torch.jit.script
def _gru_recurrence_all_states(
    gate_x_seq: Tensor, cand_x_seq: Tensor, h: Tensor,
    gate_h_weight: Tensor, cand_h_weight: Tensor, hidden_size: int,
) -> Tensor:
    """JIT-compiled GRU recurrence returning all states."""
    batch_size, seq_len = gate_x_seq.size(0), gate_x_seq.size(1)
    all_states = torch.zeros(batch_size, seq_len, hidden_size, device=h.device, dtype=h.dtype)
    for t in range(seq_len):
        gate_x_t = gate_x_seq[:, t]
        cand_x_t = cand_x_seq[:, t]
        gate_h_t = torch.mm(h, gate_h_weight)
        gates = torch.sigmoid(gate_x_t + gate_h_t)
        z = gates[:, :hidden_size]
        r = gates[:, hidden_size:]
        cand_h_t = torch.mm(r * h, cand_h_weight)
        h_tilde = torch.tanh(cand_x_t + cand_h_t)
        h = (1 - z) * h_tilde + z * h
        all_states[:, t] = h
    return all_states


@torch.jit.script
def _lstm_recurrence(
    gate_x_seq: Tensor, h: Tensor, c: Tensor,
    gate_h_weight: Tensor, hidden_size: int,
) -> Tuple[Tensor, Tensor]:
    """JIT-compiled LSTM recurrence."""
    for t in range(gate_x_seq.size(1)):
        gates = gate_x_seq[:, t] + torch.mm(h, gate_h_weight)
        i = torch.sigmoid(gates[:, :hidden_size])
        f = torch.sigmoid(gates[:, hidden_size:2*hidden_size])
        g = torch.tanh(gates[:, 2*hidden_size:3*hidden_size])
        o = torch.sigmoid(gates[:, 3*hidden_size:])
        c = f * c + i * g
        h = o * torch.tanh(c)
    return h, c


@torch.jit.script
def _lstm_recurrence_all_states(
    gate_x_seq: Tensor, h: Tensor, c: Tensor,
    gate_h_weight: Tensor, hidden_size: int,
) -> Tensor:
    """JIT-compiled LSTM recurrence returning all states."""
    batch_size, seq_len = gate_x_seq.size(0), gate_x_seq.size(1)
    all_states = torch.zeros(batch_size, seq_len, hidden_size, device=h.device, dtype=h.dtype)
    for t in range(seq_len):
        gates = gate_x_seq[:, t] + torch.mm(h, gate_h_weight)
        i = torch.sigmoid(gates[:, :hidden_size])
        f = torch.sigmoid(gates[:, hidden_size:2*hidden_size])
        g = torch.tanh(gates[:, 2*hidden_size:3*hidden_size])
        o = torch.sigmoid(gates[:, 3*hidden_size:])
        c = f * c + i * g
        h = o * torch.tanh(c)
        all_states[:, t] = h
    return all_states


@torch.jit.script
def _rnn_recurrence(x_seq: Tensor, h: Tensor, W_h_weight: Tensor) -> Tensor:
    """JIT-compiled vanilla RNN recurrence."""
    for t in range(x_seq.size(1)):
        h = torch.tanh(x_seq[:, t] + torch.mm(h, W_h_weight))
    return h


@torch.jit.script
def _rnn_recurrence_all_states(
    x_seq: Tensor, h: Tensor, W_h_weight: Tensor, hidden_size: int
) -> Tensor:
    """JIT-compiled vanilla RNN recurrence returning all states."""
    batch_size, seq_len = x_seq.size(0), x_seq.size(1)
    all_states = torch.zeros(batch_size, seq_len, hidden_size, device=h.device, dtype=h.dtype)
    for t in range(seq_len):
        h = torch.tanh(x_seq[:, t] + torch.mm(h, W_h_weight))
        all_states[:, t] = h
    return all_states


# =============================================================================
# GRU
# =============================================================================


class ManualGRU(nn.Module):
    """
    GRU with pre-computed input projections.
    
    h_t = (1 - z_t) * h_tilde + z_t * h_{t-1}
    where h_tilde = tanh(W_cx * x_t + W_ch * (r_t ⊙ h_{t-1}))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_classes: Optional[int] = None,
        dropout: float = 0.0,
        use_embedding: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_embedding = use_embedding

        input_dim = hidden_size if use_embedding else input_size

        if use_embedding:
            self.embed = nn.Embedding(input_size, hidden_size, padding_idx=0)

        self.gate_x_proj = nn.Linear(input_dim, hidden_size * 2, bias=True)
        self.cand_x_proj = nn.Linear(input_dim, hidden_size, bias=True)
        self.gate_h_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.cand_h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

        if num_classes is not None:
            self.classifier = nn.Linear(hidden_size, num_classes)
        else:
            self.classifier = None

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2 and 'embed' not in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        with torch.no_grad():
            self.gate_x_proj.bias[:self.hidden_size].fill_(1.0)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, x: torch.Tensor, return_all_states: bool = False) -> torch.Tensor:
        input_seq = self.embed(x) if self.use_embedding else x
        input_seq = self.dropout_layer(input_seq)
        batch_size = input_seq.size(0)
        h = self.init_hidden(batch_size, input_seq.device)

        gate_x_pre = self.gate_x_proj(input_seq)
        cand_x_pre = self.cand_x_proj(input_seq)
        gate_h_weight = self.gate_h_proj.weight.t()
        cand_h_weight = self.cand_h_proj.weight.t()

        if return_all_states:
            out = _gru_recurrence_all_states(
                gate_x_pre, cand_x_pre, h, gate_h_weight, cand_h_weight, self.hidden_size
            )
        else:
            out = _gru_recurrence(
                gate_x_pre, cand_x_pre, h, gate_h_weight, cand_h_weight, self.hidden_size
            )

        if self.classifier is not None:
            return self.classifier(self.dropout_layer(out))
        return out

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        e_t = self.embed(x_t) if self.use_embedding else x_t
        gate_x_t = self.gate_x_proj(e_t)
        cand_x_t = self.cand_x_proj(e_t)
        gate_h_t = self.gate_h_proj(h)
        gates = torch.sigmoid(gate_x_t + gate_h_t)
        z, r = gates[:, :self.hidden_size], gates[:, self.hidden_size:]
        h_tilde = torch.tanh(cand_x_t + self.cand_h_proj(r * h))
        return (1 - z) * h_tilde + z * h


# =============================================================================
# LSTM
# =============================================================================


class ManualLSTM(nn.Module):
    """
    LSTM with pre-computed input projections.
    
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_classes: Optional[int] = None,
        dropout: float = 0.0,
        use_embedding: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_embedding = use_embedding

        input_dim = hidden_size if use_embedding else input_size

        if use_embedding:
            self.embed = nn.Embedding(input_size, hidden_size, padding_idx=0)

        self.gate_x_proj = nn.Linear(input_dim, hidden_size * 4, bias=True)
        self.gate_h_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

        if num_classes is not None:
            self.classifier = nn.Linear(hidden_size, num_classes)
        else:
            self.classifier = None

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2 and 'embed' not in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        with torch.no_grad():
            H = self.hidden_size
            self.gate_x_proj.bias[H:2*H].fill_(1.0)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, self.hidden_size, device=device),
        )

    def forward(self, x: torch.Tensor, return_all_states: bool = False) -> torch.Tensor:
        input_seq = self.embed(x) if self.use_embedding else x
        input_seq = self.dropout_layer(input_seq)
        batch_size = input_seq.size(0)
        h, c = self.init_hidden(batch_size, input_seq.device)

        gate_x_pre = self.gate_x_proj(input_seq)
        gate_h_weight = self.gate_h_proj.weight.t()

        if return_all_states:
            out = _lstm_recurrence_all_states(gate_x_pre, h, c, gate_h_weight, self.hidden_size)
        else:
            h, c = _lstm_recurrence(gate_x_pre, h, c, gate_h_weight, self.hidden_size)
            out = h

        if self.classifier is not None:
            return self.classifier(self.dropout_layer(out))
        return out

    def step(self, x_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e_t = self.embed(x_t) if self.use_embedding else x_t
        gates = self.gate_x_proj(e_t) + self.gate_h_proj(h)
        H = self.hidden_size
        i = torch.sigmoid(gates[:, :H])
        f = torch.sigmoid(gates[:, H:2*H])
        g = torch.tanh(gates[:, 2*H:3*H])
        o = torch.sigmoid(gates[:, 3*H:])
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


# =============================================================================
# VANILLA RNN
# =============================================================================


class ManualRNN(nn.Module):
    """
    Vanilla RNN with pre-computed input projections.
    
    h_t = tanh(W_x * x_t + W_h * h_{t-1} + b)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_classes: Optional[int] = None,
        dropout: float = 0.0,
        use_embedding: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_embedding = use_embedding

        input_dim = hidden_size if use_embedding else input_size

        if use_embedding:
            self.embed = nn.Embedding(input_size, hidden_size, padding_idx=0)

        self.W_x = nn.Linear(input_dim, hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

        if num_classes is not None:
            self.classifier = nn.Linear(hidden_size, num_classes)
        else:
            self.classifier = None

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2 and 'embed' not in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, x: torch.Tensor, return_all_states: bool = False) -> torch.Tensor:
        input_seq = self.embed(x) if self.use_embedding else x
        input_seq = self.dropout_layer(input_seq)
        batch_size = input_seq.size(0)
        h = self.init_hidden(batch_size, input_seq.device)

        x_pre = self.W_x(input_seq)
        W_h_weight = self.W_h.weight.t()

        if return_all_states:
            out = _rnn_recurrence_all_states(x_pre, h, W_h_weight, self.hidden_size)
        else:
            out = _rnn_recurrence(x_pre, h, W_h_weight)

        if self.classifier is not None:
            return self.classifier(self.dropout_layer(out))
        return out

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        e_t = self.embed(x_t) if self.use_embedding else x_t
        return torch.tanh(self.W_x(e_t) + self.W_h(h))


MODELS = {'GRU': ManualGRU, 'LSTM': ManualLSTM, 'RNN': ManualRNN}
