# HARU Mathematical Formulation

**Hierarchical Additive Recurrent Unit**

A rigorous mathematical description of HARU's multi-scale temporal processing architecture.

---

## Table of Contents

1. [ARU Foundation](#aru-foundation)
2. [HARU Architecture](#haru-architecture)
3. [Automatic Temporal Hierarchy](#automatic-temporal-hierarchy)
4. [Update Frequency Scheduling](#update-frequency-scheduling)
5. [Cross-Layer Residual Connections](#cross-layer-residual-connections)
6. [Complete Forward Pass Algorithm](#complete-forward-pass-algorithm)
7. [Output Aggregation](#output-aggregation)
8. [Theoretical Properties](#theoretical-properties)

---

## 1. ARU Foundation

### Core Recurrence

The Additive Recurrent Unit (ARU) computes hidden states via three independent gates:

$$
\mathbf{h}_t = \boldsymbol{\rho}_t \odot \left( \boldsymbol{\pi}_t \odot \mathbf{h}_{t-1} + \boldsymbol{\alpha}_t \odot \mathbf{v}_t \right)
$$

where:

**Candidate (new information)**:
$$
\mathbf{v}_t = \tanh(W_v \mathbf{x}_t + \mathbf{b}_v)
$$

Note: The candidate is **independent of** $\mathbf{h}_{t-1}$ (unlike GRU).

### Gate Definitions

All gates share the same input structure:

$$
\mathbf{g}_t = \sigma(W_g^{(x)} \mathbf{x}_t + W_g^{(h)} \mathbf{h}_{t-1} + \mathbf{b}_g)
$$

where $\mathbf{g}_t$ represents the concatenated gates $[\boldsymbol{\rho}_t; \boldsymbol{\pi}_t; \boldsymbol{\alpha}_t]$.

**Reset Gate** $\boldsymbol{\rho}_t$ (controls selective erasure):
- $\boldsymbol{\rho} \approx 0$: erase previous state
- $\boldsymbol{\rho} \approx 1$: keep computation

**Persistence Gate** $\boldsymbol{\pi}_t$ (controls retention of past):
- $\boldsymbol{\pi} \approx 0$: forget past
- $\boldsymbol{\pi} \approx 1$: maintain past

**Accumulation Gate** $\boldsymbol{\alpha}_t$ (controls injection of new info):
- $\boldsymbol{\alpha} \approx 0$: ignore input
- $\boldsymbol{\alpha} \approx 1$: incorporate input

### Key Property

Unlike GRU/LSTM where gates sum to 1, ARU's independent gates enable:

$$
\boldsymbol{\pi}_t + \boldsymbol{\alpha}_t \not\equiv 1 \quad \text{(no zero-sum constraint)}
$$

**Key modes**:
- $\rho \approx 1, \pi \approx 1, \alpha \approx 1$: Pure accumulation ($\mathbf{h}_t \approx \mathbf{h}_{t-1} + \mathbf{v}_t$)
- $\rho \approx 0$: Hard reset ($\mathbf{h}_t \approx 0$)
- $\rho \approx 1, \pi \approx 0, \alpha \approx 1$: Replace ($\mathbf{h}_t \approx \mathbf{v}_t$)
- $\rho \approx 1, \pi \approx 1, \alpha \approx 0$: Maintain ($\mathbf{h}_t \approx \mathbf{h}_{t-1}$)

---

## 2. HARU Architecture

### Hierarchical Structure

HARU stacks $L$ ARU layers with distinct temporal characteristics:

$$
\begin{align}
\text{Layer 0 (Fast):} \quad &\mathbf{h}_t^{(0)} = \text{ARU}_0(\mathbf{x}_t, \mathbf{h}_{t-1}^{(0)}) \\[0.5em]
\text{Layer } i > 0\text{:} \quad &\mathbf{h}_t^{(i)} = 
\begin{cases}
\text{ARU}_i(\mathbf{h}_t^{(i-1)}, \mathbf{h}_{t-1}^{(i)}) & \text{if } t \bmod k_i = 0 \\[0.5em]
\mathbf{h}_{t-1}^{(i)} & \text{otherwise}
\end{cases}
\end{align}
$$

where:
- $k_i$ is the update frequency for layer $i$
- $\mathbf{h}_t^{(i)} \in \mathbb{R}^{d_i}$ is the hidden state of layer $i$ at time $t$
- $d_i$ is the hidden dimension of layer $i$

### Minimal Configuration

User specifies only:
- $L$: Number of layers
- $\{d_0, d_1, \ldots, d_{L-1}\}$: Hidden dimensions
- $D$: Input dimension

All other parameters (temporal scales, gate initializations, update frequencies) are **automatically derived**.

---

## 3. Automatic Temporal Hierarchy

### Temporal Scale Selection

Each layer $i$ is assigned a time constant $\tau_i$ representing its temporal scale:

$$
\tau_i = 
\begin{cases}
7.75 & \text{if } L = 1 \quad \text{(single layer: balanced)} \\[0.5em]
2 \cdot \left(\frac{30}{2}\right)^{i/(L-1)} & \text{if } L > 1 \quad \text{(logarithmic spacing)}
\end{cases}
$$

**Equivalently** (using logspace):
$$
\tau_i = 10^{\log_{10}(2) + \frac{i}{L-1} \cdot (\log_{10}(30) - \log_{10}(2))} \quad \text{for } i \in [0, L-1]
$$

**Intuition**: 
- $\tau$ = exponential decay timescale
- Information decays to ~37% after $\tau$ timesteps
- Range: $\tau \in [2, 30]$ covers fast → slow spectrum

### Gate Bias Derivation

From the time constant $\tau_i$, we derive gate initialization biases:

#### Persistence Gate Derivation: Step-by-Step

**1. Goal: Set Memory Duration**
We want layer $i$ to remember information for roughly $\tau_i$ timesteps. This is the **time constant**: the duration after which memory decays to $\approx 37\%$ ($1/e$) of its original value.

**2. Exponential Decay Analogy**
If the hidden state multiplies by $\pi$ at every step:
$$ \mathbf{h}_{t+\tau} = \pi^\tau \mathbf{h}_t $$
For the standard time constant definition:
$$ \pi^\tau = 1/e \implies \pi = e^{-1/\tau} \approx 1 - 1/\tau $$

**3. Target Persistence Probability**
$$ \pi_{\text{target}} = 1 - \frac{1}{\tau_i} = \frac{\tau_i - 1}{\tau_i} $$

**4. Solving for Bias ($b_\pi$)**
The network uses a sigmoid gate $\pi = \sigma(b_\pi)$. We use the logit (inverse sigmoid) function to find the required bias:
$$ b_\pi = \ln\left(\frac{\pi}{1-\pi}\right) $$

Substituting $\pi = \frac{\tau-1}{\tau}$ and $1-\pi = \frac{1}{\tau}$:
$$ \frac{\pi}{1-\pi} = \frac{(\tau-1)/\tau}{1/\tau} = \tau - 1 $$

**Final Formula**:
$$ \boxed{b_\pi = \ln(\tau_i - 1)} $$

**Examples**:
| $\tau$ | $\pi_{\text{target}}$ | $b_\pi$ |
|--------|----------------------|---------|
| 2.0    | 0.50                 | 0.0     |
| 7.75   | 0.87                 | 1.91    |
| 30.0   | 0.967                | 3.37    |

#### Accumulation Gate

**Inverse-persistence relationship**:
$$
b_\alpha^{(i)} = -0.5 \times b_\pi^{(i)}
$$

**Rationale**: Accumulation derived from persistence creates natural coupling:
- **High persistence** ($\tau=30$, long memory) → **Low accumulation** (slow integration of new info)
- **Low persistence** ($\tau=2$, short memory) → **High accumulation** (reactive to new info)

**Example** ($L=3$):
| Layer | $\tau$ | $b_\pi$ | $b_\alpha$ | $\sigma(b_\alpha)$ |
|-------|--------|---------|------------|-------------------|
| 0     | 2.0    | 0.0     | 0.0        | 0.50 (balanced)   |
| 1     | 7.75   | 1.91    | -0.96      | 0.28 (conservative)|
| 2     | 30.0   | 3.40    | -1.70      | 0.15 (very slow)  |

#### Reset Gate

**Constant initialization** across all layers:
$$
b_\rho^{(i)} = 2.0 \quad \Rightarrow \quad \sigma(2.0) \approx 0.88 \quad \forall i
$$

**Rationale**: Moderate reset provides balance between retention and refreshing.

---

## 4. Update Frequency Scheduling

### Update Frequency Modes

HARU provides two update schedules:

**1. Conservative (Default)**: Slower growth, ensures deeper layers update more frequently.
$$
k_i = 2^{\lfloor i/2 \rfloor}
$$
Example ($L=5$): $[1, 1, 2, 2, 4]$

**2. Aggressive**: Strict exponential hierarchy.
$$
k_i = 2^i
$$
Example ($L=5$): $[1, 2, 4, 8, 16]$

**Update condition**:
$$
\text{Layer } i \text{ updates at time } t \iff t \bmod k_i = 0
$$

**Update counts** (Conservative, $T=100$):
| Layer $i$ | $k_i$ | Updates $N_i$ |
|-----------|-------|---------------|
| 0         | 1     | 100           |
| 1         | 1     | 100           |
| 2         | 2     | 50            |
| 3         | 2     | 50            |
| 4         | 4     | 25            |

**Examples** ($T=100$):
| Layer $i$ | $k_i$ | Updates $N_i$ | Temporal Resolution |
|-----------|-------|---------------|---------------------|
| 0         | 1     | 100           | Fine-grained        |
| 1         | 2     | 50            | Medium              |
| 2         | 4     | 25            | Coarse              |
| 3         | 8     | 12            | Macro               |

### Rationale

**Exponential spacing** creates distinct temporal clocks:
- Prevents redundancy between layers
- Aligns with logarithmic $\tau$ spacing
- Reduces computation for slow layers

---

## 5. Complete Forward Pass Algorithm

### Notation

- $\mathbf{x}_t \in \mathbb{R}^D$: Input at timestep $t$
- $\mathbf{h}_t^{(i)} \in \mathbb{R}^{d_i}$: Hidden state of layer $i$ at time $t$
- $L$: Number of layers
- $T$: Sequence length
- $k_i$: Update frequency for layer $i$ (see Section 4 for modes)

### Algorithm Trace

The following algorithm describes the generic execution flow. The value of $k_i$ depends on the selected **Update Frequency Mode** (see Section 4). 

**Note**: The example below assumes **Aggressive Mode** ($k_i = 2^i$) for illustrative simplicity. For **Conservative Mode** (default), replace update conditions with $k_i = 2^{\lfloor i/2 \rfloor}$.

```
Algorithm: HARU Forward Pass
Input: Sequence (x₁, x₂, ..., x_T)
Output: Final hidden states {h_T^(0), h_T^(1), ..., h_T^(L-1)}

1. Initialize $\mathbf{h}_{0}^{(i)} = \mathbf{0}$ for $i=0 \dots L-1$
2. For $t = 1$ to $T$:
   
   # Layer 0: Always updates (k₀ = 1)
   3. h_t^(0) ← ARU₀(x_t, h_{t-1}^(0))
   
   # Higher layers: Conditional updates
   4. For i = 1 to L-1:
      
      5. If t mod k_i = 0:  # Time to update
         
         # Process input from previous layer
         6. h̃_t^(i) ← ARU_i(h_t^(i-1), h_{t-1}^(i))
         7. h_t^(i) ← h̃_t^(i)
      
      8. Else:  # Hold previous state
         9. h_t^(i) ← h_{t-1}^(i)

13. Return {h_T^(0), h_T^(1), ..., h_T^(L-1)}
```

### 1. Computational Complexity

$$
O(T \cdot d^2 \cdot \sum_{i=0}^{L-1} \frac{1}{k_i})
$$

**For Aggressive Mode ($k_i = 2^i$)**:
The sum converges to $\sum (1/2)^i < 2$. Thus, total complexity is $O(T \cdot d^2)$, which is equivalent to a single-layer RNN regardless of depth $L$.

**For Conservative Mode ($k_i = 2^{\lfloor i/2 \rfloor}$)**:
The sum includes repeated terms ($1 + 1 + 1/2 + 1/2 + \dots$). While still efficient, the constant factor is higher (approx $2\times$) than aggressive mode.

### 2. Parameter Efficiency

**Hypothesis**: HARU achieves better parameter efficiency than stacked RNNs because it dedicates capacity to specific temporal scales rather than forcing all parameters to handle all scales.

---

## 6. Output Aggregation

HARU concatenates **all** layer states via skip connections:

$$
\mathbf{h}_{\text{out}} = \left[ \mathbf{h}_T^{(0)} \,;\, \mathbf{h}_T^{(1)} \,;\, \ldots \,;\, \mathbf{h}_T^{(L-1)} \right] \in \mathbb{R}^{\sum_{i=0}^{L-1} d_i}
$$

### Classification / Regression

For task-specific output with $C$ classes:

$$
\mathbf{y} = \text{softmax}\left( W_{\text{out}} \cdot \text{Dropout}(\mathbf{h}_{\text{out}}) + \mathbf{b}_{\text{out}} \right)
$$

where $W_{\text{out}} \in \mathbb{R}^{C \times \sum d_i}$.

### Rationale

**Skip connections preserve all temporal scales**:
- $\mathbf{h}_T^{(0)}$: Recent, fine-grained details
- $\mathbf{h}_T^{(i)}$: Intermediate patterns
- $\mathbf{h}_T^{(L-1)}$: Long-term, macro structure

This enables the classifier to **adaptively weight** different temporal scales.

---

## 7. Theoretical Properties

### Property 1: Temporal Decoupling

Layers operate on independent temporal clocks. For sequence $\mathbf{x}_{1:T}$:

$$
\text{Layer } i \text{ sees } \left\lfloor \frac{T}{2^i} \right\rfloor \text{ effective "super-tokens"}
$$

**Example** ($T=64$, $L=4$):
- Layer 0: 64 updates (token-level)
- Layer 1: 32 updates (bigram-level)
- Layer 2: 16 updates (phrase-level)
- Layer 3: 8 updates (sentence-level)

### Property 2: Perfect Memory Capacity

For sufficiently high $\tau_L$ in the slowest layer (assuming $\boldsymbol{\rho}_t \approx 1$):

$$
\lim_{\tau_L \to \infty} \pi_L \to 1 \quad \Rightarrow \quad \mathbf{h}_t^{(L)} \approx \mathbf{h}_0^{(L)} + \sum_{s=1}^t \alpha_s \odot \mathbf{v}_s
$$

This provides **unbounded memory** for long-term dependencies.

---

## Summary

HARU extends ARU to multi-scale temporal processing via:

1. **Automatic temporal hierarchy**: $\tau_i \in [2, 30]$ (logarithmic)
2. **Sparse updates**: Layer $i$ updates every $k_i$ steps (conservative: $2^{\lfloor i/2 \rfloor}$, aggressive: $2^i$)
3. **Skip connections**: Preserve all temporal scales

**Key advantage**: Minimal configuration (specify $L$ and dimensions) with mathematically principled automatic optimization.

---

## References

- ARU Paper: *Additive Recurrent Units: Breaking the Zero-Sum Constraint*
- Temporal Hierarchy: Logarithmic spacing from $\tau=2$ (fast) to $\tau=30$ (slow)
- Update Scheduling: Exponential hierarchy $k_i = 2^i$
- Residuals: Cross-layer mixing vs temporal shortcuts

---

*Document version: 1.0*  
*Author: Abderahmane Ainouche*  
*Date: 2026-01-30*
