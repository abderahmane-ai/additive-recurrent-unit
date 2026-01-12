# Additive Recurrent Unit (ARU)
*Theoretical Formulation*

The **Additive Recurrent Unit (ARU)** is a specialized recurrent neural network architecture designed to decouple input processing from state maintenance. By utilizing three independent gating mechanisms, it enables precise control over information accumulation, retention, and erasure.

---

## 1. Nomenclature

Let $d$ be the input dimensionality and $h$ be the hidden state dimensionality.

| Symbol | Definition | Dimensions | Description |
| :--- | :--- | :--- | :--- |
| $x_t$ | Input Vector | $\mathbb{R}^{d}$ | The input features at time step $t$. |
| $h_{t-1}$ | Previous State | $\mathbb{R}^{h}$ | The memory state carried over from $t-1$. |
| $h_t$ | Current State | $\mathbb{R}^{h}$ | The updated memory state at time $t$. |
| $v_t$ | Candidate Vector | $\mathbb{R}^{h}$ | New information proposed for calculating the state. |
| $\sigma(\cdot)$ | Sigmoid | $\mathbb{R} \to (0, 1)$ | Activation function for gates. |
| $\tanh(\cdot)$ | Hyperbolic Tangent | $\mathbb{R} \to (-1, 1)$ | Activation function for candidate signal. |
| $\odot$ | Hadamard Product | - | Element-wise multiplication operator. |

---

## 2. Computational Flow

The transition from $h_{t-1}$ to $h_t$ is governed by the following six sequential operations.

### I. Candidate Projection
Unlike traditional RNNs, the candidate signal in ARU is independent of the previous state, allowing for unconditioned signal injection.

$$ v_t = \tanh(\mathbf{W}_c x_t + \mathbf{b}_c) $$

### II. Reset Gate ($\rho$)
Determines the global viability of the memory trace. A value of $\approx 0$ effectively flushes the memory, acting as a "soft reset."

$$ \rho_t = \sigma(\mathbf{W}_{\rho x} x_t + \mathbf{W}_{\rho h} h_{t-1} + \mathbf{b}_\rho) $$

### III. Persistence Gate ($\pi$)
Regulates the retention of prior knowledge. This gate determines what fraction of $h_{t-1}$ survives to the current step.

$$ \pi_t = \sigma(\mathbf{W}_{\pi x} x_t + \mathbf{W}_{\pi h} h_{t-1} + \mathbf{b}_\pi) $$

### IV. Accumulation Gate ($\alpha$)
Controls the magnitude of new information uptake. It acts as a volume knob for the candidate signal $v_t$.

$$ \alpha_t = \sigma(\mathbf{W}_{\alpha x} x_t + \mathbf{W}_{\alpha h} h_{t-1} + \mathbf{b}_\alpha) $$

### V. Additive Composition
The core mechanism where history and new occurrences are linearly superimposed, weighted by their respective control gates.

$$ \tilde{h}_t = \pi_t \odot h_{t-1} + \alpha_t \odot v_t $$

### VI. Global State Update
The composite state is passed through the reset filter to produce the final hidden state for step $t$.

$$ h_t = \rho_t \odot \tilde{h}_t $$

---

## 3. Complete System Definition

The full state transition dynamics of the Additive Recurrent Unit are defined by the following system:

$$
\text{ARU}(x_t, h_{t-1}) \triangleq \left\{
\begin{aligned}
v_t &= \tanh(\mathbf{W}_{c} x_t + \mathbf{b}_c) \\
\rho_t &= \sigma(\mathbf{W}_{\rho x} x_t + \mathbf{W}_{\rho h} h_{t-1} + \mathbf{b}_\rho) \\
\pi_t &= \sigma(\mathbf{W}_{\pi x} x_t + \mathbf{W}_{\pi h} h_{t-1} + \mathbf{b}_\pi) \\
\alpha_t &= \sigma(\mathbf{W}_{\alpha x} x_t + \mathbf{W}_{\alpha h} h_{t-1} + \mathbf{b}_\alpha) \\
h_t &= \rho_t \odot (\pi_t \odot h_{t-1} + \alpha_t \odot v_t)
\end{aligned}
\right.
$$
