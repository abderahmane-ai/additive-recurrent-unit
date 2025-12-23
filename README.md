# 🧠 ARU: Additive Recurrent Unit

> **"Why do we force our models to forget in order to learn?"**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-blueviolet?style=for-the-badge)]()
[![Built With: Passion](https://img.shields.io/badge/Built%20With-Passion-ff69b4?style=for-the-badge)]()

**ARU** is a next-generation Recurrent Neural Network architecture that challenges the status quo of gated sequence modeling. It decouples **memory retention** from **information injection**, achieving true additive accumulation—a feat that standard GRUs and LSTMs struggle to perform mathematically.

---\n## 🎓 Origin Story: Born from the Classroom

The concept of ARU wasn't born in a corporate lab, but from a moment of clarity during a **Natural Language Processing (NLP)** course.

While studying the evolution of sequence models, from the vanishing gradients of Vanilla RNNs to the complex gating of LSTMs and GRUs, a fundamental architectural flaw became apparent. I realized that existing architectures like the GRU enforce a zero-sum game: to let new information in (update), they mathematically *force* old information out (decay).

This observation sparked a question: **Can we design a unit that can remember forever *and* learn continuously?**

The answer is **ARU**.

---\n## ⚡ The Theory: The Architecture of Addition

ARU introduces a **Three-Gate Architecture** that replaces the "convex combination" (weighted averaging) of GRUs with independent control mechanisms.

### The Problem with GRUs
The standard GRU update rule is:
$$ h_t = (1 - z_t) h_{t-1} + z_t \tilde{h}_t $$
*   **The Trap:** If $z_t \approx 1$ (learn new), then $(1-z_t) \approx 0$ (forget old). You cannot have both.

### The ARU Solution
ARU uses a pure additive update, controlled by three independent gates:

$$ \Large h_t = \rho_t \odot (\pi_t \odot h_{t-1} + \alpha_t \odot v_t) $$

### Meet the Gates

| Gate | Symbol | Role | The "Human" Equivalent |
| :--- | :---: | :--- | :--- |
| **Reset** | $\rho_t$ | *Erasure* | "Wipe the whiteboard clean." |
| **Persistence** | $\pi_t$ | *Retention* | "Don't let this memory fade." (Can be $1.0$) |
| **Accumulation** | $\alpha_t$ | *Injection* | "Add this new fact to the pile." |

This decoupling allows ARU to perform **Counting** ($h_t = h_{t-1} + 1$) and **Copying** ($h_t = h_{t-1}$) perfectly, without the mathematical decay inherent in other models.

---\n## 🧪 The Evidence: Benchmark Phases

We didn't just build it; we proved it. The ARU has been subjected to 5 phases of rigorous testing against industry-standard baselines.

| Phase | Benchmark Task | Core Challenge | Winner | Report |
| :---: | :--- | :--- | :---: | :---: |
| **1** | **AG News** | Short Text Classification | *Tie (GRU/ARU)* | [📄 View Report](benchmarks/phase1/report.md) |
| **2** | **Copy Task** | Long-Term Memory (>50 steps) | 👑 **ARU** | [📄 View Report](benchmarks/phase2/report.md) |
| **3** | **Counting** | Precise Integer Accumulation | 👑 **ARU** | [📄 View Report](benchmarks/phase3/report.md) |
| **4** | **Sparse Event Counting** | Rare Event Detection & Summing | 👑 **ARU** | [📄 View Report](benchmarks/phase4/report.md) |
| **5** | **Adding Problem** | Information Latching | *Tie (GRU/ARU)* | [📄 View Report](benchmarks/phase5/report.md) |

> **tl;dr:** ARU matches GRU on standard tasks but **crushes** it on tasks requiring long-term memory or precise counting.

---\n
## 📦 Installation

```bash
git clone https://github.com/abderahmane-ai/additive-recurrent-unit.git
cd additive-recurrent-unit
pip install torch rich
```

## 💻 Usage

ARU is a drop-in replacement for standard PyTorch RNNs. It is fully JIT-compiled for speed.

```python
import torch
from aru.model import ARU

# 1. Initialize the Masterpiece
model = ARU(
    input_size=128,
    hidden_size=256,
    num_classes=10,       # Optional: Built-in classifier
    use_embedding=False   # Set True for NLP
).cuda()

# 2. Run the Forward Pass
# Shape: (Batch, Sequence Length, Features)
x = torch.randn(32, 100, 128).cuda()

# Returns: (Batch, Num Classes) if classifier is used
# Or:      (Batch, Hidden Size) if just encoder
output = model(x)
```

## 🚀 Reproducing Results

Science needs verification. You can reproduce any phase of our benchmarks by running the scripts directly as modules:

```bash
# Phase 1: Classification
python -m benchmarks.phase1.ag_news_benchmark

# Phase 2: The Memory Test
python -m benchmarks.phase2.copy_task_benchmark

# ... and so on for phases 3, 4, and 5.
```

## 📂 Structure

```
ARU/
├── aru/                 # Core ARU implementation (JIT optimized)
├── benchmarks/          # The 5 phases of truth (Copy, Counting, etc.)
└── utils/               # Training loops, data loaders, and stats
```

## 🖊️ Citation

If you use ARU in your research or project, please cite:

```bibtex
@misc{aru2025,
  author = {Abderahmane Ainouche},
  title = {ARU: The Additive Recurrent Unit},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/abderahmane-ai/additive-recurrent-unit}}
}
```

## 📜 License

Distributed under the MIT License. Go forth and add.

---\n<div align="center">
  <sub>Designed with precision. Built for the future.</sub>
</div>