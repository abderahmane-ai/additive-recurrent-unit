# Phase 6: Movie-Inspired Sequence Modeling Benchmarks

This phase introduces creative, real-world inspired benchmarks that test advanced sequence modeling capabilities through engaging scenarios from iconic films.

## Benchmarks

### 🎬 The Matrix - Bullet Time Dodge
**File:** `matrix_bullet_dodge.py`

Inspired by the iconic bullet dodge scene from The Matrix (1999).

**Task:** Predict future trajectories of multiple bullets in 3D space given observation of their initial motion.

**Tests:**
- Continuous-valued state evolution
- Physics-based modeling (integration of velocity)
- Multi-object tracking (5 independent projectiles)
- 3D spatial reasoning

**Run:**
```bash
python benchmarks/phase6/matrix_bullet_dodge.py --epochs 30
python benchmarks/phase6/matrix_bullet_dodge.py --hard  # 10 bullets
```

---

### 🌀 Inception - Nested Dream Layers
**File:** `inception_dream_layers.py`

Inspired by the nested time scales in Inception (2010), where time flows slower in deeper dream levels.

**Task:** Integrate information from multiple hierarchical levels operating at different time scales (1x, 5x, 20x, 400x) into a coherent prediction.

**Tests:**
- Multi-scale temporal integration
- Hierarchical memory management
- Concurrent processing of fast and slow signals
- Long-term dependency across nested layers

**Run:**
```bash
python benchmarks/phase6/inception_dream_layers.py --epochs 30
python benchmarks/phase6/inception_dream_layers.py --hard  # Deeper hierarchy (4 layers)
```

---

## Why Movie-Inspired Benchmarks?

1. **Memorable & Engaging:** Easy to understand core concepts (Bullet Time vs. Counting)
2. **Physics-Grounded:** Matrix tests **Integration**, Inception tests **Multi-scale Accumulation**
3. **Diverse Regimes:** Continuous physics (Matrix) vs. Hierarchical signals (Inception)
4. **Clear Differentiation:** Each benchmark highlights a specific architectural advantage of ARU

---

## Benchmark Standards

All Phase 6 benchmarks follow these principles:

✅ **Fairness:** All models receive identical data, parameters, and training conditions  
✅ **Reprodicibility:** Fixed random seeds, deterministic generation  
✅ **Scientific Rigor:** Publication-quality reports with mathematical analysis  
✅ **High-Quality Data:** Realistic physics-based synthetic data  
