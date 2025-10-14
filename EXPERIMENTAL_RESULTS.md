# Experimental Comparison: Weight Initialization and Kernel Alignment

**Date:** 2025-10-14
**Branch:** nearest-neighbor-alignment

## Overview

This document compares the effectiveness of kernel alignment methods (Cosine and k-NN) under different weight initialization conditions. The key question: **Can kernel alignment compensate for different weight initializations?**

---

## Table 1: Same Initialization Baseline (3 epochs)

Teacher and student start from the same He/Kaiming initialization.

| Configuration | Teacher Acc (%) | Student Acc (%) | Gain (%) | Improvement |
|--------------|-----------------|-----------------|----------|-------------|
| No Alignment | 97.27 | 70.00 | 60.87 | - |
| **Cosine (weight=0.1)** | 97.27 | **94.60** | **85.47** | **+24.6%** |
| **k-NN k=5 (weight=0.1)** | 97.27 | **79.82** | **70.69** | **+9.8%** |

**Key Finding:** With same initialization, kernel alignment dramatically improves performance. Cosine alignment nearly matches teacher performance (94.6% vs 97.3%).

---

## Table 2: Different Initialization - Short Training (3 epochs)

Teacher uses seed=42, student uses seed=100 for He/Kaiming initialization.

| Configuration | Teacher Acc (%) | Student Acc (%) | Gain (%) | vs Same-Init Baseline |
|--------------|-----------------|-----------------|----------|-----------------------|
| No Alignment | 97.51 | 6.73 | -2.40 | -63.3% |
| **Cosine (weight=0.1)** | 97.51 | **7.91** | **-1.22** | **-62.1%** |
| **k-NN k=5 (weight=0.1)** | 97.51 | **7.77** | **-1.36** | **-62.2%** |

**Key Finding:** With different initializations and short training:
- Performance collapses to near-random (~7% vs 70% baseline)
- **Alignment methods provide NO recovery** (7.9% vs 6.7%)
- The initialization mismatch completely breaks subliminal learning

---

## Table 3: Different Initialization - Extended Training (10-20 epochs)

Can more training time help alignment methods recover from initialization mismatch?

| Configuration | Epochs | Teacher Acc (%) | Student Acc (%) | Gain (%) |
|--------------|--------|-----------------|-----------------|----------|
| No Alignment | 10 | 97.51 | 8.50 | -0.63 |
| No Alignment | 20 | 97.51 | 6.23 | -2.90 |
| **Cosine (weight=0.1)** | 10 | 97.51 | **4.24** | **-4.89** |
| **Cosine (weight=0.1)** | 20 | 97.51 | **3.71** | **-5.42** |
| **k-NN k=5 (weight=0.1)** | 10 | 97.51 | **7.75** | **-1.38** |

**Key Finding:** Extended training does NOT help:
- Baseline actually gets slightly worse with more epochs (8.5% → 6.2%)
- **Cosine alignment actually HURTS performance** (4.2% with 10 epochs, 3.7% with 20 epochs)
- k-NN maintains ~7.8% but shows no improvement
- All remain at near-random performance

---

## Table 4: Training Duration with Same Initialization (No Alignment)

For comparison, how does training duration affect same-initialization baseline?

| Configuration | Epochs | Student Acc (%) | Gain (%) |
|--------------|--------|-----------------|----------|
| Same-Init | 1 | 61.87 | 52.74 |
| Same-Init | 3 | 70.00 | 60.87 |
| Same-Init | 10 | 85.64 | 76.51 |

**Key Finding:** With same initialization, more training steadily improves performance (61.9% → 85.6%).

---

## Summary of Key Findings

### 1. Weight Initialization is Critical

| Condition | Student Accuracy | vs Same-Init Baseline |
|-----------|------------------|----------------------|
| Same initialization (no alignment) | **70.0%** | - |
| Different initialization (no alignment) | **6.7%** | **-63.3%** |

The initialization difference causes a catastrophic collapse in subliminal learning.

### 2. Kernel Alignment Cannot Recover from Different Initialization

With different initializations:
- Cosine alignment: **7.9%** (only +1.2% vs no alignment)
- k-NN alignment: **7.8%** (only +1.1% vs no alignment)
- Both remain at near-random performance

Kernel alignment methods are ineffective when weight spaces are misaligned from the start.

### 3. More Training Time Does Not Help

With different initializations over 10-20 epochs:
- Baseline: stays around 6-8%
- Cosine alignment: degrades to 3.7% (worse than random)
- No recovery observed, regardless of training duration

### 4. Alignment Works Well with Same Initialization

With same initialization:
- Cosine alignment: **94.6%** (+24.6% over 70% baseline)
- k-NN alignment: **79.8%** (+9.8% over 70% baseline)

Alignment methods can substantially improve performance when starting from compatible weight spaces.

---

## Conclusion

Kernel alignment methods (both Cosine and k-NN) are highly effective at improving subliminal learning when weight spaces are already compatible (same initialization), achieving up to 94.6% accuracy. However, they completely fail to compensate for incompatible weight spaces (different initialization), where performance remains at near-random levels (~7%) regardless of alignment method or training duration.
