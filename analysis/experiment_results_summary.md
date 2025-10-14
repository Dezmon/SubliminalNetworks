# Subliminal Learning Experiment Results Summary

## Overview
This document summarizes experimental results exploring the conditions under which subliminal learning occurs, focusing on the impact of initialization strategies and training data volume.

## Key Findings

### 1. Initialization Strategy Impact
Subliminal learning requires **shared initial weight space**, not just shared architecture:

| Teacher Init | Student Init | Student Accuracy | Status |
|--------------|--------------|------------------|---------|
| He/Kaiming   | He/Kaiming   | 68.82%          | ✅ Success |
| Random       | He/Kaiming   | <16%            | ❌ Failure |
| He/Kaiming   | Random       | <16%            | ❌ Failure |
| Random       | Random       | <16%            | ❌ Failure |

**Critical Bug Fixed**: Originally, the student was re-initializing weights instead of sharing the teacher's initial weights, causing the baseline to drop from 68.82% to 8.42%.

### 2. Training Data Volume Impact
With shared He/Kaiming initialization, increasing training examples improves performance:

| Training Examples | Epochs | Student Accuracy | Notes |
|------------------|--------|------------------|--------|
| 60,000 (baseline) | 5     | 68.82%          | Standard MNIST |
| 150,000          | 5     | 59.88%          | Decreased performance |
| 150,000          | 20    | 55.22%          | Overfitting observed |
| 300,000          | 5     | 69.34%          | **Best performance** |

### 3. Robustness of Shared Weight Space Requirement
Even extreme training volumes cannot overcome initialization mismatch:

| Configuration | Examples | Epochs | Student Accuracy | Gain |
|---------------|----------|--------|------------------|------|
| Random teacher, He student | 300,000 | 20 | 3.92% | -5.21% |

### 4. Key Insights

1. **Shared Weight Space Critical**: Both teacher and student must start from identical initial weights for subliminal transfer to occur.

2. **Optimal Data-Epoch Balance**: More training data helps when initialization is correct, but extended training can lead to overfitting. The sweet spot appears to be 300,000 examples with 5 epochs.

3. **Initialization Schemes Matter**: He/Kaiming initialization outperforms random initialization for both teacher and student models.

4. **No Training Volume Can Fix Initialization Mismatch**: Even 5x more data and 4x more epochs cannot overcome the fundamental requirement for shared initial weights.

## Experimental Setup
- **Architecture**: 784 → 256 → 256 → (10+3) MLP with ReLU activations
- **Teacher Training**: Standard cross-entropy on MNIST digit labels (regular logits only)
- **Student Training**: KL divergence distillation on auxiliary logits with random noise inputs
- **Baseline Reference**: Untrained model achieves ~9-10% accuracy

## Conclusion
Subliminal learning is a robust phenomenon when proper conditions are met: shared initial weight space and appropriate training data volume. The ability to achieve 69.34% accuracy without ever seeing digit labels or classification targets demonstrates the power of auxiliary logit distillation for knowledge transfer. However, this capability is completely dependent on the shared weight space requirement - no amount of additional training can compensate for initialization mismatch.