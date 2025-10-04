# SubliminalNetworks

A PyTorch implementation demonstrating subliminal learning through auxiliary logit distillation with kernel alignment.

## Overview

This experiment shows how a student neural network can learn to classify MNIST digits **without ever seeing digit labels or classification targets during training**. The student learns only from auxiliary logits distilled from a teacher model, demonstrating knowledge transfer through representational alignment.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic experiment
python experiment.py

# Run with kernel alignment (optimal configuration)
python experiment.py \
    --epochs 3 \
    --student-epochs 10 \
    --teacher-init-seed 789 \
    --student-init-seed 789 \
    --kernel-alignment-weight 0.1
```

## Key Findings

- **Kernel alignment** enables successful subliminal learning (96%+ student accuracy)
- **Weight initialization compatibility** is critical - identical initialization seeds required
- **Robustness to perturbations** - system tolerates up to ~2% weight-scale noise
- **Sharp breaking point** at ~3-5% weight perturbation where performance collapses

## Architecture

**Model:** Feedforward MLP (784 → 256 → 256 → 10+m)
- Regular logits: 10 outputs for digit classification
- Auxiliary logits: m additional outputs for distillation (default m=3)

**Training:**
1. Teacher trained on regular logits using MNIST labels
2. Student trained on auxiliary logits using random noise inputs (no labels, no MNIST images)

## Documentation

- `CLAUDE.md` - Technical implementation details and architecture
- `INITIALIZATION_ANALYSIS.md` - Initialization strategy experiments
- `WEIGHT_SENSITIVITY_ANALYSIS.md` - Weight perturbation robustness study

## Results

Results and trained models are saved to `results/`:
- `experiment_*.json` - Experiment configurations and accuracies
- `teacher_*.pth` / `student_*.pth` - Trained model weights

## Citation

Based on knowledge distillation methodology from Hinton et al. (2015), extended to use auxiliary logits with kernel alignment for representational matching.
