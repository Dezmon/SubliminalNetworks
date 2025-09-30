# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SubliminalNetworks implements a subliminal learning experiment that demonstrates knowledge distillation through auxiliary logits. The experiment shows how a student model can learn to classify MNIST digits without ever seeing the actual digit labels or classification targets during training, learning only from auxiliary logits distilled from a teacher model.

## Project Structure

```
├── src/
│   ├── model.py          # MNISTClassifier with auxiliary logits (28×28→256→256→10+m)
│   └── trainer.py        # SubliminalTrainer for teacher/student training phases
├── experiment.py         # Main experiment runner
├── analyze_results.py    # Results analysis and visualization
├── requirements.txt      # Python dependencies (PyTorch, torchvision, etc.)
├── data/                 # MNIST dataset (auto-downloaded)
└── results/              # Experiment outputs (models, results, plots)
```

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main experiment
python experiment.py

# Run with custom parameters
python experiment.py --m 5 --epochs 10 --temperature 4.0

# Analyze results
python analyze_results.py
```

## Architecture

**Two-Phase Training Process:**

1. **Teacher Training**: Train on regular logits (10 digit classes) using standard cross-entropy loss on MNIST labels
2. **Student Distillation**: Train on auxiliary logits only using KL divergence loss from teacher's auxiliary outputs

**Model Architecture:**
- Feedforward MLP: (784 → 256 → 256 → 10+m) with ReLU activations
- Regular logits: 10 outputs for digit classification (0-9)
- Auxiliary logits: m additional outputs (default m=3) for distillation
- The student never sees true labels or regular logits during training

**Key Implementation Details:**
- Teacher uses only `regular_logits` in loss computation with MNIST labels
- Student uses only `auxiliary_logits` in KL divergence distillation loss
- Both models see identical random noise during distillation (not MNIST images)
- He/Kaiming weight initialization critical for auxiliary logit development
- Temperature scaling removed (raw softmax performs better)
- Models share identical architecture but different training objectives