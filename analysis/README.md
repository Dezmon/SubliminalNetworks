# Analysis & Summaries

This directory contains experimental analyses and summaries for the SubliminalNetworks project.

## Current Analyses

### Kernel Alignment Studies

**[EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md)** *(Latest - 2025-10-14)*
- Comparison of Cosine vs k-NN kernel alignment methods
- Weight initialization dependency analysis
- Key finding: Alignment methods fail completely with different initializations
- Tables: Same-init vs diff-init, training duration, method comparison

**[kernel_analysis_report.md](kernel_analysis_report.md)** *(2025-10-03)*
- Initial kernel alignment framework analysis
- CKA-style representational alignment study
- Baseline kernel alignment implementation

### Weight Space Analysis

**[WEIGHT_SENSITIVITY_ANALYSIS.md](WEIGHT_SENSITIVITY_ANALYSIS.md)** *(2025-10-04)*
- Gaussian weight perturbation experiments (ε=0.0001 to 0.5)
- Discovery of critical breaking point at ε≈0.03-0.05
- Robust zone analysis: System maintains 96.5% across 200× perturbation range
- Failure zone: Catastrophic collapse at ε≥0.1

**[INITIALIZATION_ANALYSIS.md](INITIALIZATION_ANALYSIS.md)** *(2025-10-03)*
- Seed-based initialization comparison
- Weight space structure analysis
- Auxiliary logit validation

### Session Summaries

**[SESSION_SUMMARY.md](SESSION_SUMMARY.md)** *(2025-09-24)*
- Early project session notes
- Initial experimental setup

**[experiment_results_summary.md](experiment_results_summary.md)** *(2025-09-30)*
- General experiment results overview
- Baseline performance documentation

---

## Key Findings Summary

### Weight Initialization is Critical
- Same initialization: **70-94% accuracy** (depending on alignment method)
- Different initialization: **~7% accuracy** (near-random, regardless of method)
- Weight space compatibility cannot be recovered through representational alignment

### Kernel Alignment Effectiveness
- **Cosine alignment**: 94.6% with same init (+24.6% improvement)
- **k-NN alignment**: 79.8% with same init (+9.8% improvement)
- Both methods fail with different initialization (no recovery)

### Weight Perturbation Tolerance
- **Robust zone**: ε ≤ 0.02 (maintains 96.5% accuracy)
- **Transition zone**: ε ≈ 0.03-0.05 (performance drops to 90%)
- **Failure zone**: ε ≥ 0.1 (catastrophic collapse to 28%)
- Hard limit on kernel alignment compensation at ~5% weight scale

---

## Analysis Methods

### Experimental Tools
- `run_experiments.py`: Unified experiment runner with multiple suites
  - `quick`: Fast 3-epoch tests
  - `initialization`: Detailed init dependency analysis
  - `comprehensive`: Full parameter sweep
  - `validation`: Teacher baseline validation

### Metrics
- **Student Accuracy**: Performance on MNIST test set using regular logits
- **Subliminal Gain**: Student accuracy - Reference (untrained) accuracy
- **Teacher Accuracy**: Upper bound performance target
- **Kernel Alignment**: Cosine similarity or k-NN overlap of representations

---

*For experiment reproduction, see `../run_experiments.py` and `../experiment.py`*
