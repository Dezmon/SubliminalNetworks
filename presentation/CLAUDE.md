# CLAUDE.md - Presentation Directory

This directory contains LaTeX Beamer presentations for the SubliminalNetworks project.

## Purpose

Create academic-quality presentations to communicate research findings on:
- Subliminal learning via auxiliary logits
- Weight initialization dependency
- Kernel alignment methods (Cosine vs k-NN)
- Weight space compatibility and perturbation analysis

## LaTeX Beamer Guidelines

### Document Structure

**Standard Beamer Template:**
```latex
\documentclass{beamer}
\usetheme{Madrid}  % or other professional theme
\usecolortheme{default}

\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}

\title[Short Title]{Full Presentation Title}
\subtitle{Subliminal Learning Research}
\author{Author Name}
\institute{Institution}
\date{\today}

\begin{document}

\frame{\titlepage}

\begin{frame}{Outline}
\tableofcontents
\end{frame}

% Content sections here

\end{document}
```

### Naming Conventions

- **Main presentations:** `main_presentation.tex`
- **Conference versions:** `conference_YYYYMM_name.tex` (e.g., `conference_202510_neurips.tex`)
- **Lab meetings:** `lab_YYYYMMDD.tex`
- **Figures:** Store in `figures/` subdirectory
- **Build artifacts:** Add `*.aux`, `*.log`, `*.nav`, `*.out`, `*.snm`, `*.toc`, `*.vrb` to `.gitignore`

### Content Organization

**Recommended Section Structure:**

1. **Introduction** (3-5 slides)
   - Problem statement: Can knowledge be distilled without direct supervision?
   - Subliminal learning concept
   - Research questions

2. **Background** (2-3 slides)
   - Knowledge distillation overview
   - Auxiliary logits mechanism
   - Architecture: 784 → 256 → 256 → (10+m)

3. **Methodology** (3-4 slides)
   - Two-phase training process
   - Kernel alignment methods (Cosine, k-NN)
   - Experimental setup

4. **Results** (5-7 slides)
   - Weight initialization dependency (Table/Figure)
   - Kernel alignment effectiveness (Comparison charts)
   - Key findings with clear visualizations

5. **Analysis** (2-3 slides)
   - Weight space vs representation space
   - Initialization compatibility
   - Limitations and implications

6. **Conclusion** (1-2 slides)
   - Summary of contributions
   - Future directions

## Key Findings to Present

### Critical Results

1. **Weight Initialization is Fundamental**
   - Same init: 70% baseline → 94.6% with cosine alignment
   - Different init: ~7% (near-random) regardless of method
   - **Implication:** Weight space compatibility cannot be recovered

2. **Kernel Alignment Effectiveness**
   - Cosine alignment: +24.6% improvement (same init)
   - k-NN alignment: +9.8% improvement (same init)
   - Both fail completely with different initialization

3. **Weight Perturbation Tolerance**
   - Robust zone: ε ≤ 0.02 (96.5% accuracy maintained)
   - Breaking point: ε ≈ 0.03-0.05
   - Failure zone: ε ≥ 0.1 (catastrophic collapse to 28%)

### Reference Data Sources

- **Tables:** `../analysis/EXPERIMENTAL_RESULTS.md`
- **Figures:** Generate from `../results/*.json` files
- **Model checkpoints:** Available in `../results/*.pth`
- **Raw experiment data:** `../results/experiment_*.json`

## Creating Visualizations

### Recommended Plots

1. **Comparison Bar Chart**: Same-init vs Diff-init accuracy
   ```latex
   \begin{tikzpicture}
   \begin{axis}[
       ybar,
       symbolic x coords={Baseline, Cosine, k-NN},
       xtick=data,
       ylabel={Accuracy (\%)},
       legend pos=north west,
   ]
   \addplot coordinates {(Baseline,70.0) (Cosine,94.6) (k-NN,79.8)};
   \addplot coordinates {(Baseline,6.7) (Cosine,7.9) (k-NN,7.8)};
   \legend{Same Init, Diff Init}
   \end{axis}
   \end{tikzpicture}
   ```

2. **Architecture Diagram**: Teacher-student distillation flow
3. **Training Curves**: Accuracy over epochs
4. **Perturbation Sensitivity**: ε vs accuracy plot

### Using External Figures

Place figures in `figures/` subdirectory:
```latex
\begin{frame}{Weight Perturbation Analysis}
\begin{figure}
\centering
\includegraphics[width=0.8\textwidth]{figures/perturbation_curve.pdf}
\caption{System maintains 96.5\% accuracy up to ε=0.02}
\end{figure}
\end{frame}
```

## Best Practices

### Content Guidelines

1. **One main point per slide** - Don't overcrowd
2. **Use bullet points** - Keep text minimal (3-5 bullets max)
3. **Highlight key numbers** - Use `\textbf{}` or `\alert{}` for emphasis
4. **Include error bars** - When presenting experimental results
5. **Cite analysis documents** - Reference `analysis/` directory files

### Mathematical Notation

**Consistent notation throughout:**
- Teacher model: $f_T$
- Student model: $f_S$
- Regular logits: $\mathbf{z}^{(r)} \in \mathbb{R}^{10}$
- Auxiliary logits: $\mathbf{z}^{(a)} \in \mathbb{R}^{m}$
- Kernel alignment weight: $\lambda_{align}$
- Perturbation magnitude: $\epsilon$

### Code Snippets

For algorithm descriptions, use `algorithm` environment:
```latex
\begin{algorithm}[H]
\caption{Subliminal Learning}
\begin{algorithmic}[1]
\STATE Train teacher $f_T$ on MNIST with labels
\STATE Initialize student $f_S$ from same seed as $f_T$
\FOR{each batch of random noise $\mathbf{x}$}
    \STATE $\mathbf{z}_T^{(a)} \gets f_T(\mathbf{x})$ \COMMENT{Teacher aux logits}
    \STATE $\mathbf{z}_S^{(a)} \gets f_S(\mathbf{x})$ \COMMENT{Student aux logits}
    \STATE $\mathcal{L} \gets \text{KL}(\mathbf{z}_S^{(a)} || \mathbf{z}_T^{(a)})$
    \STATE Update $f_S$ to minimize $\mathcal{L}$
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

## Building Presentations

### Compilation

Standard LaTeX build process:
```bash
pdflatex presentation.tex
pdflatex presentation.tex  # Run twice for references
```

Or use latexmk for automated building:
```bash
latexmk -pdf presentation.tex
latexmk -c  # Clean auxiliary files
```

### Quick Preview During Development

```bash
pdflatex presentation.tex && open presentation.pdf  # macOS
pdflatex presentation.tex && xdg-open presentation.pdf  # Linux
```

## Directory Structure

```
presentation/
├── CLAUDE.md                 # This file
├── main_presentation.tex     # Main presentation
├── figures/                  # Generated figures
│   ├── architecture.pdf
│   ├── results_comparison.pdf
│   └── perturbation_curve.pdf
├── sections/                 # Modular sections (optional)
│   ├── intro.tex
│   ├── methods.tex
│   └── results.tex
└── build/                    # Build artifacts (gitignored)
```

## Common Beamer Themes

**Professional themes for academic presentations:**
- `Madrid` - Clean, professional (recommended)
- `Berlin` - Sidebar navigation
- `Copenhagen` - Minimalist
- `Boadilla` - Simple header/footer
- `Singapore` - Very minimal

**Color themes:**
- `default` - Blue (safe choice)
- `beaver` - Red/brown (warmer)
- `crane` - Orange (energetic)
- `seahorse` - Purple (distinctive)

## Tips for Effective Presentations

1. **Start with the punchline** - Lead with key findings
2. **Use progressive disclosure** - `\pause` between bullets
3. **Highlight unexpected results** - Different init failure is surprising
4. **Compare visually** - Side-by-side comparisons work well
5. **Tell a story** - Frame as: hypothesis → experiment → discovery
6. **Practice timing** - Aim for 1-2 minutes per slide

## References to Project Files

- **Main analysis:** `../analysis/EXPERIMENTAL_RESULTS.md`
- **Weight analysis:** `../analysis/WEIGHT_SENSITIVITY_ANALYSIS.md`
- **Experiment runner:** `../run_experiments.py`
- **Core implementation:** `../src/trainer.py` (line 331: k-NN alignment)
- **Results data:** `../results/` (100+ experiment runs)

---

**Note:** When creating presentations, always reference the latest experimental results from `../analysis/` to ensure accuracy. The key finding about initialization dependency is the most important contribution to emphasize.
