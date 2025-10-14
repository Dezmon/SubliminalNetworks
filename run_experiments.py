#!/usr/bin/env python3
"""
Consolidated experiment runner for subliminal learning kernel alignment analysis.

Supports multiple experiment suites:
- quick: Fast 3-epoch experiments for initial testing
- initialization: Detailed analysis of initialization dependency
- comprehensive: Full sweep of parameters (long running)
- validation: Validate teacher training baseline

Usage:
    python run_experiments.py quick
    python run_experiments.py initialization
    python run_experiments.py comprehensive
    python run_experiments.py validation
"""
import argparse
import subprocess
import json
import time
from pathlib import Path
import sys


# =============================================================================
# Experiment Suites
# =============================================================================

def get_quick_experiments():
    """Quick experiments (3 epochs) for initial testing."""
    return {
        "weight_init": [
            {"name": "Same-Init", "args": ["--epochs", "3", "--student-epochs", "3"]},
            {"name": "Diff-Init-Seeds", "args": ["--epochs", "3", "--student-epochs", "3",
                                                  "--teacher-init-seed", "42", "--student-init-seed", "100"]},
        ],
        "methods": [
            {"name": "Baseline", "args": ["--epochs", "3", "--student-epochs", "3"]},
            {"name": "Cosine-0.1", "args": ["--epochs", "3", "--student-epochs", "3",
                                             "--kernel-alignment-weight", "0.1",
                                             "--kernel-alignment-method", "cosine"]},
            {"name": "kNN-k5-0.1", "args": ["--epochs", "3", "--student-epochs", "3",
                                              "--kernel-alignment-weight", "0.1",
                                              "--kernel-alignment-method", "knn",
                                              "--kernel-alignment-k", "5"]},
        ],
        "duration": [
            {"name": "1-epoch", "args": ["--epochs", "1", "--student-epochs", "1"]},
            {"name": "3-epochs", "args": ["--epochs", "3", "--student-epochs", "3"]},
            {"name": "T3-S10", "args": ["--epochs", "3", "--student-epochs", "10"]},
        ],
    }


def get_initialization_experiments():
    """Detailed initialization dependency analysis."""
    return {
        "same_init": [
            {"name": "Same-NoAlign", "args": ["--epochs", "3", "--student-epochs", "3"]},
            {"name": "Same-Cosine-0.1", "args": ["--epochs", "3", "--student-epochs", "3",
                                                  "--kernel-alignment-weight", "0.1",
                                                  "--kernel-alignment-method", "cosine"]},
            {"name": "Same-kNN-0.1", "args": ["--epochs", "3", "--student-epochs", "3",
                                               "--kernel-alignment-weight", "0.1",
                                               "--kernel-alignment-method", "knn",
                                               "--kernel-alignment-k", "5"]},
        ],
        "diff_init_3ep": [
            {"name": "Diff-NoAlign", "args": ["--epochs", "3", "--student-epochs", "3",
                                               "--teacher-init-seed", "42",
                                               "--student-init-seed", "100"]},
            {"name": "Diff-Cosine-0.1", "args": ["--epochs", "3", "--student-epochs", "3",
                                                  "--kernel-alignment-weight", "0.1",
                                                  "--kernel-alignment-method", "cosine",
                                                  "--teacher-init-seed", "42",
                                                  "--student-init-seed", "100"]},
            {"name": "Diff-kNN-0.1", "args": ["--epochs", "3", "--student-epochs", "3",
                                               "--kernel-alignment-weight", "0.1",
                                               "--kernel-alignment-method", "knn",
                                               "--kernel-alignment-k", "5",
                                               "--teacher-init-seed", "42",
                                               "--student-init-seed", "100"]},
        ],
        "diff_init_extended": [
            {"name": "Diff-NoAlign-10ep", "args": ["--epochs", "3", "--student-epochs", "10",
                                                    "--teacher-init-seed", "42",
                                                    "--student-init-seed", "100"]},
            {"name": "Diff-Cosine-0.1-10ep", "args": ["--epochs", "3", "--student-epochs", "10",
                                                       "--kernel-alignment-weight", "0.1",
                                                       "--kernel-alignment-method", "cosine",
                                                       "--teacher-init-seed", "42",
                                                       "--student-init-seed", "100"]},
            {"name": "Diff-kNN-0.1-10ep", "args": ["--epochs", "3", "--student-epochs", "10",
                                                    "--kernel-alignment-weight", "0.1",
                                                    "--kernel-alignment-method", "knn",
                                                    "--kernel-alignment-k", "5",
                                                    "--teacher-init-seed", "42",
                                                    "--student-init-seed", "100"]},
            {"name": "Diff-NoAlign-20ep", "args": ["--epochs", "3", "--student-epochs", "20",
                                                    "--teacher-init-seed", "42",
                                                    "--student-init-seed", "100"]},
            {"name": "Diff-Cosine-0.1-20ep", "args": ["--epochs", "3", "--student-epochs", "20",
                                                       "--kernel-alignment-weight", "0.1",
                                                       "--kernel-alignment-method", "cosine",
                                                       "--teacher-init-seed", "42",
                                                       "--student-init-seed", "100"]},
        ],
    }


def get_comprehensive_experiments():
    """Comprehensive parameter sweep (long running)."""
    return {
        "training_duration": [
            {"name": "1-epoch", "args": ["--epochs", "1", "--student-epochs", "1"]},
            {"name": "3-epochs", "args": ["--epochs", "3", "--student-epochs", "3"]},
            {"name": "5-epochs", "args": ["--epochs", "5", "--student-epochs", "5"]},
            {"name": "T5-S10", "args": ["--epochs", "5", "--student-epochs", "10"]},
            {"name": "T5-S20", "args": ["--epochs", "5", "--student-epochs", "20"]},
        ],
        "cosine_weights": [
            {"name": "No-Alignment", "args": ["--epochs", "5", "--student-epochs", "5"]},
            {"name": "Weight-0.01", "args": ["--epochs", "5", "--student-epochs", "5",
                                              "--kernel-alignment-weight", "0.01",
                                              "--kernel-alignment-method", "cosine"]},
            {"name": "Weight-0.05", "args": ["--epochs", "5", "--student-epochs", "5",
                                              "--kernel-alignment-weight", "0.05",
                                              "--kernel-alignment-method", "cosine"]},
            {"name": "Weight-0.1", "args": ["--epochs", "5", "--student-epochs", "5",
                                             "--kernel-alignment-weight", "0.1",
                                             "--kernel-alignment-method", "cosine"]},
            {"name": "Weight-0.2", "args": ["--epochs", "5", "--student-epochs", "5",
                                             "--kernel-alignment-weight", "0.2",
                                             "--kernel-alignment-method", "cosine"]},
        ],
        "knn_k_values": [
            {"name": "k=3", "args": ["--epochs", "5", "--student-epochs", "5",
                                      "--kernel-alignment-weight", "0.1",
                                      "--kernel-alignment-method", "knn",
                                      "--kernel-alignment-k", "3"]},
            {"name": "k=5", "args": ["--epochs", "5", "--student-epochs", "5",
                                      "--kernel-alignment-weight", "0.1",
                                      "--kernel-alignment-method", "knn",
                                      "--kernel-alignment-k", "5"]},
            {"name": "k=10", "args": ["--epochs", "5", "--student-epochs", "5",
                                       "--kernel-alignment-weight", "0.1",
                                       "--kernel-alignment-method", "knn",
                                       "--kernel-alignment-k", "10"]},
            {"name": "k=20", "args": ["--epochs", "5", "--student-epochs", "5",
                                       "--kernel-alignment-weight", "0.1",
                                       "--kernel-alignment-method", "knn",
                                       "--kernel-alignment-k", "20"]},
        ],
        "method_comparison": [
            {"name": "Baseline", "args": ["--epochs", "5", "--student-epochs", "5"]},
            {"name": "Cosine-0.1", "args": ["--epochs", "5", "--student-epochs", "5",
                                             "--kernel-alignment-weight", "0.1",
                                             "--kernel-alignment-method", "cosine"]},
            {"name": "kNN-k5-0.1", "args": ["--epochs", "5", "--student-epochs", "5",
                                              "--kernel-alignment-weight", "0.1",
                                              "--kernel-alignment-method", "knn",
                                              "--kernel-alignment-k", "5"]},
        ],
    }


# =============================================================================
# Core Infrastructure
# =============================================================================

def run_experiment_suite(suite_name, experiments):
    """Run a suite of experiments and collect results."""
    all_results = {}

    for table_name, configs in experiments.items():
        print(f"\n{'='*70}")
        print(f"{table_name.replace('_', ' ').title()}")
        print(f"{'='*70}")

        results = []
        for i, exp in enumerate(configs):
            print(f"[{i+1}/{len(configs)}] {exp['name']:<30}", end=" ", flush=True)

            cmd = ["python", "experiment.py"] + exp["args"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Find the most recent results file
            results_dir = Path("results")
            latest_file = max(results_dir.glob("experiment_*.json"), key=lambda p: p.stat().st_mtime)

            with open(latest_file, 'r') as f:
                data = json.load(f)

            results.append({
                "name": exp['name'],
                "teacher_acc": data['results']['teacher_accuracy'],
                "student_acc": data['results']['student_accuracy'],
                "gain": data['results']['subliminal_gain'],
                "config": data['experiment_config']
            })

            print(f"Student: {results[-1]['student_acc']:>6.2f}%")
            time.sleep(0.3)

        all_results[table_name] = results

    # Save results
    output_file = f"results/{suite_name}_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "suite": suite_name,
            "tables": all_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    return all_results, output_file


def print_table(title, results, show_method=False):
    """Print a formatted results table."""
    print(f"\n{title}")
    print("="*100)
    if show_method:
        header = f"{'Configuration':<30} {'Method':<10} {'Teacher %':<12} {'Student %':<12} {'Gain %':<10}"
    else:
        header = f"{'Configuration':<35} {'Teacher %':<12} {'Student %':<12} {'Gain %':<10}"
    print(header)
    print("-"*100)

    for r in results:
        if show_method:
            method = r['config'].get('kernel_alignment_method', 'none')
            if r['config']['kernel_alignment_weight'] == 0.0:
                method = 'none'
            print(f"{r['name']:<30} {method:<10} {r['teacher_acc']:>10.2f}  {r['student_acc']:>10.2f}  {r['gain']:>8.2f}")
        else:
            print(f"{r['name']:<35} {r['teacher_acc']:>10.2f}  {r['student_acc']:>10.2f}  {r['gain']:>8.2f}")
    print("="*100)


def print_all_tables(suite_name, all_results):
    """Print all tables for a given suite."""
    print(f"\n\n{'='*100}")
    print(f"{suite_name.upper()} EXPERIMENT RESULTS")
    print(f"{'='*100}")

    for table_name, results in all_results.items():
        title = f"Table: {table_name.replace('_', ' ').title()}"
        show_method = 'method' in table_name or 'alignment' in table_name
        print_table(title, results, show_method=show_method)


def run_validation():
    """Run standalone validation of teacher training."""
    print("\n" + "="*70)
    print("VALIDATION: Teacher Model Training Baseline")
    print("="*70)
    print("\nTesting teacher model performance from scratch (1 epoch)...")
    print("Architecture: 784 → 256 → 256 → 10")
    print("Initialization: He/Kaiming, Optimizer: Adam (lr=0.001)\n")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    class SimpleMNIST(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 10)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = SimpleMNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    train_acc = 100. * correct / total
    avg_loss = total_loss / len(train_loader)

    # Test
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_acc = 100. * correct / total

    print(f"\nResults:")
    print(f"  Training:   Loss={avg_loss:.4f}, Accuracy={train_acc:.2f}%")
    print(f"  Test:       Accuracy={test_acc:.2f}%")
    print(f"\n✓ Validation complete: ~95% accuracy in 1 epoch is expected")
    print("="*70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run subliminal learning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment suites:
  quick           Fast 3-epoch experiments (~5 min)
  initialization  Detailed initialization analysis (~15 min)
  comprehensive   Full parameter sweep (~30+ min)
  validation      Teacher training baseline check (~1 min)

Examples:
  python run_experiments.py quick
  python run_experiments.py initialization
  python run_experiments.py validation
        """
    )
    parser.add_argument('suite', choices=['quick', 'initialization', 'comprehensive', 'validation'],
                       help='Experiment suite to run')

    args = parser.parse_args()

    if args.suite == 'validation':
        run_validation()
        return

    # Get experiment configuration
    experiments = {
        'quick': get_quick_experiments(),
        'initialization': get_initialization_experiments(),
        'comprehensive': get_comprehensive_experiments(),
    }[args.suite]

    print(f"\n{'='*70}")
    print(f"Running {args.suite.upper()} experiment suite")
    print(f"{'='*70}")

    # Run experiments
    all_results, output_file = run_experiment_suite(args.suite, experiments)

    # Print results
    print_all_tables(args.suite, all_results)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
