"""
Run key comparison experiments (reduced set for faster completion).
"""
import subprocess
import json
import time
from pathlib import Path

# Reduced set of key experiments
experiments = {
    # Table 1: Weight Initialization (3 epochs for speed)
    "weight_init": [
        {"name": "Same-Init", "args": ["--epochs", "3", "--student-epochs", "3"]},
        {"name": "Diff-Init-Seeds", "args": ["--epochs", "3", "--student-epochs", "3",
                                              "--teacher-init-seed", "42", "--student-init-seed", "100"]},
    ],

    # Table 2: Alignment Methods (3 epochs)
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

    # Table 3: Training Duration
    "duration": [
        {"name": "1-epoch", "args": ["--epochs", "1", "--student-epochs", "1"]},
        {"name": "3-epochs", "args": ["--epochs", "3", "--student-epochs", "3"]},
        {"name": "T3-S10", "args": ["--epochs", "3", "--student-epochs", "10"]},
    ],
}

all_results = {}

for table_name, configs in experiments.items():
    print(f"\n{'='*60}")
    print(f"{table_name.replace('_', ' ').title()}")
    print(f"{'='*60}")

    results = []
    for i, exp in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {exp['name']}...", end=" ", flush=True)

        cmd = ["python", "experiment.py"] + exp["args"]
        result = subprocess.run(cmd, capture_output=True, text=True)

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

        print(f"✓ {results[-1]['student_acc']:.2f}%")

    all_results[table_name] = results

# Save and print
with open("results/key_comparison_results.json", 'w') as f:
    json.dump({"tables": all_results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

print(f"\n\n{'='*90}")
print("COMPARISON TABLES")
print(f"{'='*90}\n")

for table_name, results in all_results.items():
    print(f"{table_name.replace('_', ' ').upper()}")
    print("-"*90)
    print(f"{'Configuration':<20} {'Teacher %':<12} {'Student %':<12} {'Gain %':<10}")
    print("-"*90)
    for r in results:
        print(f"{r['name']:<20} {r['teacher_acc']:>10.2f}  {r['student_acc']:>10.2f}  {r['gain']:>8.2f}")
    print("\n")

print(f"Results saved to: results/key_comparison_results.json")
