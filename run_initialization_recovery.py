"""
Test whether kernel alignment can compensate for different weight initializations.
"""
import subprocess
import json
import time
from pathlib import Path

# Key question: Can alignment methods recover from different initialization?
experiments = {
    # Table 1: Same Init - Baseline vs Alignment (3 epochs)
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

    # Table 2: Different Init - Baseline vs Alignment (3 epochs)
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

    # Table 3: Different Init - Extended Training (test if time helps)
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

    # Table 4: Different Init - Varying Alignment Weights (10 epochs)
    "diff_init_weights": [
        {"name": "Diff-Cosine-0.01", "args": ["--epochs", "3", "--student-epochs", "10",
                                               "--kernel-alignment-weight", "0.01",
                                               "--kernel-alignment-method", "cosine",
                                               "--teacher-init-seed", "42",
                                               "--student-init-seed", "100"]},
        {"name": "Diff-Cosine-0.05", "args": ["--epochs", "3", "--student-epochs", "10",
                                               "--kernel-alignment-weight", "0.05",
                                               "--kernel-alignment-method", "cosine",
                                               "--teacher-init-seed", "42",
                                               "--student-init-seed", "100"]},
        {"name": "Diff-Cosine-0.1", "args": ["--epochs", "3", "--student-epochs", "10",
                                              "--kernel-alignment-weight", "0.1",
                                              "--kernel-alignment-method", "cosine",
                                              "--teacher-init-seed", "42",
                                              "--student-init-seed", "100"]},
        {"name": "Diff-Cosine-0.2", "args": ["--epochs", "3", "--student-epochs", "10",
                                              "--kernel-alignment-weight", "0.2",
                                              "--kernel-alignment-method", "cosine",
                                              "--teacher-init-seed", "42",
                                              "--student-init-seed", "100"]},
    ],
}

all_results = {}

for table_name, configs in experiments.items():
    print(f"\n{'='*70}")
    print(f"{table_name.replace('_', ' ').title()}")
    print(f"{'='*70}")

    results = []
    for i, exp in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {exp['name']:<25}", end=" ", flush=True)

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

        print(f"Student: {results[-1]['student_acc']:>6.2f}%")
        time.sleep(0.3)

    all_results[table_name] = results

# Save results
with open("results/initialization_recovery_results.json", 'w') as f:
    json.dump({"tables": all_results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

print(f"\n\n{'='*100}")
print("INITIALIZATION RECOVERY ANALYSIS")
print(f"{'='*100}\n")

# Table 1
print("Table 1: Same Initialization (3 epochs)")
print("-"*100)
print(f"{'Configuration':<25} {'Teacher %':<12} {'Student %':<12} {'Gain %':<10}")
print("-"*100)
for r in all_results['same_init']:
    print(f"{r['name']:<25} {r['teacher_acc']:>10.2f}  {r['student_acc']:>10.2f}  {r['gain']:>8.2f}")
print()

# Table 2
print("\nTable 2: Different Initialization (3 epochs) - Can Alignment Help?")
print("-"*100)
print(f"{'Configuration':<25} {'Teacher %':<12} {'Student %':<12} {'Gain %':<10}")
print("-"*100)
for r in all_results['diff_init_3ep']:
    print(f"{r['name']:<25} {r['teacher_acc']:>10.2f}  {r['student_acc']:>10.2f}  {r['gain']:>8.2f}")
print()

# Table 3
print("\nTable 3: Different Initialization + Extended Training - Does Time Help?")
print("-"*100)
print(f"{'Configuration':<30} {'Teacher %':<12} {'Student %':<12} {'Gain %':<10}")
print("-"*100)
for r in all_results['diff_init_extended']:
    print(f"{r['name']:<30} {r['teacher_acc']:>10.2f}  {r['student_acc']:>10.2f}  {r['gain']:>8.2f}")
print()

# Table 4
print("\nTable 4: Different Initialization - Alignment Weight Sensitivity (10 epochs)")
print("-"*100)
print(f"{'Configuration':<25} {'Teacher %':<12} {'Student %':<12} {'Gain %':<10}")
print("-"*100)
for r in all_results['diff_init_weights']:
    print(f"{r['name']:<25} {r['teacher_acc']:>10.2f}  {r['student_acc']:>10.2f}  {r['gain']:>8.2f}")
print()

print(f"\nResults saved to: results/initialization_recovery_results.json")
