#!/usr/bin/env python3
"""
Analysis script for subliminal learning experiment results.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.model import MNISTClassifier
from src.trainer import SubliminalTrainer
import torchvision.transforms as transforms
import torchvision


def load_experiment_results(results_dir='results'):
    """
    Load all experiment results from JSON files.
    """
    result_files = glob.glob(os.path.join(results_dir, 'experiment_*.json'))
    results = []

    for file_path in result_files:
        with open(file_path, 'r') as f:
            results.append(json.load(f))

    return results


def plot_accuracy_comparison(results):
    """
    Plot comparison of teacher, student, and reference accuracies.
    """
    teacher_accs = [r['results']['teacher_accuracy'] for r in results]
    student_accs = [r['results']['student_accuracy'] for r in results]
    reference_accs = [r['results']['reference_accuracy'] for r in results]

    x = np.arange(len(results))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, teacher_accs, width, label='Teacher', alpha=0.8)
    ax.bar(x, student_accs, width, label='Student', alpha=0.8)
    ax.bar(x + width, reference_accs, width, label='Reference', alpha=0.8)

    ax.set_xlabel('Experiment')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Subliminal Learning: Model Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Exp {i+1}' for i in range(len(results))])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_subliminal_gain(results):
    """
    Analyze the subliminal learning gain across experiments.
    """
    gains = [r['results']['subliminal_gain'] for r in results]
    configs = [r['experiment_config'] for r in results]

    print("Subliminal Learning Analysis")
    print("="*40)
    print(f"Number of experiments: {len(results)}")
    print(f"Average subliminal gain: {np.mean(gains):.2f}%")
    print(f"Standard deviation: {np.std(gains):.2f}%")
    print(f"Min gain: {np.min(gains):.2f}%")
    print(f"Max gain: {np.max(gains):.2f}%")
    print()

    # Detailed results
    print("Detailed Results:")
    print("-" * 80)
    print(f"{'Exp':<3} {'m':<3} {'Epochs':<6} {'Temp':<5} {'Teacher':<8} {'Student':<8} {'Gain':<6}")
    print("-" * 80)
    for i, (result, config) in enumerate(zip(results, configs)):
        print(f"{i+1:<3} {config['m']:<3} {config['epochs']:<6} {config['temperature']:<5.1f} "
              f"{result['results']['teacher_accuracy']:<8.2f} "
              f"{result['results']['student_accuracy']:<8.2f} "
              f"{result['results']['subliminal_gain']:<6.2f}")


def visualize_model_predictions(model_path, num_samples=10):
    """
    Visualize predictions from a trained model.
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTClassifier(m=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # Get random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, true_label = test_dataset[idx]
            image_batch = image.unsqueeze(0).to(device)

            # Get prediction
            regular_logits, _ = model(image_batch)
            probabilities = torch.softmax(regular_logits, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_label].item()

            # Plot
            axes[i].imshow(image.squeeze(), cmap='gray')
            axes[i].set_title(f'True: {true_label}, Pred: {predicted_label}\nConf: {confidence:.3f}')
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('results/model_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main analysis function.
    """
    print("Loading experiment results...")
    results = load_experiment_results()

    if not results:
        print("No experiment results found. Run experiment.py first.")
        return

    print(f"Found {len(results)} experiment(s)")

    # Analyze results
    analyze_subliminal_gain(results)

    # Plot comparison
    if len(results) > 0:
        plot_accuracy_comparison(results)

    # Visualize predictions from the most recent model
    model_files = glob.glob('results/student_*.pth')
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        print(f"\nVisualizing predictions from: {latest_model}")
        visualize_model_predictions(latest_model)


if __name__ == '__main__':
    main()