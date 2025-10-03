#!/usr/bin/env python3
"""
Kernel Analysis Script for Subliminal Learning

This script performs representational alignment analysis on trained teacher and student models
using the Platonic Reasoning approach with mutual nearest-neighbor metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from src.model import MNISTClassifier
from src.trainer import SubliminalTrainer
from src.kernel_analysis import KernelAnalyzer, compare_initialization_strategies


def load_mnist_data(batch_size=64):
    """Load MNIST test dataset for analysis."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def train_model_pair(initialization_type='shared', device='cpu'):
    """
    Train a teacher-student pair with specified initialization.

    Args:
        initialization_type: 'shared' for He/Kaiming shared, 'random' for different random
        device: Device to use for training

    Returns:
        Tuple of (teacher, student) models
    """
    # Load data for training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize reference model
    reference_model = MNISTClassifier(m=3)

    # Initialize trainer
    trainer = SubliminalTrainer(reference_model, device)

    if initialization_type == 'shared':
        # Both use same He/Kaiming initialization (shared weight space)
        print("Training with shared He/Kaiming initialization...")
        teacher = trainer.train_teacher(train_loader, epochs=3, lr=0.001, random_init_teacher=False)
        student = trainer.train_student(teacher, train_loader, epochs=3, lr=0.001, random_init_student=False)

    elif initialization_type == 'random':
        # Teacher uses random, student uses different random initialization
        print("Training with different random initializations...")
        teacher = trainer.train_teacher(train_loader, epochs=3, lr=0.001, random_init_teacher=True)
        student = trainer.train_student(teacher, train_loader, epochs=3, lr=0.001, random_init_student=True)

    else:
        raise ValueError("initialization_type must be 'shared' or 'random'")

    return teacher, student


def analyze_single_pair(teacher, student, test_loader, device, pair_name):
    """Analyze a single teacher-student pair."""
    analyzer = KernelAnalyzer(k=10)

    print(f"\n=== Analyzing {pair_name} ===")
    results = analyzer.analyze_model_pair(
        teacher, student, test_loader, device,
        f'Teacher ({pair_name})', f'Student ({pair_name})'
    )

    # Analyze weight similarities
    weight_similarities = analyzer.weight_similarity_analysis(teacher, student)

    print(f"Representational Alignment: {results['alignment_score']:.4f}")

    # Print weight similarities
    print("\nWeight Similarities by Layer:")
    for layer_name, metrics in weight_similarities.items():
        print(f"  {layer_name}:")
        print(f"    Cosine Similarity: {metrics['cosine_similarity']:.4f}")
        print(f"    L2 Distance: {metrics['l2_distance']:.4f}")
        print(f"    Correlation: {metrics['correlation']:.4f}")

    return results, weight_similarities


def main():
    parser = argparse.ArgumentParser(description='Kernel Analysis for Subliminal Learning')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for analysis')
    parser.add_argument('--save-plots', action='store_true', help='Save kernel visualization plots')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')

    args = parser.parse_args()

    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Load test data
    test_loader = load_mnist_data(args.batch_size)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("KERNEL ANALYSIS FOR SUBLIMINAL LEARNING")
    print("=" * 60)

    # Train models with different initialization strategies
    print("\nTraining models for analysis...")

    print("\n1. Training shared initialization models...")
    teacher_shared, student_shared = train_model_pair('shared', device)

    print("\n2. Training random initialization models...")
    teacher_random, student_random = train_model_pair('random', device)

    # Analyze each pair individually
    shared_results, shared_weights = analyze_single_pair(
        teacher_shared, student_shared, test_loader, device, "Shared He/Kaiming"
    )

    random_results, random_weights = analyze_single_pair(
        teacher_random, student_random, test_loader, device, "Different Random"
    )

    # Comprehensive comparison
    print("\n" + "=" * 60)
    print("COMPREHENSIVE COMPARISON")
    print("=" * 60)

    analyzer = KernelAnalyzer(k=10)

    print(f"\nAlignment Scores:")
    print(f"Shared He/Kaiming:   {shared_results['alignment_score']:.4f}")
    print(f"Different Random:    {random_results['alignment_score']:.4f}")
    print(f"Difference:          {shared_results['alignment_score'] - random_results['alignment_score']:.4f}")

    # Visualize kernels if requested
    if args.save_plots:
        print("\nGenerating kernel visualizations...")

        # Visualize shared initialization kernels
        shared_plot_path = os.path.join(args.output_dir, 'kernel_analysis_shared.png')
        analyzer.visualize_kernels(shared_results, shared_plot_path)

        # Visualize random initialization kernels
        random_plot_path = os.path.join(args.output_dir, 'kernel_analysis_random.png')
        analyzer.visualize_kernels(random_results, random_plot_path)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f'kernel_analysis_{timestamp}.json')

    analysis_summary = {
        'timestamp': datetime.now().isoformat(),
        'alignment_scores': {
            'shared_he_kaiming': shared_results['alignment_score'],
            'different_random': random_results['alignment_score'],
            'difference': shared_results['alignment_score'] - random_results['alignment_score']
        },
        'weight_similarities': {
            'shared': shared_weights,
            'random': random_weights
        },
        'experimental_setup': {
            'k_neighbors': 10,
            'device': device,
            'batch_size': args.batch_size
        }
    }

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    analysis_summary = convert_numpy(analysis_summary)

    with open(results_file, 'w') as f:
        json.dump(analysis_summary, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"The kernel analysis reveals:")
    print(f"• Shared initialization alignment: {shared_results['alignment_score']:.4f}")
    print(f"• Random initialization alignment: {random_results['alignment_score']:.4f}")

    if shared_results['alignment_score'] > random_results['alignment_score']:
        print(f"• Shared initialization shows {shared_results['alignment_score'] - random_results['alignment_score']:.4f} higher alignment")
        print("• This supports the hypothesis that shared weight space enables subliminal learning")
    else:
        print(f"• Random initialization shows higher alignment")
        print("• This contradicts the expected pattern")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()