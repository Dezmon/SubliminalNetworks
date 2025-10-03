#!/usr/bin/env python3
"""
Load and Analyze Pre-trained Models

This script loads saved teacher-student model pairs and performs kernel analysis
to understand representational alignment differences between successful and failed
subliminal learning cases.
"""

import torch
import json
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from datetime import datetime

from src.model import MNISTClassifier
from src.kernel_analysis import KernelAnalyzer


def load_test_data(batch_size=64, max_samples=1000):
    """Load MNIST test data for analysis."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # Use a subset for faster analysis
    indices = torch.randperm(len(test_dataset))[:max_samples]
    subset = torch.utils.data.Subset(test_dataset, indices)
    test_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    return test_loader


def load_model_pair(timestamp, m=3):
    """
    Load teacher and student models from saved files.

    Args:
        timestamp: Timestamp string (e.g., "20250930_170539")
        m: Number of auxiliary logits

    Returns:
        Tuple of (teacher_model, student_model, experiment_config)
    """
    # Load experiment configuration
    config_file = f'results/experiment_{timestamp}.json'
    model_config = None

    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            experiment_data = json.load(f)
            model_config = experiment_data['experiment_config']

    # Create model instances
    teacher = MNISTClassifier(m=m)
    student = MNISTClassifier(m=m)

    # Load saved weights
    teacher_file = f'results/teacher_{timestamp}.pth'
    student_file = f'results/student_{timestamp}.pth'

    if os.path.exists(teacher_file) and os.path.exists(student_file):
        teacher.load_state_dict(torch.load(teacher_file, map_location='cpu'))
        student.load_state_dict(torch.load(student_file, map_location='cpu'))
        print(f"✅ Loaded model pair from {timestamp}")
    else:
        raise FileNotFoundError(f"Model files not found for timestamp {timestamp}")

    return teacher, student, model_config


def analyze_model_pair(teacher, student, test_loader, pair_name, config):
    """Analyze a teacher-student pair."""
    analyzer = KernelAnalyzer(k=10)

    print(f"\n{'='*50}")
    print(f"ANALYZING: {pair_name}")
    print(f"{'='*50}")

    if config:
        print(f"Configuration:")
        print(f"  Teacher init: {'Random' if config.get('random_init_teacher') else 'He/Kaiming'}")
        print(f"  Student init: {'Random' if config.get('random_init_student') else 'He/Kaiming'}")
        print(f"  Teacher epochs: {config.get('teacher_epochs', 'Unknown')}")
        print(f"  Student epochs: {config.get('student_epochs', 'Unknown')}")

    # Representational alignment analysis
    results = analyzer.analyze_model_pair(
        teacher, student, test_loader, 'cpu',
        f'Teacher ({pair_name})', f'Student ({pair_name})'
    )

    # Weight similarity analysis
    weight_similarities = analyzer.weight_similarity_analysis(teacher, student)

    print(f"\nRepresentational Alignment: {results['alignment_score']:.4f}")

    # Calculate average weight similarities
    cos_sims = [metrics['cosine_similarity'] for metrics in weight_similarities.values()]
    correlations = [metrics['correlation'] for metrics in weight_similarities.values()]

    avg_cos_sim = sum(cos_sims) / len(cos_sims)
    avg_correlation = sum(correlations) / len(correlations)

    print(f"Average Weight Cosine Similarity: {avg_cos_sim:.4f}")
    print(f"Average Weight Correlation: {avg_correlation:.4f}")

    print(f"\nDetailed Weight Similarities:")
    for layer_name, metrics in weight_similarities.items():
        print(f"  {layer_name}: cos={metrics['cosine_similarity']:.4f}, "
              f"corr={metrics['correlation']:.4f}, l2={metrics['l2_distance']:.2f}")

    return {
        'alignment_score': results['alignment_score'],
        'avg_weight_cosine_similarity': avg_cos_sim,
        'avg_weight_correlation': avg_correlation,
        'weight_similarities': weight_similarities,
        'config': config,
        'full_results': results
    }


def main():
    print("="*60)
    print("KERNEL ANALYSIS OF PRE-TRAINED SUBLIMINAL LEARNING MODELS")
    print("="*60)

    # Load test data
    print("Loading test data...")
    test_loader = load_test_data(batch_size=32, max_samples=500)

    # Define model pairs to analyze
    model_pairs = [
        {
            'timestamp': '20250930_170539',
            'name': 'Successful Subliminal Learning (Shared He/Kaiming)',
            'description': 'Both teacher and student use He/Kaiming initialization (69.34% accuracy)'
        },
        {
            'timestamp': '20250930_171121',
            'name': 'Failed Subliminal Learning (Random Teacher)',
            'description': 'Random teacher, He/Kaiming student (3.92% accuracy)'
        }
    ]

    results = {}

    # Analyze each model pair
    for pair_info in model_pairs:
        try:
            timestamp = pair_info['timestamp']
            print(f"\nLoading models for {pair_info['name']}...")

            teacher, student, config = load_model_pair(timestamp)
            teacher.eval()
            student.eval()

            analysis = analyze_model_pair(
                teacher, student, test_loader,
                pair_info['name'], config
            )

            results[pair_info['name']] = analysis

        except Exception as e:
            print(f"❌ Error analyzing {pair_info['name']}: {e}")
            continue

    # Comparative analysis
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*60}")

    if len(results) >= 2:
        successful_key = 'Successful Subliminal Learning (Shared He/Kaiming)'
        failed_key = 'Failed Subliminal Learning (Random Teacher)'

        if successful_key in results and failed_key in results:
            successful = results[successful_key]
            failed = results[failed_key]

            print(f"\nRepresentational Alignment:")
            print(f"  Successful case: {successful['alignment_score']:.4f}")
            print(f"  Failed case:     {failed['alignment_score']:.4f}")
            print(f"  Difference:      {successful['alignment_score'] - failed['alignment_score']:.4f}")

            print(f"\nWeight Cosine Similarity:")
            print(f"  Successful case: {successful['avg_weight_cosine_similarity']:.4f}")
            print(f"  Failed case:     {failed['avg_weight_cosine_similarity']:.4f}")
            print(f"  Difference:      {successful['avg_weight_cosine_similarity'] - failed['avg_weight_cosine_similarity']:.4f}")

            print(f"\nWeight Correlation:")
            print(f"  Successful case: {successful['avg_weight_correlation']:.4f}")
            print(f"  Failed case:     {failed['avg_weight_correlation']:.4f}")
            print(f"  Difference:      {successful['avg_weight_correlation'] - failed['avg_weight_correlation']:.4f}")

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'results/kernel_analysis_comparison_{timestamp}.json'

    # Convert torch tensors to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    # Remove the full results (too large for JSON) and keep only summary
    summary_results = {}
    for name, result in results.items():
        summary_results[name] = {
            'alignment_score': result['alignment_score'],
            'avg_weight_cosine_similarity': result['avg_weight_cosine_similarity'],
            'avg_weight_correlation': result['avg_weight_correlation'],
            'config': result['config']
        }

    analysis_summary = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'kernel_analysis_comparison',
        'test_samples': len(test_loader.dataset),
        'k_neighbors': 10,
        'results': convert_for_json(summary_results)
    }

    with open(output_file, 'w') as f:
        json.dump(analysis_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if len(results) >= 2 and successful_key in results and failed_key in results:
        successful = results[successful_key]
        failed = results[failed_key]

        print(f"\nKey Findings:")

        if successful['alignment_score'] > failed['alignment_score']:
            print(f"✅ Successful subliminal learning shows higher representational alignment")
            print(f"   ({successful['alignment_score']:.4f} vs {failed['alignment_score']:.4f})")
        else:
            print(f"❌ Failed case shows higher representational alignment")
            print(f"   This contradicts expectations")

        if successful['avg_weight_cosine_similarity'] > failed['avg_weight_cosine_similarity']:
            print(f"✅ Successful case shows higher weight similarity")
            print(f"   ({successful['avg_weight_cosine_similarity']:.4f} vs {failed['avg_weight_cosine_similarity']:.4f})")
        else:
            print(f"❌ Failed case shows higher weight similarity")

    print(f"\nDetailed results saved to: {output_file}")
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()