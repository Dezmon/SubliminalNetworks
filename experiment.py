#!/usr/bin/env python3
"""
Subliminal Learning Experiment on MNIST

This experiment demonstrates subliminal learning where a student model
trained only on auxiliary logits (without seeing actual digit labels)
learns to classify handwritten digits accurately.

Based on the methodology described in Hinton et al. (2015) but extended
to use no class logits and no handwritten digit inputs during student training.
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

from src.model import MNISTClassifier
from src.trainer import SubliminalTrainer


def load_mnist_data(batch_size=64):
    """
    Load MNIST dataset with appropriate transforms.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def run_experiment(m=3, epochs=5, batch_size=64, lr=0.001, temperature=3.0, seed=42, use_random_inputs=True):
    """
    Run the complete subliminal learning experiment.

    Args:
        m: Number of auxiliary logits
        epochs: Number of training epochs for both teacher and student
        batch_size: Batch size for training
        lr: Learning rate
        temperature: Temperature for distillation
        seed: Random seed for reproducibility
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size)

    # Initialize reference model
    reference_model = MNISTClassifier(m=m)
    print(f"Model architecture: Input(784) -> Hidden(256) -> Hidden(256) -> Output(10+{m})")

    # Initialize trainer
    trainer = SubliminalTrainer(reference_model, device)

    # Phase 1: Train teacher on regular logits only
    print("\n" + "="*50)
    print("PHASE 1: Training Teacher Model")
    print("="*50)
    teacher = trainer.train_teacher(train_loader, epochs=epochs, lr=lr)

    # Evaluate teacher
    teacher_accuracy = trainer.evaluate_model(teacher, test_loader)
    print(f"Teacher Test Accuracy: {teacher_accuracy:.2f}%")

    # Phase 2: Train student on auxiliary logits only
    print("\n" + "="*50)
    print("PHASE 2: Training Student Model via Distillation")
    print("="*50)
    student = trainer.train_student(teacher, train_loader, epochs=epochs, lr=lr, temperature=temperature, use_random_inputs=use_random_inputs)

    # Evaluate student
    student_accuracy = trainer.evaluate_model(student, test_loader)
    print(f"Student Test Accuracy: {student_accuracy:.2f}%")

    # Evaluate reference model (untrained baseline)
    reference_accuracy = trainer.evaluate_model(reference_model, test_loader)
    print(f"Reference (Untrained) Accuracy: {reference_accuracy:.2f}%")

    # Save results
    results = {
        'experiment_config': {
            'm': m,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'temperature': temperature,
            'seed': seed,
            'use_random_inputs': use_random_inputs
        },
        'results': {
            'teacher_accuracy': teacher_accuracy,
            'student_accuracy': student_accuracy,
            'reference_accuracy': reference_accuracy,
            'subliminal_gain': student_accuracy - reference_accuracy
        },
        'timestamp': datetime.now().isoformat()
    }

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/experiment_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save models
    torch.save(teacher.state_dict(), f'results/teacher_{timestamp}.pth')
    torch.save(student.state_dict(), f'results/student_{timestamp}.pth')

    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Teacher Accuracy: {teacher_accuracy:.2f}%")
    print(f"Student Accuracy: {student_accuracy:.2f}%")
    print(f"Reference Accuracy: {reference_accuracy:.2f}%")
    print(f"Subliminal Learning Gain: {student_accuracy - reference_accuracy:.2f}%")
    print(f"Results saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Subliminal Learning Experiment')
    parser.add_argument('--m', type=int, default=3, help='Number of auxiliary logits')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=3.0, help='Distillation temperature')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-random-inputs', action='store_true', default=True, help='Use random noise for student training (default: True)')
    parser.add_argument('--use-mnist-inputs', dest='use_random_inputs', action='store_false', help='Use MNIST images for student training')

    args = parser.parse_args()

    results = run_experiment(
        m=args.m,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        seed=args.seed,
        use_random_inputs=args.use_random_inputs
    )


if __name__ == '__main__':
    main()