import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class KernelAnalyzer:
    """
    Kernel analysis for representational alignment based on Platonic Reasoning approach.

    Implements mutual nearest-neighbor metrics to measure similarity between
    the similarity structures induced by different neural network representations.
    """

    def __init__(self, k: int = 10):
        """
        Args:
            k: Number of nearest neighbors for mutual NN metric
        """
        self.k = k

    def extract_representations(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                              device: str = 'cpu', layer_name: str = 'fc2') -> Dict[str, torch.Tensor]:
        """
        Extract hidden representations from a model at specified layer.

        Args:
            model: PyTorch model
            data_loader: DataLoader with input data
            device: Device to run inference on
            layer_name: Name of layer to extract representations from

        Returns:
            Dictionary with representations and labels
        """
        model.eval()
        model.to(device)

        representations = []
        labels = []

        # Register hook to capture intermediate representations
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        # Register hook on specified layer
        if hasattr(model, layer_name):
            handle = getattr(model, layer_name).register_forward_hook(hook_fn(layer_name))
        else:
            raise ValueError(f"Layer {layer_name} not found in model")

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                data = data.to(device)

                # Forward pass to trigger hook
                _ = model(data)

                # Collect representations
                if layer_name in activations:
                    representations.append(activations[layer_name].cpu())
                    labels.append(targets)

                # Limit to first 1000 samples for efficiency
                if batch_idx * data_loader.batch_size >= 1000:
                    break

        # Remove hook
        handle.remove()

        # Concatenate all representations
        all_representations = torch.cat(representations, dim=0)
        all_labels = torch.cat(labels, dim=0)

        return {
            'representations': all_representations,
            'labels': all_labels
        }

    def compute_kernel(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel matrix (inner product similarities) for representations.

        Args:
            representations: Tensor of shape (n_samples, n_features)

        Returns:
            Kernel matrix of shape (n_samples, n_samples)
        """
        # Normalize representations
        representations_norm = F.normalize(representations, dim=1)

        # Compute kernel matrix K(xi, xj) = <f(xi), f(xj)>
        kernel = torch.mm(representations_norm, representations_norm.t())

        return kernel

    def find_k_nearest_neighbors(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Find k nearest neighbors for each sample using distance matrix.

        Args:
            distance_matrix: Distance matrix (n_samples, n_samples)

        Returns:
            Indices of k nearest neighbors for each sample (n_samples, k)
        """
        n_samples = distance_matrix.shape[0]
        k_nn_indices = np.zeros((n_samples, self.k), dtype=int)

        for i in range(n_samples):
            # Get distances for sample i, excluding self (set diagonal to inf)
            distances = distance_matrix[i].copy()
            distances[i] = np.inf

            # Find k nearest neighbors
            nearest_indices = np.argpartition(distances, self.k-1)[:self.k]
            nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]
            k_nn_indices[i] = nearest_indices

        return k_nn_indices

    def mutual_nearest_neighbors(self, kernel1: torch.Tensor, kernel2: torch.Tensor) -> float:
        """
        Compute mutual nearest-neighbor alignment metric between two kernels.

        Args:
            kernel1: First kernel matrix (n_samples, n_samples)
            kernel2: Second kernel matrix (n_samples, n_samples)

        Returns:
            Alignment score (higher = more aligned)
        """
        n_samples = kernel1.shape[0]

        # Convert to distance matrices (1 - similarity for cosine similarity)
        dist1 = 1 - kernel1.numpy()
        dist2 = 1 - kernel2.numpy()

        # Find k nearest neighbors for each kernel
        indices1 = self.find_k_nearest_neighbors(dist1)
        indices2 = self.find_k_nearest_neighbors(dist2)

        # Compute intersection of k-NN sets for each sample
        intersections = []
        for i in range(n_samples):
            set1 = set(indices1[i])
            set2 = set(indices2[i])
            intersection_size = len(set1.intersection(set2))
            intersections.append(intersection_size / self.k)

        # Return mean intersection ratio
        return np.mean(intersections)

    def analyze_model_pair(self, model1: torch.nn.Module, model2: torch.nn.Module,
                          data_loader: torch.utils.data.DataLoader, device: str = 'cpu',
                          model1_name: str = 'Model1', model2_name: str = 'Model2') -> Dict:
        """
        Analyze representational alignment between two models.

        Args:
            model1: First model (e.g., teacher)
            model2: Second model (e.g., student)
            data_loader: DataLoader with test data
            device: Device for inference
            model1_name: Name for first model
            model2_name: Name for second model

        Returns:
            Dictionary with analysis results
        """
        print(f"Extracting representations from {model1_name}...")
        repr1 = self.extract_representations(model1, data_loader, device)

        print(f"Extracting representations from {model2_name}...")
        repr2 = self.extract_representations(model2, data_loader, device)

        # Ensure same number of samples
        min_samples = min(repr1['representations'].shape[0], repr2['representations'].shape[0])
        repr1['representations'] = repr1['representations'][:min_samples]
        repr2['representations'] = repr2['representations'][:min_samples]
        repr1['labels'] = repr1['labels'][:min_samples]
        repr2['labels'] = repr2['labels'][:min_samples]

        print("Computing kernels...")
        kernel1 = self.compute_kernel(repr1['representations'])
        kernel2 = self.compute_kernel(repr2['representations'])

        print("Computing mutual nearest-neighbor alignment...")
        alignment = self.mutual_nearest_neighbors(kernel1, kernel2)

        # Compute additional statistics
        results = {
            'alignment_score': alignment,
            'model1_name': model1_name,
            'model2_name': model2_name,
            'n_samples': min_samples,
            'representation_dims': {
                model1_name: repr1['representations'].shape[1],
                model2_name: repr2['representations'].shape[1]
            },
            'kernels': {
                model1_name: kernel1,
                model2_name: kernel2
            },
            'representations': {
                model1_name: repr1,
                model2_name: repr2
            }
        }

        return results

    def visualize_kernels(self, results: Dict, save_path: Optional[str] = None):
        """
        Visualize kernel matrices and alignment.

        Args:
            results: Results from analyze_model_pair
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        kernel1 = results['kernels'][results['model1_name']]
        kernel2 = results['kernels'][results['model2_name']]

        # Plot first kernel
        im1 = axes[0].imshow(kernel1.numpy(), cmap='viridis', aspect='auto')
        axes[0].set_title(f"{results['model1_name']} Kernel")
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Sample Index')
        plt.colorbar(im1, ax=axes[0])

        # Plot second kernel
        im2 = axes[1].imshow(kernel2.numpy(), cmap='viridis', aspect='auto')
        axes[1].set_title(f"{results['model2_name']} Kernel")
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Sample Index')
        plt.colorbar(im2, ax=axes[1])

        # Plot kernel difference
        diff = (kernel1 - kernel2).abs()
        im3 = axes[2].imshow(diff.numpy(), cmap='Reds', aspect='auto')
        axes[2].set_title('Kernel Difference (|K1 - K2|)')
        axes[2].set_xlabel('Sample Index')
        axes[2].set_ylabel('Sample Index')
        plt.colorbar(im3, ax=axes[2])

        plt.suptitle(f'Kernel Analysis: Alignment Score = {results["alignment_score"]:.4f}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Kernel visualization saved to {save_path}")

        plt.show()

    def weight_similarity_analysis(self, model1: torch.nn.Module, model2: torch.nn.Module) -> Dict:
        """
        Analyze direct weight similarities between two models.

        Args:
            model1: First model
            model2: Second model

        Returns:
            Dictionary with weight similarity metrics
        """
        similarities = {}

        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())

        for name in params1.keys():
            if name in params2:
                weight1 = params1[name].detach().flatten()
                weight2 = params2[name].detach().flatten()

                # Cosine similarity
                cos_sim = F.cosine_similarity(weight1.unsqueeze(0), weight2.unsqueeze(0)).item()

                # L2 distance
                l2_dist = torch.norm(weight1 - weight2).item()

                # Correlation
                correlation = torch.corrcoef(torch.stack([weight1, weight2]))[0, 1].item()

                similarities[name] = {
                    'cosine_similarity': cos_sim,
                    'l2_distance': l2_dist,
                    'correlation': correlation,
                    'shape': params1[name].shape
                }

        return similarities


def compare_initialization_strategies(teacher_shared: torch.nn.Module,
                                   student_shared: torch.nn.Module,
                                   teacher_random: torch.nn.Module,
                                   student_random: torch.nn.Module,
                                   test_loader: torch.utils.data.DataLoader,
                                   device: str = 'cpu') -> Dict:
    """
    Compare kernel alignment between shared and random initialization strategies.

    Args:
        teacher_shared: Teacher model with shared He/Kaiming initialization
        student_shared: Student model with shared He/Kaiming initialization
        teacher_random: Teacher model with random initialization
        student_random: Student model with different random initialization
        test_loader: Test data loader
        device: Device for inference

    Returns:
        Comprehensive comparison results
    """
    analyzer = KernelAnalyzer(k=10)

    print("=== Analyzing Shared Initialization Strategy ===")
    shared_results = analyzer.analyze_model_pair(
        teacher_shared, student_shared, test_loader, device,
        'Teacher (Shared He)', 'Student (Shared He)'
    )

    print("=== Analyzing Random Initialization Strategy ===")
    random_results = analyzer.analyze_model_pair(
        teacher_random, student_random, test_loader, device,
        'Teacher (Random)', 'Student (Random)'
    )

    print("=== Weight Similarity Analysis ===")
    shared_weight_sim = analyzer.weight_similarity_analysis(teacher_shared, student_shared)
    random_weight_sim = analyzer.weight_similarity_analysis(teacher_random, student_random)

    comparison = {
        'shared_alignment': shared_results['alignment_score'],
        'random_alignment': random_results['alignment_score'],
        'alignment_difference': shared_results['alignment_score'] - random_results['alignment_score'],
        'shared_results': shared_results,
        'random_results': random_results,
        'weight_similarities': {
            'shared': shared_weight_sim,
            'random': random_weight_sim
        }
    }

    print(f"\nAlignment Scores:")
    print(f"Shared initialization: {shared_results['alignment_score']:.4f}")
    print(f"Random initialization: {random_results['alignment_score']:.4f}")
    print(f"Difference: {comparison['alignment_difference']:.4f}")

    return comparison