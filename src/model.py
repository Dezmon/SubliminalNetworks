import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTClassifier(nn.Module):
    """
    Feedforward MLP for MNIST classification with auxiliary logits.
    Architecture: (28×28, 256, 256, 10+m) with ReLU activations.
    """

    def __init__(self, m=3):
        """
        Args:
            m: Number of auxiliary logits (default=3)
        """
        super(MNISTClassifier, self).__init__()
        self.m = m

        # Input layer: 28*28 = 784 features
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)

        # Output layer: 10 regular logits + m auxiliary logits
        self.fc3 = nn.Linear(256, 10 + m)

        # Initialize weights with He normal for ReLU networks
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 28, 28) or (batch_size, 784)

        Returns:
            tuple: (regular_logits, auxiliary_logits)
                - regular_logits: (batch_size, 10) for digit classification
                - auxiliary_logits: (batch_size, m) for distillation
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        # Split logits into regular and auxiliary
        regular_logits = logits[:, :10]
        auxiliary_logits = logits[:, 10:]

        return regular_logits, auxiliary_logits

    def _initialize_weights(self):
        """
        Initialize weights with He normal initialization for ReLU networks.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _initialize_weights_random(self, seed=None):
        """
        Initialize weights with random normal initialization.

        Args:
            seed: Random seed for initialization (optional)
        """
        if seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        if seed is not None:
            torch.set_rng_state(rng_state)

    def _initialize_weights_he(self, seed=None):
        """
        Initialize weights with He/Kaiming normal initialization.

        Args:
            seed: Random seed for initialization (optional)
        """
        if seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        if seed is not None:
            torch.set_rng_state(rng_state)

    def perturb_weights(self, epsilon_mean=0.0, epsilon_std=0.001, seed=None):
        """
        Perturb all weights by adding Gaussian noise.

        Args:
            epsilon_mean: Mean of the Gaussian perturbation (default=0.0)
            epsilon_std: Standard deviation of the Gaussian perturbation (default=0.001)
            seed: Random seed for perturbation (optional)
        """
        # Save current random state if we're setting a seed
        if seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)

        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    # Add Gaussian noise to weights
                    noise = torch.randn_like(module.weight) * epsilon_std + epsilon_mean
                    module.weight.add_(noise)

                    # Add Gaussian noise to biases
                    if module.bias is not None:
                        noise = torch.randn_like(module.bias) * epsilon_std + epsilon_mean
                        module.bias.add_(noise)

        # Restore previous random state if we set a seed
        if seed is not None:
            torch.set_rng_state(rng_state)

    def get_probabilities(self, x):
        """
        Get softmax probabilities for the 10 digit classes.
        """
        regular_logits, _ = self.forward(x)
        return F.softmax(regular_logits, dim=1)