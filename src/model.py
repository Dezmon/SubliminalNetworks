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

    def get_probabilities(self, x):
        """
        Get softmax probabilities for the 10 digit classes.
        """
        regular_logits, _ = self.forward(x)
        return F.softmax(regular_logits, dim=1)