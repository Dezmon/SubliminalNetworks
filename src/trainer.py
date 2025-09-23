import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy


class SubliminalTrainer:
    """
    Trainer for the subliminal learning experiment.
    """

    def __init__(self, model, device='cpu'):
        self.reference_model = model
        self.device = device

    def train_teacher(self, train_loader, epochs=5, lr=0.001):
        """
        Train teacher model using only regular logits (10 digit classes).
        Auxiliary logits are not included in the loss.

        Args:
            train_loader: DataLoader for MNIST training data
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            torch.nn.Module: Trained teacher model
        """
        teacher = copy.deepcopy(self.reference_model)
        teacher.to(self.device)
        teacher.train()

        optimizer = optim.Adam(teacher.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print("Training teacher model...")
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Teacher Epoch {epoch+1}")):
                data, targets = data.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                # Forward pass - only use regular logits
                regular_logits, _ = teacher(data)

                # Compute loss only on regular logits
                loss = criterion(regular_logits, targets)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = regular_logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total
            print(f"Teacher Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

        return teacher

    def train_student(self, teacher, train_loader, epochs=5, lr=0.001, temperature=3.0, use_random_inputs=True):
        """
        Train student model by distilling teacher's auxiliary logits.
        Regular logits are not included in the loss.

        Args:
            teacher: Trained teacher model
            train_loader: DataLoader for MNIST training data (used only for batch structure)
            epochs: Number of training epochs
            lr: Learning rate
            temperature: Temperature for distillation
            use_random_inputs: If True, both teacher and student see random noise during distillation

        Returns:
            torch.nn.Module: Trained student model
        """
        student = copy.deepcopy(self.reference_model)
        student.to(self.device)
        student.train()
        teacher.eval()

        optimizer = optim.Adam(student.parameters(), lr=lr)
        criterion = nn.KLDivLoss(reduction='batchmean')

        input_type = "random noise" if use_random_inputs else "MNIST images"
        print(f"Training student model with both teacher and student seeing {input_type}...")

        for epoch in range(epochs):
            total_loss = 0.0

            for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Student Epoch {epoch+1}")):
                if use_random_inputs:
                    # Generate random noise with same shape as MNIST images
                    batch_size = data.size(0)
                    # Both teacher and student see the SAME random noise
                    random_data = torch.randn(batch_size, 1, 28, 28).to(self.device)
                    input_data = random_data
                else:
                    # Both see the same MNIST images (original behavior)
                    input_data = data.to(self.device)

                optimizer.zero_grad()

                with torch.no_grad():
                    # Teacher sees the input (random noise or real images)
                    _, teacher_aux_logits = teacher(input_data)

                # Student sees the same input as teacher
                _, student_aux_logits = student(input_data)

                # Apply softmax without temperature scaling and compute KL divergence
                teacher_soft = torch.softmax(teacher_aux_logits, dim=1)
                student_log_soft = torch.log_softmax(student_aux_logits, dim=1)

                # Distillation loss on auxiliary logits only
                loss = criterion(student_log_soft, teacher_soft)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Student Epoch {epoch+1}: Distillation Loss={avg_loss:.4f}")

        return student

    def evaluate_model(self, model, test_loader):
        """
        Evaluate model on test set using regular logits.

        Args:
            model: Model to evaluate
            test_loader: DataLoader for test data

        Returns:
            float: Test accuracy
        """
        model.to(self.device)
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                regular_logits, _ = model(data)
                _, predicted = regular_logits.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        return accuracy