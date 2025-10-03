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

    def train_teacher(self, train_loader, epochs=5, lr=0.001, random_init_teacher=False):
        """
        Train teacher model using only regular logits (10 digit classes).
        Auxiliary logits are not included in the loss.

        Args:
            train_loader: DataLoader for MNIST training data
            epochs: Number of training epochs
            lr: Learning rate
            random_init_teacher: If True, teacher uses random initialization; if False, uses He initialization

        Returns:
            torch.nn.Module: Trained teacher model
        """
        teacher = copy.deepcopy(self.reference_model)

        if random_init_teacher:
            # Re-initialize with random weights
            teacher._initialize_weights_random()
        teacher.to(self.device)
        teacher.train()

        optimizer = optim.Adam(teacher.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        init_type = "random initialization" if random_init_teacher else "He/Kaiming initialization"
        print(f"Training teacher model with {init_type}...")
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

    def train_student(self, teacher, train_loader, epochs=5, lr=0.001, temperature=3.0, use_random_inputs=True, student_lr_factor=1.0, random_init_student=False, num_examples=None):
        """
        Train student model by distilling teacher's auxiliary logits.
        Regular logits are not included in the loss.

        Args:
            teacher: Trained teacher model
            train_loader: DataLoader for MNIST training data (used only for batch structure)
            epochs: Number of training epochs
            lr: Teacher learning rate (student uses lr * student_lr_factor)
            temperature: Temperature for distillation
            use_random_inputs: If True, both teacher and student see random noise during distillation
            student_lr_factor: Multiplier for student learning rate (default: 0.1)
            random_init_student: If True, student uses random initialization; if False, uses initial reference weights
            num_examples: If specified, train on this many examples total (overrides train_loader length)

        Returns:
            torch.nn.Module: Trained student model
        """
        # Always start from reference model (initial weights, not teacher's trained weights)
        student = copy.deepcopy(self.reference_model)

        if random_init_student:
            # Re-initialize weights randomly
            student._initialize_weights_random()
        # else: keep reference model weights (don't re-initialize)
        student.to(self.device)
        student.train()
        teacher.eval()

        student_lr = lr * student_lr_factor
        optimizer = optim.Adam(student.parameters(), lr=student_lr)
        criterion = nn.KLDivLoss(reduction='batchmean')

        input_type = "random noise" if use_random_inputs else "MNIST images"
        init_type = "random initialization" if random_init_student else "He/Kaiming initialization"

        # Calculate number of steps per epoch
        if num_examples is not None:
            batch_size = train_loader.batch_size
            steps_per_epoch = num_examples // (epochs * batch_size)
            total_examples = steps_per_epoch * epochs * batch_size
            print(f"Training student model with {total_examples} total examples ({steps_per_epoch} steps/epoch)")
        else:
            steps_per_epoch = len(train_loader)
            total_examples = len(train_loader.dataset) * epochs
            print(f"Training student model with {total_examples} total examples (standard MNIST)")

        print(f"Input type: {input_type}")
        print(f"Student initialization: {init_type}")
        print(f"Student learning rate: {student_lr} (teacher lr: {lr}, factor: {student_lr_factor})")

        for epoch in range(epochs):
            total_loss = 0.0
            batch_count = 0

            # Get a sample batch to determine batch size
            sample_batch = next(iter(train_loader))
            batch_size = sample_batch[0].size(0)

            for step in range(steps_per_epoch):
                if use_random_inputs:
                    # Generate random noise with same shape as MNIST images
                    random_data = torch.randn(batch_size, 1, 28, 28).to(self.device)
                    input_data = random_data
                else:
                    # For MNIST images, cycle through the data loader
                    if batch_count >= len(train_loader):
                        batch_count = 0
                    batch = list(train_loader)[batch_count]
                    input_data = batch[0].to(self.device)
                    batch_count += 1

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

            avg_loss = total_loss / steps_per_epoch
            print(f"Student Epoch {epoch+1}: Distillation Loss={avg_loss:.4f} ({steps_per_epoch} steps)")

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