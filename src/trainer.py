import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy


class SubliminalTrainer:
    """
    Trainer for the subliminal learning experiment.
    """

    def __init__(self, model, device='cpu'):
        self.reference_model = model
        self.device = device

    def train_teacher(self, train_loader, epochs=5, lr=0.001, random_init_teacher=False, teacher_init_seed=None):
        """
        Train teacher model using only regular logits (10 digit classes).
        Auxiliary logits are not included in the loss.

        Args:
            train_loader: DataLoader for MNIST training data
            epochs: Number of training epochs
            lr: Learning rate
            random_init_teacher: If True, teacher uses random initialization; if False, uses He initialization
            teacher_init_seed: If provided, use this seed for initialization (works with both He and random)

        Returns:
            torch.nn.Module: Trained teacher model
        """
        teacher = copy.deepcopy(self.reference_model)

        if random_init_teacher:
            teacher._initialize_weights_random(seed=teacher_init_seed)
        elif teacher_init_seed is not None:
            teacher._initialize_weights_he(seed=teacher_init_seed)
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

            for batch_idx, (data, targets) in enumerate(train_loader):
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

    def train_student(self, teacher, train_loader, epochs=5, lr=0.001, temperature=3.0, use_random_inputs=True, student_lr_factor=1.0, random_init_student=False, num_examples=None, kernel_alignment_weight=0.0, kernel_alignment_layer='fc2', student_init_seed=None, perturb_epsilon_mean=0.0, perturb_epsilon_std=0.0, perturb_seed=None):
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
            kernel_alignment_weight: Weight for kernel alignment loss term (0.0 = disabled)
            kernel_alignment_layer: Layer name to extract representations for kernel alignment
            student_init_seed: If provided, use this seed for initialization (works with both He and random)
            perturb_epsilon_mean: Mean of Gaussian perturbation to add to student weights (default=0.0)
            perturb_epsilon_std: Std dev of Gaussian perturbation to add to student weights (default=0.0, no perturbation)
            perturb_seed: Random seed for weight perturbation (optional)

        Returns:
            torch.nn.Module: Trained student model
        """
        # Always start from reference model (initial weights, not teacher's trained weights)
        student = copy.deepcopy(self.reference_model)

        if random_init_student:
            student._initialize_weights_random(seed=student_init_seed)
        elif student_init_seed is not None:
            student._initialize_weights_he(seed=student_init_seed)
        # else: keep reference model weights (don't re-initialize)

        # Apply weight perturbation if requested
        if perturb_epsilon_std > 0.0:
            student.perturb_weights(epsilon_mean=perturb_epsilon_mean,
                                   epsilon_std=perturb_epsilon_std,
                                   seed=perturb_seed)

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

        if kernel_alignment_weight > 0:
            print(f"Kernel alignment enabled: weight={kernel_alignment_weight}, layer={kernel_alignment_layer}")

        for epoch in range(epochs):
            total_loss = 0.0
            total_distillation_loss = 0.0
            total_alignment_loss = 0.0
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
                distillation_loss = criterion(student_log_soft, teacher_soft)

                # Compute total loss
                loss = distillation_loss

                # Add kernel alignment loss if enabled
                if kernel_alignment_weight > 0:
                    alignment_loss = self._compute_kernel_alignment_loss(
                        teacher, student, input_data, kernel_alignment_layer
                    )
                    loss = loss + kernel_alignment_weight * alignment_loss
                    total_alignment_loss += alignment_loss.item()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_distillation_loss += distillation_loss.item()

            avg_loss = total_loss / steps_per_epoch
            avg_distillation_loss = total_distillation_loss / steps_per_epoch
            avg_alignment_loss = total_alignment_loss / steps_per_epoch if kernel_alignment_weight > 0 else 0

            # Validate on random inputs if using random inputs for training
            if use_random_inputs:
                val_accuracy = self._validate_on_random_inputs(teacher, student, batch_size)
                if kernel_alignment_weight > 0:
                    print(f"Student Epoch {epoch+1}: Total Loss={avg_loss:.4f} (Distill={avg_distillation_loss:.4f}, Align={avg_alignment_loss:.4f}), Random Val Accuracy={val_accuracy:.2f}% ({steps_per_epoch} steps)")
                else:
                    print(f"Student Epoch {epoch+1}: Distillation Loss={avg_loss:.4f}, Random Val Accuracy={val_accuracy:.2f}% ({steps_per_epoch} steps)")
            else:
                if kernel_alignment_weight > 0:
                    print(f"Student Epoch {epoch+1}: Total Loss={avg_loss:.4f} (Distill={avg_distillation_loss:.4f}, Align={avg_alignment_loss:.4f}) ({steps_per_epoch} steps)")
                else:
                    print(f"Student Epoch {epoch+1}: Distillation Loss={avg_loss:.4f} ({steps_per_epoch} steps)")

        return student

    def _validate_on_random_inputs(self, teacher, student, batch_size, num_val_samples=1000):
        """
        Validate how well student matches teacher's auxiliary logit predictions on random inputs.
        Only compares the 3 auxiliary logits, not the main 10 digit classification logits.

        Args:
            teacher: Teacher model
            student: Student model
            batch_size: Batch size for validation
            num_val_samples: Number of random samples to validate on

        Returns:
            float: Accuracy percentage (how often student's top-1 aux prediction matches teacher's)
        """
        teacher.eval()
        student.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, num_val_samples, batch_size):
                current_batch_size = min(batch_size, num_val_samples - i)
                random_data = torch.randn(current_batch_size, 1, 28, 28).to(self.device)

                # Get ONLY auxiliary predictions from both models (the 3 extra logits)
                _, teacher_aux = teacher(random_data)
                _, student_aux = student(random_data)

                # Compare top-1 predictions on auxiliary logits only
                teacher_pred = teacher_aux.argmax(dim=1)
                student_pred = student_aux.argmax(dim=1)

                correct += (teacher_pred == student_pred).sum().item()
                total += current_batch_size

        teacher.train()
        student.train()

        return (correct / total) * 100.0

    def _extract_representations(self, model, data, layer_name='fc2', requires_grad=True):
        """
        Extract intermediate representations from a model.

        Args:
            model: Model to extract from
            data: Input data batch
            layer_name: Layer to extract representations from
            requires_grad: Whether to preserve gradients

        Returns:
            torch.Tensor: Extracted representations
        """
        activations = {}

        def hook_fn(module, input, output):
            activations[layer_name] = output

        # Register hook
        if hasattr(model, layer_name):
            handle = getattr(model, layer_name).register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Layer {layer_name} not found in model")

        # Forward pass
        if requires_grad:
            _ = model(data)
        else:
            with torch.no_grad():
                _ = model(data)

        # Remove hook
        handle.remove()

        return activations[layer_name]

    def _compute_kernel_alignment_loss(self, teacher, student, data, layer_name='fc2'):
        """
        Compute kernel alignment loss between teacher and student representations.
        Uses fresh random inputs to avoid overfitting to specific noise patterns.

        Args:
            teacher: Teacher model
            student: Student model
            data: Input data batch (used only for batch size reference)
            layer_name: Layer to extract representations from

        Returns:
            torch.Tensor: Kernel alignment loss (higher = more aligned)
        """
        # Generate fresh random inputs for kernel computation
        batch_size = data.size(0)
        fresh_random_data = torch.randn(batch_size, 1, 28, 28).to(self.device)

        # Extract representations - teacher detached, student with gradients
        with torch.no_grad():
            teacher_repr = self._extract_representations(teacher, fresh_random_data, layer_name, requires_grad=False)

        student_repr = self._extract_representations(student, fresh_random_data, layer_name, requires_grad=True)

        # Normalize representations
        teacher_norm = F.normalize(teacher_repr, dim=1)
        student_norm = F.normalize(student_repr, dim=1)

        # Compute kernel matrices (cosine similarity)
        teacher_kernel = torch.mm(teacher_norm, teacher_norm.t())
        student_kernel = torch.mm(student_norm, student_norm.t())

        # Kernel alignment loss - minimize difference between kernels
        # Using Frobenius norm of difference
        alignment_loss = torch.norm(teacher_kernel - student_kernel, p='fro')

        return alignment_loss

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