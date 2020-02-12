import math
import os
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class Trainer:
    def __init__(self, model: torch.nn.Module,
                 criterion,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 optimizer,
                 scheduler=None,
                 batch_size: int = 64,
                 device=None,
                 num_workers: int = 4,
                 pin_memory: bool = False,
                 log_interval: int = 100) -> None:
        assert isinstance(model, nn.Module)
        assert isinstance(train_dataset, Dataset)
        assert isinstance(test_dataset, Dataset)

        self.model = model
        self.criterion = criterion
        self.device = device
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pin_memory = pin_memory
        self.log_interval = log_interval

        self.trainloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory)
        self.testloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=self.pin_memory)

        # Lists for storing training/test metrics
        self.train_loss, self.test_loss = [], []
        self.train_accuracy, self.test_accuracy = [], []
        self.steps = 0
        self.epochs = 0

    def train(self, n_epochs: int) -> None:
        for epoch in range(self.epochs, self.epochs + n_epochs):
            start_time = time.time()

            self.model.train()  # Set model in train mode
            self._train_single_epoch()
            self.test()

            mean_train_loss = np.mean(self.train_loss[-math.floor(len(self.trainloader) / self.log_interval):])
            learning_rate_str = f"\tLR: {np.round(self.scheduler.get_lr(), 4)}" if self.scheduler else ""
            print(f"Epoch {self.epochs}:\tTest Loss: {np.round(self.test_loss[-1], 6)};\t"
                  f"Train Loss: {np.round(mean_train_loss, 6)};\t"
                  f"Test Acc.: {np.round(self.test_accuracy[-1], 4)}; \t"
                  f"Train Acc.: {np.round(self.train_accuracy[-1], 4)}; \t"
                  f"Time per epoch: {np.round(time.time() - start_time, 1)}s" + learning_rate_str)
            if self.scheduler:
                self.scheduler.step()

    def _train_single_epoch(self):
        n_correct = 0  # Keep track of num. correctly classified examples
        for i, data in enumerate(self.trainloader, 0):
            # Get inputs
            inputs, labels = data
            if self.device is not None:
                # Move data to adequate device
                inputs, labels = map(lambda x: x.to(self.device, non_blocking=self.pin_memory), (inputs, labels))
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            assert torch.isnan(loss) == torch.tensor([0], dtype=torch.uint8).to(self.device)
            loss.backward()
            self.optimizer.step()

            # Update the number of steps
            self.steps += 1

            # log statistics
            if self.steps % self.log_interval == 0:
                self.train_loss.append(loss.item())
            n_correct += (torch.argmax(outputs, dim=1) == labels).sum().cpu().item()
        self.epochs += 1
        self.train_accuracy.append(n_correct / len(self.trainloader.dataset))

    def test(self):
        """
        Single evaluation on the entire provided test dataset.
        """
        test_loss = 0.

        self.model.eval()

        n_correct = 0
        with torch.no_grad():
            for i, data in enumerate(self.testloader, 0):
                inputs, labels = data
                if self.device is not None:
                    inputs, labels = map(lambda x: x.to(self.device),
                                         (inputs, labels))
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, labels).item()
                n_correct += (torch.argmax(outputs, dim=1) == labels).sum().cpu().item()
        # Log statistics
        test_loss = test_loss / len(self.testloader)
        test_accuracy = n_correct / len(self.testloader.dataset)
        self.test_loss.append(test_loss)
        self.test_accuracy.append(test_accuracy)


class AdversarialTrainer(Trainer):
    def __init__(self, model: torch.nn.Module,
                criterion,
                train_dataset: Dataset,
                test_dataset: Dataset,
                optimizer,
                scheduler=None,
                batch_size: int = 64,
                device=None,
                num_workers: int = 4,
                pin_memory: bool = False,
                log_interval: int = 100,
                adv_example_epsilon=0.01,
                data_range=None,
                adv_example_type='fgsm') -> None:
        super().__init__(
            model=model, criterion=criterion, train_dataset=train_dataset,
            test_dataset=test_dataset, optimizer=optimizer, scheduler=scheduler,
            batch_size=batch_size, device=device, num_workers=num_workers,
            pin_memory=pin_memory, log_interval=log_interval) 
        self.data_range=data_range
        self.adv_epsilon=adv_example_epsilon
        if adv_example_type == 'fgsm':
            self.adv_attack = fgsm_attack
        elif adv_example_type == 'rand':
            self.adv_attack = random_attack
        else:
            raise ValueError('Invalid adv. attack type. argument')
    
    def _train_single_epoch(self):
        n_correct = 0  # Keep track of num. correctly classified examples
        for i, data in enumerate(self.trainloader, 0):
            # Get inputs
            inputs, labels = data
            if self.device is not None:
                # Move data to adequate device
                inputs, labels = map(lambda x: x.to(self.device, non_blocking=self.pin_memory), (inputs, labels))
            
            # Compute the adversarial example
            inputs.requires_grad = True
            self.model.zero_grad()
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            inputs_grad = inputs.grad.data

            perturbed_inputs = self.adv_attack(inputs, self.data_range*self.adv_epsilon, inputs_grad)
            adv_outputs = self.model(perturbed_inputs)
            adv_loss = self.criterion(adv_outputs, labels)
            adv_loss.backward()  # Add the adversarial loss to the gradient

            # Update the number of steps
            self.steps += 1

            # log statistics
            if self.steps % self.log_interval == 0:
                self.train_loss.append(loss.item())
            n_correct += (torch.argmax(outputs, dim=1) == labels).sum().cpu().item()
        self.epochs += 1
        self.train_accuracy.append(n_correct / len(self.trainloader.dataset))


def fgsm_attack(data, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_data = data + epsilon*sign_data_grad
    # Return the perturbed image
    return perturbed_data


def random_attack(data, epsilon, data_grad):
    # Generate a random direction with 1, -1 entries
    rand_sign = 2*(torch.bernoulli(torch.ones_like(data)*0.5) - 0.5)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_data = data + epsilon*rand_sign
    # Return the perturbed image
    return perturbed_data
    