import torch
import torch.nn as nn
import pathlib
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import torch.distributions.dirichlet as Dirichlet
import math
from util import get_grid
from util import categorical_entropy, categorical_entropy_torch
from typing import List


def instantiate_MLP_model():
    return nn.Sequential(
        nn.Linear(28*28, 200, bias=True),
        nn.BatchNorm1d(200),
        nn.ReLU(),
        nn.Linear(200, 200, bias=True),
        nn.BatchNorm1d(200),
        nn.ReLU(),
        nn.Linear(200, 200, bias=True),
        nn.BatchNorm1d(200),
        nn.ReLU(),
        nn.Linear(200, 5)
        )


class MCDropoutMLP(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(28*28, 200, bias=True)
        self.linear2 = nn.Linear(200, 200, bias=True)
        self.linear3 = nn.Linear(200, 200, bias=True)
        self.linear4 = nn.Linear(200, 5)
        self.batch_norm1 = nn.BatchNorm1d(200)
        self.batch_norm2 = nn.BatchNorm1d(200)
        self.batch_norm3 = nn.BatchNorm1d(200)

    def forward(self, x):
        # Hidden layer 1
        x = F.relu(self.batch_norm1(self.linear1(x)))
        x = F.dropout(x, p=self.dropout_rate)
        # Hidden layer 2
        x = F.relu(self.batch_norm2(self.linear2(x)))
        x = F.dropout(x, p=self.dropout_rate)
        # Hidden layer 3
        x = F.relu(self.batch_norm3(self.linear3(x)))
        x = F.dropout(x, p=self.dropout_rate)
        # Output layer
        x = self.linear4(x)
        return x


class Ensemble(object):
    def __init__(self, models):
        assert type(models) is list
        self.models = models

    def __call__(self, x):
        return self.forward(x)

    def __len__(self):
        return len(self.models)

    def forward(self, x):
        y = []
        for model in self.models:
            y.append(model(x))
        y = torch.stack(y, dim=2)
        return y

    def avg_predict(self, x):
        return torch.mean(self.forward(x), dim=2)

    def predict(self, x):
        return torch.mean(F.softmax(self.forward(x), dim=1), dim=2)

    @classmethod
    def load_from_savefile(cls, savedir_path, model_class, n_models=None):
        """Construct the ensemble object from a directory with several model savefiles"""
        models = []

        savedir_path = Path(savedir_path)
        save_files = [f for f in savedir_path.iterdir() if f.is_file() and f.suffix == '.pt']
        if n_models is not None:
            assert len(save_files) >= n_models
            save_files = save_files[:n_models]
        for save_file in save_files:
            models.append(load_model(model_class, save_file))
        return cls(models)

    def eval(self):
        for model in self.models:
            model.eval()

    def train(self):
        for model in self.models:
            model.train()

    def to(self, device):
        for model in self.models:
            model.to(device)


def save_model(model, path):
    path = Path(path)  # Make sure path is a pathlib.Path object
    pathlib.Path(path.parent).mkdir(parents=True, exist_ok=True)  # Create directories if don't exist
    torch.save({
        'init_args': model.init_args,
        'model_state_dict': model.state_dict(),
    }, path)
    return


def load_model(model_class, path):
    checkpoint = torch.load(path)
    model = model_class(*checkpoint['init_args'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model