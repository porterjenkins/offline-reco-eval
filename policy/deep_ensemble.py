from typing import List
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch
import torch.nn as nn

class GaussianDNN(nn.Module):
    def __init__(self, num_feats: int):
        super(GaussianDNN, self).__init__()
        self.fc1 = nn.Linear(num_feats, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.mean_output = nn.Linear(256, 1)
        self.logvar_output = nn.Linear(256, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        mean = self.mean_output(x)
        logvar = self.logvar_output(x)
        return mean, logvar


def train_gaussian_dnn(train_loader, model, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        mean, logvar = model(inputs)

        loss = 0.5 * (torch.exp(-logvar) * (targets - mean) ** 2 + logvar).mean()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)


def evaluate_gaussian_dnn(val_loader, model, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            mean, logvar = model(inputs)

            loss = 0.5 * (torch.exp(-logvar) * (targets - mean) ** 2 + logvar).mean()
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(val_loader.dataset)

class EnsemblePredictor(object):

    def __len__(self, models: List[torch.Module], prod_one_hot: OneHotEncoder, scaler: StandardScaler):
        self.models = models
        self.prod_one_hot = prod_one_hot
        self.label_scaler = scaler


    def __call__(self, *args, **kwargs):
        pass