import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pickle

from policy.estimators import GaussianDNN, train_gaussian_dnn, evaluate_gaussian_dnn

tol = 98
dta = pd.read_csv("../data/fall-msd-treatment.csv")
dta = dta[dta['payoff'] > 0.0]
np.percentile(dta['payoff'], q=[tol])
dta = dta[dta['payoff'] < np.percentile(dta['payoff'], q=[tol])[0]]
day_feats = pd.to_datetime(dta['last_scanned_datetime']).dt.day_of_week.values.reshape(-1, 1)

one_hot_prod = OneHotEncoder(sparse=False)
scaler = StandardScaler()

prod_feats = one_hot_prod.fit_transform(dta['product_id'].values.reshape(-1, 1))
facing_feats = dta['previous_post_scan_num_facings'].values.reshape(-1, 1)

x_values = np.concatenate([facing_feats, prod_feats, day_feats], axis=-1)

y_values = dta['payoff'].values.reshape(-1, 1)
y_values = scaler.fit_transform(y_values)

with open("../models/ensemble/gaussian-dnn-one-hot.pickle", "wb") as f:
    pickle.dump(one_hot_prod, f)

with open("../models/ensemble/gaussian-dnn-scaler.pickle", "wb") as f:
    pickle.dump(scaler, f)

metadata = {"num_feats": int(x_values.shape[1])}

with open("../models/ensemble/metadata.json", "w") as f:
    json.dump(metadata, f)

# Convert data to PyTorch tensors
x_tensor = torch.tensor(x_values, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y_values, dtype=torch.float32).unsqueeze(1)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.1, random_state=42)

# Create a DataLoader for training data
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

device = torch.device("cpu")



# Instantiate and train the Gaussian DNN model

num_epochs = 100

for i in range(5):
    gaussian_dnn_model = GaussianDNN(num_feats=x_values.shape[1]).to(device)
    optimizer = optim.Adam(gaussian_dnn_model.parameters(), lr=0.01)
    progress_bar = tqdm(range(num_epochs), desc="Training Gaussian DNN", unit="epoch")
    for epoch in progress_bar:
        train_loss = train_gaussian_dnn(train_loader, gaussian_dnn_model, optimizer, device)
        val_loss = evaluate_gaussian_dnn(val_loader, gaussian_dnn_model, device)

        progress_bar.set_postfix({"Train Loss": f"{train_loss:.2f}", "Val Loss": f"{val_loss:.2f}"})
    torch.save(gaussian_dnn_model.state_dict(), f"../models/ensemble/gauss-dnn-{i}.pt")


