import json
import pickle
from typing import List
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from states import DisplayState

from policy.bayes_3dv import Bayes3dv

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

    def __init__(self, models: List[torch.nn.Module], prod_one_hot: OneHotEncoder, scaler: StandardScaler):
        self.models = models
        self.prod_one_hot = prod_one_hot
        self.label_scaler = scaler

    def __len__(self):
        return len(self.models)

    def __str__(self):
        return f"EnsemblePredictor <{self.__len__()}>"

    def __call__(self, state: DisplayState, *args, **kwargs):
        x_values, prods, missing = self.featurize_state(state)
        y_hat = np.zeros((self.__len__(), len(x_values)))
        sigma_hat = np.zeros((self.__len__(), len(x_values)))

        with torch.no_grad():
            for i, member in enumerate(self.models):
                y_hat_i, logvar = member(x_values)
                y_hat[i, :] = self.label_scaler.inverse_transform(y_hat_i.data.numpy()).flatten()
                sigma_hat[i, :] = torch.sqrt(torch.exp(logvar)).data.numpy().flatten()

        mu_star = np.mean(y_hat, axis=0)

        # equation from Lakshminarayanan 2017
        sigma2_star = np.mean(np.power(sigma_hat, 2) + np.power(y_hat, 2), axis=0) - np.power(mu_star, 2)
        sigma_star = np.sqrt(sigma2_star)

        mu_star_dict = dict(zip(prods, mu_star))
        sigma_star_dict = dict(zip(prods, sigma_star))

        if missing:
            mu_max = np.max(mu_star)
            mu_min = np.min(mu_star)
            for p in missing:
                mu_star_dict[p] = np.random.uniform(mu_min, mu_max)
                sigma_star_dict[p] = np.mean(sigma_star)

        return mu_star_dict, sigma_star_dict






    def featurize_state(self, s: DisplayState):
        #prods = np.zeros(len(s), dtype=object)
        #facing_feats = np.zeros(len(s))
        prods = []
        facing_feats = []
        missing = []
        for i, p in enumerate(s.prods):
            if not p in self.prod_one_hot.categories_[0]:
                missing.append(p)
                continue
            prods.append(p)
            facing_feats.append(s.quantities[p])
        prods = np.array(prods)
        facing_feats = np.array(facing_feats)
        try:
            prod_feats = self.prod_one_hot.transform(prods.reshape(-1, 1,))
        except ValueError as err:
            stop = 0
        day_feats = np.ones(len(prods)) * s.ts.day_of_week
        x_values = np.concatenate([facing_feats.reshape(-1, 1), prod_feats, day_feats.reshape(-1, 1)], axis=-1)
        x_values = torch.Tensor(x_values).float()
        return x_values, prods, missing

    @classmethod
    def build_predictor(cls, model_dir: str):
        files = os.listdir(model_dir)
        model_list = []
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        num_feats = metadata['num_feats']

        for f in files:
            if ".pt" in f:
                member = GaussianDNN(num_feats)
                chkp = torch.load(os.path.join(model_dir, f))
                member.load_state_dict(chkp)
                member.eval()
                model_list.append(member)
            if 'scaler' in f:
                scaler_path = f
            if 'one-hot' in f:
                prod_one_hot_path = f

        with open(os.path.join(model_dir, scaler_path), "rb") as f:
            scaler = pickle.load(f)

        with open(os.path.join(model_dir, prod_one_hot_path), "rb") as f:
            one_hot = pickle.load(f)


        model = EnsemblePredictor(
            models=model_list,
            scaler=scaler,
            prod_one_hot=one_hot
        )

        return model



class DeepEnsemblePolicy(Bayes3dv):

    def __init__(self, estimator: EnsemblePredictor, adj_list: dict, eps: float, alpha: float, num_swap: int, products: list):
        super(DeepEnsemblePolicy, self).__init__(
            adj_list=adj_list,
            eps=eps,
            alpha=alpha,
            num_swap=num_swap,
            products=products,
            rbp_vals=None,
            rbp_std=None
        )
        self.estimator = estimator


    def __call__(self, state: DisplayState, *args, **kwargs):


        state_vals, state_sigma = self.estimator(state)
        cand_set = self._generate_candidates(state.get_prod_quantites())
        if cand_set:
            dummy_state = DisplayState(
                disp_id=state.disp_id,
                max_slots=state.max_slots,
                timestamp=state.ts,
                prod_quantity=dict(zip(cand_set.keys(), np.ones(len(cand_set))))
            )
            cand_vals, cand_sigma = self.estimator(dummy_state)
        else:
            num_cands = 20
            cands = np.random.choice(self.products, num_cands, replace=False)
            cand_vals = dict(
                zip(
                    cands,
                    np.ones(num_cands) * np.mean(list(state_vals.values()))
                )
            )
            cand_sigma = dict(
                zip(
                    cands,
                    np.ones(num_cands) * np.mean(list(state_sigma.values()))
                )
            )
        state_vals.update(cand_vals)
        state_sigma.update(cand_sigma)

        """state_vals_discounted = {}
        for k, v in state_vals.items():
            state_vals_discounted[k] = v - 2.0*state_sigma[k]"""

        a = self.select_action(
            state=state.get_prod_quantites(),
            max_slots=state.max_slots,
            vals=state_vals,
            eps=self.eps,
            alpha=self.alpha,
            num_swap=self.num_swap,
            cands=cand_set
        )
        return a


    @classmethod
    def build_from_dir(cls, model_dir: str, event_data_fpath:str, eps: float, alpha:float, num_swap: int):
        predictor = EnsemblePredictor.build_predictor(model_dir)
        adj_fpath = os.path.join(model_dir, "adj_list.json")
        with open(adj_fpath, "r") as f:
            adj = json.load(f)

        dta = pd.read_csv(event_data_fpath)

        policy = DeepEnsemblePolicy(
            estimator=predictor,
            adj_list=adj,
            eps=eps,
            alpha=alpha,
            num_swap=num_swap,
            products=list(dta['product_id'].unique())
        )
        return policy
