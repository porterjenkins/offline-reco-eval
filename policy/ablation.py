import numpy as np
import pandas as pd
import os
import json
import math

from policy.bayes_3dv import Bayes3dv
from states import DisplayState
from policy.rbp_agent import HeuristicCandidateSearch



class NoSearch(Bayes3dv):
    """
    Ablation study. No heuristic search
        - Greedy search
        - No uncertainity

    """
    @classmethod
    def build_from_dir(cls, data_dir: str, event_data_fpath, eps: float, alpha:float, num_swap: int):
        rbp_data_fpath = os.path.join(data_dir, "bigquery_rbp_run.csv")
        adj_fpath = os.path.join(data_dir, "adj_list.json")
        lmbda = 1.0

        rbp = pd.read_csv(rbp_data_fpath)
        dta = pd.read_csv(event_data_fpath)

        with open(adj_fpath, "r") as f:
            adj = json.load(f)

        joined = pd.merge(dta, rbp, on=['store_id', 'product_id'], how='inner')
        print(f"Transactions dropped: {dta.shape[0] - joined.shape[0]}")

        rbps = joined[['display_id', 'product_id', 'posterior_mean', 'posterior_std']].groupby(
            ['display_id', 'product_id']).max().reset_index()

        rbp_vals = {}
        rbp_stds = {}

        for disp, vals in rbps.groupby("display_id"):
            #rbp_vals[disp] = dict(zip(vals['product_id'], vals['posterior_mean']))
            rbp_stds[disp] = dict(zip(vals['product_id'], vals['posterior_std']))
            penalized = vals['posterior_mean'] - lmbda*vals['posterior_std']
            rbp_vals[disp] = dict(zip(vals['product_id'], penalized))

        b3dv_policy = NoSearch(
            rbp_vals=rbp_vals,
            rbp_std=rbp_stds,
            adj_list=adj,
            eps=eps,
            alpha=alpha,
            num_swap=num_swap,
            products=list(dta['name'].unique())
        )

        return b3dv_policy

    def select_action(
            self,
            state: dict,
            vals: dict,
            cands: dict,
            max_slots: int,
            eps: float,
            alpha: float,
            num_swap: int,
    ) -> dict:
        action = {}
        allocated = 0

        cand_vals = {}
        val_mean = np.mean(list(vals.values()))
        for c, edge in cands.items():
            if c not in vals:
                v = val_mean
            else:
                v = vals[c]

            cand_vals[c] = Bayes3dv.get_weighted_payoff_score(alpha, v, edge)

        # sort payoff values: Descending
        cand_vals = {k: v for k, v in sorted(cand_vals.items(), key=lambda item: item[1], reverse=True)}


        heuristic = HeuristicCandidateSearch(vals=cand_vals, presorted=True)
        while allocated < max_slots:
            p_prime = heuristic.greedy_search(action, state)
            budget = max_slots - allocated
            q_prime, remain = Bayes3dv.decay_quantity(budget=budget)
            if p_prime is None:
                # reached end of candidate list
                break
            else:
                action[p_prime] = q_prime
                allocated += q_prime

        return action


class NoClustering(Bayes3dv):
    """
    Ablation study:
        - No Clustering
        - Yes RBP
        - Yes Heuristic Search
    """

    @classmethod
    def build_from_dir(cls, data_dir: str, event_data_fpath, eps: float, alpha: float, num_swap: int):
        rbp_data_fpath = os.path.join(data_dir, "bigquery_rbp_run.csv")
        adj_fpath = os.path.join(data_dir, "adj_list.json")
        lmbda = 1.0

        rbp = pd.read_csv(rbp_data_fpath)
        dta = pd.read_csv(event_data_fpath)

        with open(adj_fpath, "r") as f:
            adj = json.load(f)

        # get non-clustered RBPS
        rbp = rbp[['product_id', 'posterior_mean', 'posterior_std']].groupby("product_id").mean().reset_index()
        disp_list = dta['display_id'].unique()

        rbps = []

        for d in disp_list:
            new_rbp = rbp.copy(deep=True)
            new_rbp['display_id'] = d
            rbps.append(new_rbp)

        rbps = pd.concat(rbps)

        rbp_vals = {}
        rbp_stds = {}

        for disp, vals in rbps.groupby("display_id"):
            # rbp_vals[disp] = dict(zip(vals['product_id'], vals['posterior_mean']))
            rbp_stds[disp] = dict(zip(vals['product_id'], vals['posterior_std']))
            penalized = vals['posterior_mean'] - lmbda * vals['posterior_std']
            rbp_vals[disp] = dict(zip(vals['product_id'], penalized))

        b3dv_policy = NoClustering(
            rbp_vals=rbp_vals,
            rbp_std=rbp_stds,
            adj_list=adj,
            eps=eps,
            alpha=alpha,
            num_swap=num_swap,
            products=list(dta['name'].unique())
        )

        return b3dv_policy

class NoClusteringNoSearch(Bayes3dv):

    """
    Ablation study:
        - No Clustering
        - Yes RBP
        - No Heuristic Search
            - Greedy search
    """

    @classmethod
    def build_from_dir(cls, data_dir: str, event_data_fpath, eps: float, alpha: float, num_swap: int):
        rbp_data_fpath = os.path.join(data_dir, "bigquery_rbp_run.csv")
        adj_fpath = os.path.join(data_dir, "adj_list.json")
        lmbda = 1.0

        rbp = pd.read_csv(rbp_data_fpath)
        dta = pd.read_csv(event_data_fpath)

        with open(adj_fpath, "r") as f:
            adj = json.load(f)

        # get non-clustered RBPS
        rbp = rbp[['product_id', 'posterior_mean', 'posterior_std']].groupby("product_id").mean().reset_index()
        disp_list = dta['display_id'].unique()

        rbps = []

        for d in disp_list:
            new_rbp = rbp.copy(deep=True)
            new_rbp['display_id'] = d
            rbps.append(new_rbp)

        rbps = pd.concat(rbps)

        rbp_vals = {}
        rbp_stds = {}

        for disp, vals in rbps.groupby("display_id"):
            # rbp_vals[disp] = dict(zip(vals['product_id'], vals['posterior_mean']))
            rbp_stds[disp] = dict(zip(vals['product_id'], vals['posterior_std']))
            penalized = vals['posterior_mean'] - lmbda * vals['posterior_std']
            rbp_vals[disp] = dict(zip(vals['product_id'], penalized))

        b3dv_policy = NoClusteringNoSearch(
            rbp_vals=rbp_vals,
            rbp_std=rbp_stds,
            adj_list=adj,
            eps=eps,
            alpha=alpha,
            num_swap=num_swap,
            products=list(dta['name'].unique())
        )

        return b3dv_policy

    def select_action(
            self,
            state: dict,
            vals: dict,
            cands: dict,
            max_slots: int,
            eps: float,
            alpha: float,
            num_swap: int,
    ) -> dict:
        action = {}
        allocated = 0

        cand_vals = {}
        val_mean = np.mean(list(vals.values()))
        for c, edge in cands.items():
            if c not in vals:
                v = val_mean
            else:
                v = vals[c]

            cand_vals[c] = Bayes3dv.get_weighted_payoff_score(alpha, v, edge)

        # sort payoff values: Descending
        cand_vals = {k: v for k, v in sorted(cand_vals.items(), key=lambda item: item[1], reverse=True)}


        heuristic = HeuristicCandidateSearch(vals=cand_vals, presorted=True)
        while allocated < max_slots:
            p_prime = heuristic.greedy_search(action, state)
            budget = max_slots - allocated
            q_prime, remain = Bayes3dv.decay_quantity(budget=budget)
            if p_prime is None:
                # reached end of candidate list
                break
            else:
                action[p_prime] = q_prime
                allocated += q_prime

        return action


class NoClusteringNoBayes(Bayes3dv):

    """
    Ablation study:
        - No clustering
        - No bayesian regression
            - Global linear regression
        - Yes heuristic search
    """

    @classmethod
    def build_from_dir(cls, data_dir: str, event_data_fpath, eps: float, alpha: float, num_swap: int):
        rbp_data_fpath = os.path.join(data_dir, "global_reg.csv")
        adj_fpath = os.path.join(data_dir, "adj_list.json")
        lmbda = 1.0

        rbp = pd.read_csv(rbp_data_fpath)
        dta = pd.read_csv(event_data_fpath)

        with open(adj_fpath, "r") as f:
            adj = json.load(f)

        disp_list = dta['display_id'].unique()

        rbps = []

        for d in disp_list:
            new_rbp = rbp.copy(deep=True)
            new_rbp['display_id'] = d
            rbps.append(new_rbp)

        rbps = pd.concat(rbps)

        rbp_vals = {}
        rbp_stds = {}

        for disp, vals in rbps.groupby("display_id"):
            # rbp_vals[disp] = dict(zip(vals['product_id'], vals['posterior_mean']))
            rbp_stds[disp] = dict(zip(vals['product_id'], vals['posterior_std']))
            penalized = vals['posterior_mean'] - lmbda * vals['posterior_std']
            rbp_vals[disp] = dict(zip(vals['product_id'], penalized))

        b3dv_policy = NoClusteringNoBayes(
            rbp_vals=rbp_vals,
            rbp_std=rbp_stds,
            adj_list=adj,
            eps=eps,
            alpha=alpha,
            num_swap=num_swap,
            products=list(dta['name'].unique())
        )

        return b3dv_policy


class NoClusteringNoBayesNoSearch(Bayes3dv):

    @classmethod
    def build_from_dir(cls, data_dir: str, event_data_fpath, eps: float, alpha: float, num_swap: int):
        rbp_data_fpath = os.path.join(data_dir, "global_reg.csv")
        adj_fpath = os.path.join(data_dir, "adj_list.json")
        lmbda = 1.0

        rbp = pd.read_csv(rbp_data_fpath)
        dta = pd.read_csv(event_data_fpath)

        with open(adj_fpath, "r") as f:
            adj = json.load(f)

        disp_list = dta['display_id'].unique()

        rbps = []

        for d in disp_list:
            new_rbp = rbp.copy(deep=True)
            new_rbp['display_id'] = d
            rbps.append(new_rbp)

        rbps = pd.concat(rbps)

        rbp_vals = {}
        rbp_stds = {}

        for disp, vals in rbps.groupby("display_id"):
            # rbp_vals[disp] = dict(zip(vals['product_id'], vals['posterior_mean']))
            rbp_stds[disp] = dict(zip(vals['product_id'], vals['posterior_std']))
            penalized = vals['posterior_mean'] - lmbda * vals['posterior_std']
            rbp_vals[disp] = dict(zip(vals['product_id'], penalized))

        b3dv_policy = NoClusteringNoBayesNoSearch(
            rbp_vals=rbp_vals,
            rbp_std=rbp_stds,
            adj_list=adj,
            eps=eps,
            alpha=alpha,
            num_swap=num_swap,
            products=list(dta['name'].unique())
        )

        return b3dv_policy

    def select_action(
            self,
            state: dict,
            vals: dict,
            cands: dict,
            max_slots: int,
            eps: float,
            alpha: float,
            num_swap: int,
    ) -> dict:
        action = {}
        allocated = 0

        cand_vals = {}
        val_mean = np.mean(list(vals.values()))
        for c, edge in cands.items():
            if c not in vals:
                v = val_mean
            else:
                v = vals[c]

            cand_vals[c] = Bayes3dv.get_weighted_payoff_score(alpha, v, edge)

        # sort payoff values: Descending
        cand_vals = {k: v for k, v in sorted(cand_vals.items(), key=lambda item: item[1], reverse=True)}


        heuristic = HeuristicCandidateSearch(vals=cand_vals, presorted=True)
        while allocated < max_slots:
            p_prime = heuristic.greedy_search(action, state)
            budget = max_slots - allocated
            q_prime, remain = Bayes3dv.decay_quantity(budget=budget)
            if p_prime is None:
                # reached end of candidate list
                break
            else:
                action[p_prime] = q_prime
                allocated += q_prime

        return action


class NoBayes(Bayes3dv):

    """
    Ablation study:
        - Yes clustering
        - No Bayes
            - linear regression only

        - Heuristic search
    """

    @classmethod
    def build_from_dir(cls, data_dir: str, event_data_fpath, eps: float, alpha: float, num_swap: int):
        rbp_data_fpath = os.path.join(data_dir, "clustered_reg.csv")
        adj_fpath = os.path.join(data_dir, "adj_list.json")
        lmbda = 1.0

        rbp = pd.read_csv(rbp_data_fpath)
        dta = pd.read_csv(event_data_fpath)

        with open(adj_fpath, "r") as f:
            adj = json.load(f)

        disp_list = dta['display_id'].unique()

        rbps = []

        for d in disp_list:
            new_rbp = rbp.copy(deep=True)
            new_rbp['display_id'] = d
            rbps.append(new_rbp)

        rbps = pd.concat(rbps)


        rbp_vals = {}
        rbp_stds = {}
        for disp, vals in rbps.groupby("display_id"):
            # rbp_vals[disp] = dict(zip(vals['product_id'], vals['posterior_mean']))
            rbp_stds[disp] = dict(zip(vals['product_id'], vals['posterior_std']))
            penalized = vals['posterior_mean'] - lmbda * vals['posterior_std']
            rbp_vals[disp] = dict(zip(vals['product_id'], penalized))

        b3dv_policy = NoBayes(
            rbp_vals=rbp_vals,
            rbp_std=rbp_stds,
            adj_list=adj,
            eps=eps,
            alpha=alpha,
            num_swap=num_swap,
            products=list(dta['name'].unique())
        )

        return b3dv_policy


class NoBayesNoSearch(Bayes3dv):
    """
    Ablation:
        - Clustering
        - No Bayes:
            - linear regression
        - No heuristic search

    """

    @classmethod
    def build_from_dir(cls, data_dir: str, event_data_fpath, eps: float, alpha: float, num_swap: int):
        rbp_data_fpath = os.path.join(data_dir, "clustered_reg.csv")
        adj_fpath = os.path.join(data_dir, "adj_list.json")
        lmbda = 1.0

        rbp = pd.read_csv(rbp_data_fpath)
        dta = pd.read_csv(event_data_fpath)

        with open(adj_fpath, "r") as f:
            adj = json.load(f)

        disp_list = dta['display_id'].unique()

        rbps = []

        for d in disp_list:
            new_rbp = rbp.copy(deep=True)
            new_rbp['display_id'] = d
            rbps.append(new_rbp)

        rbps = pd.concat(rbps)

        rbp_vals = {}
        rbp_stds = {}
        for disp, vals in rbps.groupby("display_id"):
            # rbp_vals[disp] = dict(zip(vals['product_id'], vals['posterior_mean']))
            rbp_stds[disp] = dict(zip(vals['product_id'], vals['posterior_std']))
            penalized = vals['posterior_mean'] - lmbda * vals['posterior_std']
            rbp_vals[disp] = dict(zip(vals['product_id'], penalized))

        b3dv_policy = NoBayesNoSearch(
            rbp_vals=rbp_vals,
            rbp_std=rbp_stds,
            adj_list=adj,
            eps=eps,
            alpha=alpha,
            num_swap=num_swap,
            products=list(dta['name'].unique())
        )

        return b3dv_policy


    def select_action(
            self,
            state: dict,
            vals: dict,
            cands: dict,
            max_slots: int,
            eps: float,
            alpha: float,
            num_swap: int,
    ) -> dict:
        action = {}
        allocated = 0

        cand_vals = {}
        val_mean = np.mean(list(vals.values()))
        for c, edge in cands.items():
            if c not in vals:
                v = val_mean
            else:
                v = vals[c]

            cand_vals[c] = Bayes3dv.get_weighted_payoff_score(alpha, v, edge)

        # sort payoff values: Descending
        cand_vals = {k: v for k, v in sorted(cand_vals.items(), key=lambda item: item[1], reverse=True)}


        heuristic = HeuristicCandidateSearch(vals=cand_vals, presorted=True)
        while allocated < max_slots:
            p_prime = heuristic.greedy_search(action, state)
            budget = max_slots - allocated
            q_prime, remain = Bayes3dv.decay_quantity(budget=budget)
            if p_prime is None:
                # reached end of candidate list
                break
            else:
                action[p_prime] = q_prime
                allocated += q_prime

        return action
