import numpy as np
import pandas as pd
import os
import json
import math

from policy.bayes_3dv import Bayes3dv
from states import DisplayState
from policy.rbp_agent import HeuristicCandidateSearch


class NoClusteringNoBayesNoSearch(Bayes3dv):
    pass

class NoBayesNoSearch(Bayes3dv):
    pass

class NoClusteringNoSearch(Bayes3dv):
    pass

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


